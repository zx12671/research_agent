# -*- coding: utf-8 -*-
"""
e2e_agentic_ef.py: 把"证据前移(ef, top10+整档回捞)"接入 exp1_agentic_rag 生产链路。

背景
----
上一轮已在"轻量直答 LLM"口径下定位 ef 降分根因 = 上下文稀释(DISTRACT)：
  - base 证据已够/已能答好的题，批量回捞低相关碎片反而丢要点（降分）；
  - base 证据不足的题(缺口型)，回捞补上关键证据则升分。
本轮把 ef 真正接到生产 agentic pipeline（TaskAnalyzer→Planner→Organizer→Solver）上，
验证在 完整 agentic 链路 下，ef 是仍复现轻量直答的降分/升分规律，还是被 organizer/solver
的"证据组织+原文保留"吸收掉（即生产链路比轻量直答更稳健）。

方法
----
- 不改生产代码。通过自定义 ExternalRetriever 在"稳定知识接口"层注入回捞块：
    AgenticRAGEngine.answer()
        → EFExternalRetriever.retrieve(q, k=10)   # top10 + 整档回捞(同 ef 口径)
        → pre_retrieved_chunks (含回捞块)
        → AdaptiveAgenticPipeline.run()            # 其余全走生产逻辑
- 与生产一致：外部检索固定 k=10，下游 organizer/solver 全套保留。
- 对照：base = 原 engine（纯 top10）；ef = 加了回捞块的 engine。
- 复用 _diag_degrade_mechanism 的 FOCUS 6 题（3 降分 + 3 对照），保证可比。
  每条件跑 1 次完整 agentic（内部已含 analyzer/planner/decide/verify/solver 多轮 LLM）。
"""
import os, re, sys, json, time, statistics
import csv
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)
_LINS_MAIN = os.path.abspath(os.path.join(_PJ, "..", "LINS-main"))
if _LINS_MAIN not in sys.path:
    sys.path.insert(0, _LINS_MAIN)

from experiments.config import DEEPSEEK_KEY, LLM_NAME, INDUSTRYBENCH_CSV, register_paths
register_paths()

from experiments.exp1_agentic_rag import (
    AgenticRAGEngine,
    IndustrialRetriever,
    RetrievedDocument,
)
from metrics.industrybench_scorer import RuleBasedScorer

CSV = INDUSTRYBENCH_CSV or os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
MAX_PER_DOC = 50
JT_RANK_TH = 0.02

# 与 _diag_degrade_mechanism 完全一致：q 前缀 → (标志, 备注)
FOCUS = {
    "在相同体积下，纤维滤料与传统滤砂相比": ("DEGRADE", "光纤滤料截污"),
    "在污水池专用聚脲防水防腐涂料的施工中": ("DEGRADE", "聚脲防水涂料"),
    "插座端子在冲压成型后、电镀前的质量控制": ("DEGRADE", "插座端子"),
    "在切削塑性材料如不锈钢或铝合金时": ("CTRL_UP", "切削鳞片"),
    "在安装新45型电压表时": ("CTRL_UP2", "电压表"),
    "桥式整流器规格标识中的IF(av)参数": ("CTRL_UP3", "桥式整流器"),
}


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def _content(c):
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "") or ""


class EFExternalRetriever(IndustrialRetriever):
    """
    覆盖 external retriever：top10 + 整档回捞（与 ef 口径一致）。
    回捞块以 RetrievedDocument 追加进 result.documents → 后续 pre_retrieved_chunks
    包含它们 → organizer/solver 全部可见。citation 带 *ef* 标记便于归因。
    """
    def retrieve(self, query, k=None):
        result = super().retrieve(query, k)
        if not result or not result.documents:
            return result
        # 收集已出现的 top10 doc + content，避免回捞重复
        top_docs = set()
        top_contents = set()
        for d in result.documents:
            if d.document_id:
                top_docs.add(d.document_id)
            if d.content:
                top_contents.add(d.content)
        # 底层 OpenDomainRetriever 提供 chunks 字典
        chunks = getattr(self._retriever, "chunks", None)
        if not chunks:
            return result
        # 整档回捞：对每个 top10 doc 的其余块按 jt(query,块) 排序 top50, 过滤 th
        next_rank = max((d.rank for d in result.documents), default=0) + 1
        added = []
        for d in result.documents:
            did = d.document_id or ""
            if not did:
                continue
            same = []
            for cid, data in chunks.items():
                dd = data.get("document_id") or data.get("doc_id") or ""
                if dd == did:
                    same.append((cid, data))
            same.sort(key=lambda x: x[1].get("chunk_index", 0))
            scored = []
            for cid, data in same:
                txt = _content(data)
                if not txt or txt in top_contents:
                    continue
                scored.append((jt(query, txt), txt, cid))
            scored.sort(key=lambda x: x[0], reverse=True)
            for s, txt, cid in scored:
                if s < JT_RANK_TH:
                    continue
                top_contents.add(txt)
                added.append(RetrievedDocument(
                    chunk_id=cid,
                    document_id=did,
                    source=getattr(d, "source", ""),
                    content=txt,
                    score=round(s, 4),
                    rank=next_rank,
                    industry=getattr(d, "industry", ""),
                    capability=getattr(d, "capability", ""),
                    citation=f"[{next_rank}*ef]",
                ))
                next_rank += 1
                if len(added) >= MAX_PER_DOC * 4:  # 上限保护，防止某文档超大
                    break
            if len(added) >= MAX_PER_DOC * 4:
                break
        if added:
            result.documents.extend(added)
            # 同步辅助列表，保持 RetrievalResult 数据一致性
            for d in added:
                result.chunk_ids.append(d.chunk_id)
                result.scores.append(d.score)
                result.sources.append(d.source)
                result.citations.append(d.citation or f"[{d.rank}]")
        return result


class EFAgenticRAGEngine(AgenticRAGEngine):
    """生产 AgenticRAGEngine 的仅替换外部检索层版本：ef = top10 + 回捞。"""
    def _init_external_retriever(self):
        try:
            r = EFExternalRetriever(corpus_dir=self.corpus_dir)
            print("[EFAgenticRAGEngine] External retriever (top10 + 全档回捞) ready")
            return r
        except Exception as e:
            print(f"[EFAgenticRAGEngine] WARNING: EF external retriever init failed: {e}")
            return None


def _load_rows():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick_rows():
    rows_all = _load_rows()
    out = {}
    # 直接从 CSV 按前缀精确匹配（与轻量直答的 FOCUS 保持一致）
    for pre, (flag, note) in FOCUS.items():
        match = next((x for x in rows_all if (x.get("question") or "").startswith(pre)), None)
        if match is not None:
            out[pre] = (flag, note, match)
    return out


def main():
    selected = pick_rows()
    print(f"e2e agentic 链路验证 ef: N={len(selected)}题 x {2}条件（每条件 1 次完整 agentic）\n")

    # 生产基线 engine（base，纯 top10）
    base_engine = AgenticRAGEngine(
        deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False,
    )
    # ef engine（top10 + 回捞）
    ef_engine = EFAgenticRAGEngine(
        deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False,
    )
    scorer = RuleBasedScorer()

    report = {}
    for pre, (flag, note, row_m) in selected.items():
        q = row_m.get("question", "")
        ref = row_m.get("answer", "")
        print(f"\n===== [{flag}] {note} =====")

        # ---- base（生产原样）----
        t0 = time.time()
        ans_b, ret_b = base_engine.answer(q, retrieval_k=10)
        tb = time.time() - t0
        sc_b = scorer.rule_based_score(q, ref, ans_b)
        n_b = len(ret_b.documents) if ret_b else 0

        # ---- ef（top10+回捞）----
        t0 = time.time()
        ans_e, ret_e = ef_engine.answer(q, retrieval_k=10)
        te = time.time() - t0
        sc_e = scorer.rule_based_score(q, ref, ans_e)
        n_e = len(ret_e.documents) if ret_e else 0

        # 统计回捞块数 (citation 形如 "[13*ef]"，以 "*ef" 为标记)
        n_extra = sum(1 for d in (ret_e.documents if ret_e else [])
                      if "*ef" in (d.citation or ""))

        print(f"  base: score={sc_b} 证据数={n_b} ({tb:.0f}s)")
        print(f"  ef  : score={sc_e} 证据数={n_e} (含回捞 {n_extra}) ({te:.0f}s)")

        report[pre] = {
            "flag": flag, "note": note, "q": q[:40],
            "base": {"score": sc_b, "n_evidence": n_b, "time_s": round(tb, 1)},
            "ef": {"score": sc_e, "n_evidence": n_e, "n_extra_backfill": n_extra, "time_s": round(te, 1)},
            "delta": round(sc_e - sc_b, 1),
            "base_answer": ans_b[:500],
            "ef_answer": ans_e[:500],
        }
        # 简单打印答案首行便于人工核验
        print(f"  base_ans: {ans_b[:120]!r}")
        print(f"  ef_ans  : {ans_e[:120]!r}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("汇总（agentic 生产链路）")
    print("=" * 70)
    agg_count = {"UP": 0, "DOWN": 0, "SAME": 0}
    for pre, r in report.items():
        tag = "涨↑" if r["delta"] > 0 else ("跌↓" if r["delta"] < 0 else "持平—")
        if r["delta"] > 0: agg_count["UP"] += 1
        elif r["delta"] < 0: agg_count["DOWN"] += 1
        else: agg_count["SAME"] += 1
        print(f"  [{r['flag']:<8}] {r['note']:<10} base={r['base']['score']} "
              f"ef={r['ef']['score']} delta={tag} "
              f"证据 {r['base']['n_evidence']}->{r['ef']['n_evidence']} "
              f"(回捞{r['ef']['n_extra_backfill']})")
    print(f"\n  涨/跌/持平 = {agg_count['UP']}/{agg_count['DOWN']}/{agg_count['SAME']}")

    out = os.path.join(_LINS, "results", "e2e_agentic_ef.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"agg": agg_count, "rows": report}, f, ensure_ascii=False, indent=2)
    print("\n已写:", out)


if __name__ == "__main__":
    main()
