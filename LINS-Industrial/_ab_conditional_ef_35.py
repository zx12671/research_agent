# -*- coding: utf-8 -*-
"""
_ab_conditional_ef_35.py: 条件式回捞 vs 无条件式回捞的 35 题量化（agentic 生产链路）。

背景 / 动机
----------
docs/agentic_ef_end_to_end.md 的"建议A + 后续量化项"结论：
  - ef（top10+整档回捞）在 agentic 生产链路下 0 降分、1/6 升分、5/6 持平；
  - 在 agentic 链路里固定 top10 已覆盖绝大多数判定事实，ef 的主要价值是"兜底缺口型升分"；
  - 因此应**仅对判定为缺口型的问题启用回捞**，对饱和型只保留 top10，避免无效 token 开销；
  - 后续用量化项：在更大样本（≥30）上按 agentic 口径复核"缺口型升分率"，并测量 ef 引入的 token/时延成本。

本脚本把该"后续量化项"落实为 35 题 × 三条件（base / ef_无条件下回捞 / ef_条件式回捞）的对比。

方法（最小侵入，不改生产代码）
----------------------------
- 复用 exp1_agentic_rag 的生产 AgenticRAGEngine，仅在其"稳定知识接口"层替换外部检索器：
    * base    = 原 engine：OpenDomainRetriever 纯 top10
    * ef_unc  = EFExternalRetriever：top10 + 无条件整档回捞（同 _e2e_agentic_ef 口径）
    * ef_ctr  = CtrEFExternalRetriever：top10 + 仅当"检索层自足度信号=缺口型"时才回捞
- 门控信号（LLM 无关、纯检索层、在线可得）：
    * 对 base top10 的 bge-cos score 计算
      mu_top = mean(top[:3])   # 命中强度
      mu_tail= mean(top[-3:])  # 长尾水平
      headroom = mu_top - mu_tail  # 证据是否集中在顶层
    * 缺口型（值得回捞）：mu_top < TH_SAT（base 顶层证据弱 → 大概率缺关键证据）
    * 饱和型（跳过回捞）：mu_top >= TH_SAT（base 顶层证据已足够强 → 不再注入低相关碎片）
- 35 题池：直接读取 results/probe_evidence_forward_e2e_35.json 的 q 文本，保证与既有
  evidence-forward 35 题结论可比（7 能力 × 5 题）。
- 评分：RuleBasedScorer.rule_based_score（与生产一致）。
- token 代理：evidence 输入 token ≈ sum(len(content)//2)，用于对比条件式回捞省下的证据 tokens。

运行（约 35 题 × 3 次完整 agentic，LLM 调用量大，建议后台跑并盯日志）：
    cd LINS-Industrial
    python _ab_conditional_ef_35.py
"""
import os, re, json, csv, time, statistics

_LINS = os.path.dirname(os.path.abspath(__file__))
import sys
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import DEEPSEEK_KEY, LLM_NAME, INDUSTRYBENCH_CSV, register_paths
register_paths()

from experiments.exp1_agentic_rag import (
    AgenticRAGEngine,
    IndustrialRetriever,
    RetrievalResult,
    RetrievedDocument,
)
from metrics.industrybench_scorer import RuleBasedScorer

# ---- 预注册（PREREG）参数 ----
TH_SAT = 0.25       # 缺口阈值：base top3 平均 bge-cos < 0.25 → 缺口型，触发回捞
MAX_PER_DOC = 50    # ef 整档回捞每档封顶（与 _e2e_agentic_ef 一致）
JT_RANK_TH = 0.02   # ef 每档相似度过滤下限（与 _e2e_agentic_ef 一致）

E2E35 = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")
OUT = os.path.join(_LINS, "results", "ab_conditional_ef_35.json")


# ---------- 检索层自足度信号（LLM 无关） ----------
def self_sufficiency_signal(scores):
    """由 base top10 的 score 计算门控信号。scores 为按 rank 升序的 bge-cos 列表。"""
    s = sorted([float(x) for x in scores], reverse=True)[:10]
    if not s:
        return {"mu_top": 0.0, "mu_tail": 0.0, "headroom": 0.0, "max_s": 0.0, "n": 0}
    n = len(s)
    ktop = min(3, n)
    ktail = min(3, n)
    mu_top = sum(s[:ktop]) / ktop
    mu_tail = sum(s[n - ktail:]) / ktail
    return {
        "mu_top": round(mu_top, 4),
        "mu_tail": round(mu_tail, 4),
        "headroom": round(mu_top - mu_tail, 4),
        "max_s": round(max(s), 4),
        "n": n,
    }


def is_gap_saturated(sig, th=TH_SAT):
    """返回 (is_gap: bool, label: str)。缺口型 True -> 回捞。"""
    gap = sig["mu_top"] < th
    return gap, ("GAP" if gap else "SAT")


# ---------- 无条件回捞检索器（与 _e2e_agentic_ef 的 EFExternalRetriever 一致） ----------
@staticmethod
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


@staticmethod
def _jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def _content(c):
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "") or ""


class EFExternalRetriever(IndustrialRetriever):
    """top10 + 无条件整档回捞（与既有 ef 口径完全一致）。"""

    def retrieve(self, query, k=None):
        result = super().retrieve(query, k)
        if not result or not result.documents:
            return result
        added = self._backfill(result, query)
        self._attach(result, added)
        return result

    def _backfill(self, result, query):
        top_docs = set()
        top_contents = set()
        for d in result.documents:
            if d.document_id:
                top_docs.add(d.document_id)
            if d.content:
                top_contents.add(d.content)
        chunks = getattr(self._retriever, "chunks", None)
        if not chunks:
            return []
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
                scored.append((_jt(query, txt), txt, cid))
            scored.sort(key=lambda x: x[0], reverse=True)
            for s, txt, cid in scored:
                if s < JT_RANK_TH:
                    continue
                top_contents.add(txt)
                added.append(RetrievedDocument(
                    chunk_id=cid, document_id=did,
                    source=getattr(d, "source", ""),
                    content=txt, score=round(s, 4), rank=next_rank,
                    industry=getattr(d, "industry", ""),
                    capability=getattr(d, "capability", ""),
                    citation=f"[{next_rank}*ef]",
                ))
                next_rank += 1
                if len(added) >= MAX_PER_DOC * 4:
                    break
            if len(added) >= MAX_PER_DOC * 4:
                break
        return added

    def _attach(self, result, added):
        if added:
            result.documents.extend(added)
            for d in added:
                result.chunk_ids.append(d.chunk_id)
                result.scores.append(d.score)
                result.sources.append(d.source)
                result.citations.append(d.citation or f"[{d.rank}]")


class CtrEFExternalRetriever(EFExternalRetriever):
    """top10 + 条件式整档回捞：仅当检索层自足度=缺口型才回捞，饱和型保持纯 top10。
    门控在 base top10 之后、回捞之前执行，纯检索层信号、零额外 LLM。"""

    def retrieve(self, query, k=None):
        # 先取纯 base top10（最低层检索，不经 EF 的无条件回捞），计算门控信号
        result = IndustrialRetriever.retrieve(self, query, k)
        if not result or not result.documents:
            return result
        sig = self_sufficiency_signal(result.scores)
        gap, label = is_gap_saturated(sig, TH_SAT)
        result.__gate__ = {"gap": gap, "label": label, "signal": sig}
        if not gap:
            return result              # 饱和型：保持纯 top10，不注入回捞碎片
        # 缺口型：走 EF 的无条件回捞逻辑（复用 EFExternalRetriever.retrieve）
        return EFExternalRetriever.retrieve(self, query, k)


class EFAgenticRAGEngine(AgenticRAGEngine):
    """仅替换外部检索层：无条件回捞。"""
    def _init_external_retriever(self):
        try:
            r = EFExternalRetriever(corpus_dir=self.corpus_dir)
            return r
        except Exception as e:
            print(f"[EFEngine] WARNING: init failed: {e}")
            return None


class CEFAgenticRAGEngine(AgenticRAGEngine):
    """仅替换外部检索层：条件式回捞（门控）。"""
    def _init_external_retriever(self):
        try:
            r = CtrEFExternalRetriever(corpus_dir=self.corpus_dir)
            return r
        except Exception as e:
            print(f"[CEFEngine] WARNING: init failed: {e}")
            return None


# ---------- 35 题池（与既有 evidence-forward 35 题一致，保证可比） ----------
def load_35_pool():
    with open(E2E35, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows") or data
    pool = []
    for r in rows:
        q = (r.get("q") or "").strip()
        if q:
            pool.append({"q": q, "cap": r.get("cap", ""), "cls": r.get("cls", "")})
    # 去重 + 保序
    seen, out = set(), []
    for item in pool:
        if item["q"] in seen:
            continue
        seen.add(item["q"])
        out.append(item)
    return out


def tokens_of_docs(docs):
    return sum(len(_content(d)) // 2 for d in docs)


def n_extra(docs):
    return sum(1 for d in docs if "*ef" in (d.citation or ""))


def score_est_tokens(ans):
    return len(ans or "") // 2


def main():
    pool = load_35_pool()
    print(f"条件式 vs 无条件 ef | 题数={len(pool)} x 3 条件（agentic 生产链路）")
    print(f"门控: GAP iff mu_top < {TH_SAT}（检索层自足度）\n")

    base_engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    ef_engine = EFAgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    cef_engine = CEFAgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)

    scorer = RuleBasedScorer()
    refs = _load_refs()

    report, results = [], {"rows": []}

    for idx, item in enumerate(pool, 1):
        q = item["q"]
        ref = refs.get(q, "")
        print(f"\n[{idx}/{len(pool)}] ({item['cap']}) {q[:30]}...")

        # ---- base（生产原样 top10）----
        t0 = time.time()
        ans_b, ret_b = base_engine.answer(q, retrieval_k=10)
        tb = time.time() - t0
        sc_b = scorer.rule_based_score(q, ref, ans_b)
        tc_b = tokens_of_docs(ret_b.documents if ret_b else []) + score_est_tokens(ans_b)

        # ---- ef 无条件（top10 + 整档回捞）----
        t0 = time.time()
        ans_u, ret_u = ef_engine.answer(q, retrieval_k=10)
        tu = time.time() - t0
        sc_u = scorer.rule_based_score(q, ref, ans_u)
        nu = n_extra(ret_u.documents if ret_u else [])
        tc_u = tokens_of_docs(ret_u.documents if ret_u else []) + score_est_tokens(ans_u)

        # ---- ef 条件式（门控回捞）----
        t0 = time.time()
        ans_c, ret_c = cef_engine.answer(q, retrieval_k=10)
        tc = time.time() - t0
        sc_c = scorer.rule_based_score(q, ref, ans_c)
        nc = n_extra(ret_c.documents if ret_c else [])
        sig = getattr(ret_c, "__gate__", {}).get("signal", {})
        gap, label = is_gap_saturated(sig or {"mu_top": 0.0}, TH_SAT)
        tc_ce = tokens_of_docs(ret_c.documents if ret_c else []) + score_est_tokens(ans_c)

        n_b = len(ret_b.documents) if ret_b else 0
        n_u = len(ret_u.documents) if ret_u else 0
        n_c = len(ret_c.documents) if ret_c else 0

        print(f"  base sc={sc_b} 证据={n_b} tok~{tc_b} ({tb:.0f}s)")
        print(f"  ef_u  sc={sc_u} 证据={n_u}(回捞{nu}) tok~{tc_u} ({tu:.0f}s)")
        print(f"  ef_c  sc={sc_c} 证据={n_c}(回捞{nc}) tok~{tc_ce} gate={label} mu_top={sig.get('mu_top') if sig else '?'} ({tc:.0f}s)")

        results["rows"].append({
            "q": q[:40], "cap": item["cap"], "cls": item["cls"],
            "gate_signal": sig, "gate_label": label, "gate_gap": gap,
            "base": {"score": sc_b, "n_evi": n_b, "tokens": int(tc_b), "time_s": round(tb, 1)},
            "ef_uncond": {"score": sc_u, "n_evi": n_u, "n_extra": nu, "tokens": int(tc_u), "time_s": round(tu, 1)},
            "ef_cond": {"score": sc_c, "n_evi": n_c, "n_extra": nc, "tokens": int(tc_ce), "time_s": round(tc, 1)},
        })

    # ---- 汇总 ----
    rows = results["rows"]
    def cmp(a, b):  # 分数比较：高更好
        return 1 if b > a else (-1 if b < a else 0)

    agg = {
        "vs_base": {
            "ef_uncond_updown_same": [sum(1 for r in rows if cmp(r["base"]["score"], r["ef_uncond"]["score"]) > 0),
                                       sum(1 for r in rows if cmp(r["base"]["score"], r["ef_uncond"]["score"]) < 0),
                                       sum(1 for r in rows if cmp(r["base"]["score"], r["ef_uncond"]["score"]) == 0)],
            "ef_cond_updown_same": [sum(1 for r in rows if cmp(r["base"]["score"], r["ef_cond"]["score"]) > 0),
                                     sum(1 for r in rows if cmp(r["base"]["score"], r["ef_cond"]["score"]) < 0),
                                     sum(1 for r in rows if cmp(r["base"]["score"], r["ef_cond"]["score"]) == 0)],
        },
        "backfill_triggers": {
            "ef_uncond": sum(r["ef_uncond"]["n_extra"] for r in rows),
            "ef_cond": sum(r["ef_cond"]["n_extra"] for r in rows),
            "gate_gap_n": sum(1 for r in rows if r["gate_gap"]),
            "gate_sat_n": sum(1 for r in rows if not r["gate_gap"]),
        },
        "token": {
            "base_total": sum(r["base"]["tokens"] for r in rows),
            "ef_uncond_total": sum(r["ef_uncond"]["tokens"] for r in rows),
            "ef_cond_total": sum(r["ef_cond"]["tokens"] for r in rows),
        },
        "time_s": {
            "base_total": round(sum(r["base"]["time_s"] for r in rows), 1),
            "ef_uncond_total": round(sum(r["ef_uncond"]["time_s"] for r in rows), 1),
            "ef_cond_total": round(sum(r["ef_cond"]["time_s"] for r in rows), 1),
        },
    }
    results["agg"] = agg

    print("\n" + "=" * 64)
    print("汇总（agentic 生产链路，N=%d）" % len(rows))
    print("=" * 64)
    print(f"gate: GAP={agg['backfill_triggers']['gate_gap_n']}  SAT={agg['backfill_triggers']['gate_sat_n']}")
    print(f"回捞块: ef_unc={agg['backfill_triggers']['ef_uncond']}  ef_cond={agg['backfill_triggers']['ef_cond']}")
    print(f"证据 tokens: base={agg['token']['base_total']}  ef_unc={agg['token']['ef_uncond_total']}  ef_cond={agg['token']['ef_cond_total']}")
    print(f"总耗时(s): base={agg['time_s']['base_total']}  ef_unc={agg['time_s']['ef_uncond_total']}  ef_cond={agg['time_s']['ef_cond_total']}")
    up, down, same = agg["vs_base"]["ef_cond_updown_same"]
    print(f"vs base 涨/跌/持平:  ef_uncond={agg['vs_base']['ef_uncond_updown_same']}  ef_cond=[{up},{down},{same}]")

    # 分 gate 子组看条件式是否保留缺口型升分
    gap_rows = [r for r in rows if r["gate_gap"]]
    sat_rows = [r for r in rows if not r["gate_gap"]]
    def _updown(grp, key):
        base_s, cond_s = grp["base"]["score"], grp[key]["score"]
        return 1 if cond_s > base_s else (-1 if cond_s < base_s else 0)
    print(f"\n[GAP 子组 n={len(gap_rows)}]  ef_uncond 升/降/平 = "
          f"{[sum(1 for g in gap_rows if _updown(g,'ef_uncond')>0), sum(1 for g in gap_rows if _updown(g,'ef_uncond')<0), sum(1 for g in gap_rows if _updown(g,'ef_uncond')==0)]}")
    print(f"[GAP 子组]        ef_cond  升/降/平 = "
          f"{[sum(1 for g in gap_rows if _updown(g,'ef_cond')>0), sum(1 for g in gap_rows if _updown(g,'ef_cond')<0), sum(1 for g in gap_rows if _updown(g,'ef_cond')==0)]}")
    print(f"[SAT 子组 n={len(sat_rows)}]  ef_cond  升/降/平 = "
          f"{[sum(1 for s in sat_rows if _updown(s,'ef_cond')>0), sum(1 for s in sat_rows if _updown(s,'ef_cond')<0), sum(1 for s in sat_rows if _updown(s,'ef_cond')==0)]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写:", OUT)


def _load_refs():
    """返回支持「截断 q → 完整 CSV question 前缀回填」的 ref 查表。

    背景(见 docs/agentic_conditional_ef_ab.md 根因诊断)：35 题池的 q 是完整 question 的
    [截断短语]（探测阶段为省 token 截断存下），直接 rows.get(q) 因字符串不相等必然 0 命中，
    导致 rule_based_score(q, ref="", ans) 全 0（标签退化）。本函数保留精确匹配，并在 miss 时
    用「q 是某完整 question 的唯一前缀(>=10字)」做回填，命中即补上该题的参考答案。
    """
    rows = {}
    with open(INDUSTRYBENCH_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            if q:
                rows[q] = r.get("answer", "")
    q_full = list(rows.keys())

    class RefDict(dict):
        """dict.get 优先精确；未命中则做截断前缀回填（唯一前缀匹配）取真实 ref。"""
        def get(self, key, default=""):
            v = dict.get(self, key, default)
            if v:
                return v
            if isinstance(key, str) and len(key) >= 10:
                cands = [x for x in q_full if x.startswith(key)]
                if len(cands) == 1:
                    return rows[cands[0]]
            return default
    return RefDict(rows)


if __name__ == "__main__":
    main()
