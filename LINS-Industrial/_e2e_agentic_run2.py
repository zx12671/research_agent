# -*- coding: utf-8 -*-
"""_e2e_agentic_run2.py — 把 run2 两步式检索接入【生产 agentic 链路】并留痕。

背景
----
run2 = 两步式检索（EF 改进版）：① 现有 dense 初步检索取宽池；② 真实 DeepSeek 仅凭
【问题+候选块】评估适配分（绝不碰 GT/答案），取高相关锚块的 document_id 反向回捞同源
文档其余块，再融合排序。已在 `_diag_recall_run2.py` 于检索层证明：cov@10 41.6% vs
best baseline(single/hybrid) 35.7%（+5.9pp），机制 = 补全>排序。

本脚本把 run2 真正接到生产 `AgenticRAGEngine` 的全链路上（analyzer→planner→organizer
→reason→solver，外加 verify/decide 等 advanced 图节点），验证在"组织+推理+多轮 LLM"
真实端到端条件下，run2 检索候选是否仍带来答案分增益（或是否被 organizer/solver 吸收）。

方法（与 _e2e_agentic_ef.py 同构，最小侵入）
----
- 生产 engine：`AgenticRAGEngine`（base，纯 top10）。
- run2 engine：仅覆盖 `_init_external_retriever()` 返回自定义 `Run2ExternalRetriever`，
  其 `retrieve()` 调用底层 `OpenDomainRetriever.two_stage_retrieve(...)` 并把
  `RetrievedChunk` 转为生产 `RetrievedDocument`，citation 带 `*run2` 便于归因回捞。
- 其余全部走生产 `answer(retrieval_k=10)`；external 证据经 SHORTCUT 直接喂给
  retrieve_1（S2-P0 已实现短路），organizer/solver 全套保留。

用法:
  python _e2e_agentic_run2.py --per_cap 2 [--pool 50] [--seed 7]
输出:
  results/e2e_agentic_run2/e2e_run2_{ts}.json  （题级 base/run2 分数 + 证据数 + 答案）
"""
import os, sys, json, time, random, argparse
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiments.config import DEEPSEEK_KEY, LLM_NAME, INDUSTRYBENCH_CSV, register_paths
register_paths()

from experiments.exp1_agentic_rag import (
    AgenticRAGEngine,
    IndustrialRetriever,
    RetrievedDocument,
    RetrievalResult,
)
from metrics.industrybench_scorer import RuleBasedScorer
from _diag_recall_run2 import LLMRelevanceRanker

CSV = INDUSTRYBENCH_CSV or os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
OUT_DIR = os.path.join(_LINS, "results", "e2e_agentic_run2")
CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]


class Run2ExternalRetriever(IndustrialRetriever):
    """生产 external retriever 的 run2 两步式检索版（默认 v4）。

    retrieve() 用底层 OpenDomainRetriever 的 two-stage 检索变体产出候选 top-k（含
    文档级反向补全），再转成生产 RetrievedDocument。二阶段 ranker（真实 DeepSeek）注入，
    打分仅凭【问题+候选块】，不碰 GT/答案。

    version: 选择 two-stage 变体
      - "v1": two_stage_retrieve（回捞块 rel 恒 0，补全进不了 top-k）
      - "v4": two_stage_retrieve_v4（去重到 3 篇文档回捞、anchor_bonus 只给第 1 篇；
              21 题 LLM 检索层 cov@10 43.7%＞v1/v2 41.6%，唯一实质正增益变体）
    """

    def __init__(self, corpus_dir=None, retriever_k=10, ranker=None,
                 pool_k=50, n_anchor=3, version="v4"):
        super().__init__(corpus_dir=corpus_dir, retriever_k=retriever_k)
        self.ranker = ranker
        self.pool_k = pool_k
        self.n_anchor = n_anchor
        self.version = version
        self._last_expanded = ""

    def _call_two_stage(self, query, k):
        if self.version == "v4":
            return self._retriever.two_stage_retrieve_v4(
                query, k=k, pool_k=self.pool_k,
                n_anchor=self.n_anchor, ranker=self.ranker,
            )
        return self._retriever.two_stage_retrieve(
            query, k=k, pool_k=self.pool_k,
            n_anchor=self.n_anchor, ranker=self.ranker,
        )

    def retrieve(self, query, k=None):
        self._ensure_initialized()
        if k is None:
            k = self.retriever_k
        result = RetrievalResult(query=query)
        if not self._initialized or self._retriever is None:
            print("[WARN] 检索器未就绪，返回空结果")
            return result

        st = time.time()
        try:
            rr = self._call_two_stage(query, k)
            self._last_expanded = rr.query_expanded or ""
            result.query_expanded = self._last_expanded
            result.timing_ms = rr.timing_ms
            for chunk in rr.chunks:
                doc = RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    content=chunk.content,
                    score=chunk.score,
                    rank=chunk.rank,
                    industry=chunk.industry,
                    capability=chunk.capability,
                    citation=f"[{chunk.rank}*run2]",
                )
                result.documents.append(doc)
                result.chunk_ids.append(chunk.chunk_id)
                result.scores.append(chunk.score)
                result.sources.append(chunk.source)
                result.citations.append(f"[{chunk.rank}*run2]")
        except Exception as e:
            print(f"[ERROR] run2 检索失败: {e!r}")
        result.timing_ms = (time.time() - st) * 1000
        return result



class Run2AgenticRAGEngine(AgenticRAGEngine):
    """生产 engine 的仅替换外部检索层版本：run2 两步式检索（默认 v4）。"""

    def __init__(self, pool_k=50, n_anchor=3, ranker=None, version="v4", **kwargs):
        self._run2_pool_k = pool_k
        self._run2_n_anchor = n_anchor
        self._run2_ranker = ranker
        self._run2_version = version
        super().__init__(**kwargs)

    def _init_external_retriever(self):
        try:
            r = Run2ExternalRetriever(
                corpus_dir=self.corpus_dir,
                ranker=self._run2_ranker,
                pool_k=self._run2_pool_k,
                n_anchor=self._run2_n_anchor,
                version=self._run2_version,
            )
            print(f"[Run2AgenticRAGEngine] External retriever (run2 two-stage/{self._run2_version}) ready")
            return r
        except Exception as e:
            print(f"[Run2AgenticRAGEngine] WARNING: run2 external retriever init failed: {e!r}")
            return None



def _load_rows():
    import csv
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stratify(rows, per_cap, seed=7):
    random.seed(seed)
    by_cap = {c: [] for c in CAPS}
    for r in rows:
        c = r.get("capability") or ""
        if c in by_cap:
            by_cap[c].append(r)
    sample = []
    for c in CAPS:
        pool = by_cap.get(c, [])
        if pool:
            sample += random.sample(pool, min(per_cap, len(pool)))
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cap", type=int, default=2, help="每 capability 抽样数（题目数=per_cap*7）")
    ap.add_argument("--pool", type=int, default=50, help="run2 第一步候选池宽")
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--version", type=str, default="v4", choices=["v1", "v4"],
                    help="run2 两步式检索变体（生产端到端默认 v4）")
    args = ap.parse_args()

    rows = _load_rows()
    sample = stratify(rows, args.per_cap, seed=args.seed)
    print(f"[样本] {len(sample)} 题; cap 分布={dict(Counter(r['capability'] for r in sample))}")

    ranker = LLMRelevanceRanker(api_key=DEEPSEEK_KEY, model=LLM_NAME)

    base_engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    run2_engine = Run2AgenticRAGEngine(
        deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False,
        pool_k=args.pool, n_anchor=args.n_anchor, ranker=ranker,
        version=args.version,
    )
    scorer = RuleBasedScorer()

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = {}
    t0all = time.time()
    for idx, row in enumerate(sample, 1):
        q = row.get("question", "")
        ref = row.get("answer", "")
        cap = row.get("capability", "")
        if not q or not ref:
            continue
        print(f"\n===== [{idx}/{len(sample)}] {cap} =====")

        ta = time.time()
        ans_b, ret_b = base_engine.answer(q, retrieval_k=10)
        tb = time.time() - ta
        sc_b = scorer.rule_based_score(q, ref, ans_b)
        n_b = len(ret_b.documents) if ret_b else 0

        ta = time.time()
        ans_r, ret_r = run2_engine.answer(q, retrieval_k=10)
        tr = time.time() - ta
        sc_r = scorer.rule_based_score(q, ref, ans_r)
        n_r = len(ret_r.documents) if ret_r else 0
        n_backfill = 0
        if ret_r and getattr(ret_r, "query_expanded", ""):
            import re as _re
            _m = _re.search(r"backfill=(\d+)", ret_r.query_expanded or "")
            if _m:
                n_backfill = int(_m.group(1))

        print(f"  base: score={sc_b} 证据={n_b} ({tb:.0f}s)")
        print(f"  run2: score={sc_r} 证据={n_r} (含补全{n_backfill}) ({tr:.0f}s)")

        report[row.get("id", "")] = {
            "cap": cap, "q": q[:60],
            "base": {"score": sc_b, "n_evidence": n_b, "time_s": round(tb, 1)},
            "run2": {"score": sc_r, "n_evidence": n_r,
                     "n_backfill": n_backfill, "time_s": round(tr, 1)},
            "delta": round(sc_r - sc_b, 1),
            "base_answer": ans_b[:400], "run2_answer": ans_r[:400],
        }
        if idx % max(1, len(sample) // 4) == 0:
            print(f"  .. 进度 {idx}/{len(sample)} (总耗时 {time.time()-t0all:.0f}s)")

    # 汇总
    up = sum(1 for r in report.values() if r["delta"] > 0)
    down = sum(1 for r in report.values() if r["delta"] < 0)
    same = sum(1 for r in report.values() if r["delta"] == 0)
    base_avg = sum(r["base"]["score"] for r in report.values()) / max(1, len(report))
    run2_avg = sum(r["run2"]["score"] for r in report.values()) / max(1, len(report))
    base_ne = sum(r["base"]["n_evidence"] for r in report.values()) / max(1, len(report))
    run2_ne = sum(r["run2"]["n_evidence"] for r in report.values()) / max(1, len(report))
    run2_nb = sum(r["run2"]["n_backfill"] for r in report.values()) / max(1, len(report))

    print("\n" + "=" * 72)
    print("汇总（生产 agentic 链路）：base(top10) vs run2(两步式LLM补全)")
    print("=" * 72)
    for rid, r in report.items():
        tag = "涨↑" if r["delta"] > 0 else ("跌↓" if r["delta"] < 0 else "持平—")
        print(f"  [{r['cap'][:8]:<9}] base={r['base']['score']} run2={r['run2']['score']} "
              f"{tag} 证据 {r['base']['n_evidence']}->{r['run2']['n_evidence']} "
              f"(补全{r['run2']['n_backfill']})")
    print(f"\n  涨/跌/持平 = {up}/{down}/{same}")
    print(f"  平均分: base={base_avg:.2f} -> run2={run2_avg:.2f}  (Δ={run2_avg-base_avg:+.2f})")
    print(f"  平均证据: base={base_ne:.1f} -> run2={run2_ne:.1f} (其中 run2 补全 {run2_nb:.1f})")

    out = os.path.join(OUT_DIR, f"e2e_run2_{args.version}_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "agg": {"up": up, "down": down, "same": same,
                                               "base_avg_score": round(base_avg, 3),
                                               "run2_avg_score": round(run2_avg, 3),
                                               "delta_avg": round(run2_avg - base_avg, 3)},
                   "rows": report}, f, ensure_ascii=False, indent=2)
    print("\n已写:", out)


if __name__ == "__main__":
    main()
