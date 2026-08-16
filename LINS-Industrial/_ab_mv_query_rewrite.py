# -*- coding: utf-8 -*-
"""
_ab_mv_query_rewrite.py — 语义级多视角 Query 改写的三臂 A/B（完整 AdaptiveAgenticPipeline + 句覆盖口径）。

目标（P1：补全度主杠杆）
------------------------
验证"语义级多视角 query 改写"在**完整 AdaptiveAgenticPipeline** 上对证据池句覆盖率与
最终答案质量的真实增益。三臂：

    base       = 完整 pipeline，external 检索为原样 single dense（生产现状）
    mv_dense   = 多视角改写 → 各视角 dense retrieve → fusion 融合
    mv_hybrid  = 多视角改写 → 各视角 hybrid(dense+BM25jieba) → fusion 融合

注入方式（与 _e2e_agentic_ef.py 同构，不改生产）：
    自定义 MVExternalRetriever(IndustrialRetriever) 覆写 retrieve()，在"稳定知识接口"层
    注入多视角融合结果 → AgenticRAGEngine.answer() 全链路消费该证据池。
    三臂共用 AgenticRAGEngine（完整 analyzer/planner/organize/reason/verify/solver）。

口径（多口径并列，句覆盖为主）：
    主口径 = retrieval.recall_metrics.sent_coverage（证据池 = answer() 返回的 retrieval_result.documents，
            即 external 注入喂给 agent 的唯一证据源，可复现）。
    交叉参照 = 整段 IoU hit@k；最终答案质量（--real 下用 RuleBasedScorer）。

LLM 双模式（复用 _system_flow_quant.build_llm）：
    --mock（默认）: pipeline 用 MockLLM + 改写器 mock 确定性视角，零成本，专测证据池句覆盖传导。
    --real       : pipeline 用真实 DeepSeek + 改写器真实 LLM，测最终答案质量（建议 --max-q 限题）。

用法（LINS-Industrial 下）：
    python _ab_mv_query_rewrite.py --n 30 --seed 7            # mock 全量
    python _ab_mv_query_rewrite.py --real --max-q 8            # 真实子集
"""
import os
import csv
import sys
import json
import time
import random
import argparse
import statistics
from collections import Counter, defaultdict
from typing import Optional

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

from experiments.config import register_paths
register_paths()

from experiments.exp1_agentic_rag import (
    AgenticRAGEngine, IndustrialRetriever, RetrievedDocument, RetrievalResult,
)
from retrieval.recall_metrics import (
    sent_coverage, doc_hit_position, get_content, coverage_summary,
)

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
DOC_IOU_TH = 0.20
TOPK = [1, 3, 5, 10]
COV_KS = [10, 20, 50]

# ----------------------------------------------------------------------
# 多视角改写 external retriever（在稳定知识接口层注入多视角融合）
# ----------------------------------------------------------------------
class MVExternalRetriever(IndustrialRetriever):
    """
    覆写 retrieve()：多视角改写 → 各视角 dense/hybrid 检索 → fusion 融合。
    view_mode: "dense"(纯 dense) / "hybrid"(dense + BM25jieba, sparse_w=0.2)。
    """

    def __init__(self, rewriter=None, corpus_dir=None, retriever_k=10,
                 view_mode="dense", use_ked=False, topk_per_view=20):
        super().__init__(corpus_dir=corpus_dir, retriever_k=retriever_k)
        self.rewriter = rewriter
        self.view_mode = view_mode
        self.use_ked = use_ked
        self.topk_per_view = topk_per_view

    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """多视角改写 + 分视角检索 + fusion 融合，返回 base 同构的 RetrievalResult。"""
        self._ensure_initialized()
        if k is None:
            k = self.retriever_k
        result = RetrievalResult(query=query)
        if not self._initialized or self._retriever is None:
            return result

        start_time = time.time()
        views = [query]
        if self.rewriter is not None:
            try:
                views = self.rewriter.rewrite(query) or [query]
            except Exception as e:
                print(f"  [WARN] 改写器失败，退化单查询: {e}")
                views = [query]

        fusion = self._retriever._fusion_merge

        per_view_results = []
        for v in views[:4]:
            try:
                if self.view_mode == "hybrid":
                    rr = self._retriever.hybrid_retrieve(
                        v, k=self.topk_per_view, use_ked=self.use_ked,
                        use_multi_query=False, use_sparse=True,
                        sparse_pool=50, dense_weight=0.5, sparse_weight=0.2,
                    )
                else:
                    rr = self._retriever.retrieve(v, k=self.topk_per_view,
                                                  use_ked=self.use_ked)
                per_view_results.append(rr)
            except Exception as e:
                print(f"  [WARN] 视角检索失败 '{v[:30]}...' : {e}")

        if not per_view_results:
            return result

        # 融合
        merged = fusion(per_view_results, k)
        merged.query = query
        result.query_expanded = (
            f"mv({','.join(views[:4])})" if len(views) > 1 else "single"
        )
        result.timing_ms = (time.time() - start_time) * 1000

        # 转成 base 同构的 RetrievedDocument 列表（供 _DocWrapper 消费）
        for chunk in getattr(merged, "chunks", []):
            doc = RetrievedDocument(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
                content=chunk.content,
                score=chunk.score,
                rank=getattr(chunk, "rank", 0),
                industry=getattr(chunk, "industry", ""),
                capability=getattr(chunk, "capability", ""),
                citation=f"[{getattr(chunk, 'rank', 0)}]",
            )
            result.documents.append(doc)
            result.chunk_ids.append(chunk.chunk_id)
            result.scores.append(chunk.score)
            result.sources.append(chunk.source)
            result.citations.append(doc.citation)

        return result


# ----------------------------------------------------------------------
# 三臂 engine：用 MVExternalRetriever 覆盖 external 检索
# ----------------------------------------------------------------------
class _EngineFactory(AgenticRAGEngine):
    """按臂注入不同 external retriever 的引擎。"""
    _factory_mode = "base"

    def _init_llm(self):
        # 让 pipeline 的 analyzer/planner/solver 也能用传入的 mock/real llm，
        # 而非硬编码真实 DeepSeek —— 保证 mock 模式零成本、确定性。
        if getattr(self, "_pipeline_llm", None) is not None:
            self._llm_client = self._pipeline_llm
            return
        super()._init_llm()

    def _init_external_retriever(self):
        if self._factory_mode in ("mv_dense", "mv_hybrid"):
            return MVExternalRetriever(
                rewriter=getattr(self, "_mv_rewriter", None),
                corpus_dir=self.corpus_dir,
                view_mode="dense" if self._factory_mode == "mv_dense" else "hybrid",
                use_ked=getattr(self, "_mv_use_ked", False),
                topk_per_view=20,
            )
        return super()._init_external_retriever()


def _make_engine(mode, llm, deepseek_key, model_name):
    """构造对应臂的 engine。mock 时改写器走确定性视角；real 时改写器走真实 LLM。"""
    from agentic.query_rewriter import SemanticQueryRewriter

    is_mock = getattr(llm, "mode", "mock") == "mock"
    rewriter = SemanticQueryRewriter(
        llm_client=(None if is_mock else llm),
        model_name=model_name, num_views=3,
        mock=is_mock,
    )
    eng = _EngineFactory(deepseek_key=deepseek_key, model_name=model_name,
                         verbose=False)
    eng._factory_mode = mode
    eng._mv_rewriter = rewriter
    # mock 时让 pipeline 的 analyzer/planner/solver 也走 MockLLM（零成本、确定性）；
    # real 时透传真实 llm 给 pipeline。
    eng._pipeline_llm = llm if is_mock else None
    # AgenticRAGEngine 初始化时已建 base external retriever；这里用多视角版覆盖
    eng._external_retriever = eng._init_external_retriever()
    return eng


# ----------------------------------------------------------------------
# 样本与评估
# ----------------------------------------------------------------------
def sample_rows(n, seed):
    random.seed(seed)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    s = []
    for cap, lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP, len(lst)))

    return s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    sample = sample_rows(args.n, args.seed)
    if args.max_q:
        sample = sample[:args.max_q]
    print(f"[样本] {len(sample)} 题 | cap: {dict(Counter(r.get('capability') for r in sample))} | real={args.real}")

    from experiments.config import DEEPSEEK_KEY, LLM_NAME
    from _system_flow_quant import build_llm

    llm = build_llm(real=args.real)
    print(f"[LLM] mode={getattr(llm, 'mode', 'mock')}")

    arms = ["base", "mv_dense", "mv_hybrid"]
    engines = {m: _make_engine(m, llm, DEEPSEEK_KEY or "mock", LLM_NAME) for m in arms}

    cov = {m: {k: [] for k in COV_KS} for m in arms}
    agg = {m: {"n": 0, "hit": {kk: 0 for kk in TOPK}} for m in arms}
    cap10 = {m: defaultdict(list) for m in arms}
    score_rows = {m: [] for m in arms}

    scorer = None
    if args.real:
        from metrics.industrybench_scorer import RuleBasedScorer
        scorer = RuleBasedScorer()

    t0 = time.time()
    for idx, row in enumerate(sample, 1):
        q = row.get("question", "")
        kt = row.get("knowledge_text", "")
        cap = row.get("capability") or ""
        ref = row.get("answer", "")

        for m in arms:
            ans, ret = engines[m].answer(q, retrieval_k=10)
            ev = list(getattr(ret, "documents", []) or [])
            for k in COV_KS:
                cov[m][k].append(sent_coverage(kt, ev[:k])[0])
            c10 = cov[m][10][-1]
            cap10[m][cap].append(c10)
            if ev:
                agg[m]["n"] += 1
                hr = doc_hit_position(ev, kt, DOC_IOU_TH)
                for kk in TOPK:
                    if hr is not None and hr <= kk:
                        agg[m]["hit"][kk] += 1
            if scorer is not None and ref:
                score_rows[m].append(scorer.rule_based_score(q, ref, ans))

        if idx % 5 == 0 or idx == len(sample):
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    # ---- 主口径：证据池句覆盖率 ----
    print("\n[主口径] 证据池句覆盖率 (入口 x K; mean/med/cov>=50%)")
    print(f"{'臂':<12}" + "".join(f"{('K='+str(k)):>26}" for k in COV_KS))
    for m in arms:
        line = f"{m:<12}"
        for k in COV_KS:
            s = coverage_summary(cov[m][k])
            line += f"{s['mean']:.3f}/{s['median']:.3f}/{s['good_rate']:.0%}".rjust(26)
        print(line)

    print("\n[差分] cov@10 均值 (mv_vs_base)")
    for m in ("mv_dense", "mv_hybrid"):
        b = statistics.mean(cov["base"][10]); v = statistics.mean(cov[m][10])
        print(f"  {m:<10} base={b:.4f} {m}={v:.4f} delta={v-b:+.4f}")

    print("\n[交叉] 整段 IoU hit@k (证据池 top-10)")
    print(f"{'臂':<12}" + "".join(f"{('hit@'+str(kk)):>10}" for kk in TOPK) + f"{'n':>6}")
    for m in arms:
        a = agg[m]; line = f"{m:<12}"
        for kk in TOPK:
            line += f"{(a['hit'][kk]/a['n'] if a['n'] else 0):>10.0%}"
        line += f"{a['n']:>6}"
        print(line)

    if scorer is not None:
        print("\n[答案质量 --real] RuleBasedScorer mean")
        for m in arms:
            if score_rows[m]:
                print(f"  {m:<10} n={len(score_rows[m])} mean={statistics.mean(score_rows[m]):.3f}")

    out = os.path.join(_LINS, "results",
                       f"ab_mv_query_rewrite_{'real' if args.real else 'mock'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "real": bool(args.real),
            "cov": {m: {str(k): cov[m][k] for k in COV_KS} for m in arms},
            "agg": {m: {kk: v for kk, v in agg[m].items() if kk != "hit"} for m in arms},
            "score_rows": score_rows if scorer is not None else {},
            "cap10_mean": {m: {c: round(statistics.mean(v), 4) for c, v in cap10[m].items()} for m in arms},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[已写] {out}")


if __name__ == "__main__":
    main()
