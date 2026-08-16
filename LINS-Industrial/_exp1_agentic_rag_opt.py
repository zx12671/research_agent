# -*- coding: utf-8 -*-
"""
_exp1_agentic_rag_opt.py — exp1_agentic_rag 的优化组 A/B（同题库、同评分、仅检索注入不同）。

对照关系
--------
参照组 = experiments/exp1_agentic_rag.py（生产默认：纯 dense + KED，StrategyPlanner hybrid=False）
优化组 = 本脚本（在"稳定知识接口"层注入三种优化点，其余完全对齐参照组）：

  ① jieba 中文分词接入 _tokenize —— 使 BM25_AVAILABLE=True（已在 retrieval/retriever.py 修复，
    本脚本不重复改，仅保证 hybrid_retrieve(use_sparse=True) 真正走 BM25 分支）。
  ② sparse_weight 默认 0.8→0.2 —— 每视角 hybrid_retrieve 透传 sparse_weight=0.2，
    实现 BM25"只补不扰"（_ab_bm25_jieba_grid 结论里的甜点）。
  ③ 多视角 query 改写 —— SemanticQueryRewriter 依任务生成 2~3 个语义子视角
    （query_rewriter.py），各视角跑 hybrid → _fusion_merge 融合 → 注入 external 检索。

注入机制（与 _ab_mv_query_rewrite.py / _e2e_agentic_ef.py 同构，不改 production）：
    自定义 MVHybridExternalRetriever(IndustrialRetriever) 覆写 retrieve()，AgenticRAGEngine.answer()
    全链路在 analyzer/planner/organize/reason/verify/solver 消费该证据池。
    pipeline 用真实 DeepSeek（与参照组引擎同款初始化），故唯一差异即 external 检索注入。

口径（完全对齐参照组 run_agentic_rag_experiment）：
    数据  load_question_dataset(INDUSTRYBENCH_CSV, num_samples)（同题库，取前 N 行）
    评分  EnhancedScorer.score(...) + evaluate_retrieval_semantic(...)（RuleBased，SV 违规）
    报告  report.md / detailed_results.json / results.csv（字段与参照组逐字段一致）

用法（LINS-Industrial 下）：
    python _exp1_agentic_rag_opt.py --num_samples 30 --run_id exp1_agentic_rag_opt_mvhyb30
    # 可选 --optimization mv_hybrid|hybrid   默认 mv_hybrid（多视角×hybrid×sparse0.2 全量叠加）
"""
import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from experiments.config import register_paths
register_paths()

from experiments.exp1_agentic_rag import (
    AgenticRAGEngine, IndustrialRetriever, RetrievedDocument, RetrievalResult,
    EnhancedScorer, load_question_dataset,
)
from experiments.config import RESULTS_DIR, INDUSTRYBENCH_CSV, DEEPSEEK_KEY, LLM_NAME
from agentic.organizer import EvidenceOrganizer
from experiments.ab_organizer_eval import PassThroughOrganizer


class RerankTruncOrganizer(EvidenceOrganizer):
    """EvidenceOrganizer 子类：execute() 固定开启 L2 词法重排+截断（S4 去噪提纯档）。

    对标 _ab_organize_util.py 的 V3：把词法相关证据置顶并从低分端截断，
    回收"证据在但失焦(证据被分散/顺序靠后 + 噪声稀释)"的 S4→S5 传导缺口。
    检索完全保持参照组 dense/KED 默认，不做 hybrid/mv 污染，纯组织侧对照。
    """

    def execute(self, directive, documents, question="", task_type=None,
                rerank=True, truncate=True, signal="lex", max_chars=6000,
                rerank_weight_lex=0.7, rerank_weight_score=0.3, w_cap=0.3):
        return super().execute(
            directive=directive, documents=documents, question=question,
            task_type=task_type, rerank=rerank, truncate=truncate,
            signal=signal, max_chars=max_chars,
            rerank_weight_lex=rerank_weight_lex,
            rerank_weight_score=rerank_weight_score, w_cap=w_cap,
        )



# ----------------------------------------------------------------------
# 优化的 external retriever（多视角 × hybrid × sparse_weight=0.2 注入）
# ----------------------------------------------------------------------
class MVHybridExternalRetriever(IndustrialRetriever):
    """覆写 retrieve()：多视角改写 → 各视角 hybrid(dense+BM25jieba, sparsew=0.2) → fusion 融合。"""

    def __init__(self, rewriter=None, corpus_dir=None, retriever_k=10,
                 view_mode="hybrid", use_ked=False, topk_per_view=20,
                 sparse_weight=0.2):
        super().__init__(corpus_dir=corpus_dir, retriever_k=retriever_k)
        self.rewriter = rewriter
        self.view_mode = view_mode
        self.use_ked = use_ked
        self.topk_per_view = topk_per_view
        self.sparse_weight = sparse_weight

    def retrieve(self, query, k=None):
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
                        sparse_pool=50, dense_weight=1.0,
                        sparse_weight=self.sparse_weight,
                    )
                else:
                    rr = self._retriever.retrieve(v, k=self.topk_per_view,
                                                  use_ked=self.use_ked)
                per_view_results.append(rr)
            except Exception as e:
                print(f"  [WARN] 视角检索失败 '{v[:30]}...' : {e}")

        if not per_view_results:
            return result

        merged = fusion(per_view_results, k)
        merged.query = query
        result.query_expanded = (
            f"mv_hybrid({','.join(views[:4])})" if len(views) > 1 else "single"
        )
        result.timing_ms = (time.time() - start_time) * 1000

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


class _OptEngineFactory(AgenticRAGEngine):
    """把优化 external retriever 注入到 AgenticRAGEngine 的引擎工厂。"""

    def _init_external_retriever(self):
        return MVHybridExternalRetriever(
            rewriter=getattr(self, "_opt_rewriter", None),
            corpus_dir=self.corpus_dir,
            view_mode=getattr(self, "_opt_view_mode", "hybrid"),
            use_ked=getattr(self, "_opt_use_ked", False),
            topk_per_view=getattr(self, "_opt_topk_per_view", 20),
            sparse_weight=float(getattr(self, "_opt_sparse_weight", 0.2)),
        )


def make_opt_engine(deepseek_key, model_name, optimization="mv_hybrid", verbose=True):
    """构造优化组引擎。优化点都由 MVHybridExternalRetriever 注入；pipeline 走真实 DeepSeek。"""
    from openai import OpenAI

    llm = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    eng = _OptEngineFactory(deepseek_key=deepseek_key, model_name=model_name,
                            verbose=verbose)
    eng._pipeline_llm = None  # pipeline 走真实 DeepSeek（与参照组引擎同款）
    eng._opt_use_ked = False
    eng._opt_topk_per_view = 20
    eng._opt_sparse_weight = 0.2

    if optimization == "hybrid":
        eng._opt_rewriter = None      # 只开 hybrid，不开多视角
        eng._opt_view_mode = "hybrid"
        eng._opt_use_ked = True
    else:                             # mv_hybrid（默认）：多视角 × hybrid
        from agentic.query_rewriter import SemanticQueryRewriter
        rewriter = SemanticQueryRewriter(
            llm_client=llm, model_name=model_name, num_views=3,
            mock=False,               # 真实 LLM 语义视角
        )
        eng._opt_rewriter = rewriter
        eng._opt_view_mode = "hybrid"
        eng._opt_use_ked = False
    # AgenticRAGEngine.__init__ 已建 base external retriever；这里用优化版覆盖
    eng._external_retriever = eng._init_external_retriever()

    if optimization in ("organize_v1", "organize_v3"):
        # ── 纯 S4 组织侧消融：检索恢复参照组 dense/KED 默认（不引入 hybrid/mv 污染对照）。
        from experiments.exp1_qa import IndustrialRetriever
        eng._external_retriever = IndustrialRetriever(corpus_dir=eng.corpus_dir)
        if optimization == "organize_v1":
            new_org = PassThroughOrganizer()
            print("  🏷️  组织档 = organize_v1 (PassThroughOrganizer, 无去重/无修剪)")
        else:
            new_org = RerankTruncOrganizer()
            print("  🏷️  组织档 = organize_v3 (EvidenceOrganizer+L2词法重排截断, signal=lex)")
        eng._pipeline.executor.organizer = new_org   # 真实执行路径（GraphExecutor）
        eng._pipeline.organizer = new_org            # 兼容（构造字段）
        eng._organizer = new_org
    return eng




def run_optimized_experiment(num_samples=30, optimization="mv_hybrid",
                             output_dir=None, run_id=None, verbose=True):
    """优化组主评估：骨架 = 参照组 run_agentic_rag_experiment，仅引擎换为优化注入版。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"exp1_agentic_rag_opt_{optimization}_{timestamp}"

    print(f"\n{'=' * 70}")
    print(f"  🏭 Exp1 Agentic RAG [OPTIMIZATION GROUP] | optimization={optimization}")
    print(f"  Run ID: {run_id} | Samples: {num_samples}")
    print(f"{'=' * 70}\n")

    print("📂 加载数据集 (与参照组同题库)...")
    question_samples = load_question_dataset(INDUSTRYBENCH_CSV, num_samples=num_samples)
    if not question_samples:
        print("[ERROR] 未加载到任何样本，退出")
        return {"error": "no_samples"}

    print("🤖 初始化优化引擎 (ma opt: mv_hybrid / hybrid)...")
    opt_engine = make_opt_engine(DEEPSEEK_KEY, LLM_NAME,
                                 optimization=optimization, verbose=verbose)

    print("📊 初始化评分器 (mode=rule)...")
    scorer = EnhancedScorer(mode="rule", api_key=DEEPSEEK_KEY)

    results = []
    total_time = 0.0
    all_semantic_hit_at_1, all_semantic_hit_at_3 = [], []
    all_semantic_hit_at_5, all_semantic_hit_at_10 = [], []
    all_semantic_mrr, all_semantic_ndcg = [], []

    print(f"\n{'─' * 70}\n📝 开始评估 ({len(question_samples)} 样本)\n{'─' * 70}")
    for idx, sample in enumerate(question_samples):
        print(f"\n[{idx + 1}/{len(question_samples)}] {sample.question[:80]}...")
        start_time_sample = time.time()
        try:
            answer, retrieval_result = opt_engine.answer(
                question=sample.question, retrieval_k=10,
            )
            knowledge_text = sample.knowledge_text or ""
            score_result = scorer.score(
                question=sample.question,
                prediction=answer,
                reference=sample.ref_answer,
                retrieved_documents=getattr(retrieval_result, 'documents', []),
                citations=getattr(retrieval_result, 'citations', []),
                knowledge_text=knowledge_text,
            )
            semantic_metrics = scorer.evaluate_retrieval_semantic(
                retrieved_chunks=getattr(retrieval_result, 'documents', []),
                knowledge_text=knowledge_text,
                threshold=0.15,
            )
            elapsed = time.time() - start_time_sample
            total_time += elapsed

            result_entry = {
                "id": sample.id,
                "question": sample.question,
                "reference": sample.ref_answer,
                "prediction": answer,
                "difficulty": sample.difficulty,
                "capability": sample.capability,
                "industry": sample.industry_primary,
                "domain": sample.domain,
                "format": sample.question_format,
                "score_raw": score_result["raw_score"],
                "score_adjusted": score_result["adjusted_score"],
                "has_violation": score_result["has_violation"],
                "violation_detail": score_result["violation_detail"],
                "explanation": score_result["explanation"],
                "time_seconds": round(elapsed, 3),
            }
            result_entry.update({
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in semantic_metrics.items()
            })
            results.append(result_entry)

            all_semantic_hit_at_1.append(semantic_metrics.get("semantic_hit@1", 0.0))
            all_semantic_hit_at_3.append(semantic_metrics.get("semantic_hit@3", 0.0))
            all_semantic_hit_at_5.append(semantic_metrics.get("semantic_hit@5", 0.0))
            all_semantic_hit_at_10.append(semantic_metrics.get("semantic_hit@10", 0.0))
            all_semantic_mrr.append(semantic_metrics.get("semantic_mrr", 0.0))
            all_semantic_ndcg.append(semantic_metrics.get("semantic_ndcg", 0.0))

            print(f"  Score: {score_result['raw_score']} | "
                  f"Adjusted: {score_result['adjusted_score']:.2f} | "
                  f"SV: {score_result['has_violation']} | Time: {elapsed:.1f}s | "
                  f"{'❌' if score_result['has_violation'] else '✅'}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue



    # ---- 汇总统计（与参照组一致）----
    n = len(results)
    if n == 0:
        print("\n[ERROR] 没有有效的评估结果")
        return {"error": "no_results"}

    raw_scores = [r["score_raw"] for r in results]
    adjusted_scores = [r["score_adjusted"] for r in results]
    total_raw = sum(raw_scores)
    total_adjusted = sum(adjusted_scores)
    avg_raw = total_raw / n
    avg_adjusted = total_adjusted / n
    max_possible = 3 * n
    accuracy_raw = total_raw / max_possible if max_possible > 0 else 0.0
    accuracy_adjusted = total_adjusted / max_possible if max_possible > 0 else 0.0
    violations = sum(1 for r in results if r["has_violation"])
    violation_percent = (violations / n) * 100 if n > 0 else 0.0

    avg_semantic_hit_at_1 = sum(all_semantic_hit_at_1) / n if n else 0.0
    avg_semantic_hit_at_3 = sum(all_semantic_hit_at_3) / n if n else 0.0
    avg_semantic_hit_at_5 = sum(all_semantic_hit_at_5) / n if n else 0.0
    avg_semantic_hit_at_10 = sum(all_semantic_hit_at_10) / n if n else 0.0
    avg_semantic_mrr = sum(all_semantic_mrr) / n if n else 0.0
    avg_semantic_ndcg = sum(all_semantic_ndcg) / n if n else 0.0

    difficulty_scores = {}
    for r in results:
        diff = r["difficulty"]
        d = difficulty_scores.setdefault(diff, {"sum": 0, "count": 0, "violations": 0})
        d["sum"] += r["score_raw"]
        d["count"] += 1
        if r["has_violation"]:
            d["violations"] += 1

    capability_scores = {}
    for r in results:
        cap = r["capability"]
        d = capability_scores.setdefault(cap, {"sum": 0, "count": 0, "violations": 0})
        d["sum"] += r["score_raw"]
        d["count"] += 1
        if r["has_violation"]:
            d["violations"] += 1

    print(f"\n{'=' * 70}\n  📊 实验结果汇总 (optimization={optimization})\n{'=' * 70}")
    print(f"  运行 ID: {run_id}")
    print(f"  样本数: {n}")
    print(f"  总分 (raw): {total_raw}/{max_possible} = {accuracy_raw:.2%}")
    print(f"  总分 (adjusted): {total_adjusted:.1f}/{max_possible} = {accuracy_adjusted:.2%}")
    print(f"  平均分 (raw): {avg_raw:.2f}/3.0")
    print(f"  平均分 (adjusted): {avg_adjusted:.2f}/3.0")
    print(f"  SV 违规: {violations}/{n} ({violation_percent:.1f}%)")
    print(f"  总耗时: {total_time:.1f}s")
    if n:
        print(f"  平均耗时: {total_time / n:.1f}s/样本")
    print(f"\n  📈 语义检索指标:")
    print(f"    Semantic Hit@1: {avg_semantic_hit_at_1:.4f}")
    print(f"    Semantic Hit@3: {avg_semantic_hit_at_3:.4f}")
    print(f"    Semantic Hit@5: {avg_semantic_hit_at_5:.4f}")
    print(f"    Semantic Hit@10: {avg_semantic_hit_at_10:.4f}")
    print(f"    Semantic MRR: {avg_semantic_mrr:.4f}")
    print(f"    Semantic NDCG: {avg_semantic_ndcg:.4f}")
    print(f"\n  📈 按难度:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulty_scores:
            d = difficulty_scores[diff]
            print(f"    {diff}: {d['count']} samples, avg={d['sum']/d['count']:.2f}/3.0, "
                  f"violations={d['violations']}")
    print(f"\n  📈 按能力:")


    # ---- 保存结果（report.md / detailed_results.json / results.csv）----
    opt_out = output_dir or os.path.join(RESULTS_DIR, "experiments", run_id)
    os.makedirs(opt_out, exist_ok=True)

    detailed_path = os.path.join(opt_out, "detailed_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果已保存: {detailed_path}")

    report_path = os.path.join(opt_out, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Exp1 Agentic RAG 优化组实验报告\n\n")
        f.write(f"- **模式**: Agentic RAG (Task-Aware) + optimization=`{optimization}`\n")
        f.write(f"- **优化点**: hybrid(BM25-jieba, sparse_w=0.2) + 多视角改写\n")
        f.write(f"- **运行 ID**: {run_id}\n")
        f.write(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **样本数**: {n}\n\n")
        f.write("## 总体结果\n\n")
        f.write("| 指标 | 值 |\n|------|-----|\n")
        f.write(f"| 总分 (raw) | {total_raw}/{max_possible} ({accuracy_raw:.2%}) |\n")
        f.write(f"| 总分 (adjusted) | {total_adjusted:.1f}/{max_possible} ({accuracy_adjusted:.2%}) |\n")
        f.write(f"| 平均分 (raw) | {avg_raw:.2f}/3.0 |\n")
        f.write(f"| 平均分 (adjusted) | {avg_adjusted:.2f}/3.0 |\n")
        f.write(f"| SV 违规 | {violations}/{n} ({violation_percent:.1f}%) |\n")
        f.write(f"| 总耗时 | {total_time:.1f}s |\n")
        f.write(f"| 平均耗时 | {total_time / n:.1f}s |\n\n")
        f.write("## 语义检索指标\n\n")
        f.write("| 指标 | 值 |\n|------|-----|\n")
        f.write(f"| Semantic Hit@1 | {avg_semantic_hit_at_1:.4f} |\n")
        f.write(f"| Semantic Hit@3 | {avg_semantic_hit_at_3:.4f} |\n")
        f.write(f"| Semantic Hit@5 | {avg_semantic_hit_at_5:.4f} |\n")
        f.write(f"| Semantic Hit@10 | {avg_semantic_hit_at_10:.4f} |\n")
        f.write(f"| Semantic MRR | {avg_semantic_mrr:.4f} |\n")
        f.write(f"| Semantic NDCG | {avg_semantic_ndcg:.4f} |\n\n")
        f.write("## 按难度\n\n")
        f.write("| 难度 | 样本数 | 平均分 | Violations |\n|------|--------|--------|------------|\n")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_scores:
                d = difficulty_scores[diff]
                f.write(f"| {diff} | {d['count']} | {d['sum']/d['count']:.2f} | {d['violations']} |\n")
        f.write("\n## 按能力\n\n")
        f.write("| 能力 | 样本数 | 平均分 | Violations |\n|------|--------|--------|------------|\n")
        for cap, d in sorted(capability_scores.items()):
            f.write(f"| {cap} | {d['count']} | {d['sum']/d['count']:.2f} | {d['violations']} |\n")
    print(f"📄 实验报告已保存: {report_path}")

    csv_path_out = os.path.join(opt_out, "results.csv")
    if results:
        with open(csv_path_out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"📄 CSV 文件已保存: {csv_path_out}")

    stats = {
        "run_id": run_id, "num_samples": n,
        "total_raw_score": total_raw, "total_adjusted_score": total_adjusted,
        "max_possible_score": max_possible,
        "accuracy_raw": accuracy_raw, "accuracy_adjusted": accuracy_adjusted,
        "avg_raw_score": avg_raw, "avg_adjusted_score": avg_adjusted,
        "violations": violations,
        "avg_semantic_hit_at_1": avg_semantic_hit_at_1,
        "avg_semantic_hit_at_3": avg_semantic_hit_at_3,
        "avg_semantic_hit_at_5": avg_semantic_hit_at_5,
        "avg_semantic_hit_at_10": avg_semantic_hit_at_10,
        "avg_semantic_mrr": avg_semantic_mrr,
        "avg_semantic_ndcg": avg_semantic_ndcg,
        "optimization": optimization,
        "total_time": total_time,
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description="Exp1 Agentic RAG 优化组")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--optimization", default="mv_hybrid",
                    choices=["mv_hybrid", "hybrid", "organize_v1", "organize_v3"])
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    stats = run_optimized_experiment(
        num_samples=args.num_samples,
        optimization=args.optimization,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(f"\n{'=' * 70}")
    print(f"  优化组完成! optimization={args.optimization} | "
          f"Run ID: {stats.get('run_id', 'N/A')} | "
          f"Accuracy: {stats.get('accuracy_raw', 0):.2%}")
    print(f"{'=' * 70}")
    return stats


if __name__ == "__main__":
    main()
