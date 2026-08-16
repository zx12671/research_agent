"""
ab_fix_stats.py: 修复 A/B 评估统计 + 跑 20 个样本

修复内容:
1. results_v1 和 results_v2 现在分别存储 V1/V2 数据 (之前指向同一个列表)
2. 新增: 平均 Prompt 长度、Evidence 长度跟踪
3. 新增: 平均耗时统计
4. 修复 wins 计算使用正确的列表

然后跑 20 个样本输出详细结果。
"""

import os
import sys
import json
import time
import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime

# ─── 路径注册 ───
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, '..'))

from experiments.config import (
    DEEPSEEK_KEY,
    LLM_NAME,
    INDUSTRYBENCH_CSV,
    KNOWLEDGE_CORPUS_DIR,
    RESULTS_DIR,
    register_paths,
)
from experiments.exp1_agentic_rag import (
    load_question_dataset,
    AgenticRAGEngine,
    EnhancedScorer,
    RetrievedDocument,
    MODES,
)

from agentic.task_types import (
    OrganizedEvidence,
    ExecutionDirective,
    TaskType,
)
from agentic.organizer import EvidenceOrganizer
from retrieval.retriever import OpenDomainRetriever

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Organizer V1 — PassThrough
# ============================================================

class PassThroughOrganizer:
    """Organizer V1 (baseline): Pass-through organizer."""

    def __init__(self):
        logger.info("[PassThroughOrganizer] V1 initialized")

    def execute(
        self,
        directive: Optional[ExecutionDirective],
        documents: List[Any],
        question: str = "",
        task_type: Optional[str] = None,
    ) -> OrganizedEvidence:
        if not documents:
            return OrganizedEvidence(
                documents=[], groups={},
                original_count=0, organized_count=0,
                organization_method="pass_through_v1",
            )
        original_count = len(documents)
        all_docs = list(documents)
        groups = self._group_by_topic(all_docs, question)
        logger.info(
            f"[PassThroughOrganizer] \U0001f4c4 {original_count} docs \u2192 "
            f"{len(groups)} groups (V1, no filter)"
        )
        return OrganizedEvidence(
            documents=all_docs,
            groups=groups,
            original_count=original_count,
            organized_count=len(all_docs),
            organization_method="pass_through_v1",
        )

    def _group_by_topic(self, documents: List[Any], question: str = "") -> Dict[str, List[Any]]:
        from collections import defaultdict
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for doc in documents:
            industry = getattr(doc, "industry", "general")
            capability = getattr(doc, "capability", "")
            key = f"{industry}/{capability}" if capability else industry
            grouped[key].append(doc)
        return dict(sorted(grouped.items()))


# ============================================================
# 2. A/B 评估器 (修复版)
# ============================================================

class FixedABOrganizerEvaluator:
    """
    修复版 Organizer A/B 评估器。

    修复:
    - results_v1 / results_v2 各自独立存储 (不再指向同一列表)
    - 新增 prompt_length, evidence_length 跟踪
    - 修复 wins 计算
    """

    def __init__(
        self,
        deepseek_key: str,
        num_samples: int = 20,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.deepseek_key = deepseek_key
        self.num_samples = num_samples
        self.verbose = verbose
        self.output_dir = output_dir or os.path.join(
            RESULTS_DIR, "organizer_ab_fixed",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.classification_labels = {
            "selection": "选型",
            "comparison": "比较",
            "diagnosis": "诊断",
            "procedure": "流程",
            "calculation": "计算",
            "standard_interpretation": "标准解读",
            "explanation": "解释",
            "general": "通用",
        }

    def evaluate(self) -> Dict[str, Any]:
        register_paths()

        # ---- 加载数据 ----
        logger.info(f"\U0001f4c2 加载数据集 (n={self.num_samples})...")
        csv_path = INDUSTRYBENCH_CSV
        question_samples = load_question_dataset(csv_path, num_samples=self.num_samples)

        if not question_samples:
            logger.error("❌ 未加载到任何样本")
            return {"error": "no_samples"}

        logger.info(f"✅ 已加载 {len(question_samples)} 个样本")

        # ---- 初始化引擎 ----
        logger.info("🤖 初始化 Agentic RAG Engine (V1)...")
        engine_v1 = self._build_engine("v1")
        logger.info("🤖 初始化 Agentic RAG Engine (V2)...")
        engine_v2 = self._build_engine("v2")
        logger.info("✅ 引擎初始化完成")

        scorer = EnhancedScorer(mode="rule")

        # ---- 独立的结果列表 (修复点 1) ----
        results_v1: List[Dict] = []
        results_v2: List[Dict] = []

        total_v1_time = 0.0
        total_v2_time = 0.0

        for idx, sample in enumerate(question_samples):
            question = sample.question
            reference = getattr(sample, 'answer', '')
            knowledge_text = getattr(sample, 'knowledge_text', '')
            task_type_str = getattr(sample, 'task_type', 'general')

            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{idx+1}/{len(question_samples)}] {question[:80]}...")
            logger.info(f"  任务类型: {task_type_str}")

            # ---- 运行 V1 (baseline) ----
            try:
                t1 = time.time()
                answer_v1, retrieval_v1 = engine_v1.answer(question=question)
                elapsed_v1 = time.time() - t1
                total_v1_time += elapsed_v1
                evidence_stats_v1 = self._extract_evidence_stats(engine_v1)
                prompt_len_v1 = self._extract_prompt_length(engine_v1)
                score_v1 = scorer.score(
                    question=question,
                    prediction=answer_v1,
                    reference=reference,
                    retrieved_documents=retrieval_v1.documents if hasattr(retrieval_v1, 'documents') else [],
                )
            except Exception as e:
                logger.error(f"  ❌ V1 失败: {e}")
                answer_v1 = f"[ERROR] {e}"
                elapsed_v1 = 0.0
                evidence_stats_v1 = {"doc_count": 0, "group_count": 0, "evidence_length": 0}
                prompt_len_v1 = 0
                score_v1 = {"raw_score": 0.0, "adjusted_score": 0.0, "has_violation": False, "explanation": str(e)}

            # ---- 运行 V2 (当前) ----
            try:
                t2 = time.time()
                answer_v2, retrieval_v2 = engine_v2.answer(question=question)
                elapsed_v2 = time.time() - t2
                total_v2_time += elapsed_v2
                evidence_stats_v2 = self._extract_evidence_stats(engine_v2)
                prompt_len_v2 = self._extract_prompt_length(engine_v2)
                score_v2 = scorer.score(
                    question=question,
                    prediction=answer_v2,
                    reference=reference,
                    retrieved_documents=retrieval_v2.documents if hasattr(retrieval_v2, 'documents') else [],
                )
            except Exception as e:
                logger.error(f"  ❌ V2 失败: {e}")
                answer_v2 = f"[ERROR] {e}"
                elapsed_v2 = 0.0
                evidence_stats_v2 = {"doc_count": 0, "group_count": 0, "evidence_length": 0}
                prompt_len_v2 = 0
                score_v2 = {"raw_score": 0.0, "adjusted_score": 0.0, "has_violation": False, "explanation": str(e)}

            # ---- 独立存储 V1/V2 (修复: 不再共享 result_entry) ----
            entry_v1 = {
                "index": idx + 1,
                "question": question,
                "reference": reference[:200] if reference else "",
                "knowledge_text": knowledge_text[:200] if knowledge_text else "",
                "task_type": task_type_str,
                "answer": answer_v1[:500] if answer_v1 else "",
                "answer_length": len(answer_v1 or ""),
                "time_seconds": round(elapsed_v1, 2),
                "score": score_v1,
                "evidence": evidence_stats_v1,
                "prompt_length": prompt_len_v1,
            }
            entry_v2 = {
                "index": idx + 1,
                "question": question,
                "reference": reference[:200] if reference else "",
                "knowledge_text": knowledge_text[:200] if knowledge_text else "",
                "task_type": task_type_str,
                "answer": answer_v2[:500] if answer_v2 else "",
                "answer_length": len(answer_v2 or ""),
                "time_seconds": round(elapsed_v2, 2),
                "score": score_v2,
                "evidence": evidence_stats_v2,
                "prompt_length": prompt_len_v2,
            }
            results_v1.append(entry_v1)
            results_v2.append(entry_v2)

            # ---- 打印中间结果 ----
            self._print_interim_result(idx + 1, entry_v1, entry_v2)

        # ---- 汇总统计分析 (使用正确的两个独立列表) ----
        stats = self._compute_statistics(results_v1, results_v2, total_v1_time, total_v2_time)

        # ---- 生成报告 ----
        self._generate_report(stats, results_v1, results_v2)

        return stats

    def _build_engine(self, version: str) -> AgenticRAGEngine:
        register_paths()
        from openai import OpenAI
        from retrieval.retriever import OpenDomainRetriever

        llm_client = OpenAI(
            api_key=self.deepseek_key,
            base_url="https://api.deepseek.com",
        )

        retriever = OpenDomainRetriever(project_root=_PROJECT_ROOT)
        try:
            retriever.load_from_manifest(
                manifest_path=os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json")
            )
        except FileNotFoundError:
            retriever.load_from_manifest()

        from agentic import TaskAnalyzer, StrategyPlanner, TaskSolver

        analyzer = TaskAnalyzer(
            llm_client=llm_client,
            model_name=LLM_NAME,
            temperature=0.1,
        )
        planner = StrategyPlanner(
            llm_client=llm_client,
            model_name=LLM_NAME,
        )
        solver = TaskSolver(
            llm_client=llm_client,
            model_name=LLM_NAME,
            temperature=0.3,
        )

        if version == "v1":
            organizer = PassThroughOrganizer()
            logger.info("  \U0001f3f7\ufe0f  Organizer = V1 (PassThrough, baseline)")
        else:
            organizer = EvidenceOrganizer()
            logger.info("  \U0001f3f7\ufe0f  Organizer = V2 (EvidenceOrganizer, current)")

        from agentic.pipeline import AdaptiveAgenticPipeline

        pipeline = AdaptiveAgenticPipeline(
            analyzer=analyzer,
            planner=planner,
            retriever=retriever,
            organizer=organizer,
            solver=solver,
            llm_client=llm_client,
        )

        engine = AgenticRAGEngine(
            deepseek_key=self.deepseek_key,
            verbose=self.verbose,
        )
        engine._pipeline = pipeline
        engine._analyzer = analyzer
        engine._planner = planner
        engine._organizer = organizer
        engine._retriever = retriever
        engine._solver = solver
        engine._llm_client = llm_client

        return engine

    def _extract_evidence_stats(self, engine: AgenticRAGEngine) -> Dict[str, Any]:
        """
        从 pipeline 执行结果中提取证据统计。
        
        使用 result_dict 中的以下字段（由 pipeline 的 _handle_reason 设置）:
        - evidence_str: 组织后的证据字符串
        - group_count: 实际 group 数量
        - document_count: 文档数量  
        - evidence_length: 证据字符串长度
        """
        pipeline_result = engine.last_pipeline_result
        if pipeline_result is None:
            return {"doc_count": 0, "group_count": 0, "organization_method": "unknown", "evidence_length": 0}

        result_dict = getattr(pipeline_result, "result_dict", {})
        
        # --- 优先使用 pipeline 新字段 ---
        doc_count = int(result_dict.get("document_count", 0))
        group_count = int(result_dict.get("group_count", 0))
        evidence_length = int(result_dict.get("evidence_length", 0))
        
        # fallback: 如果 pipeline 没提供新字段，使用旧字段
        if doc_count == 0:
            doc_count = int(result_dict.get("num_evidence", 0))
        if evidence_length == 0:
            evidence_str = result_dict.get("evidence_str", "") or result_dict.get("evidence", "")
            evidence_length = len(evidence_str) if evidence_str else 0

        organizer = getattr(engine, "_organizer", None)
        org_method = "unknown"
        if isinstance(organizer, PassThroughOrganizer):
            org_method = "pass_through_v1"
        elif isinstance(organizer, EvidenceOrganizer):
            org_method = "evidence_organizer_v2"

        # fallback: 如果 group_count 仍为 0，从日志解析
        if group_count == 0:
            execution_nodes = getattr(pipeline_result, "execution_nodes", [])
            organize_nodes = [
                n for n in execution_nodes
                if isinstance(n, dict) and n.get("type") == "organize"
            ]
            if organize_nodes:
                msg = organize_nodes[-1].get("message", "")
                match = re.search(r"(\d+) groups", msg)
                if match:
                    group_count = int(match.group(1))

        return {
            "doc_count": doc_count,
            "group_count": group_count,
            "evidence_length": evidence_length,
            "organization_method": org_method,
        }

    def _extract_prompt_length(self, engine: AgenticRAGEngine) -> int:
        """从 pipeline 执行结果中提取 prompt 长度"""
        pipeline_result = engine.last_pipeline_result
        if pipeline_result is None:
            return 0
        result_dict = getattr(pipeline_result, "result_dict", {})
        prompt_length = result_dict.get("prompt_length", 0)
        if isinstance(prompt_length, int):
            return prompt_length
        try:
            return int(prompt_length)
        except (ValueError, TypeError):
            return 0

    def _print_interim_result(self, idx: int, entry_v1: Dict, entry_v2: Dict) -> None:
        score_v1 = entry_v1["score"].get("adjusted_score", 0.0) if isinstance(entry_v1["score"], dict) else 0.0
        score_v2 = entry_v2["score"].get("adjusted_score", 0.0) if isinstance(entry_v2["score"], dict) else 0.0
        diff = score_v2 - score_v1
        diff_mark = "\u2705" if diff > 0.05 else ("\u274c" if diff < -0.05 else "\u27a1\ufe0f")

        logger.info(f"  对比: V1={score_v1:.3f} vs V2={score_v2:.3f} ({diff:+.3f}) {diff_mark}")
        logger.info(f"  证据: V1={entry_v1['evidence']['doc_count']}docs/{entry_v1['evidence']['group_count']}groups "
                     f"vs V2={entry_v2['evidence']['doc_count']}docs/{entry_v2['evidence']['group_count']}groups")
        logger.info(f"  证据长度: V1={entry_v1['evidence']['evidence_length']} vs V2={entry_v2['evidence']['evidence_length']}")
        logger.info(f"  Prompt长度: V1={entry_v1.get('prompt_length', 0)} vs V2={entry_v2.get('prompt_length', 0)}")
        logger.info(f"  时间: V1={entry_v1['time_seconds']}s vs V2={entry_v2['time_seconds']}s")

    def _compute_statistics(
        self,
        results_v1: List[Dict],
        results_v2: List[Dict],
        total_v1_time: float,
        total_v2_time: float,
    ) -> Dict[str, Any]:
        """计算 A/B 对比统计数据 (修复: 使用独立的 results_v1 / results_v2 列表)"""
        v1_scores = []
        v2_scores = []
        v1_docs = []
        v2_docs = []
        v1_groups = []
        v2_groups = []
        v1_lengths = []
        v2_lengths = []
        v1_times = []
        v2_times = []
        v1_evid_lens = []  # 新增: evidence 长度
        v2_evid_lens = []
        v1_prompt_lens = []  # 新增: prompt 长度
        v2_prompt_lens = []

        by_task = defaultdict(lambda: {"v1_scores": [], "v2_scores": [], "count": 0})

        # 按 index 对齐 V1/V2
        for entry_v1, entry_v2 in zip(results_v1, results_v2):
            task_type = entry_v1.get("task_type", "general")

            v1_s = entry_v1["score"].get("adjusted_score", 0.0) if isinstance(entry_v1["score"], dict) else 0.0
            v2_s = entry_v2["score"].get("adjusted_score", 0.0) if isinstance(entry_v2["score"], dict) else 0.0

            v1_scores.append(v1_s)
            v2_scores.append(v2_s)
            v1_docs.append(entry_v1["evidence"]["doc_count"])
            v2_docs.append(entry_v2["evidence"]["doc_count"])
            v1_groups.append(entry_v1["evidence"]["group_count"])
            v2_groups.append(entry_v2["evidence"]["group_count"])
            v1_lengths.append(entry_v1["answer_length"])
            v2_lengths.append(entry_v2["answer_length"])
            v1_times.append(entry_v1["time_seconds"])
            v2_times.append(entry_v2["time_seconds"])
            v1_evid_lens.append(entry_v1["evidence"].get("evidence_length", 0))
            v2_evid_lens.append(entry_v2["evidence"].get("evidence_length", 0))
            v1_prompt_lens.append(entry_v1.get("prompt_length", 0))
            v2_prompt_lens.append(entry_v2.get("prompt_length", 0))

            by_task[task_type]["v1_scores"].append(v1_s)
            by_task[task_type]["v2_scores"].append(v2_s)
            by_task[task_type]["count"] += 1

        stats = {
            "total_samples": len(results_v1),
            "v1_mean_score": self._safe_mean(v1_scores),
            "v2_mean_score": self._safe_mean(v2_scores),
            "score_diff": self._safe_mean(v2_scores) - self._safe_mean(v1_scores),
            # 修复: 使用 v1_scores 和 v2_scores (正确对齐的值)
            "v1_wins": sum(1 for s1, s2 in zip(v1_scores, v2_scores) if s1 > s2),
            "v2_wins": sum(1 for s1, s2 in zip(v1_scores, v2_scores) if s2 > s1),
            "ties": sum(1 for s1, s2 in zip(v1_scores, v2_scores) if abs(s1 - s2) < 0.01),
            "v1_mean_docs": self._safe_mean(v1_docs),
            "v2_mean_docs": self._safe_mean(v2_docs),
            "v1_mean_groups": self._safe_mean(v1_groups),
            "v2_mean_groups": self._safe_mean(v2_groups),
            "v1_mean_answer_length": self._safe_mean(v1_lengths),
            "v2_mean_answer_length": self._safe_mean(v2_lengths),
            "v1_mean_time": self._safe_mean(v1_times),
            "v2_mean_time": self._safe_mean(v2_times),
            "total_v1_time": round(total_v1_time, 2),
            "total_v2_time": round(total_v2_time, 2),
            "v1_mean_evidence_length": self._safe_mean(v1_evid_lens),  # 新增
            "v2_mean_evidence_length": self._safe_mean(v2_evid_lens),  # 新增
            "v1_mean_prompt_length": self._safe_mean(v1_prompt_lens),  # 新增
            "v2_mean_prompt_length": self._safe_mean(v2_prompt_lens),  # 新增
            "by_task_type": dict(by_task),
        }

        stats["v1_significantly_better"] = sum(
            1 for s1, s2 in zip(v1_scores, v2_scores) if (s1 - s2) > 0.1
        )
        stats["v2_significantly_better"] = sum(
            1 for s1, s2 in zip(v1_scores, v2_scores) if (s2 - s1) > 0.1
        )

        return stats

    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    def _generate_report(
        self,
        stats: Dict[str, Any],
        results_v1: List[Dict],
        results_v2: List[Dict],
    ) -> None:
        # 保存个体结果 (V1/V2 分开)
        results_path_v1 = os.path.join(self.output_dir, "individual_results_v1.json")
        with open(results_path_v1, "w", encoding="utf-8") as f:
            json.dump(results_v1, f, ensure_ascii=False, indent=2)
        logger.info(f"\U0001f4be V1 个体结果已保存到: {results_path_v1}")

        results_path_v2 = os.path.join(self.output_dir, "individual_results_v2.json")
        with open(results_path_v2, "w", encoding="utf-8") as f:
            json.dump(results_v2, f, ensure_ascii=False, indent=2)
        logger.info(f"\U0001f4be V2 个体结果已保存到: {results_path_v2}")

        # 生成 markdown 报告
        report_path = os.path.join(self.output_dir, "ab_comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._build_report_md(stats, results_v1, results_v2))
        logger.info(f"\U0001f4ca 对比报告已生成: {report_path}")

    def _build_report_md(
        self, stats: Dict[str, Any], results_v1: List[Dict], results_v2: List[Dict]
    ) -> str:
        lines = []

        lines.append("# Organizer A/B 对比评估报告 (修复版统计)\n")
        lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **样本数量**: {stats['total_samples']}")
        lines.append(f"- **方案 A (V1)**: PassThroughOrganizer (旧: 无过滤、无去重)")
        lines.append(f"- **方案 B (V2)**: EvidenceOrganizer (新: 有过滤、有去重、task-aware组修剪)")
        lines.append("")

        lines.append("---\n")
        lines.append("## 1. 总体得分对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | \u0394 (V2-V1) |")
        lines.append("|------|:-------------:|:------------:|:---------:|")
        lines.append(f"| 平均分 | {stats['v1_mean_score']:.4f} | {stats['v2_mean_score']:.4f} | "
                       f"{stats['score_diff']:+.4f} |")
        lines.append(f"| 胜利次数 | {stats['v1_wins']} | {stats['v2_wins']} | "
                       f"{stats['v2_wins'] - stats['v1_wins']:+d} |")
        lines.append(f"| 平局 | {stats['ties']} | {stats['ties']} | \u2014 |")
        lines.append(f"| 显著更优 (>0.1) | {stats['v1_significantly_better']} | "
                       f"{stats['v2_significantly_better']} | "
                       f"{stats['v2_significantly_better'] - stats['v1_significantly_better']:+d} |")
        lines.append("")

        lines.append("## 2. Evidence & Prompt 统计对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | \u0394 (V2-V1) |")
        lines.append("|------|:-------------:|:------------:|:---------:|")
        lines.append(f"| 平均文档数 | {stats['v1_mean_docs']:.1f} | {stats['v2_mean_docs']:.1f} | "
                       f"{stats['v2_mean_docs'] - stats['v1_mean_docs']:+.1f} |")
        lines.append(f"| 平均组数 | {stats['v1_mean_groups']:.1f} | {stats['v2_mean_groups']:.1f} | "
                       f"{stats['v2_mean_groups'] - stats['v1_mean_groups']:+.1f} |")
        lines.append(f"| 平均 Evidence 长度(字符) | {stats['v1_mean_evidence_length']:.0f} | "
                       f"{stats['v2_mean_evidence_length']:.0f} | "
                       f"{stats['v2_mean_evidence_length'] - stats['v1_mean_evidence_length']:+.0f} |")
        lines.append(f"| 平均 Prompt 长度(字符) | {stats['v1_mean_prompt_length']:.0f} | "
                       f"{stats['v2_mean_prompt_length']:.0f} | "
                       f"{stats['v2_mean_prompt_length'] - stats['v1_mean_prompt_length']:+.0f} |")
        lines.append("")

        lines.append("## 3. 性能对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | \u0394 (V2-V1) |")
        lines.append("|------|:-------------:|:------------:|:---------:|")
        lines.append(f"| 平均回答长度(字符) | {stats['v1_mean_answer_length']:.1f} | "
                       f"{stats['v2_mean_answer_length']:.1f} | "
                       f"{stats['v2_mean_answer_length'] - stats['v1_mean_answer_length']:+.1f} |")
        lines.append(f"| 平均耗时(秒) | {stats['v1_mean_time']:.2f} | {stats['v2_mean_time']:.2f} | "
                       f"{stats['v2_mean_time'] - stats['v1_mean_time']:+.2f} |")
        lines.append(f"| 总耗时(秒) | {stats['total_v1_time']:.1f} | {stats['total_v2_time']:.1f} | "
                       f"{stats['total_v2_time'] - stats['total_v1_time']:+.1f} |")
        lines.append("")

        # 按任务类型分类
        lines.append("## 4. 按任务类型细分\n")
        lines.append("| 任务类型 | 样本数 | V1 平均分 | V2 平均分 | \u0394 | 胜者 |")
        lines.append("|----------|:-----:|:---------:|:---------:|:--:|:----:|")

        sorted_tasks = sorted(
            stats['by_task_type'].items(),
            key=lambda x: x[1]['count'],
            reverse=True,
        )
        for task_name, task_data in sorted_tasks:
            count = task_data['count']
            v1_avg = self._safe_mean(task_data['v1_scores'])
            v2_avg = self._safe_mean(task_data['v2_scores'])
            diff = v2_avg - v1_avg
            winner = "V2" if diff > 0.01 else ("V1" if diff < -0.01 else "\u2014")
            label = self.classification_labels.get(task_name, task_name)
            lines.append(f"| {label} | {count} | {v1_avg:.4f} | {v2_avg:.4f} | {diff:+.4f} | {winner} |")

        lines.append("")

        # 详细结果表
        lines.append("## 5. 详细结果\n")
        lines.append("| # | 问题 | V1 分 | V2 分 | \u0394 | V1文档 | V2文档 | V1证据长度 | V2证据长度 | V1 Prompt | V2 Prompt | V1耗时 | V2耗时 |")
        lines.append("|---|------|:-----:|:-----:|:--:|:-----:|:-----:|:---------:|:---------:|:---------:|:---------:|:-----:|:-----:|")

        for entry_v1, entry_v2 in zip(results_v1, results_v2):
            v1_s = entry_v1["score"].get("adjusted_score", 0.0) if isinstance(entry_v1["score"], dict) else 0.0
            v2_s = entry_v2["score"].get("adjusted_score", 0.0) if isinstance(entry_v2["score"], dict) else 0.0
            lines.append(f"| {entry_v1['index']} | {entry_v1['question'][:40]}... | "
                           f"{v1_s:.3f} | {v2_s:.3f} | {v2_s - v1_s:+.3f} | "
                           f"{entry_v1['evidence']['doc_count']} | {entry_v2['evidence']['doc_count']} | "
                           f"{entry_v1['evidence'].get('evidence_length', 0)} | {entry_v2['evidence'].get('evidence_length', 0)} | "
                           f"{entry_v1.get('prompt_length', 0)} | {entry_v2.get('prompt_length', 0)} | "
                           f"{entry_v1['time_seconds']}s | {entry_v2['time_seconds']}s |")

        lines.append("")

        return "\n".join(lines)


# ============================================================
# 3. 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Organizer A/B 对比评估 (修复版)")
    parser.add_argument("--num-samples", type=int, default=20, help="评估样本数量")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细信息")
    args = parser.parse_args()

    print("=" * 70)
    print("  \U0001f9ea Organizer A/B 对比评估 (修复版)")
    print(f"  Samples: {args.num_samples}")
    print("=" * 70)

    register_paths()

    evaluator = FixedABOrganizerEvaluator(
        deepseek_key=DEEPSEEK_KEY,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    stats = evaluator.evaluate()

    print("\n" + "=" * 70)
    print("  \U0001f4ca A/B 评估完成!")
    print(f"  V1 平均分:    {stats.get('v1_mean_score', 'N/A')}")
    print(f"  V2 平均分:    {stats.get('v2_mean_score', 'N/A')}")
    print(f"  V2-V1 差异:   {stats.get('score_diff', 'N/A'):+.4f}")
    print(f"  V1 胜利:      {stats.get('v1_wins', 0)} 次")
    print(f"  V2 胜利:      {stats.get('v2_wins', 0)} 次")
    print(f"  V1 显著更优:  {stats.get('v1_significantly_better', 0)} 次")
    print(f"  V2 显著更优:  {stats.get('v2_significantly_better', 0)} 次")
    print(f"  平均证据长度: V1={stats.get('v1_mean_evidence_length', 0):.0f} vs V2={stats.get('v2_mean_evidence_length', 0):.0f}")
    print(f"  平均Prompt长度: V1={stats.get('v1_mean_prompt_length', 0):.0f} vs V2={stats.get('v2_mean_prompt_length', 0):.0f}")
    print(f"  平均耗时: V1={stats.get('v1_mean_time', 0):.2f}s vs V2={stats.get('v2_mean_time', 0):.2f}s")
    print(f"  结果目录: {evaluator.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
