"""
ab_organizer_eval.py: Organizer A/B 对比评估脚本

对比方案:
    - Organizer V1 (baseline): PassThroughOrganizer — 不做去重、不过滤、不修剪组
    - Organizer V2 (current): EvidenceOrganizer — 有去重、相关性过滤、task-aware组修剪

流程:
    1. 为每个问题执行完整 pipeline，**仅在 Organizer 组件做 A/B 替换**
    2. 收集 V1 和 V2 的 evidence 统计、LLM 回答、评分
    3. 生成对比报告

输出:
    - results/organizer_ab/<timestamp>/ 目录
      - individual_results.json  (每个样本的 A/B 详细结果)
      - ab_comparison_report.md  (汇总对比报告)
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

# ─── 导入依赖 ───
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

# OpenDomainRetriever 在 exp1_agentic_rag.py 内部导入，这里直接从 retrievial 导入
from retrieval.retriever import OpenDomainRetriever

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Organizer V1 — PassThrough (模拟旧 Organizer 行为)
# ============================================================

class PassThroughOrganizer:
    """
    Organizer V1 (baseline): Pass-through organizer.

    模拟旧 Organizer 行为:
        - 不做内容去重
        - 不做相关性过滤
        - 不做 task-aware group 修剪
        - 按 topic (industry/capability) 分组
        - 保留所有文档

    用于 A/B 测试的对照组。
    """

    def __init__(self):
        logger.info("[PassThroughOrganizer] 初始化 V1 (baseline) — 无过滤、无去重")

    def execute(
        self,
        directive: Optional[ExecutionDirective],
        documents: List[Any],
        question: str = "",
        task_type: Optional[str] = None,
    ) -> OrganizedEvidence:
        """Pass-through 执行: 保留所有 docs, 按 topic 分组, 不过滤"""
        if not documents:
            return OrganizedEvidence(
                documents=[], groups={},
                original_count=0, organized_count=0,
                organization_method="pass_through_v1",
            )

        original_count = len(documents)

        # V1: 不做去重，保留 ALL docs
        all_docs = list(documents)

        # V1: 按 topic 分组 (industry/capability) — 但保留所有组
        groups = self._group_by_topic(all_docs, question)

        logger.info(
            f"[PassThroughOrganizer] 📄 证据: {original_count} docs → "
            f"{len(groups)} groups (V1 pass-through, no filter)"
        )

        return OrganizedEvidence(
            documents=all_docs,
            groups=groups,
            original_count=original_count,
            organized_count=len(all_docs),
            organization_method="pass_through_v1",
        )

    def _group_by_topic(self, documents: List[Any], question: str = "") -> Dict[str, List[Any]]:
        """按 topic 分组 (与 v2 的 _group_by_topic 相同，但不做组限制)"""
        from collections import defaultdict
        grouped: Dict[str, List[Any]] = defaultdict(list)

        for doc in documents:
            industry = getattr(doc, "industry", "general")
            capability = getattr(doc, "capability", "")
            key = f"{industry}/{capability}" if capability else industry
            grouped[key].append(doc)

        return dict(sorted(grouped.items()))

    def get_context(self, organized: OrganizedEvidence) -> str:
        """从 OrganizedEvidence 构建 context 字符串 (与 v2 保持兼容)"""
        parts = []
        for group_name, docs in organized.groups.items():
            parts.append(f"## [{group_name}]")
            for i, doc in enumerate(docs):
                content = getattr(doc, "content", str(doc))
                score = getattr(doc, "score", 0.0)
                citation = getattr(doc, "citation", f"[{i+1}]")
                parts.append(f"{citation} (score={score:.3f})\n{content}")
        return "\n\n".join(parts)


# ============================================================
# 2. A/B 评估器
# ============================================================

class ABOrganizerEvaluator:
    """
    Organizer A/B 评估器。

    在同一组问题上分别用 V1 (PassThrough) 和 V2 (EvidenceOrganizer) 运行 pipeline，
    收集对比数据，生成报告。
    """

    def __init__(
        self,
        deepseek_key: str,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.deepseek_key = deepseek_key
        self.num_samples = num_samples
        self.verbose = verbose
        self.output_dir = output_dir or os.path.join(
            RESULTS_DIR, "organizer_ab",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # 分类标签
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
        """
        执行完整的 A/B 评估。

        返回:
            Dict 包含 V1/V2 的统计数据、对比结果、样本详情
        """
        register_paths()

        # ---- 加载数据 ----
        logger.info(f"📂 加载数据集 (n={self.num_samples})...")
        csv_path = INDUSTRYBENCH_CSV
        question_samples = load_question_dataset(csv_path, num_samples=self.num_samples)

        if not question_samples:
            logger.error("❌ 未加载到任何样本")
            return {"error": "no_samples"}

        logger.info(f"✅ 已加载 {len(question_samples)} 个样本")

        # ---- 初始化引擎 ----
        logger.info("🤖 初始化 Agentic RAG Engine (共享组件)...")
        engine_v1 = self._build_engine("v1")  # PassThroughOrganizer
        engine_v2 = self._build_engine("v2")  # EvidenceOrganizer (当前版本)
        logger.info("✅ 引擎初始化完成")

        # ---- 评分器 ----
        scorer = EnhancedScorer(mode="rule")

        # ---- 执行 A/B 评估 ----
        results_v1: List[Dict] = []
        results_v2: List[Dict] = []

        total_v1_time = 0.0
        total_v2_time = 0.0

        for idx, sample in enumerate(question_samples):
            question = sample.question
            reference = getattr(sample, 'answer', '')
            knowledge_text = getattr(sample, 'knowledge_text', '')
            task_type_str = getattr(sample, 'task_type', 'general')

            logger.info(f"\n{'='*60}")
            logger.info(f"[{idx+1}/{len(question_samples)}] {question[:80]}...")
            logger.info(f"  任务类型: {task_type_str}")

            # ---- 运行 V1 (baseline) ----
            try:
                t1 = time.time()
                answer_v1, retrieval_v1 = engine_v1.answer(question=question)
                elapsed_v1 = time.time() - t1
                total_v1_time += elapsed_v1

                # 提取 evidence 统计
                evidence_stats_v1 = self._extract_evidence_stats(engine_v1)

                # 评分
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
                evidence_stats_v1 = {"doc_count": 0, "group_count": 0}
                score_v1 = {"raw_score": 0.0, "adjusted_score": 0.0, "has_violation": False, "explanation": str(e)}

            # ---- 运行 V2 (当前) ----
            try:
                t2 = time.time()
                answer_v2, retrieval_v2 = engine_v2.answer(question=question)
                elapsed_v2 = time.time() - t2
                total_v2_time += elapsed_v2

                # 提取 evidence 统计
                evidence_stats_v2 = self._extract_evidence_stats(engine_v2)

                # 评分
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
                evidence_stats_v2 = {"doc_count": 0, "group_count": 0}
                score_v2 = {"raw_score": 0.0, "adjusted_score": 0.0, "has_violation": False, "explanation": str(e)}

            # ---- 收集结果 ----
            result_entry = {
                "index": idx + 1,
                "question": question,
                "reference": reference[:200] if reference else "",
                "knowledge_text": knowledge_text[:200] if knowledge_text else "",
                "task_type": task_type_str,
                "v1": {
                    "answer": answer_v1[:500] if answer_v1 else "",
                    "answer_length": len(answer_v1 or ""),
                    "time_seconds": round(elapsed_v1, 2),
                    "score": score_v1,
                    "evidence": evidence_stats_v1,
                },
                "v2": {
                    "answer": answer_v2[:500] if answer_v2 else "",
                    "answer_length": len(answer_v2 or ""),
                    "time_seconds": round(elapsed_v2, 2),
                    "score": score_v2,
                    "evidence": evidence_stats_v2,
                },
            }
            results_v1.append(result_entry)
            results_v2.append(result_entry)

            # ---- 打印中间结果 ----
            self._print_interim_result(idx + 1, result_entry)

        # ---- 汇总统计分析 ----
        stats = self._compute_statistics(results_v1, results_v2, total_v1_time, total_v2_time)

        # ---- 生成报告 ----
        self._generate_report(stats, results_v1)

        return stats

    def _build_engine(self, version: str) -> AgenticRAGEngine:
        """
        构建带指定 Organizer 版本的 AgenticRAGEngine。

        通过 monkey-patch _init_pipeline 来替换 Organizer 组件。
        """
        register_paths()

        from openai import OpenAI
        from retrieval.retriever import OpenDomainRetriever

        llm_client = OpenAI(
            api_key=self.deepseek_key,
            base_url="https://api.deepseek.com",
        )

        # 创建 Retriever
        retriever = OpenDomainRetriever(project_root=_PROJECT_ROOT)
        try:
            retriever.load_from_manifest(
                manifest_path=os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json")
            )
        except FileNotFoundError:
            retriever.load_from_manifest()

        # 创建 Analyzer 和 Planner
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

        # ── 关键：按版本选择 Organizer ──
        if version == "v1":
            organizer = PassThroughOrganizer()
            logger.info("  🏷️  Organizer = V1 (PassThrough, baseline)")
        else:
            organizer = EvidenceOrganizer()
            logger.info("  🏷️  Organizer = V2 (EvidenceOrganizer, current)")

        # 创建 Pipeline
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

        # 替换 engine 内部组件
        engine._pipeline = pipeline
        engine._analyzer = analyzer
        engine._planner = planner
        engine._organizer = organizer
        engine._retriever = retriever
        engine._solver = solver
        engine._llm_client = llm_client

        return engine

    def _extract_evidence_stats(self, engine: AgenticRAGEngine) -> Dict[str, Any]:
        """从 pipeline 执行结果中提取 evidence 统计信息"""
        pipeline_result = engine.last_pipeline_result
        if pipeline_result is None:
            return {"doc_count": 0, "group_count": 0, "organization_method": "unknown"}

        result_dict = getattr(pipeline_result, "result_dict", {})
        num_evidence = result_dict.get("num_evidence", 0)
        evidence_str = result_dict.get("evidence", "")

        # 从 engine 获取 organizer 状态
        organizer = getattr(engine, "_organizer", None)
        org_method = "unknown"

        if isinstance(organizer, PassThroughOrganizer):
            org_method = "pass_through_v1"
        elif isinstance(organizer, EvidenceOrganizer):
            org_method = "evidence_organizer_v2"

        # 从 execution_nodes 获取组织信息
        execution_nodes = getattr(pipeline_result, "execution_nodes", [])
        organize_nodes = [
            n for n in execution_nodes
            if isinstance(n, dict) and n.get("type") == "organize"
        ]

        group_count = 0
        if organize_nodes:
            msg = organize_nodes[-1].get("message", "")
            match = re.search(r"(\d+) groups", msg)
            if match:
                group_count = int(match.group(1))

        return {
            "doc_count": int(num_evidence) if num_evidence else 0,
            "group_count": group_count,
            "evidence_length": len(evidence_str) if evidence_str else 0,
            "organization_method": org_method,
        }

    def _print_interim_result(self, idx: int, entry: Dict) -> None:
        """打印单个样本的对比结果"""
        v1 = entry["v1"]
        v2 = entry["v2"]

        score_v1 = v1["score"].get("adjusted_score", 0.0) if isinstance(v1["score"], dict) else 0.0
        score_v2 = v2["score"].get("adjusted_score", 0.0) if isinstance(v2["score"], dict) else 0.0

        diff = score_v2 - score_v1
        diff_mark = "✅" if diff > 0.05 else ("❌" if diff < -0.05 else "➡️")

        logger.info(f"  对比: V1={score_v1:.3f} vs V2={score_v2:.3f} ({diff:+.3f}) {diff_mark}")
        logger.info(f"  证据: V1={v1['evidence']['doc_count']}docs/{v1['evidence']['group_count']}groups "
                     f"vs V2={v2['evidence']['doc_count']}docs/{v2['evidence']['group_count']}groups")
        logger.info(f"  时间: V1={v1['time_seconds']}s vs V2={v2['time_seconds']}s")

    def _compute_statistics(
        self,
        results_v1: List[Dict],
        results_v2: List[Dict],
        total_v1_time: float,
        total_v2_time: float,
    ) -> Dict[str, Any]:
        """计算 A/B 对比统计数据"""
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

        # 按任务类型分类
        by_task = defaultdict(lambda: {"v1_scores": [], "v2_scores": [], "count": 0})

        for entry in results_v1:
            v1 = entry["v1"]
            v2 = entry["v2"]
            task_type = entry.get("task_type", "general")

            v1_s = v1["score"].get("adjusted_score", 0.0) if isinstance(v1["score"], dict) else 0.0
            v2_s = v2["score"].get("adjusted_score", 0.0) if isinstance(v2["score"], dict) else 0.0

            v1_scores.append(v1_s)
            v2_scores.append(v2_s)
            v1_docs.append(v1["evidence"]["doc_count"])
            v2_docs.append(v2["evidence"]["doc_count"])
            v1_groups.append(v1["evidence"]["group_count"])
            v2_groups.append(v2["evidence"]["group_count"])
            v1_lengths.append(v1["answer_length"])
            v2_lengths.append(v2["answer_length"])
            v1_times.append(v1["time_seconds"])
            v2_times.append(v2["time_seconds"])

            by_task[task_type]["v1_scores"].append(v1_s)
            by_task[task_type]["v2_scores"].append(v2_s)
            by_task[task_type]["count"] += 1

        stats = {
            "total_samples": len(results_v1),
            "v1_mean_score": self._safe_mean(v1_scores),
            "v2_mean_score": self._safe_mean(v2_scores),
            "score_diff": self._safe_mean(v2_scores) - self._safe_mean(v1_scores),
            "v1_wins": sum(1 for v1, v2 in zip(v1_scores, v2_scores) if v1 > v2),
            "v2_wins": sum(1 for v1, v2 in zip(v1_scores, v2_scores) if v2 > v1),
            "ties": sum(1 for v1, v2 in zip(v1_scores, v2_scores) if abs(v1 - v2) < 0.01),
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
            "by_task_type": dict(by_task),
        }

        # 计算 V1 显著优于 V2 的样本数
        stats["v1_significantly_better"] = sum(
            1 for v1, v2 in zip(v1_scores, v2_scores) if (v1 - v2) > 0.1
        )
        stats["v2_significantly_better"] = sum(
            1 for v1, v2 in zip(v1_scores, v2_scores) if (v2 - v1) > 0.1
        )

        return stats

    def _safe_mean(self, values: List[float]) -> float:
        """安全计算平均值"""
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    def _generate_report(
        self,
        stats: Dict[str, Any],
        all_results: List[Dict],
    ) -> None:
        """生成 A/B 对比报告"""
        # 保存个体结果
        results_path = os.path.join(self.output_dir, "individual_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 个体结果已保存到: {results_path}")

        # 生成 markdown 报告
        report_path = os.path.join(self.output_dir, "ab_comparison_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._build_report_md(stats, all_results))
        logger.info(f"📊 对比报告已生成: {report_path}")

    def _build_report_md(
        self, stats: Dict[str, Any], all_results: List[Dict]
    ) -> str:
        """构建 markdown 格式的对比报告"""
        lines = []

        lines.append("# Organizer A/B 对比评估报告\n")
        lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **样本数量**: {stats['total_samples']}")
        lines.append(f"- **方案 A (V1)**: PassThroughOrganizer (旧: 无过滤、无去重)")
        lines.append(f"- **方案 B (V2)**: EvidenceOrganizer (新: 有过滤、有去重、task-aware组修剪)")
        lines.append("")

        lines.append("---\n")
        lines.append("## 1. 总体得分对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | Δ (V2-V1) |")
        lines.append("|------|:-------------:|:------------:|:---------:|")
        lines.append(f"| 平均分 | {stats['v1_mean_score']:.4f} | {stats['v2_mean_score']:.4f} | "
                       f"{stats['score_diff']:+.4f} |")
        lines.append(f"| 胜利次数 | {stats['v1_wins']} | {stats['v2_wins']} | "
                       f"{stats['v2_wins'] - stats['v1_wins']:+d} |")
        lines.append(f"| 平局 | {stats['ties']} | {stats['ties']} | — |")
        lines.append(f"| 显著更优 (>0.1) | {stats['v1_significantly_better']} | "
                       f"{stats['v2_significantly_better']} | "
                       f"{stats['v2_significantly_better'] - stats['v1_significantly_better']:+d} |")
        lines.append("")

        lines.append("## 2. Evidence 统计对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | Δ (V2-V1) |")
        lines.append("|------|:-------------:|:------------:|:---------:|")
        lines.append(f"| 平均文档数 | {stats['v1_mean_docs']:.1f} | {stats['v2_mean_docs']:.1f} | "
                       f"{stats['v2_mean_docs'] - stats['v1_mean_docs']:+.1f} |")
        lines.append(f"| 平均组数 | {stats['v1_mean_groups']:.1f} | {stats['v2_mean_groups']:.1f} | "
                       f"{stats['v2_mean_groups'] - stats['v1_mean_groups']:+.1f} |")
        lines.append("")

        lines.append("## 3. 性能对比\n")
        lines.append("| 指标 | V1 (Baseline) | V2 (Current) | Δ (V2-V1) |")
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
        lines.append("| 任务类型 | 样本数 | V1 平均分 | V2 平均分 | Δ | 胜者 |")
        lines.append("|----------|:-----:|:---------:|:---------:|:--:|:----:|")

        # 排序: 按样本数降序
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
            winner = "V2" if diff > 0.01 else ("V1" if diff < -0.01 else "—")
            label = self.classification_labels.get(task_name, task_name)
            lines.append(f"| {label} | {count} | {v1_avg:.4f} | {v2_avg:.4f} | {diff:+.4f} | {winner} |")

        lines.append("")

        # 前 10 个显著差异样本
        lines.append("## 5. Top 10 差异最大样本\n")
        diffs = []
        for entry in all_results:
            v1_s = entry["v1"]["score"].get("adjusted_score", 0.0) if isinstance(entry["v1"]["score"], dict) else 0.0
            v2_s = entry["v2"]["score"].get("adjusted_score", 0.0) if isinstance(entry["v2"]["score"], dict) else 0.0
            diffs.append((abs(v2_s - v1_s), entry["index"], entry["question"][:60],
                          v1_s, v2_s, entry["v1"]["evidence"]["doc_count"],
                          entry["v2"]["evidence"]["doc_count"],
                          entry["v1"]["evidence"]["group_count"],
                          entry["v2"]["evidence"]["group_count"]))

        diffs.sort(key=lambda x: x[0], reverse=True)
        lines.append("| # | 问题 | V1 分 | V2 分 | Δ | V1 证据 | V2 证据 | V1 组 | V2 组 |")
        lines.append("|---|------|:-----:|:-----:|:--:|:------:|:------:|:-----:|:-----:|")
        for d in diffs[:10]:
            lines.append(f"| {d[1]} | {d[2]}... | {d[3]:.3f} | {d[4]:.3f} | {d[4]-d[3]:+.3f} | "
                           f"{d[5]} | {d[6]} | {d[7]} | {d[8]} |")
        lines.append("")

        # 详细结果
        lines.append("## 6. 详细结果\n")
        lines.append("<details>")
        lines.append(f"<summary>点击展开全部 {stats['total_samples']} 个样本的详细结果</summary>\n")
        lines.append("")
        lines.append("| # | 问题 | 任务 | V1 分 | V2 分 | Δ | V1文档 | V2文档 | V1组 | V2组 |")
        lines.append("|---|------|:----:|:-----:|:-----:|:--:|:-----:|:-----:|:----:|:----:|")

        for entry in all_results:
            v1_s = entry["v1"]["score"].get("adjusted_score", 0.0) if isinstance(entry["v1"]["score"], dict) else 0.0
            v2_s = entry["v2"]["score"].get("adjusted_score", 0.0) if isinstance(entry["v2"]["score"], dict) else 0.0
            task_label = self.classification_labels.get(entry.get("task_type", ""), entry.get("task_type", ""))
            lines.append(f"| {entry['index']} | {entry['question'][:50]}... | {task_label} | "
                           f"{v1_s:.3f} | {v2_s:.3f} | {v2_s-v1_s:+.3f} | "
                           f"{entry['v1']['evidence']['doc_count']} | {entry['v2']['evidence']['doc_count']} | "
                           f"{entry['v1']['evidence']['group_count']} | {entry['v2']['evidence']['group_count']} |")

        lines.append("</details>\n")
        lines.append("")

        return "\n".join(lines)


# ============================================================
# 3. 入口
# ============================================================

def main():
    """A/B 评估入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Organizer A/B 对比评估")
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="评估样本数量 (默认: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: results/organizer_ab/<timestamp>)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="打印详细信息"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  🧪 Organizer A/B 对比评估")
    print(f"  Samples: {args.num_samples}")
    print("=" * 70)

    # 注册路径
    register_paths()

    # 创建评估器
    evaluator = ABOrganizerEvaluator(
        deepseek_key=DEEPSEEK_KEY,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # 执行评估
    stats = evaluator.evaluate()

    # 打印关键结果
    print("\n" + "=" * 70)
    print("  📊 A/B 评估完成!")
    print(f"  V1 平均分: {stats.get('v1_mean_score', 'N/A')}")
    print(f"  V2 平均分: {stats.get('v2_mean_score', 'N/A')}")
    print(f"  V2-V1 差异: {stats.get('score_diff', 'N/A'):+.4f}")
    print(f"  V1 胜利: {stats.get('v1_wins', 0)} 次")
    print(f"  V2 胜利: {stats.get('v2_wins', 0)} 次")
    print(f"  V1 显著更优: {stats.get('v1_significantly_better', 0)} 次")
    print(f"  V2 显著更优: {stats.get('v2_significantly_better', 0)} 次")
    print(f"  结果目录: {evaluator.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
