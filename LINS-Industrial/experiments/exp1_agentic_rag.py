"""
exp1_agentic_rag.py: 基于 Agentic RAG 的工业问答实验

基于 exp1_qa.py 重构，替换原管线为 Agentic RAG 架构：

原管线:
    Question → Retriever → DeepSeek → Answer

新管线:
    Question → TaskAnalyzer → StrategyPlanner → Retriever
            → EvidenceOrganizer → TaskSolver → DeepSeek → Answer

设计原则:
- 保留 exp1_qa.py 的数据加载、评分器、评估框架 (完全兼容)
- 保持 scorer 不变
- 仅替换回答生成管线
- 支持所有现有模式: quick, closed_book, rag, standard_rag + 新增 agentic_rag 模式
"""

import sys
import os

# ==== 路径修复 ====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_LINS_MAIN = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "LINS-main"))
if _LINS_MAIN not in sys.path:
    sys.path.insert(0, _LINS_MAIN)

import json
import csv
import time
import argparse
import logging
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple, Set

from experiments.config import (
    PROJECT_ROOT,
    DATA_DIR,
    KNOWLEDGE_CORPUS_DIR,
    RESULTS_DIR,
    LLM_NAME,
    DEEPSEEK_KEY,
    INDUSTRYBENCH_CSV,
    TOP_K,
    RECALL_TOP_K,
    register_paths,
)

register_paths()

# ============================================================
# 1. 模式配置 (继承 exp1_qa 所有模式 + 新增 agentic_rag)
# ============================================================

MODES = {
    "quick": {
        "name": "Oracle (Ground Truth Knowledge)",
        "desc": "注入 ground-truth knowledge_text，测量检索管线上界",
        "topk": 3,
        "use_retriever": False,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 10,
        "prompt_extra": "",
        "retrieval_k": 10,
    },
    "closed_book": {
        "name": "闭卷 (Closed-Book)",
        "desc": "无外部知识，仅依赖模型内部知识，下界参考",
        "topk": 1,
        "use_retriever": False,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 1,
        "prompt_extra": "",
    },
    "rag": {
        "name": "RAG (KED + Industrial Retriever + MAIRAG)",
        "desc": "原始 exp1 RAG 管线",
        "topk": 5,
        "use_retriever": True,
        "if_PRA": True,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 50,
        "prompt_extra": "",
        "retrieval_k": 10,
    },
    "standard_rag": {
        "name": "标准 RAG Baseline (无 PRM)",
        "desc": "Question → Retriever → Top-k → Standard Prompt → LLM",
        "topk": 5,
        "use_retriever": True,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 10,
        "prompt_extra": "",
        "retrieval_k": 10,
    },
    "agentic_rag": {
        "name": "Agentic RAG (Task-Aware, 本实验)",
        "desc": (
            "Question → TaskAnalyzer → StrategyPlanner → Retriever "
            "→ EvidenceOrganizer → TaskSolver → DeepSeek → Answer"
        ),
        "topk": 5,
        "use_retriever": True,  # Agentic pipeline handles retrieval internally
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 10,
        "prompt_extra": "",
        "retrieval_k": 10,  # Default, overridden by StrategyPlanner
    },
}

# ============================================================
# 2. 数据加载 (与 exp1_qa 完全一致)
# ============================================================

from dataclasses import dataclass


@dataclass
class QuestionSample:
    """
    仅包含问题和参考信息的数据点。
    知识库是独立的 knowledge_corpus/，不从这里读取。
    """
    id: str
    question: str
    ref_answer: str
    difficulty: str
    capability: str
    industry_primary: str
    domain: str
    question_format: str = "QA"
    knowledge_text: str = ""


def load_question_dataset(
    csv_path: str, num_samples: Optional[int] = None
) -> List[QuestionSample]:
    """加载问题数据集 (与 exp1_qa 完全一致)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_format = row.get(
                "_format",
                row.get("format", row.get("question_format", "")),
            ).strip()
            if not q_format:
                q_format = _infer_question_format(row.get("question", ""))

            knowledge_text = row.get("knowledge_text", "").strip()

            sample = QuestionSample(
                id=row.get("id", "").strip(),
                question=row.get("question", "").strip(),
                ref_answer=row.get("answer", "").strip(),
                difficulty=row.get("difficulty", "").strip().lower(),
                capability=row.get("capability", "").strip(),
                industry_primary=row.get("industry_primary", "").strip(),
                domain=row.get("domain", "").strip(),
                question_format=q_format,
                knowledge_text=knowledge_text,
            )
            if sample.question and sample.ref_answer:
                samples.append(sample)

    if num_samples:
        samples = samples[:num_samples]

    print(
        f"[INFO] 加载了 {len(samples)} 个问题样本 "
        f"(格式分布: {_format_distribution(samples)})"
    )
    return samples


def _infer_question_format(question: str) -> str:
    """从问题文本推断格式"""
    q = question.strip()
    if "___" in q or "____" in q or "＿" in q:
        return "FillBlank"
    if q.startswith(("A.", "A)")) or "A." in q[:20]:
        return "MultipleChoice"
    if any(kw in q for kw in ["多少", "计算", "数值", "参数", "mm", "kW", "A", "V", "Hz"]):
        return "Calculation"
    return "QA"


def _format_distribution(samples: List[QuestionSample]) -> str:
    counts = Counter(s.question_format for s in samples)
    return ", ".join(f"{k}={v}" for k, v in counts.most_common())


# ============================================================
# 3. 检索器 (沿用 exp1_qa 的 IndustrialRetriever)
# ============================================================

from retrieval.retriever import RetrievedChunk as _BaseRetrievedChunk


class RetrievedDocument:
    """
    检索到的单篇文档。与 exp1_qa 中 RetrievalResult 使用的类型一致。
    """
    def __init__(
        self,
        chunk_id: str = "",
        document_id: str = "",
        source: str = "",
        content: str = "",
        score: float = 0.0,
        rank: int = 0,
        industry: str = "",
        capability: str = "",
        citation: str = "",
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.source = source
        self.content = content
        self.score = score
        self.rank = rank
        self.industry = industry
        self.capability = capability
        self.citation = citation


class RetrievalResult:
    """检索返回结果 (兼容 exp1_qa 的接口)"""
    def __init__(
        self,
        query: str = "",
        query_expanded: str = "",
        documents: Optional[List[RetrievedDocument]] = None,
        timing_ms: float = 0.0,
    ):
        self.query = query
        self.query_expanded = query_expanded
        self.documents = documents or []
        self.chunk_ids = [d.chunk_id for d in self.documents]
        self.scores = [d.score for d in self.documents]
        self.sources = [d.source for d in self.documents]
        self.citations = [d.citation or f"[{d.rank}]" for d in self.documents]
        self.timing_ms = timing_ms

    def get_context(self, separator: str = "\n---\n") -> str:
        parts = []
        for d in self.documents:
            parts.append(
                f"[{d.rank}] (score={d.score:.3f}) "
                f"[{d.industry}/{d.capability}]\n{d.content}"
            )
        return separator.join(parts) if parts else ""


class IndustrialRetriever:
    """
    工业开放域检索器 (与 exp1_qa 完全一致，但返回兼容的 RetrievalResult)
    """
    def __init__(self, corpus_dir: Optional[str] = None, retriever_k: int = 10):
        self.corpus_dir = corpus_dir or os.path.join(
            _PROJECT_ROOT, "knowledge_corpus"
        )
        self.retriever_k = retriever_k
        self._retriever = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            from retrieval.retriever import OpenDomainRetriever

            self._retriever = OpenDomainRetriever(project_root=_PROJECT_ROOT)
            self._retriever.load_from_manifest(
                manifest_path=os.path.join(self.corpus_dir, "manifest.json")
            )
            self._initialized = True

            noise_count = 0
            for cid, chunk in self._retriever.chunks.items():
                if chunk.get("category") == "distractor" or cid.startswith("noise_"):
                    noise_count += 1
            total_chunks = len(self._retriever.chunks)
            print(
                f"[INFO] IndustrialRetriever 就绪: {total_chunks} chunks "
                f"(含 {noise_count} 篇噪声/干扰文档)"
            )

        except (ImportError, FileNotFoundError) as e:
            print(f"[WARN] 无法初始化 IndustrialRetriever: {e}")
            self._initialized = True

    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        self._ensure_initialized()
        if k is None:
            k = self.retriever_k

        result = RetrievalResult(query=query)

        if not self._initialized or self._retriever is None:
            print(f"[WARN] 检索器未就绪，返回空结果")
            return result

        start_time = time.time()
        try:
            retrieval_result = self._retriever.retrieve(query, k=k, use_ked=True)
            result.query_expanded = retrieval_result.query_expanded
            result.timing_ms = retrieval_result.timing_ms

            for chunk in retrieval_result.chunks:
                doc = RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    content=chunk.content,
                    score=chunk.score,
                    rank=chunk.rank,
                    industry=chunk.industry,
                    capability=chunk.capability,
                    citation=f"[{chunk.rank}]",
                )
                result.documents.append(doc)
                result.chunk_ids.append(chunk.chunk_id)
                result.scores.append(chunk.score)
                result.sources.append(chunk.source)
                result.citations.append(f"[{chunk.rank}]")

        except Exception as e:
            print(f"[ERROR] 检索失败: {e}")

        result.timing_ms = (time.time() - start_time) * 1000
        return result


# ============================================================
# 4. Agentic RAG Pipeline (核心新增模块)
# ============================================================

from agentic import (
    TaskAnalyzer,
    StrategyPlanner,
    EvidenceOrganizer,
    TaskSolver,
    TaskType,
    TaskAnalysis,
    AdaptiveAgenticPipeline,
    PipelineResult,
    GraphNode,
    ExecutionGraph,
)


class _DocWrapper:
    """包装 RetrievedDocument 使其兼容 agentic 管线的 RetrievedChunk 接口。

    将 exp1_qa.IndustrialRetriever 返回的 RetrievedDocument 对象
    转换为 agentic.pipeline 模块期望的 RetrievedChunk 兼容接口。
    """
    def __init__(self, doc):
        self.chunk_id = doc.chunk_id
        self.document_id = getattr(doc, 'document_id', '')
        self.source = getattr(doc, 'source', '')
        self.content = doc.content
        self.score = getattr(doc, 'score', 0.0)
        self.rank = getattr(doc, 'rank', 0)
        self.industry = getattr(doc, 'industry', '')
        self.capability = getattr(doc, 'capability', '')

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "content": self.content[:200] if self.content else "",
            "score": round(self.score, 4),
            "rank": self.rank,
            "industry": self.industry,
            "capability": self.capability,
        }


class AgenticRAGEngine:
    """
    Agentic RAG 引擎 (REDESIGNED: 使用 AdaptiveAgenticPipeline)。

    封装完整的自适应 Agentic 管线，提供与 exp1_qa 兼容的 answer() 接口。

    REDESIGN: 使用 AdaptiveAgenticPipeline 作为核心驱动。
    TaskAnalyzer 的输出直接影响检索、组织、推理全流程。

    使用说明:
        engine = AgenticRAGEngine(deepseek_key)
        answer, retrieval_result = engine.answer("What is ...?")
    """

    def __init__(
        self,
        deepseek_key: str,
        model_name: str = "deepseek-chat",
        analyzer_temperature: float = 0.1,
        solver_temperature: float = 0.3,
        max_evidence: int = 20,
        verbose: bool = False,
        default_retrieve_k: int = 10,
        corpus_dir: Optional[str] = None,
        neutralize_task: bool = False,
    ):
        self.deepseek_key = deepseek_key
        self.model_name = model_name
        self.verbose = verbose
        self.default_retrieve_k = default_retrieve_k
        self._llm_client = None
        self._retriever = None
        self._pipeline = None
        self._last_pipeline_result = None
        self.corpus_dir = corpus_dir
        # 旁路开关：True 时强制 TaskAnalysis 的 task 维度回落到 general，
        # 使 planner 建图 / prompt_builder 模板 / organizer 证据优先级全部走中性路径，
        # 从而 task 仅作观测、不参与行为决策。保留 TaskAnalyzer 代码供复用。
        self._neutralize_task = neutralize_task


        self._init_llm()
        self._init_pipeline(
            analyzer_temperature=analyzer_temperature,
            solver_temperature=solver_temperature,
            max_evidence=max_evidence,
        )
        # Initialize an external IndustrialRetriever for fixed retrieval,
        # independent of the Agent's internal retriever.
        self._external_retriever = self._init_external_retriever()

    def _init_external_retriever(self):
        """
        Initialize the Stable Knowledge Interface retriever.
        This retriever is called BEFORE the Agent, uses fixed k=10,
        and its results are passed to the Agent as pre_retrieved_evidence.
        The Agent's internal retrieve nodes are then skipped.
        """
        try:
            from experiments.exp1_qa import IndustrialRetriever
            retriever = IndustrialRetriever(corpus_dir=self.corpus_dir)
            print("[AgenticRAGEngine] External retriever (Stable Knowledge Interface) ready")
            return retriever
        except Exception as e:
            print(f"[AgenticRAGEngine] WARNING: External retriever init failed: {e}")
            return None

    def answer(
        self,
        question: str,
        retrieval_k: Optional[int] = None,
        use_adaptive_retrieval: bool = True,
        use_adaptive_organization: bool = True,
        use_adaptive_reasoning: bool = True,
        format: str = "",
    ) -> Tuple[str, RetrievalResult]:
        """
        执行完整自适应 Agentic RAG 管线，返回 (answer, retrieval_result).

        ARCHITECTURE (Redesigned):
            Question
               ↓
            [1] IndustrialRetriever.retrieve(question, k=10)   ← Stable Knowledge Interface
               ↓
            [2] TaskAnalyzer.analyze(question, pre_retrieved_evidence) ← Agent sees the docs
               ↓
            [3] StrategyPlanner → ExecutionGraph (organize → reason → end)
               ↓
            [4] GraphExecutor (skips ALL retrieve nodes, uses pre_retrieved_evidence)
               ↓
            [5] Answer

            The retriever is a STABLE KNOWLEDGE INTERFACE:
              - Always retrieve(question, k=10) — fixed, no Agent override
              - The Agent (Analyzer/Planner) NO LONGER controls retrieval params
              - The Agent can ONLY organize, reason, and synthesize from the fixed retrieval

        Args:
            question: 问题
            retrieval_k: 覆盖检索 top-k 数量 (仅在不使用自适应检索时生效)
            use_adaptive_retrieval: 是否使用自适应检索参数
            use_adaptive_organization: 是否使用自适应证据组织
            use_adaptive_reasoning: 是否使用自适应推理流程
            format: 题目题型/格式真值（如 CSV `_format`: 问答题/QA）。透传给
                    pipeline.run(..., format=format) → analyzer.analyze(question, format=format)，
                    落地「analyze 直接读 format」：TaskAnalysis.format 直接归一化读入 ground-truth，
                    而非 heuristic。为空串时保持既有行为。

        Returns:
            (answer_text, retrieval_result)
        """
        # ── Step 1: Fixed external retrieval (Stable Knowledge Interface) ──
        # Always use fixed k=10, independent of the Agent.
        # This mimics exp1_qa style: retriever.retrieve(question, k=10)
        retrieve_k = retrieval_k or self.default_retrieve_k
        retrieval_result = RetrievalResult(query=question)

        if self._external_retriever is not None:
            try:
                retrieval_result = self._external_retriever.retrieve(
                    question, k=retrieve_k
                )
                if self.verbose:
                    print(f"[ExternalRetriever] Retrieved {len(retrieval_result.documents)} docs (k={retrieve_k})")
            except Exception as e:
                print(f"[ExternalRetriever] Retrieval failed: {e}")
                retrieval_result = RetrievalResult(query=question)

        # Convert RetrievalResult documents to List[RetrievedChunk] for the Agent
        # so the GraphExecutor can consume them as pre_retrieved_evidence.
        pre_retrieved_chunks = []
        if retrieval_result.documents:
            for doc in retrieval_result.documents:
                # RetrievedDocument → RetrievedChunk-like object
                chunk = _DocWrapper(doc)
                pre_retrieved_chunks.append(chunk)

        # ── Step 2: Run the Agent pipeline with pre_retrieved_evidence ──
        all_adaptive = (
            use_adaptive_retrieval
            and use_adaptive_organization
            and use_adaptive_reasoning
        )

        if all_adaptive:
            pipeline_result = self._pipeline.run(
                question=question,
                retriever_kwargs=({"k": retrieval_k} if retrieval_k else None),
                pre_retrieved_evidence=pre_retrieved_chunks if pre_retrieved_chunks else None,
                format=format,
            )
        else:
            pipeline_result = self._pipeline.run_with_ablation(
                question=question,
                ablation_config={
                    "adaptive_retrieval": use_adaptive_retrieval,
                    "adaptive_organize": use_adaptive_organization,
                    "adaptive_reasoning": use_adaptive_reasoning,
                    "conditional_branching": use_adaptive_reasoning,
                    "multi_hop_retrieval": use_adaptive_retrieval and use_adaptive_organization,
                },
                retriever_kwargs=({"k": retrieval_k} if retrieval_k else None),
                pre_retrieved_evidence=pre_retrieved_chunks if pre_retrieved_chunks else None,
                format=format,
            )

        self._last_pipeline_result = pipeline_result

        if isinstance(pipeline_result, dict):
            from agentic.pipeline import PipelineResult as _wrap_result
            pipeline_result = _wrap_result(pipeline_result)
            self._last_pipeline_result = pipeline_result

        answer = pipeline_result.answer

        return answer, retrieval_result

    def _init_llm(self):
        """初始化 LLM 客户端 (DeepSeek)"""
        from openai import OpenAI

        self._llm_client = OpenAI(
            api_key=self.deepseek_key,
            base_url="https://api.deepseek.com",
        )

    def _init_pipeline(
        self,
        analyzer_temperature: float = 0.1,
        solver_temperature: float = 0.3,
        max_evidence: int = 20,
    ):
        """初始化 Adaptive Agentic Pipeline (v2 ExecutionGraph)"""
        from retrieval.retriever import OpenDomainRetriever

        # Retriever (复用现有实现)
        retriever = OpenDomainRetriever(project_root=_PROJECT_ROOT)
        try:
            retriever.load_from_manifest(
                manifest_path=os.path.join(
                    KNOWLEDGE_CORPUS_DIR, "manifest.json"
                )
            )
        except FileNotFoundError:
            # 自动发现
            retriever.load_from_manifest()

        self._retriever = retriever

        # 创建组件实例 (v2 设计: 每个组件独立实例化)
        from agentic import TaskAnalyzer, StrategyPlanner, EvidenceOrganizer, TaskSolver

        analyzer = TaskAnalyzer(
            llm_client=self._llm_client,
            model_name=self.model_name,
            temperature=analyzer_temperature,
            neutralize_task=getattr(self, "_neutralize_task", False),
        )

        planner = StrategyPlanner(
            llm_client=self._llm_client,
            model_name=self.model_name,
        )
        # 生产直接开 L2 组织（词法重排+低分端截断）：RerankTruncOrganizer。
        # 裁决复测结论：organize_v3 avg 1.700 vs base 1.667（+0.033，不显著，升5降3平22），
        # 无系统性伤害、检索/SV 无污染；保持落地。回退：改回 EvidenceOrganizer()。
        from agentic.organizer import RerankTruncOrganizer
        organizer = RerankTruncOrganizer()
        solver = TaskSolver(
            llm_client=self._llm_client,
            model_name=self.model_name,
            temperature=solver_temperature,
        )

        # 使用 AdaptiveAgenticPipeline (v2) - 接收完整组件实例
        self._pipeline = AdaptiveAgenticPipeline(
            analyzer=analyzer,
            planner=planner,
            retriever=retriever,
            organizer=organizer,
            solver=solver,
            llm_client=self._llm_client,
        )

        # 保留组件引用以备外部访问
        self._analyzer = analyzer
        self._planner = planner
        self._organizer = organizer
        self._solver = solver

    def _build_retrieval_result(
        self,
        question: str,
        pipeline_result: Any,
    ) -> RetrievalResult:
        """
        从 PipelineResult 构建 exp1_qa 兼容的 RetrievalResult。

        PipelineResult 是 SimpleNamespace，包含:
            .answer, .task_analysis, .task, .execution_graph,
            .execution_nodes, .node_count, .result_dict

        evidence 字符串从 .result_dict["evidence"] 获取，
        但转换为 RetrievalResult 所需的文档列表需要从 execution_nodes 重建。
        """
        retrieval_result = RetrievalResult(
            query=question,
            query_expanded=question,
            timing_ms=0.0,  # 管线不追踪单独检索时间
        )

        # 从 result_dict 提取有用信息
        result_dict = getattr(pipeline_result, "result_dict", {})

        # 从 execution_graph 和 execution_nodes 重建检索结果
        # execution_graph 包含检索参数信息
        execution_graph = getattr(pipeline_result, "execution_graph", {})
        graph_nodes = execution_graph.get("nodes", {}) if execution_graph else {}

        # 从 execution_nodes 日志中提取检索信息
        execution_nodes = getattr(pipeline_result, "execution_nodes", [])

        # 从 evidence 字符串重建文档列表 (逐行解析)
        evidence_str = result_dict.get("evidence", "")
        num_evidence = result_dict.get("num_evidence", 0)

        if evidence_str and num_evidence > 0:
            # 尝试从 evidence 字符串解析出文档片段
            # 格式: [N] (score=0.xxx) [industry/capability]\ncontent\n---\n
            sections = evidence_str.split("\n---\n")
            for rank, section in enumerate(sections):
                if not section.strip():
                    continue
                content = section
                score = 0.0
                # 尝试提取 score
                import re
                score_match = re.search(r"score=([\d.]+)", section)
                if score_match:
                    score = float(score_match.group(1))
                    # 移除 metadata 行只保留内容
                    content = re.sub(
                        r"^\[.*?\]\s*\(score=[\d.]+\)\s*\[.*?\].*\n",
                        "",
                        section,
                        flags=re.MULTILINE,
                    ).strip()

                retrieved_doc = RetrievedDocument(
                    chunk_id=f"chunk_{rank}",
                    document_id=f"doc_{rank}",
                    source="knowledge_corpus",
                    content=content,
                    score=score,
                    rank=rank + 1,
                    citation=f"[{rank + 1}]",
                )
                retrieval_result.documents.append(retrieved_doc)
                retrieval_result.chunk_ids.append(retrieved_doc.chunk_id)
                retrieval_result.scores.append(retrieved_doc.score)

        # 也尝试从 execution_nodes 日志中提取检索描述
        retrieval_logs = [
            n for n in execution_nodes
            if isinstance(n, dict) and n.get("type") == "retrieve"
        ]
        if retrieval_logs and not retrieval_result.documents:
            # 至少记录检索发生了
            retrieval_result.timing_ms = len(retrieval_logs) * 100

        return retrieval_result

    @property
    def last_pipeline_result(self) -> Optional[PipelineResult]:
        """获取最后一次管线执行结果 (用于详细分析)"""
        return self._last_pipeline_result


# ============================================================
# 5. 评分器 (与 exp1_qa 完全一致)
# ============================================================

from metrics.industrybench_scorer import (
    RuleBasedScorer,
    IndustryBenchScorer,
)
from eval_scripts.industrial_linkeval.retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalScoreResult,
)


class EnhancedScorer:
    """
    增强评分器 (与 exp1_qa 完全一致)
    """
    def __init__(self, mode: str = "rule", api_key: Optional[str] = None):
        self.mode = mode
        if mode == "llm":
            self.qa_scorer = IndustryBenchScorer(api_key=api_key)
            self.judge = self.qa_scorer.judge
        else:
            self.qa_scorer = RuleBasedScorer()

        self.retrieval_evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

    def score(
        self,
        question: str,
        prediction: str,
        reference: str,
        retrieved_documents: Optional[list] = None,
        citations: Optional[list] = None,
        knowledge_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mode == "llm":
            score_result = self.judge.score(question, reference, prediction)
            if knowledge_text:
                sv_result = self.judge.check_safety(knowledge_text, question, prediction)
                has_violation = sv_result.has_violation
                violation_detail = sv_result.violation_detail
            else:
                has_violation = False
                violation_detail = ""
            adjusted_score = 0.0 if has_violation else float(score_result.raw_score)
            explanation = score_result.explanation
        else:
            raw_score = self.qa_scorer.rule_based_score(question, reference, prediction)
            has_violation = False
            violation_detail = ""
            if knowledge_text:
                has_violation = self.qa_scorer.check_safety_simple(knowledge_text, prediction)
            adjusted_score = 0.0 if has_violation else float(raw_score)
            explanation = "rule-based coverage score"

        return {
            "raw_score": raw_score if isinstance(raw_score, int) else raw_score,
            "adjusted_score": adjusted_score,
            "has_violation": has_violation,
            "violation_detail": violation_detail,
            "explanation": explanation,
        }

    def evaluate_retrieval(
        self,
        retrieved_document_ids: List[str],
        relevant_document_ids: List[str],
    ) -> Dict[str, float]:
        result = self.retrievalevaluator.evaluate(
            retrieved_ids=retrieved_document_ids,
            relevant_ids={rid for rid in relevant_document_ids},
        )
        metrics = {}
        for i, k in enumerate(self.retrieval_evaluator.k_values):
            if i < len(result.recall_at_k):
                metrics[f"recall@{k}"] = result.recall_at_k[i]
            if i < len(result.precision_at_k):
                metrics[f"precision@{k}"] = result.precision_at_k[i]
        for k in self.retrieval_evaluator.k_values:
            top_k_ids = set(retrieved_document_ids[:k])
            relevant_set = {rid for rid in relevant_document_ids}
            hit = 1.0 if top_k_ids & relevant_set else 0.0
            metrics[f"hit@{k}"] = hit
        metrics["mrr"] = result.mrr
        metrics["ndcg"] = result.ndcg
        return metrics

    def evaluate_retrieval_semantic(
        self,
        retrieved_chunks: List[RetrievedDocument],
        knowledge_text: str,
        threshold: float = 0.15,
    ) -> Dict[str, float]:
        """语义检索评估 (与 exp1_qa 完全一致)"""
        if not knowledge_text or not retrieved_chunks:
            return {
                "semantic_hit@1": 0.0,
                "semantic_hit@3": 0.0,
                "semantic_hit@5": 0.0,
                "semantic_hit@10": 0.0,
                "semantic_mrr": 0.0,
                "semantic_ndcg": 0.0,
                "matched_count": 0,
                "semantic_threshold": threshold,
            }

        import re

        def _word_set(text: str) -> Set[str]:
            words = set()
            for w in re.findall(r"[a-zA-Z0-9]+", text):
                if len(w) >= 2:
                    words.add(w.lower())
            text_cn = re.sub(r"[a-zA-Z0-9\s]", "", text)
            for i in range(len(text_cn) - 1):
                words.add(text_cn[i : i + 2])
            return words

        kt_words = _word_set(knowledge_text)
        if not kt_words:
            return {
                "semantic_hit@1": 0.0,
                "semantic_hit@3": 0.0,
                "semantic_hit@5": 0.0,
                "semantic_hit@10": 0.0,
                "semantic_mrr": 0.0,
                "semantic_ndcg": 0.0,
                "matched_count": 0,
                "semantic_threshold": threshold,
            }

        relevant_flags = []
        for doc in retrieved_chunks:
            chunk_words = _word_set(doc.content)
            if not chunk_words:
                relevant_flags.append(False)
                continue
            jaccard = len(kt_words & chunk_words) / len(kt_words | chunk_words)
            relevant_flags.append(jaccard >= threshold)

        k_values = [1, 3, 5, 10]
        metrics = {}
        for k in k_values:
            top_k = relevant_flags[:k]
            metrics[f"semantic_hit@{k}"] = 1.0 if any(top_k) else 0.0

        semantic_mrr = 0.0
        for i, flag in enumerate(relevant_flags):
            if flag:
                semantic_mrr = 1.0 / (i + 1)
                break
        metrics["semantic_mrr"] = semantic_mrr

        dcg = 0.0
        idcg = 1.0
        for i, flag in enumerate(relevant_flags[:10]):
            if flag:
                dcg += 1.0 / (i + 1)
        metrics["semantic_ndcg"] = dcg / idcg if idcg > 0 else 0.0
        metrics["matched_count"] = sum(relevant_flags)
        metrics["semantic_threshold"] = threshold

        return metrics


# ============================================================
# 6. 实验主逻辑
# ============================================================

def run_agentic_rag_experiment(
    mode: str = "agentic_rag",
    num_samples: int = 100,
    scorer_mode: str = "rule",
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    corpus_dir: Optional[str] = None,
    verbose: bool = True,
    neutralize_task: bool = False,
) -> Dict[str, Any]:

    """
    运行 Agentic RAG 实验 (兼容 exp1_qa 的评估框架)。

    Pipeline:
        Dataset → Industrial Retriever → Scorer → Report

    其中 "回答生成" 阶段根据 mode 选择不同管线:
        - agentic_rag: TaskAnalyzer → StrategyPlanner → Retriever
                       → EvidenceOrganizer → TaskSolver → Answer
        - rag / standard_rag / closed_book / quick: 复用 exp1_qa 逻辑

    Args:
        mode: 实验模式 (agentic_rag, rag, closed_book, quick, standard_rag)
        num_samples: 评估样本数
        scorer_mode: 评分器模式 ("rule" 或 "llm")
        output_dir: 输出目录
        run_id: 运行 ID
        corpus_dir: 知识库目录
        verbose: 是否打印详细信息

    Returns:
        stats: 实验统计结果
    """
    register_paths()

    # ---- 准备 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"exp1_agentic_rag_{timestamp}"
    mode_info = MODES.get(mode, MODES["agentic_rag"])

    print(f"\n{'=' * 70}")
    print(f"  🏭 Exp1 Agentic RAG")
    print(f"  Mode: {mode_info['name']}")
    print(f"  Run ID: {run_id}")
    print(f"  Samples: {num_samples}")
    print(f"{'=' * 70}\n")

    # ---- 加载数据 ----
    print("📂 加载数据集...")
    csv_path = corpus_dir or INDUSTRYBENCH_CSV
    question_samples = load_question_dataset(csv_path, num_samples=num_samples)

    if not question_samples:
        print("[ERROR] 未加载到任何样本，退出")
        return {"error": "no_samples"}

    # ---- 初始化组件 ----
    is_agentic = mode == "agentic_rag"

    if is_agentic:
        # Agentic RAG 模式: 使用 AgenticRAGEngine
        print("\n🤖 初始化 Agentic RAG Engine...")
        agentic_engine = AgenticRAGEngine(
            deepseek_key=DEEPSEEK_KEY,
            model_name=LLM_NAME,
            verbose=verbose,
            neutralize_task=neutralize_task,
        )

        retriever = None  # Agentic engine handles its own retriever
    else:
        # 兼容模式: 使用原有 IndustrialLINSWrapper
        print("\n🔧 初始化 IndustrialLINSWrapper...")
        from model.model_LINS import LINS

        lins = LINS(
            LLM_name="deepseek-chat",
            assistant_LLM_name="deepseek-chat",
            retriever_name="BGE",
            DeepSeek_keys=DEEPSEEK_KEY,
            database_name="none",
        )
        retriever = IndustrialRetriever(corpus_dir=corpus_dir)

    # 评分器 (与 exp1_qa 一致)
    print(f"📊 初始化评分器 (mode={scorer_mode})...")
    scorer = EnhancedScorer(mode=scorer_mode, api_key=DEEPSEEK_KEY)

    # ---- 执行评估 ----
    results: List[Dict[str, Any]] = []
    total_time = 0.0
    all_semantic_hit_at_1: List[float] = []
    all_semantic_hit_at_3: List[float] = []
    all_semantic_hit_at_5: List[float] = []
    all_semantic_hit_at_10: List[float] = []
    all_semantic_mrr: List[float] = []
    all_semantic_ndcg: List[float] = []

    # 任务分布统计
    task_distribution = Counter()

    print(f"\n{'─' * 70}")
    print(f"📝 开始评估 ({len(question_samples)} 样本)")
    print(f"{'─' * 70}")

    for idx, sample in enumerate(question_samples):
        print(f"\n[{idx + 1}/{len(question_samples)}] {sample.question[:80]}...")
        start_time_sample = time.time()

        # ---- Step A: 生成回答 ----
        try:
            if is_agentic:
                answer, retrieval_result = agentic_engine.answer(
                    question=sample.question,
                    retrieval_k=mode_info.get("retrieval_k", 10),
                )
            else:
                # 复用 exp1_qa 模式
                use_retriever = mode_info.get("use_retriever", False)
                if mode == "quick":
                    context = sample.knowledge_text or ""
                    response, urls, passages, history, sub_qs = lins.MAIRAG(
                        question=sample.question,
                        context=context,
                        topk=mode_info["topk"],
                        if_PRA=mode_info["if_PRA"],
                        if_SKA=mode_info["if_SKA"],
                        if_QDA=mode_info["if_QDA"],
                        if_PCA=mode_info["if_PCA"],
                        recall_top_k=mode_info["recall_top_k"],
                    )
                    answer = response or ""
                    retrieval_result = RetrievalResult(
                        query=sample.question,
                        query_expanded=sample.question,
                    )

                elif mode == "closed_book":
                    cb_prompt = (
                        "You are an industrial domain question-answering assistant.\n\n"
                        "Answer the following question as accurately as possible "
                        "using your internal knowledge.\n\n"
                        "[Question]\n{question}\n\n"
                        "Provide a clear and concise final answer."
                    ).format(question=sample.question)
                    response, history = lins.chat(question=cb_prompt)
                    answer = response or ""
                    retrieval_result = RetrievalResult(
                        query=sample.question,
                        query_expanded=sample.question,
                    )

                elif mode == "rag":
                    retrieval_result = retriever.retrieve(
                        sample.question, k=mode_info.get("retrieval_k", 10)
                    )
                    if retrieval_result.documents:
                        context = retrieval_result.get_context()
                        response, urls, passages, history, sub_qs = lins.MAIRAG(
                            question=sample.question,
                            context=context,
                            topk=mode_info["topk"],
                            if_PRA=mode_info["if_PRA"],
                            if_SKA=mode_info["if_SKA"],
                            if_QDA=mode_info["if_QDA"],
                            if_PCA=mode_info["if_PCA"],
                            recall_top_k=mode_info["recall_top_k"],
                        )
                    else:
                        response, urls, passages, history, sub_qs = lins.MAIRAG(
                            question=sample.question,
                            topk=3,
                            if_PRA=False,
                            if_SKA=False,
                            if_QDA=False,
                            if_PCA=False,
                            recall_top_k=3,
                        )
                    answer = response or ""

                elif mode == "standard_rag":
                    retrieval_result = retriever.retrieve(
                        sample.question, k=mode_info.get("retrieval_k", 10)
                    )
                    if retrieval_result.documents:
                        context = retrieval_result.get_context()
                        rag_prompt = (
                            "You are an industrial domain question-answering assistant.\n\n"
                            "Use the following retrieved knowledge to answer "
                            "the question accurately.\n\n"
                            "## Retrieved Knowledge:\n{context}\n\n"
                            "## Question:\n{question}\n\n"
                            "## Instructions:\n"
                            "- Answer based on the retrieved knowledge above\n"
                            "- If the retrieved knowledge does not contain enough information, say so\n"
                            "- Provide a clear and concise answer\n"
                            "- Do NOT cite sources with numbers like [1], [2]\n\n"
                            "Answer:"
                        ).format(context=context, question=sample.question)
                        response, history = lins.chat(question=rag_prompt)
                    else:
                        closed_book_prompt = (
                            "You are an industrial domain question-answering assistant.\n\n"
                            "Answer the following question as accurately as possible "
                            "using your internal knowledge.\n\n"
                            "[Question]\n{question}\n\n"
                            "Provide a clear and concise final answer."
                        ).format(question=sample.question)
                        response, history = lins.chat(question=closed_book_prompt)
                    answer = response or ""

            # ---- Step B: 评分 ----
            knowledge_text_for_scoring = sample.knowledge_text or ""
            score_result = scorer.score(
                question=sample.question,
                prediction=answer,
                reference=sample.ref_answer,
                retrieved_documents=getattr(retrieval_result, 'documents', []),
                citations=getattr(retrieval_result, 'citations', []),
                knowledge_text=knowledge_text_for_scoring,
            )

            # 语义检索评估
            semantic_metrics = scorer.evaluate_retrieval_semantic(
                retrieved_chunks=getattr(retrieval_result, 'documents', []),
                knowledge_text=knowledge_text_for_scoring,
                threshold=0.15,
            )

            # ---- 记录结果 ----
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

            # 语义检索指标
            result_entry.update({
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in semantic_metrics.items()
            })

            results.append(result_entry)

            # 收集语义指标
            all_semantic_hit_at_1.append(semantic_metrics.get("semantic_hit@1", 0.0))
            all_semantic_hit_at_3.append(semantic_metrics.get("semantic_hit@3", 0.0))
            all_semantic_hit_at_5.append(semantic_metrics.get("semantic_hit@5", 0.0))
            all_semantic_hit_at_10.append(semantic_metrics.get("semantic_hit@10", 0.0))
            all_semantic_mrr.append(semantic_metrics.get("semantic_mrr", 0.0))
            all_semantic_ndcg.append(semantic_metrics.get("semantic_ndcg", 0.0))

            # 显示当前结果
            print(f"  Score: {score_result['raw_score']} | "
                  f"Adjusted: {score_result['adjusted_score']:.2f} | "
                  f"SV: {score_result['has_violation']} | "
                  f"Time: {elapsed:.1f}s | "
                  f"{'❌' if score_result['has_violation'] else '✅'}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ============================================================
    # 7. 汇总统计
    # ============================================================

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

    # 语义检索汇总
    avg_semantic_hit_at_1 = sum(all_semantic_hit_at_1) / n if n > 0 else 0.0
    avg_semantic_hit_at_3 = sum(all_semantic_hit_at_3) / n if n > 0 else 0.0
    avg_semantic_hit_at_5 = sum(all_semantic_hit_at_5) / n if n > 0 else 0.0
    avg_semantic_hit_at_10 = sum(all_semantic_hit_at_10) / n if n > 0 else 0.0
    avg_semantic_mrr = sum(all_semantic_mrr) / n if n > 0 else 0.0
    avg_semantic_ndcg = sum(all_semantic_ndcg) / n if n > 0 else 0.0

    # 按难度统计
    difficulty_scores = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulty_scores:
            difficulty_scores[diff] = {"sum": 0, "count": 0, "violations": 0}
        difficulty_scores[diff]["sum"] += r["score_raw"]
        difficulty_scores[diff]["count"] += 1
        if r["has_violation"]:
            difficulty_scores[diff]["violations"] += 1

    # 按能力统计
    capability_scores = {}
    for r in results:
        cap = r["capability"]
        if cap not in capability_scores:
            capability_scores[cap] = {"sum": 0, "count": 0, "violations": 0}
        capability_scores[cap]["sum"] += r["score_raw"]
        capability_scores[cap]["count"] += 1
        if r["has_violation"]:
            capability_scores[cap]["violations"] += 1

    # ============================================================
    # 输出汇总
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"  📊 实验结果汇总")
    print(f"{'=' * 70}")
    print(f"  模式: {mode_info['name']}")
    print(f"  运行 ID: {run_id}")
    print(f"  样本数: {n}")
    print(f"  总分 (raw): {total_raw}/{max_possible} = {accuracy_raw:.2%}")
    print(f"  总分 (adjusted): {total_adjusted:.1f}/{max_possible} = {accuracy_adjusted:.2%}")
    print(f"  平均分 (raw): {avg_raw:.2f}/3.0")
    print(f"  平均分 (adjusted): {avg_adjusted:.2f}/3.0")
    print(f"  SV 违规: {violations}/{n} ({violation_percent:.1f}%)")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均耗时: {total_time / n:.1f}s/样本" if n > 0 else "")
    print(f"\n  📈 语义检索指标:")
    print(f"    Semantic Hit@1: {avg_semantic_hit_at_1:.4f}")
    print(f"    Semantic Hit@3: {avg_semantic_hit_at_3:.4f}")
    print(f"    Semantic Hit@5: {avg_semantic_hit_at_5:.4f}")
    print(f"    Semantic Hit@10: {avg_semantic_hit_at_10:.4f}")
    print(f"    Semantic MRR: {avg_semantic_mrr:.4f}")
    print(f"    Semantic NDCG: {avg_semantic_ndcg:.4f}")

    # 按难度汇总
    print(f"\n  📈 按难度:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulty_scores:
            d = difficulty_scores[diff]
            avg_d = d["sum"] / d["count"] if d["count"] > 0 else 0.0
            print(f"    {diff}: {d['count']} samples, avg={avg_d:.2f}/3.0, "
                  f"violations={d['violations']}")

    # 按能力汇总
    print(f"\n  📈 按能力:")
    for cap, d in sorted(capability_scores.items()):
        avg_c = d["sum"] / d["count"] if d["count"] > 0 else 0.0
        print(f"    {cap}: {d['count']} samples, avg={avg_c:.2f}/3.0, "
              f"violations={d['violations']}")

    print(f"\n{'=' * 70}")

    # ============================================================
    # 保存结果
    # ============================================================

    output_dir = output_dir or os.path.join(
        RESULTS_DIR, "experiments", run_id
    )
    os.makedirs(output_dir, exist_ok=True)

    # 详细结果 JSON
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果已保存: {detailed_path}")

    # 报告
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Exp1 Agentic RAG 实验报告\n\n")
        f.write(f"- **模式**: {mode_info['name']}\n")
        f.write(f"- **运行 ID**: {run_id}\n")
        f.write(f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **样本数**: {n}\n\n")
        f.write(f"## 总体结果\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|-----|\n")
        f.write(f"| 总分 (raw) | {total_raw}/{max_possible} ({accuracy_raw:.2%}) |\n")
        f.write(f"| 总分 (adjusted) | {total_adjusted:.1f}/{max_possible} ({accuracy_adjusted:.2%}) |\n")
        f.write(f"| 平均分 (raw) | {avg_raw:.2f}/3.0 |\n")
        f.write(f"| 平均分 (adjusted) | {avg_adjusted:.2f}/3.0 |\n")
        f.write(f"| SV 违规 | {violations}/{n} ({violation_percent:.1f}%) |\n")
        f.write(f"| 总耗时 | {total_time:.1f}s |\n")
        f.write(f"| 平均耗时 | {total_time / n:.1f}s |\n\n")
        f.write(f"## 语义检索指标\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|-----|\n")
        f.write(f"| Semantic Hit@1 | {avg_semantic_hit_at_1:.4f} |\n")
        f.write(f"| Semantic Hit@3 | {avg_semantic_hit_at_3:.4f} |\n")
        f.write(f"| Semantic Hit@5 | {avg_semantic_hit_at_5:.4f} |\n")
        f.write(f"| Semantic Hit@10 | {avg_semantic_hit_at_10:.4f} |\n")
        f.write(f"| Semantic MRR | {avg_semantic_mrr:.4f} |\n")
        f.write(f"| Semantic NDCG | {avg_semantic_ndcg:.4f} |\n\n")
        f.write(f"## 按难度\n\n")
        f.write(f"| 难度 | 样本数 | 平均分 | Violations |\n")
        f.write(f"|------|--------|--------|------------|\n")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_scores:
                d = difficulty_scores[diff]
                avg_d = d["sum"] / d["count"] if d["count"] > 0 else 0.0
                f.write(f"| {diff} | {d['count']} | {avg_d:.2f} | {d['violations']} |\n")
        f.write(f"\n## 按能力\n\n")
        f.write(f"| 能力 | 样本数 | 平均分 | Violations |\n")
        f.write(f"|------|--------|--------|------------|\n")
        for cap, d in sorted(capability_scores.items()):
            avg_c = d["sum"] / d["count"] if d["count"] > 0 else 0.0
            f.write(f"| {cap} | {d['count']} | {avg_c:.2f} | {d['violations']} |\n")

    print(f"📄 实验报告已保存: {report_path}")

    # CSV 导出
    csv_path_out = os.path.join(output_dir, "results.csv")
    if results:
        with open(csv_path_out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"📄 CSV 文件已保存: {csv_path_out}")

    # ============================================================
    # 统计汇总
    # ============================================================

    stats = {
        "run_id": run_id,
        "mode": mode,
        "mode_name": mode_info["name"],
        "num_samples": n,
        "total_raw_score": total_raw,
        "total_adjusted_score": total_adjusted,
        "max_possible_score": max_possible,
        "accuracy_raw": accuracy_raw,
        "accuracy_adjusted": accuracy_adjusted,
        "avg_raw_score": avg_raw,
        "avg_adjusted_score": avg_adjusted,
        "violations": violations,
        "violation_percent": violation_percent,
        "total_time_seconds": total_time,
        "avg_time_seconds": total_time / n if n > 0 else 0.0,
        "semantic_hit_at_1": avg_semantic_hit_at_1,
        "semantic_hit_at_3": avg_semantic_hit_at_3,
        "semantic_hit_at_5": avg_semantic_hit_at_5,
        "semantic_hit_at_10": avg_semantic_hit_at_10,
        "semantic_mrr": avg_semantic_mrr,
        "semantic_ndcg": avg_semantic_ndcg,
        "difficulty_breakdown": {
            d: {
                "count": info["count"],
                "avg_score": info["sum"] / info["count"],
                "violations": info["violations"],
            }
            for d, info in difficulty_scores.items()
        },
        "capability_breakdown": {
            c: {
                "count": info["count"],
                "avg_score": info["sum"] / info["count"],
                "violations": info["violations"],
            }
            for c, info in capability_scores.items()
        },
    }

    return stats


# ============================================================
# 8. CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp1: Agentic RAG - Task-aware Industrial RAG Evaluation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="agentic_rag",
        choices=list(MODES.keys()),
        help="Evaluation mode (default: agentic_rag)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="rule",
        choices=["rule", "llm"],
        help="Scorer mode (default: rule)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default=None,
        help="Knowledge corpus directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    parser.add_argument(
        "--neutralize-task",
        action="store_true",
        default=False,
        help=(
            "Neutralize the task dimension of TaskAnalyzer. When set, the "
            "8-class task falls back to 'general', so planner graph-building, "
            "prompt-builder template selection and organizer evidence priority "
            "all take the neutral path (task treated as observation-only).",
        ),
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # ---------- 运行实验 ----------
    stats = run_agentic_rag_experiment(
        mode=args.mode,
        num_samples=args.num_samples,
        scorer_mode=args.scorer,
        output_dir=args.output_dir,
        run_id=args.run_id,
        corpus_dir=args.corpus_dir,
        verbose=args.verbose,
        neutralize_task=getattr(args, "neutralize_task", False),
    )


    print(f"\n{'=' * 70}")
    print(f"  Exp1 Agentic RAG Complete!")
    print(f"  Mode: {MODES[args.mode]['name']}")
    print(f"  Run ID: {stats.get('run_id', 'N/A')}")
    print(f"  Accuracy: {stats.get('accuracy_raw', 0):.2%}")
    print(f"{'=' * 70}")

    return stats


if __name__ == "__main__":
    main()
