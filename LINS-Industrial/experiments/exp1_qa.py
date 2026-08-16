"""
exp1_qa.py: 实验一 - Industrial Open-Domain RAG Evaluation

评估 IndustrialLINS 在开放域工业知识问答场景下的表现。

核心架构变化:
  1. Dataset ⟂ Knowledge Corpus — 问题数据集与知识库彻底分离，不再假设"一问一文档"
  2. Open-Domain Retriever — 使用 industrial_retriever 从全量知识库检索 top-k
  3. RetrievalResult 对象 — 结构化保存检索结果、chunk_ids、scores、sources、citations
  4. 检索指标 + QA 指标同时输出
  5. Question Format 感知 — 自动加载 _format 字段
  6. Scorer API 升级 — score(question, prediction, reference, retrieved_documents, citations)
  7. 三种模式: quick(Oracle上限) / closed_book(下界) / rag(主实验)
  8. 知识源无关 — 仅依赖 knowledge_corpus

评估模式:
  - quick (Oracle 开卷): 注入 ground-truth knowledge_text，上限参考
  - closed_book (闭卷): 无外部知识，仅依赖模型自身
  - rag (检索增强): Question → KED → Industrial Retriever → Knowledge Corpus → Top-k → MAIRAG → Answer

输出:
  - JSON 详细结果 (含 retrieval_results, qa_metrics, retrieval_metrics)
  - Markdown 格式报告
"""

import os
import sys
import json
import csv
import time
import argparse
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

# ===== 路径配置 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')

for p in [PROJECT_ROOT, LINS_MAIN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics.industrybench_scorer import (
    EvalSample, FinalEvalResult, IndustryBenchScorer, RuleBasedScorer
)
from eval_scripts.industrial_linkeval.retrieval_evaluator import (
    RetrievalEvaluator, RetrievalScoreResult
)


# ============================================================
# 0. 新增数据结构 (Task 7: RetrievalResult)
# ============================================================

@dataclass
class RetrievedDocument:
    """单条检索结果文档"""
    chunk_id: str
    document_id: str = ""
    source: str = ""
    content: str = ""
    score: float = 0.0
    rank: int = 0
    industry: str = ""
    capability: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class RetrievalResult:
    """一次检索的完整结构化结果"""
    query: str
    query_expanded: str = ""
    documents: List[RetrievedDocument] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    timing_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_expanded": self.query_expanded,
            "num_results": len(self.documents),
            "chunk_ids": self.chunk_ids,
            "scores": [round(s, 4) for s in self.scores],
            "sources": self.sources,
            "citations": self.citations,
            "timing_ms": round(self.timing_ms, 1),
            "documents": [d.to_dict() for d in self.documents],
        }
    
    def get_context(self, separator: str = "\n---\n") -> str:
        """将所有结果拼接为 LLM 上下文字符串"""
        parts = []
        for d in self.documents:
            parts.append(f"[{d.rank}] (score={d.score:.3f}) [{d.industry}/{d.capability}]\n{d.content}")
        return separator.join(parts) if parts else ""
    
    def get_relevant_ids_for_metric(self) -> Set[str]:
        """获取检索到的 document_id 集合（用于评估指标）"""
        return {d.document_id for d in self.documents if d.document_id}


# ============================================================
# 1. 评估模式定义 (Task 8: 三模式区分)
# ============================================================

QA_MODES = {
    "quick": {
        "name": "Oracle 开卷 (Quick)",
        "desc": "直接注入 ground-truth knowledge_text，上限参考。不使用检索器。",
        "topk": 3,
        "use_retriever": False,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 10,
        "prompt_extra": "",
    },
    "rag": {
        "name": "开放域检索增强 (RAG)",
        "desc": "Question → KED → Industrial Retriever → Knowledge Corpus → Top-k → MAIRAG → Answer",
        "topk": 10,
        "use_retriever": True,
        "if_PRA": True,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 50,
        "prompt_extra": "\n请在你的回答中标注引用编号如 [1], [2] 以标明信息来源。",
        "retrieval_k": 10,  # 检索 top-k 数量
    },
    "closed_book": {
        "name": "闭卷 (Closed-Book)",
        "desc": "无外部知识，仅依赖模型内部知识，下界参考。",
        "topk": 1,
        "use_retriever": False,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 1,
        "prompt_extra": "",
    },
    "standard_rag": {
        "name": "标准 RAG Baseline (无 PRM)",
        "desc": "Question → Industrial Retriever → Top-k → Standard RAG Prompt → chat() 直接回答。\n与 rag 模式共享完全相同的检索管线，但使用纯 LLM 而非 MAIRAG。\n用于消融实验：衡量 MAIRAG(PRM+Rerank) 相对于普通 RAG 的提升。",
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
}



# ============================================================
# 2. 数据加载 (Task 1: 分离 Dataset 和 Knowledge Corpus)
# ============================================================

@dataclass
class QuestionSample:
    """
    仅包含问题和参考信息的数据点。
    
    注意: 不包含 knowledge_text。
    知识库是独立的 knowledge_corpus/，不从这里读取。
    """
    id: str
    question: str
    ref_answer: str
    difficulty: str                # easy / medium / hard
    capability: str                # 能力维度
    industry_primary: str          # 主行业
    domain: str                    # 领域
    question_format: str = "QA"    # (Task 5) QA / FillBlank / MultipleChoice / Calculation
    knowledge_text: str = ""       # 仅用于 quick (Oracle) 模式


def load_question_dataset(csv_path: str, num_samples: Optional[int] = None) -> List[QuestionSample]:
    """
    加载问题数据集 (Task 1: 不含知识库依赖)
    
    仅加载: id, question, answer, difficulty, capability, industry_primary, domain, _format
    不加载知识库内容 —— 知识库由独立的 Industrial Retriever 管理。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 加载问题格式 (Task 5)
            q_format = row.get('_format', row.get('format', row.get('question_format', ''))).strip()
            if not q_format:
                q_format = _infer_question_format(row.get('question', ''))
            
            # knowledge_text 仅保留用于 quick (Oracle) 模式
            knowledge_text = row.get('knowledge_text', '').strip()
            
            sample = QuestionSample(
                id=row.get('id', '').strip(),
                question=row.get('question', '').strip(),
                ref_answer=row.get('answer', '').strip(),
                difficulty=row.get('difficulty', '').strip().lower(),
                capability=row.get('capability', '').strip(),
                industry_primary=row.get('industry_primary', '').strip(),
                domain=row.get('domain', '').strip(),
                question_format=q_format,
                knowledge_text=knowledge_text,
            )
            if sample.question and sample.ref_answer:
                samples.append(sample)

    if num_samples:
        samples = samples[:num_samples]

    print(f"[INFO] 加载了 {len(samples)} 个问题样本 (格式分布: {_format_distribution(samples)})")
    return samples


def _infer_question_format(question: str) -> str:
    """从问题文本推断格式 (Task 5)"""
    q = question.strip()
    if '___' in q or '____' in q or '＿' in q:
        return "FillBlank"
    if q.startswith(('A.', 'A)')) or 'A.' in q[:20]:
        return "MultipleChoice"
    if any(kw in q for kw in ['多少', '计算', '数值', '参数', 'mm', 'kW', 'A', 'V', 'Hz']):
        return "Calculation"
    return "QA"


def _format_distribution(samples: List[QuestionSample]) -> str:
    """打印格式分布"""
    counts = Counter(s.question_format for s in samples)
    return ", ".join(f"{k}={v}" for k, v in counts.most_common())


# ============================================================
# 3. 工业开放域检索器 (Task 2: 替换默认 LINS 检索)
# ============================================================

class IndustrialRetriever:
    """
    工业开放域检索器。
    
    Pipeline:
        Question → KED → Embedding → FAISS Search → Top-k → RetrievalResult
    
    不假设"一个问题对应一篇文档"。
    从全量 knowledge_corpus 中检索最相关的 top-k 文本块。
    """
    
    def __init__(self, corpus_dir: str = None, retriever_k: int = 10):
        self.corpus_dir = corpus_dir or os.path.join(PROJECT_ROOT, 'knowledge_corpus')
        self.retriever_k = retriever_k
        self._retriever = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """延迟初始化检索器"""
        if self._initialized:
            return
        
        try:
            from retrieval.retriever import OpenDomainRetriever
            self._retriever = OpenDomainRetriever(project_root=PROJECT_ROOT)
            self._retriever.load_from_manifest(
                manifest_path=os.path.join(self.corpus_dir, 'manifest.json')
            )
            self._initialized = True
            
            # ===== Report: Noise corpus statistics =====
            noise_count = 0
            for cid, chunk in self._retriever.chunks.items():
                if chunk.get("category") == "distractor" or cid.startswith("noise_"):
                    noise_count += 1
            total_chunks = len(self._retriever.chunks)
            print(f"[INFO] IndustrialRetriever 就绪: {total_chunks} chunks (含 {noise_count} 篇噪声/干扰文档)")
            
        except (ImportError, FileNotFoundError) as e:
            print(f"[WARN] 无法初始化 IndustrialRetriever: {e}")
            print("[WARN] 将使用空检索结果，仅 closed_book 模式可用")
            self._initialized = True  # 标记已尝试
    
    def retrieve(self, query: str, k: int = None) -> RetrievalResult:
        """
        执行开放域检索。
        
        Args:
            query: 查询文本
            k: top-k 数量（默认使用配置值）
        
        Returns:
            RetrievalResult (Task 7)
        """
        self._ensure_initialized()
        
        if k is None:
            k = self.retriever_k
        
        result = RetrievalResult(query=query)
        
        if not self._initialized or self._retriever is None:
            print(f"[WARN] 检索器未就绪，返回空结果")
            return result
        
        start_time = time.time()
        
        try:
            # 调用 OpenDomainRetriever (含 KED + Embedding + FAISS)
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
                )
                result.documents.append(doc)
                result.chunk_ids.append(chunk.chunk_id)
                result.scores.append(chunk.score)
                result.sources.append(chunk.source)
                # 生成引用编号
                citation = f"[{chunk.rank}]"
                result.citations.append(citation)
        
        except Exception as e:
            print(f"[ERROR] 检索失败: {e}")
        
        result.timing_ms = (time.time() - start_time) * 1000
        
        return result


# ============================================================
# 4. LINS 模型包装器 (Task 2: 集成 IndustrialRetriever)
# ============================================================

class IndustrialLINSWrapper:
    """
    IndustrialLINS 模型包装器。
    
    三种模式:
      - quick (Oracle): 注入 ground-truth knowledge_text
      - closed_book: 无外部知识
      - rag: IndustrialRetriever → KED → Knowledge Corpus → MAIRAG
    """
    
    def __init__(self, deepseek_key: str):
        self.deepseek_key = deepseek_key
        self.lins = None
        self.retriever = None
        self._init_lins()
    
    def _init_lins(self):
        """初始化 LINS 模型"""
        from model.model_LINS import LINS
        self.lins = LINS(
            LLM_name='deepseek-chat',
            assistant_LLM_name='deepseek-chat',
            retriever_name='BGE',
            DeepSeek_keys=self.deepseek_key,
            database_name='none'
        )
    
    def _init_retriever(self):
        """初始化工业检索器"""
        if self.retriever is None:
            self.retriever = IndustrialRetriever()
    
    def answer_quick(self, question: str, knowledge_text: str) -> Tuple[str, RetrievalResult]:
        """
        Quick (Oracle) 模式: 直接注入 ground-truth knowledge_text。
        
        不使用检索器，直接通过 context 参数注入知识。
        MAIRAG 将跳过内部检索，直接使用传入的 evidence。
        """
        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=question,
            context=knowledge_text,    # ← 直接传 context，跳过检索
            topk=3,
            if_PRA=False, if_SKA=False, if_QDA=False, if_PCA=False,
            recall_top_k=10
        )
        
        # Oracle 模式无检索结果
        retrieval_result = RetrievalResult(
            query=question,
            query_expanded=question,
        )
        
        return response, retrieval_result
    
    def answer_closed_book(self, question: str) -> Tuple[str, RetrievalResult]:
        """
        闭卷模式 (Closed-Book): 仅依赖模型内部知识。
        
        关键区别:
        - 不使用 MAIRAG (避免任何检索管线干扰)
        - 使用 LINS.chat() 直接调用 LLM
        - Prompt 不暗示存在知识库，仅要求用内部知识回答
        
        这是纯粹的 Closed-book DeepSeek Baseline:
            Question → DeepSeek → Answer
        
        作为 RAG 实验的下界 (lower bound) 参考。
        """
        # ===== 纯净的闭卷 Prompt（不暗示存在外部知识库）=====
        closed_book_prompt = """You are an industrial domain question-answering assistant.

Answer the following question as accurately as possible using your internal knowledge.

[Question]
{question}

Provide a clear and concise final answer.""".format(question=question)
        
        # 直接调用 LINS.chat()，绕过 MAIRAG 的所有检索管线
        response, history = self.lins.chat(question=closed_book_prompt)
        
        retrieval_result = RetrievalResult(
            query=question,
            query_expanded=question,
        )
        
        return response or "", retrieval_result
    
    def answer_rag(self, question: str, retrieval_k: int = 10) -> Tuple[str, RetrievalResult]:
        """
        RAG 模式 (主实验):
        
        Pipeline:
            Question → KED → Industrial Retriever → Knowledge Corpus
            → Top-k Chunks → Context → MAIRAG → Answer
        
        Args:
            question: 问题文本
            retrieval_k: 检索 top-k 数量
        
        Returns:
            (answer, retrieval_result)
        """
        self._init_retriever()
        
        # Step 1: 开放域检索 (Task 2)
        retrieval_result = self.retriever.retrieve(question, k=retrieval_k)
        
        if retrieval_result.documents:
            # Step 2: 构建检索上下文
            context = retrieval_result.get_context()
            
            # Step 3: 将 Evidence 作为 context 注入 LINS
            # - context=context: 注入外部检索证据，MAIRAG 会识别并跳过内部检索
            # - if_PRA=True: 使用 PRM 对外部证据进行相关性重排和过滤
            # - if_SKA=False / if_QDA=False / if_PCA=False: 禁用非必要模块
            # - recall_top_k=50: 从 50 个候选块中精炼至 topk=5 个最相关块
            response, urls, passages, history, sub_qs = self.lins.MAIRAG(
                question=question,
                context=context,
                topk=5,
                if_PRA=True,   # 对外部检索结果进行 PRM 相关性精炼
                if_SKA=False,
                if_QDA=False,
                if_PCA=False,
                recall_top_k=50
            )
        else:
            # 检索无结果，降级为闭卷
            response, urls, passages, history, sub_qs = self.lins.MAIRAG(
                question=question,
                topk=3,
                if_PRA=False, if_SKA=False, if_QDA=False, if_PCA=False,
                recall_top_k=3
            )
        
        return response, retrieval_result
    
    def answer_rag_baseline(self, question: str, retrieval_k: int = 10) -> Tuple[str, RetrievalResult]:
        """
        标准 RAG Baseline (无 PRM):
        
        Pipeline:
            Question → Industrial Retriever → Top-k → Standard RAG Prompt → chat() → Answer
        
        与 answer_rag() 共享完全相同的检索管线，但:
        - 不使用 MAIRAG (避免 PRM 重排/过滤)
        - 使用 LINS.chat() 直接调用 LLM
        - 使用纯净的 RAG prompt (只问"根据检索内容回答")
        - 不要求 citation (与 closed_book 一致)
        
        用于消融实验: MAIRAG(PRM+Rerank) vs 普通 RAG 的公平对比。
        
        Args:
            question: 问题文本
            retrieval_k: 检索 top-k 数量
        
        Returns:
            (answer, retrieval_result)
        """
        self._init_retriever()
        
        # Step 1: 开放域检索 (与 answer_rag 完全一致)
        retrieval_result = self.retriever.retrieve(question, k=retrieval_k)
        
        if retrieval_result.documents:
            # Step 2: 构建检索上下文
            context = retrieval_result.get_context()
            
            # Step 3: 纯净的 RAG Prompt（不使用 MAIRAG）
            rag_prompt = """You are an industrial domain question-answering assistant.
            
Use the following retrieved knowledge to answer the question accurately.

## Retrieved Knowledge:
{context}

## Question:
{question}

## Instructions:
- Answer based on the retrieved knowledge above
- If the retrieved knowledge does not contain enough information, say so
- Provide a clear and concise answer
- Do NOT cite sources with numbers like [1], [2]

Answer:""".format(context=context, question=question)
            
            # 使用 chat() 直接调用 LLM
            response, history = self.lins.chat(question=rag_prompt)
        else:
            # 检索无结果，降级为闭卷
            closed_book_prompt = """You are an industrial domain question-answering assistant.

Answer the following question as accurately as possible using your internal knowledge.

[Question]
{question}

Provide a clear and concise final answer.""".format(question=question)
            
            response, history = self.lins.chat(question=closed_book_prompt)
        
        return response or "", retrieval_result


# ============================================================
# 5. 评分器 API 升级 (Task 6)
# ============================================================

class EnhancedScorer:
    """
    增强评分器: score(question, prediction, reference, retrieved_documents, citations)
    
    同时支持:
    - QA 正确性评分 (复用 IndustryBenchScorer / RuleBasedScorer)
    - 检索质量评估 (Recall@k, MRR, NDCG)
    - 引用准确性评估
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
        """
        综合评分 (Task 6: 新 API)。
        
        Args:
            question: 问题
            prediction: 模型回答
            reference: 参考答案
            retrieved_documents: 检索到的文档列表 (RetrievedDocument)
            citations: 引用列表
            knowledge_text: 仅用于 quick (Oracle) 模式的 SV 检查
        
        Returns:
            score_dict: {
                "raw_score": int,        # 0-3
                "adjusted_score": float,  # SV-adjusted
                "has_violation": bool,
                "violation_detail": str,
                "explanation": str,
            }
        """
        # Step 1: QA 正确性评分
        if self.mode == "llm":
            score_result = self.judge.score(question, reference, prediction)
            
            # SV 检查
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
            explanation = f"rule-based coverage score"
        
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
        """
        评估检索质量 (Task 4).
        
        Args:
            retrieved_document_ids: 检索到的文档 ID 列表（按排名）
            relevant_document_ids: 相关的文档 ID 列表
        
        Returns:
            metrics: {"recall@k": ..., "mrr": ..., "ndcg": ..., "hit@k": ...}
        """
        result = self.retrieval_evaluator.evaluate(
            retrieved_ids=retrieved_document_ids,
            relevant_ids={rid for rid in relevant_document_ids},
        )
        
        metrics = {}
        for i, k in enumerate(self.retrieval_evaluator.k_values):
            if i < len(result.recall_at_k):
                metrics[f"recall@{k}"] = result.recall_at_k[i]
            if i < len(result.precision_at_k):
                metrics[f"precision@{k}"] = result.precision_at_k[i]
        
        # Hit@k
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
        """
        语义检索评估 (Exp1 当前版本).
        
        由于 RAG 模式下 knowledge_text 和检索结果的 document_id 无法直接匹配，
        使用文本重叠度 (Jaccard) 作为语义相关性判断。
        
        重要说明:
        - 此方法使用 Hit@k / Success@k 而非严格 Recall@k
        - 因为当前数据集没有 ground-truth chunk IDs，无法计算真正的 Recall
        - 指标命名统一使用 semantic_ 前缀，避免与严格 IR 指标混淆
        - 后续论文最终实验应构建 Ground Truth Chunk IDs 后改用 Recall
        
        Args:
            retrieved_chunks: 检索到的文档列表 (含 content)
            knowledge_text: ground-truth 知识文本
            threshold: Jaccard 相似度阈值，超过则视为相关
        
        Returns:
            metrics: {
                "semantic_hit@1", "semantic_hit@3", "semantic_hit@5", "semantic_hit@10",
                "semantic_mrr", "semantic_ndcg",
                "matched_count", "semantic_threshold"
            }
        """
        if not knowledge_text or not retrieved_chunks:
            return {
                "semantic_hit@1": 0.0, "semantic_hit@3": 0.0,
                "semantic_hit@5": 0.0, "semantic_hit@10": 0.0,
                "semantic_mrr": 0.0, "semantic_ndcg": 0.0,
                "matched_count": 0, "semantic_threshold": threshold,
            }
        
        # 构建 knowledge 的词集
        def _word_set(text: str) -> Set[str]:
            """提取文本的词集合（中文按字 bigram + 英文按词）"""
            import re
            words = set()
            # 英文/数字词
            for w in re.findall(r'[a-zA-Z0-9]+', text):
                if len(w) >= 2:
                    words.add(w.lower())
            # 中文按字 bigram
            text_cn = re.sub(r'[a-zA-Z0-9\s]', '', text)
            for i in range(len(text_cn) - 1):
                words.add(text_cn[i:i+2])
            return words
        
        kt_words = _word_set(knowledge_text)
        if not kt_words:
            return {
                "semantic_hit@1": 0.0, "semantic_hit@3": 0.0,
                "semantic_hit@5": 0.0, "semantic_hit@10": 0.0,
                "semantic_mrr": 0.0, "semantic_ndcg": 0.0,
                "matched_count": 0, "semantic_threshold": threshold,
            }
        
        # 逐块评估相关性
        relevant_flags = []
        for doc in retrieved_chunks:
            chunk_words = _word_set(doc.content)
            if not chunk_words:
                relevant_flags.append(False)
                continue
            jaccard = len(kt_words & chunk_words) / len(kt_words | chunk_words)
            relevant_flags.append(jaccard >= threshold)
        
        # 计算指标
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            top_k = relevant_flags[:k]
            # Hit@k / Success@k: 只要 Top-k 中至少有一个相关块即为 1，否则为 0
            metrics[f"semantic_hit@{k}"] = 1.0 if any(top_k) else 0.0
        
        # Semantic MRR: 第一个相关块的排名的倒数
        semantic_mrr = 0.0
        for i, flag in enumerate(relevant_flags):
            if flag:
                semantic_mrr = 1.0 / (i + 1)
                break
        metrics["semantic_mrr"] = semantic_mrr
        
        # Semantic NDCG (binary relevance, simplified)
        dcg = 0.0
        idcg = 1.0  # ideal: first result is relevant at rank 1
        for i, flag in enumerate(relevant_flags[:10]):
            if flag:
                dcg += 1.0 / (i + 1)  # simplified: 1/(rank) instead of 1/log2(rank+1)
        semantic_ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics["semantic_ndcg"] = semantic_ndcg
        
        metrics["matched_count"] = sum(relevant_flags)
        metrics["semantic_threshold"] = threshold
        
        return metrics



# ============================================================
# 辅助函数
# ============================================================

def _make_doc_id(knowledge_text: str) -> str:
    """从知识文本生成文档 ID（用于检索评估）"""
    if not knowledge_text:
        return "__unknown__"
    return f"doc_{hashlib.md5(knowledge_text.encode('utf-8')).hexdigest()[:12]}"


def _layer_stats(results: List[Dict], key: str) -> Dict[str, Dict]:
    """按分层维度统计评分"""
    layers: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        val = r.get(key, "unknown")
        score = r.get("adjusted_score", 0.0)
        layers[val].append(score)
    
    out = {}
    for k, vals in sorted(layers.items()):
        out[k] = {
            "avg_score": sum(vals) / max(len(vals), 1),
            "count": len(vals),
            "scores": vals,
        }
    return out


# ============================================================
# 6. 实验一核心函数
# ============================================================

def run_qa_experiment(
    mode: str,
    num_samples: int = 100,
    scorer_mode: str = "rule",
    deepseek_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    corpus_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行开放域 QA 评估实验。
    
    Pipeline:
        Question Dataset → Question → KED → Industrial Retriever
        → Knowledge Corpus → Top-k Chunks → MAIRAG → Answer
        → Industrial Link-Eval → Evaluation Report
    
    Args:
        mode: quick / rag / closed_book
        num_samples: 评估样本数
        scorer_mode: rule 或 llm
        deepseek_key: API Key
        output_dir: 输出目录
        run_id: 运行ID
        corpus_dir: 知识库目录 (仅 RAG 模式使用)
    
    Returns:
        summary dict (含 QA 指标 + 检索指标)
    """
    if deepseek_key is None:
        deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '<DEEPSEEK_API_KEY_FROM_ENV>')
    if not deepseek_key:
        raise ValueError("未设置 DEEPSEEK_API_KEY")

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments', f'exp1_qa_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    mode_info = QA_MODES[mode]
    mode_name = mode_info["name"]
    use_retriever = mode_info["use_retriever"]

    print("\n" + "=" * 70)
    print(f"📊 实验一: 开放域工业 RAG 评估 | 模式={mode_name} | 样本数={num_samples}")
    print(f"   模式说明: {mode_info['desc']}")
    print("=" * 70)

    # ========== Step 1: 加载问题数据集 (Task 1) ==========
    csv_path = os.path.join(DATA_DIR, 'huggingface_dataset.csv')
    question_samples = load_question_dataset(csv_path, num_samples)
    
    if not question_samples:
        print("[ERROR] 未加载到有效样本")
        return {"error": "no_samples"}

    # ========== Step 2: 初始化 IndustrialLINS ==========
    print(f"\n[初始化] IndustrialLINS 模型...")
    model = IndustrialLINSWrapper(deepseek_key)

    # ========== Step 3: 初始化增强评分器 ==========
    scorer_mode_name = "Rule-based" if scorer_mode == "rule" else "LLM Judge"
    print(f"[初始化] 增强评分器: {scorer_mode_name}")
    scorer = EnhancedScorer(mode=scorer_mode, api_key=deepseek_key)

    # ========== Step 4: 运行评估 ==========
    print(f"\n{'─' * 70}")
    print(f"运行模式: {mode_name}")
    print(f"使用检索器: {'✅' if use_retriever else '❌'}")
    print(f"{'─' * 70}")

    results: List[Dict[str, Any]] = []
    total_time = 0.0
    
    # 检索指标聚合 (Exp1: semantic_ 前缀，避免与严格 IR 指标混淆)
    all_semantic_hit_at_1 = []
    all_semantic_hit_at_3 = []
    all_semantic_hit_at_5 = []
    all_semantic_hit_at_10 = []
    all_semantic_mrr = []
    all_semantic_ndcg = []

    for idx, sample in enumerate(question_samples):
        start_time = time.time()

        # ===== Step 4a: 生成回答 + 检索 =====
        retrieval_result: RetrievalResult = RetrievalResult(query=sample.question)
        
        if mode == "quick":
            # Oracle: 注入 ground-truth knowledge_text
            answer, retrieval_result = model.answer_quick(sample.question, sample.knowledge_text)
            
        elif mode == "closed_book":
            # 闭卷: 无外部知识
            answer, retrieval_result = model.answer_closed_book(sample.question)
            
        elif mode == "standard_rag":
            # 标准 RAG Baseline (无 PRM): 共享检索管线，使用 chat() 而非 MAIRAG
            retrieval_k = mode_info.get("retrieval_k", 10)
            answer, retrieval_result = model.answer_rag_baseline(sample.question, retrieval_k=retrieval_k)
            
        else:  # rag (Task 8: 主实验)
            # RAG: Industrial Retriever → Knowledge Corpus → Top-k → MAIRAG
            retrieval_k = mode_info.get("retrieval_k", 10)
            answer, retrieval_result = model.answer_rag(sample.question, retrieval_k=retrieval_k)

        elapsed = time.time() - start_time
        total_time += elapsed

        # ===== Step 4b: QA 评分（含 QuestionType 感知）=====
        # 对填空题/计算题做预处理：从 prediction 中提取关键数值作为补充 reference
        enhanced_ref = sample.ref_answer
        if sample.question_format in ("FillBlank", "Calculation"):
            # 提取 ref_answer 中的数值部分，用于数值匹配
            import re
            ref_nums = re.findall(r'[-+]?\d*\.?\d+', sample.ref_answer)
            if ref_nums:
                enhanced_ref = sample.ref_answer + f" (数值参考: {', '.join(ref_nums)})"

        score_dict = scorer.score(
            question=sample.question,
            prediction=answer,
            reference=enhanced_ref,
            retrieved_documents=retrieval_result.documents,
            citations=retrieval_result.citations,
            knowledge_text=sample.knowledge_text if mode == "quick" else None,
        )

        # ===== Step 4c: 检索评分 (Task 4, 仅 RAG 模式) =====
        # 使用语义近似度评估而非精确 ID 匹配：
        #   RAG 模式下 knowledge_text hash ID 无法与 FAISS chunk document_id 匹配，
        #   改用 Jaccard 文本重叠度判断检索结果是否包含相关知识
        retrieval_metrics = {}
        if use_retriever and retrieval_result.documents and sample.knowledge_text:
            semantic_metrics = scorer.evaluate_retrieval_semantic(
                retrieved_chunks=retrieval_result.documents,
                knowledge_text=sample.knowledge_text,
                threshold=0.15,
            )
            retrieval_metrics = semantic_metrics
            
            # 聚合 (Exp1: semantic_ 前缀)
            all_semantic_hit_at_1.append(retrieval_metrics.get("semantic_hit@1", 0.0))
            all_semantic_hit_at_3.append(retrieval_metrics.get("semantic_hit@3", 0.0))
            all_semantic_hit_at_5.append(retrieval_metrics.get("semantic_hit@5", 0.0))
            all_semantic_hit_at_10.append(retrieval_metrics.get("semantic_hit@10", 0.0))
            all_semantic_mrr.append(retrieval_metrics.get("semantic_mrr", 0.0))
            all_semantic_ndcg.append(retrieval_metrics.get("semantic_ndcg", 0.0))

        # ===== Step 4d: 保存完整结果 (Task 3) =====
        sample_result = {
            "sample_id": sample.id,
            "question": sample.question,
            "question_format": sample.question_format,
            "difficulty": sample.difficulty,
            "capability": sample.capability,
            "industry": sample.industry_primary,
            "ref_answer": sample.ref_answer,
            "model_answer": answer,
            "model_answer_preview": answer[:200] if answer else "",
            
            # QA 评分
            "raw_score": score_dict["raw_score"],
            "adjusted_score": score_dict["adjusted_score"],
            "has_violation": score_dict["has_violation"],
            "violation_detail": score_dict["violation_detail"],
            "score_explanation": score_dict["explanation"],
            
            # 检索结果 (Task 3)
            "retrieval_status": "retrieved" if retrieval_result.documents else "no_result",
            "retrieved_chunks": retrieval_result.chunk_ids,
            "retrieved_document_ids": [d.document_id or d.chunk_id for d in retrieval_result.documents],
            "retrieval_scores": retrieval_result.scores,
            "retrieved_sources": retrieval_result.sources,
            "citations": retrieval_result.citations,
            "retrieval_timing_ms": retrieval_result.timing_ms,
            
            # 检索指标
            "retrieval_metrics": retrieval_metrics,
            
            # 检索文档预览
            "retrieved_documents_preview": [
                f"[{d.rank}] {d.source}:{d.chunk_id[:16]}... (score={d.score:.3f})"
                for d in retrieval_result.documents[:5]
            ],
            
            # 元数据
            "generation_time_seconds": elapsed,
        }
        
        results.append(sample_result)

        # 进度打印
        if (idx + 1) % 10 == 0:
            last_qa = score_dict["raw_score"]
            last_ret = retrieval_metrics.get("semantic_mrr", 0.0) if retrieval_metrics else "-"
            print(f"  [{idx+1}/{len(question_samples)}] 完成 | time={elapsed:.1f}s | "
                  f"score={last_qa} | mrr={last_ret}")

    # ========== Step 5: 统计汇总 ==========
    print(f"\n{'─' * 70}")
    print("📈 统计汇总")
    
    # 5a. QA 总体指标
    raw_scores = [r["raw_score"] for r in results]
    adj_scores = [r["adjusted_score"] for r in results]
    sv_count_val = sum(1 for r in results if r["has_violation"])

    stats: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "mode_name": mode_name,
        "scorer": scorer_mode_name,
        "num_samples": len(results),
        "timestamp": datetime.now().isoformat(),
        
        # QA 指标
        "qa_metrics": {
            "avg_raw_score": sum(raw_scores) / max(len(raw_scores), 1),
            "avg_adjusted_score": sum(adj_scores) / max(len(adj_scores), 1),
            "sv_count": sv_count_val,
            "sv_rate": sv_count_val / max(len(results), 1),
            "score_distribution": dict(Counter(raw_scores)),
        },
        
        # 检索指标 (Task 4, 仅 RAG)
        "retrieval_metrics": {},
        
        # 性能指标
        "total_time_seconds": total_time,
        "avg_time_per_sample": total_time / max(len(results), 1),
        
        # 问题格式分布
        "format_distribution": dict(Counter(s.question_format for s in question_samples)),
    }
    
    # 打印 QA 指标
    qa = stats["qa_metrics"]
    print(f"\n📊 QA 指标:")
    print(f"  平均原始分: {qa['avg_raw_score']:.3f} / 3.0")
    print(f"  平均调整分: {qa['avg_adjusted_score']:.3f} / 3.0")
    print(f"  安全违规: {qa['sv_count']}/{len(results)} ({qa['sv_rate']*100:.1f}%)")
    print(f"  分数分布: {qa['score_distribution']}")
    
    # 打印检索指标 (Exp1)
    if use_retriever and all_semantic_hit_at_1:
        retrieval_stats = {
            "avg_semantic_hit@1": sum(all_semantic_hit_at_1) / len(all_semantic_hit_at_1),
            "avg_semantic_hit@3": sum(all_semantic_hit_at_3) / len(all_semantic_hit_at_3),
            "avg_semantic_hit@5": sum(all_semantic_hit_at_5) / len(all_semantic_hit_at_5),
            "avg_semantic_hit@10": sum(all_semantic_hit_at_10) / len(all_semantic_hit_at_10),
            "avg_semantic_mrr": sum(all_semantic_mrr) / len(all_semantic_mrr),
            "avg_semantic_ndcg": sum(all_semantic_ndcg) / len(all_semantic_ndcg),
        }
        stats["retrieval_metrics"] = retrieval_stats
        
        print(f"\n📊 检索指标 (Exp1 Semantic):")
        print(f"  semantic_hit@1: {retrieval_stats['avg_semantic_hit@1']:.3f}")
        print(f"  semantic_hit@3: {retrieval_stats['avg_semantic_hit@3']:.3f}")
        print(f"  semantic_hit@5: {retrieval_stats['avg_semantic_hit@5']:.3f}")
        print(f"  semantic_hit@10: {retrieval_stats['avg_semantic_hit@10']:.3f}")
        print(f"  semantic_mrr: {retrieval_stats['avg_semantic_mrr']:.3f}")
        print(f"  semantic_ndcg: {retrieval_stats['avg_semantic_ndcg']:.3f}")

    # ========== Step 6: 保存结果 ==========
    output = {
        "experiment": "exp1_qa",
        "mode": mode,
        "mode_info": mode_info,
        "num_samples": len(results),
        "num_errors": sum(1 for r in results if r.get("has_violation", False)),
        "qa_metrics": stats["qa_metrics"],
        "retrieval_metrics": stats.get("retrieval_metrics", {}),
        "format_distribution": stats["format_distribution"],
    }
    if use_retriever and all_semantic_hit_at_1:
        output["retrieval_metrics"] = stats["retrieval_metrics"]

    # 保存详细结果
    detailed_path = os.path.join(output_dir, f"detailed_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细结果已保存: {detailed_path}")

    # 保存汇总
    summary_path = os.path.join(output_dir, f"summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"📄 汇总结果已保存: {summary_path}")

    # 保存 Markdown 报告
    report_path = _write_markdown_report(output_dir, stats, results, mode_info)
    print(f"📄 报告已保存: {report_path}")

    print(f"\n{'=' * 70}")
    print(f"  🎯 Exp1 完成! 模式={mode_name}, 样本={len(results)}")
    print(f"{'=' * 70}")

    return stats


def _write_markdown_report(
    output_dir: str,
    stats: Dict[str, Any],
    results: List[Dict[str, Any]],
    mode_info: Dict[str, Any],
) -> str:
    """生成 Markdown 格式报告"""
    report_path = os.path.join(output_dir, "report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Exp1 QA Evaluation Report\n\n")
        f.write(f"**Mode**: {stats['mode_name']} ({stats['mode']})\n\n")
        f.write(f"**Samples**: {stats['num_samples']}\n\n")
        f.write(f"**Timestamp**: {stats['timestamp']}\n\n")
        
        # QA 指标表
        qa = stats["qa_metrics"]
        f.write("## QA Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Avg Raw Score | {qa['avg_raw_score']:.3f} / 3.0 |\n")
        f.write(f"| Avg Adjusted Score | {qa['avg_adjusted_score']:.3f} / 3.0 |\n")
        f.write(f"| SV Count | {qa['sv_count']} |\n")
        f.write(f"| SV Rate | {qa['sv_rate']*100:.1f}% |\n")
        f.write(f"| Score Distribution | {qa['score_distribution']} |\n\n")
        
        # 检索指标表
        if stats.get("retrieval_metrics"):
            rm = stats["retrieval_metrics"]
            f.write("## Retrieval Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in rm.items():
                f.write(f"| {k} | {v:.3f} |\n")
            f.write("\n")
        
        # 样本结果表（预览前10条）
        f.write("## Sample Results (Preview)\n\n")
        f.write("| ID | Question | Raw Score | Adj Score | SV |\n")
        f.write("|----|----------|-----------|-----------|-----|\n")
        for r in results[:10]:
            q_repr = r["question"][:60].replace("|", "｜")
            sv = "⚠️" if r["has_violation"] else "✅"
            f.write(f"| {r['sample_id']} | {q_repr} | {r['raw_score']} | {r['adjusted_score']} | {sv} |\n")
        f.write("\n")
        
        # 分层统计
        f.write("## Layered Stats\n\n")
        for key in ["difficulty", "capability", "industry", "question_format"]:
            f.write(f"### By {key}\n\n")
            stats_by = _layer_stats(results, key)
            f.write("| Layer | Count | Avg Score |\n")
            f.write("|-------|-------|-----------|\n")
            for k, v in stats_by.items():
                f.write(f"| {k} | {v['count']} | {v['avg_score']:.3f} |\n")
            f.write("\n")
        
        f.write(f"---\n*Generated by exp1_qa.py*\n")
    
    return report_path


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="Exp1: QA Performance Evaluation")
    parser.add_argument("--mode", choices=["quick", "closed_book", "rag", "standard_rag"], default="rag",
                       help="评估模式 (default: rag)")

    parser.add_argument("--num", type=int, default=100, help="评估样本数 (default: 100)")
    parser.add_argument("--scorer", choices=["rule", "llm"], default="rule",
                       help="评分器模式 (default: rule)")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--run-id", default=None, help="运行 ID")
    parser.add_argument("--corpus", default=None, help="知识库目录")
    args = parser.parse_args()

    result = run_qa_experiment(
        mode=args.mode,
        num_samples=args.num,
        scorer_mode=args.scorer,
        output_dir=args.output,
        run_id=args.run_id,
        corpus_dir=args.corpus,
    )

    return result


if __name__ == "__main__":
    main()
