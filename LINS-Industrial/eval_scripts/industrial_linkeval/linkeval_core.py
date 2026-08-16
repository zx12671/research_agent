"""
linkeval_core.py: IndustrialLinkEval 核心引擎 + 统一评估报告

整合所有评估器:
1. 自动检测问题格式 (FormatDetector)
2. 按格式路由到对应评估器 (QA / Blank / MC)
3. 独立运行检索评估 (RetrievalEvaluator)
4. 生成统一的 EvalReport
"""

import json
import os
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


from .format_detector import FormatDetector, QuestionFormat
from .qa_evaluator import QAEvaluator, QAScoreResult
from .blank_evaluator import BlankEvaluator, BlankScoreResult
from .mc_evaluator import MCEvaluator, MCScoreResult
from .retrieval_evaluator import RetrievalEvaluator, RetrievalScoreResult


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class EvalSample:
    """
    单个评估样本的统一数据结构
    
    支持所有问题格式的输入
    """
    id: str = ""
    question: str = ""
    model_answer: str = ""
    reference_answer: str = ""
    question_format: QuestionFormat = QuestionFormat.UNKNOWN
    
    # QA 额外字段
    evidence_texts: List[str] = field(default_factory=list)
    llm_judge_score: float = 0.0
    
    # Fill-in-Blank 额外字段
    extracted_blank_answer: str = ""
    
    # Multiple Choice 额外字段
    options: Dict[str, str] = field(default_factory=dict)
    predicted_option: str = ""
    correct_option: str = ""
    
    # 检索评估字段
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)
    
    # 元信息
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """
    统一评估报告
    
    包含所有格式的评估结果
    """
    
    # 元信息
    eval_name: str = "IndustrialLinkEval Report"
    timestamp: str = ""
    total_samples: int = 0
    
    # 格式分布
    format_distribution: Dict[str, int] = field(default_factory=dict)
    
    # QA 指标
    qa_count: int = 0
    qa_avg_semantic_similarity: float = 0.0
    qa_avg_llm_judge: float = 0.0
    qa_avg_evidence_consistency: float = 0.0
    qa_details: List[Dict] = field(default_factory=list)
    
    # Fill-in-Blank 指标
    blank_count: int = 0
    blank_exact_match_accuracy: float = 0.0
    blank_normalized_accuracy: float = 0.0
    blank_synonym_accuracy: float = 0.0
    blank_avg_match_score: float = 0.0
    blank_details: List[Dict] = field(default_factory=list)
    
    # Multiple Choice 指标
    mc_count: int = 0
    mc_accuracy: float = 0.0
    mc_details: List[Dict] = field(default_factory=list)
    
    # 检索指标
    retrieval_count: int = 0
    retrieval_recall_at_k: List[float] = field(default_factory=list)
    retrieval_precision_at_k: List[float] = field(default_factory=list)
    retrieval_mrr: float = 0.0
    retrieval_ndcg: float = 0.0
    retrieval_k_values: List[int] = field(default_factory=list)
    retrieval_details: List[Dict] = field(default_factory=list)
    
    # 原始数据
    raw_scores: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save(self, path: str):
        """保存到 JSON 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        print(f"[SAVED] 报告已保存至: {path}")
    
    def to_markdown(self) -> str:
        """生成 Markdown 格式报告"""
        lines = []
        lines.append(f"# {self.eval_name}")
        lines.append(f"")
        lines.append(f"**生成时间**: {self.timestamp}")
        lines.append(f"**总样本数**: {self.total_samples}")
        lines.append(f"")
        
        # 格式分布
        lines.append("## 📊 格式分布")
        lines.append("")
        lines.append(f"| 格式 | 数量 | 占比 |")
        lines.append(f"|------|------|------|")
        for fmt, count in sorted(self.format_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(self.total_samples, 1) * 100
            lines.append(f"| {fmt} | {count} | {pct:.1f}% |")
        lines.append("")
        
        # QA 结果
        if self.qa_count > 0:
            lines.append("## 💬 QA 评估结果")
            lines.append("")
            lines.append(f"**样本数**: {self.qa_count}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 平均语义相似度 | {self.qa_avg_semantic_similarity:.4f} |")
            lines.append(f"| 平均 LLM Judge 评分 | {self.qa_avg_llm_judge:.4f} (0-3) |")
            lines.append(f"| 平均证据一致性 | {self.qa_avg_evidence_consistency:.4f} |")
            lines.append("")
        
        # Fill-in-Blank 结果
        if self.blank_count > 0:
            lines.append("## ✏️ 填空题评估结果")
            lines.append("")
            lines.append(f"**样本数**: {self.blank_count}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 精确匹配准确率 | {self.blank_exact_match_accuracy:.4f} |")
            lines.append(f"| 标准化匹配准确率 | {self.blank_normalized_accuracy:.4f} |")
            lines.append(f"| 同义词匹配准确率 | {self.blank_synonym_accuracy:.4f} |")
            lines.append(f"| 平均匹配得分 | {self.blank_avg_match_score:.4f} |")
            lines.append("")
        
        # Multiple Choice 结果
        if self.mc_count > 0:
            lines.append("## ✅ 选择题评估结果")
            lines.append("")
            lines.append(f"**样本数**: {self.mc_count}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 准确率 | {self.mc_accuracy:.4f} ({self.mc_accuracy*100:.1f}%) |")
            lines.append("")
        
        # 检索结果
        if self.retrieval_count > 0:
            lines.append("## 🔍 检索评估结果")
            lines.append("")
            lines.append(f"**查询数**: {self.retrieval_count}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for i, k in enumerate(self.retrieval_k_values):
                if i < len(self.retrieval_recall_at_k):
                    lines.append(f"| Recall@{k} | {self.retrieval_recall_at_k[i]:.4f} |")
            for i, k in enumerate(self.retrieval_k_values):
                if i < len(self.retrieval_precision_at_k):
                    lines.append(f"| Precision@{k} | {self.retrieval_precision_at_k[i]:.4f} |")
            lines.append(f"| MRR | {self.retrieval_mrr:.4f} |")
            lines.append(f"| NDCG | {self.retrieval_ndcg:.4f} |")
            lines.append("")
        
        lines.append("---")
        lines.append(f"*由 LINS-Industrial LinkEval 自动生成*")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """打印摘要到控制台"""
        print("=" * 65)
        print(f"  {self.eval_name}")
        print(f"  timestamp: {self.timestamp}")
        print(f"  total_samples: {self.total_samples}")
        print("=" * 65)
        
        # 格式分布
        print(f"\n  📊 Format Distribution:")
        for fmt, count in sorted(self.format_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(self.total_samples, 1) * 100
            print(f"    {fmt:<20} {count:>4} ({pct:5.1f}%)")
        
        # QA
        if self.qa_count > 0:
            print(f"\n  💬 QA ({self.qa_count} samples):")
            print(f"    Semantic Similarity:     {self.qa_avg_semantic_similarity:.4f}")
            print(f"    LLM Judge Score:         {self.qa_avg_llm_judge:.4f}")
            print(f"    Evidence Consistency:    {self.qa_avg_evidence_consistency:.4f}")
        
        # Blank
        if self.blank_count > 0:
            print(f"\n  ✏️  Fill-in-Blank ({self.blank_count} samples):")
            print(f"    Exact Match Accuracy:    {self.blank_exact_match_accuracy:.4f}")
            print(f"    Normalized Accuracy:     {self.blank_normalized_accuracy:.4f}")
            print(f"    Synonym Accuracy:        {self.blank_synonym_accuracy:.4f}")
            print(f"    Avg Match Score:         {self.blank_avg_match_score:.4f}")
        
        # MC
        if self.mc_count > 0:
            print(f"\n  ✅ Multiple Choice ({self.mc_count} samples):")
            print(f"    Accuracy:                {self.mc_accuracy:.4f} ({self.mc_accuracy*100:.1f}%)")
        
        # Retrieval
        if self.retrieval_count > 0:
            print(f"\n  🔍 Retrieval ({self.retrieval_count} queries):")
            for i, k in enumerate(self.retrieval_k_values):
                if i < len(self.retrieval_recall_at_k):
                    print(f"    Recall@{k:<4}              {self.retrieval_recall_at_k[i]:.4f}")
            for i, k in enumerate(self.retrieval_k_values):
                if i < len(self.retrieval_precision_at_k):
                    print(f"    Precision@{k:<3}           {self.retrieval_precision_at_k[i]:.4f}")
            print(f"    MRR:                     {self.retrieval_mrr:.4f}")
            print(f"    NDCG:                    {self.retrieval_ndcg:.4f}")
        
        print("\n" + "=" * 65)


# ============================================================
# 2. IndustrialLinkEval 核心引擎
# ============================================================

class IndustrialLinkEval:
    """
    工业版 Link-Eval 核心引擎
    
    功能:
    1. 自动检测每条问题的格式 (QA / Fill-in-Blank / Multiple Choice)
    2. 按格式路由到对应评估器
    3. 整合所有指标生成统一评估报告
    
    使用示例:
        evaluator = IndustrialLinkEval()
        report = evaluator.evaluate(
            questions=["螺栓直径是多少mm？", "M8螺栓的扭矩是___Nm"],
            answers=["14mm", "M8螺栓的扭矩是47Nm"],
            references=["14", "47"],
            formats=None,  # 自动检测
        )
        report.print_summary()
        report.save("report.json")
    """
    
    def __init__(
        self,
        use_semantic: bool = True,
        use_llm_judge: bool = False,
        use_evidence: bool = True,
        use_synonym: bool = True,
        retrieval_k: List[int] = None,
        embedding_model: str = "BGE",
        llm_model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        eval_name: str = "IndustrialLinkEval Report",
    ):
        """
        Args:
            use_semantic: 是否启用语义相似度评估 (QA)
            use_llm_judge: 是否启用 LLM Judge 评分 (QA)
            use_evidence: 是否启用证据一致性评估 (QA)
            use_synonym: 是否启用同义词匹配 (Blank)
            retrieval_k: 检索指标 k 值列表
            embedding_model: 嵌入模型名称
            llm_model: LLM Judge 模型
            api_key: API Key
            eval_name: 报告名称
        """
        self.eval_name = eval_name
        self.format_detector = FormatDetector()
        self.qa_evaluator = QAEvaluator(
            use_semantic=use_semantic,
            use_llm_judge=use_llm_judge,
            use_evidence=use_evidence,
            embedding_model=embedding_model,
            llm_model=llm_model,
            api_key=api_key,
        )
        self.blank_evaluator = BlankEvaluator(use_synonym=use_synonym)
        self.mc_evaluator = MCEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator(k_values=retrieval_k or [1, 3, 5, 10])
    
    def evaluate(
        self,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        formats: Optional[List[Union[str, QuestionFormat]]] = None,
        correct_options: Optional[List[str]] = None,
        evidence_list: Optional[List[List[str]]] = None,
        retrieved_ids: Optional[List[List[str]]] = None,
        relevant_ids: Optional[List[List[str]]] = None,
        metadata: Optional[List[Dict]] = None,
        samples: Optional[List[EvalSample]] = None,
    ) -> EvalReport:
        """
        执行完整评估
        
        Args 有两种传入方式:
        1. 通过 lists: questions, answers, references, formats, ...
        2. 通过 EvalSample 列表: samples
        
        Returns:
            EvalReport
        """
        # 从 samples 或 lists 转为统一格式
        if samples is not None:
            processed = self._from_samples(samples)
        else:
            processed = self._from_lists(
                questions=questions or [],
                answers=answers or [],
                references=references or [],
                formats=formats,
                correct_options=correct_options,
                evidence_list=evidence_list,
                metadata=metadata,
            )
        
        questions = processed["questions"]
        answers = processed["answers"]
        references = processed["references"]
        formats = processed["formats"]
        correct_options = processed["correct_options"]
        ev_list = processed["evidence_list"]
        meta_list = processed["metadata"]
        
        n = len(questions)
        
        # 初始化报告
        report = EvalReport(
            eval_name=self.eval_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=n,
        )
        
        # 按格式分组
        qa_indices = []
        blank_indices = []
        mc_indices = []
        format_counts = Counter()
        
        for i, fmt in enumerate(formats):
            format_counts[fmt.value if isinstance(fmt, QuestionFormat) else str(fmt)] += 1
            if fmt == QuestionFormat.QA:
                qa_indices.append(i)
            elif fmt == QuestionFormat.FILL_IN_BLANK:
                blank_indices.append(i)
            elif fmt == QuestionFormat.MULTIPLE_CHOICE:
                mc_indices.append(i)
        
        report.format_distribution = dict(format_counts)
        
        # === QA 评估 ===
        if qa_indices:
            qa_qs = [questions[i] for i in qa_indices]
            qa_as = [answers[i] for i in qa_indices]
            qa_rs = [references[i] for i in qa_indices]
            qa_ev = [ev_list[i] for i in qa_indices] if ev_list else None
            
            qa_results = self.qa_evaluator.evaluate(qa_qs, qa_as, qa_rs, qa_ev)
            
            report.qa_count = len(qa_results)
            report.qa_avg_semantic_similarity = float(
                sum(r.semantic_similarity for r in qa_results) / len(qa_results)
            ) if qa_results else 0.0
            report.qa_avg_llm_judge = float(
                sum(r.llm_judge_score for r in qa_results) / len(qa_results)
            ) if qa_results else 0.0
            report.qa_avg_evidence_consistency = float(
                sum(r.evidence_consistency for r in qa_results) / len(qa_results)
            ) if qa_results else 0.0
            
            report.qa_details = [
                {
                    "index": qa_indices[i],
                    "semantic_similarity": r.semantic_similarity,
                    "llm_judge_score": r.llm_judge_score,
                    "evidence_consistency": r.evidence_consistency,
                    "raw_scores": r.raw_scores,
                }
                for i, r in enumerate(qa_results)
            ]
        
        # === Fill-in-Blank 评估 ===
        if blank_indices:
            blank_qs = [questions[i] for i in blank_indices]
            blank_as = [answers[i] for i in blank_indices]
            blank_rs = [references[i] for i in blank_indices]
            
            blank_results = self.blank_evaluator.evaluate(blank_qs, blank_as, blank_rs)
            
            report.blank_count = len(blank_results)
            report.blank_exact_match_accuracy = self.blank_evaluator.get_accuracy(blank_results)
            report.blank_synonym_accuracy = self.blank_evaluator.get_synonym_accuracy(blank_results)
            report.blank_avg_match_score = float(
                sum(r.match_score for r in blank_results) / len(blank_results)
            ) if blank_results else 0.0
            
            # Normalized accuracy = exact match + normalized match
            norm_correct = sum(1 for r in blank_results if r.normalized_match)
            report.blank_normalized_accuracy = norm_correct / len(blank_results) if blank_results else 0.0
            
            report.blank_details = [
                {
                    "index": blank_indices[i],
                    "exact_match": r.exact_match,
                    "normalized_match": r.normalized_match,
                    "synonym_match": r.synonym_match,
                    "match_score": r.match_score,
                    "raw_scores": r.raw_scores,
                }
                for i, r in enumerate(blank_results)
            ]
        
        # === Multiple Choice 评估 ===
        if mc_indices:
            mc_as = [answers[i] for i in mc_indices]
            mc_correct = [correct_options[i] for i in mc_indices] if correct_options else []
            
            if mc_correct:
                mc_results = self.mc_evaluator.evaluate(mc_as, mc_correct)
                report.mc_count = len(mc_results)
                report.mc_accuracy = self.mc_evaluator.get_accuracy(mc_results)
                
                report.mc_details = [
                    {
                        "index": mc_indices[i],
                        "is_correct": r.is_correct,
                        "predicted": r.predicted,
                        "expected": r.expected,
                        "confidence": r.confidence,
                    }
                    for i, r in enumerate(mc_results)
                ]
        
        # === 检索评估 ===
        if retrieved_ids and relevant_ids:
            retrieval_metrics = self.retrieval_evaluator.evaluate_batch(
                retrieved_ids,
                [set(rids) for rids in relevant_ids],
            )
            
            report.retrieval_count = len(retrieved_ids)
            report.retrieval_k_values = self.retrieval_evaluator.k_values
            report.retrieval_recall_at_k = [
                retrieval_metrics.get(f"recall@{k}", 0.0)
                for k in self.retrieval_evaluator.k_values
            ]
            report.retrieval_precision_at_k = [
                retrieval_metrics.get(f"precision@{k}", 0.0)
                for k in self.retrieval_evaluator.k_values
            ]
            report.retrieval_mrr = retrieval_metrics.get("mrr", 0.0)
            report.retrieval_ndcg = retrieval_metrics.get("ndcg", 0.0)
            
            if "_per_query" in retrieval_metrics:
                report.retrieval_details = [
                    {
                        "query_idx": i,
                        "recall_at_k": r.recall_at_k,
                        "mrr": r.mrr,
                        "ndcg": r.ndcg,
                    }
                    for i, r in enumerate(retrieval_metrics["_per_query"])
                ]
        
        # 原始数据
        report.raw_scores = {
            "qa_raw": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in report.qa_details],
            "blank_raw": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in report.blank_details],
            "mc_raw": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in report.mc_details],
        }
        
        return report
    
    def _from_samples(self, samples: List[EvalSample]) -> Dict:
        """从 EvalSample 列表提取数据"""
        questions = []
        answers = []
        references = []
        formats = []
        correct_options = []
        evidence_list = []
        metadata = []
        
        for s in samples:
            questions.append(s.question)
            answers.append(s.model_answer)
            references.append(s.reference_answer)
            
            if s.question_format == QuestionFormat.UNKNOWN:
                fmt = self.format_detector.detect(s.question)
            else:
                fmt = s.question_format
            formats.append(fmt)
            
            correct_options.append(s.correct_option)
            evidence_list.append(s.evidence_texts)
            metadata.append(s.metadata)
        
        return {
            "questions": questions,
            "answers": answers,
            "references": references,
            "formats": formats,
            "correct_options": correct_options,
            "evidence_list": evidence_list,
            "metadata": metadata,
        }
    
    def _from_lists(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
        formats: Optional[List[Union[str, QuestionFormat]]] = None,
        correct_options: Optional[List[str]] = None,
        evidence_list: Optional[List[List[str]]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> Dict:
        """从 lists 提取数据"""
        n = len(questions)
        
        # 自动检测格式
        if formats is None:
            detected_formats = self.format_detector.detect_batch(questions)
        else:
            detected_formats = []
            for f in formats:
                if isinstance(f, str):
                    detected_formats.append(QuestionFormat(f))
                else:
                    detected_formats.append(f)
        
        # 填充默认值
        if correct_options is None:
            correct_options = [""] * n
        if evidence_list is None:
            evidence_list = [[] for _ in range(n)]
        if metadata is None:
            metadata = [{} for _ in range(n)]
        
        return {
            "questions": questions,
            "answers": answers,
            "references": references,
            "formats": detected_formats,
            "correct_options": correct_options,
            "evidence_list": evidence_list,
            "metadata": metadata,
        }
    
    def evaluate_rag_pipeline(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
        retrieved_passages: List[List[str]],
        relevant_passages: List[List[str]],
        **kwargs,
    ) -> EvalReport:
        """
        端到端评估 RAG 管线（同时评估生成质量 + 检索质量）
        
        Args:
            questions: 问题列表
            answers: 模型生成回答列表
            references: 参考答案列表
            retrieved_passages: 每个查询检索到的段落列表
            relevant_passages: 每个查询的相关段落列表
            **kwargs: 传递给 evaluate() 的额外参数
        
        Returns:
            EvalReport (含生成质量 + 检索指标)
        """
        # 1. 评估生成质量（自动检测格式）
        report = self.evaluate(
            questions=questions,
            answers=answers,
            references=references,
            evidence_list=retrieved_passages,
            **kwargs,
        )
        
        # 2. 评估检索质量
        import hashlib
        
        all_retrieved_ids = []
        all_relevant_ids = []
        
        for ret_p, rel_p in zip(retrieved_passages, relevant_passages):
            # 用 MD5 哈希作为 ID
            rids = [hashlib.md5(p.encode()).hexdigest() for p in ret_p]
            rel_set = {hashlib.md5(p.encode()).hexdigest() for p in rel_p}
            all_retrieved_ids.append(rids)
            all_relevant_ids.append(rel_set)
        
        retrieval_metrics = self.retrieval_evaluator.evaluate_batch(
            all_retrieved_ids, all_relevant_ids
        )
        
        # 3. 合并到报告
        report.retrieval_count = len(retrieved_passages)
        report.retrieval_k_values = self.retrieval_evaluator.k_values
        report.retrieval_recall_at_k = [
            retrieval_metrics.get(f"recall@{k}", 0.0)
            for k in self.retrieval_evaluator.k_values
        ]
        report.retrieval_precision_at_k = [
            retrieval_metrics.get(f"precision@{k}", 0.0)
            for k in self.retrieval_evaluator.k_values
        ]
        report.retrieval_mrr = retrieval_metrics.get("mrr", 0.0)
        report.retrieval_ndcg = retrieval_metrics.get("ndcg", 0.0)
        
        return report
