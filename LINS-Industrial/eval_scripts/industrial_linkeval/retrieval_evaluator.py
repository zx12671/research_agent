"""
retrieval_evaluator.py: 检索评估器

评估指标:
- Recall@k: 前 k 个检索结果中相关文档的召回率
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field


@dataclass
class RetrievalScoreResult:
    """检索评估结果"""
    recall_at_k: List[float] = field(default_factory=list)  # Recall@[1,3,5,10]
    mrr: float = 0.0                                        # Mean Reciprocal Rank
    ndcg: float = 0.0                                       # NDCG
    precision_at_k: List[float] = field(default_factory=list)  # Precision@[1,3,5,10]
    reciprocal_rank: float = 0.0                             # Reciprocal Rank
    raw_scores: Dict[str, float] = field(default_factory=dict)


class RetrievalEvaluator:
    """
    检索评估器
    
    评估信息检索系统的排名质量
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: 计算 Recall/Precision 的 k 值列表，默认 [1, 3, 5, 10]
        """
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]],
    ) -> RetrievalScoreResult:
        """
        评估检索结果
        
        Args:
            retrieved_ids: 检索结果文档 ID 列表（按排名顺序）
            relevant_ids: 相关文档 ID 集合
        
        Returns:
            RetrievalScoreResult
        """
        result = RetrievalScoreResult()
        
        # 1. Recall@k
        recalls = []
        precisions = []
        for k in self.k_values:
            if k > len(retrieved_ids):
                k = len(retrieved_ids)
            retrieved_k = set(retrieved_ids[:k])
            
            # Recall@k
            if relevant_ids:
                recall = len(retrieved_k & relevant_ids) / len(relevant_ids)
            else:
                recall = 0.0
            recalls.append(recall)
            
            # Precision@k
            if k > 0:
                precision = len(retrieved_k & relevant_ids) / k
            else:
                precision = 0.0
            precisions.append(precision)
        
        result.recall_at_k = recalls
        result.precision_at_k = precisions
        
        # 2. MRR / Reciprocal Rank
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                rr = 1.0 / rank
                break
        result.reciprocal_rank = rr
        result.mrr = rr  # 单查询时 MRR = RR
        
        # 3. NDCG
        ndcg = self._compute_ndcg(retrieved_ids, relevant_ids)
        result.ndcg = ndcg
        
        # 原始分数
        result.raw_scores = {
            **{f"recall@{k}": r for k, r in zip(self.k_values, recalls)},
            **{f"precision@{k}": p for k, p in zip(self.k_values, precisions)},
            "mrr": rr,
            "ndcg": ndcg,
        }
        
        return result
    
    def evaluate_batch(
        self,
        all_retrieved_ids: List[List[str]],
        all_relevant_ids: List[Set[str]],
    ) -> Dict[str, Any]:
        """
        批量评估（多查询求平均）
        
        Args:
            all_retrieved_ids: 每个查询的检索结果 ID 列表
            all_relevant_ids: 每个查询的相关文档 ID 集合
        
        Returns:
            平均后的指标 dict
        """
        n = len(all_retrieved_ids)
        if n == 0:
            return {}
        
        all_results = [
            self.evaluate(rids, rels)
            for rids, rels in zip(all_retrieved_ids, all_relevant_ids)
        ]
        
        # 计算平均
        avg_recall = {}
        for idx, k in enumerate(self.k_values):
            vals = [r.recall_at_k[idx] for r in all_results]
            avg_recall[f"recall@{k}"] = float(np.mean(vals))
        
        avg_precision = {}
        for idx, k in enumerate(self.k_values):
            vals = [r.precision_at_k[idx] for r in all_results]
            avg_precision[f"precision@{k}"] = float(np.mean(vals))
        
        avg_mrr = float(np.mean([r.mrr for r in all_results]))
        avg_ndcg = float(np.mean([r.ndcg for r in all_results]))
        
        return {
            **avg_recall,
            **avg_precision,
            "mrr": avg_mrr,
            "ndcg": avg_ndcg,
            "num_queries": n,
            "_per_query": all_results,
        }
    
    def _compute_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:
        """计算 NDCG"""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            # 相关度：相关为 1，不相关为 0
            rel = 1.0 if doc_id in relevant_ids else 0.0
            if i == 0:
                dcg += rel
            else:
                dcg += rel / math.log2(i + 1)
        
        # IDCG (理想 DCG)
        n_rel = min(len(relevant_ids), len(retrieved_ids))
        idcg = 0.0
        for i in range(n_rel):
            if i == 0:
                idcg += 1.0
            else:
                idcg += 1.0 / math.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_rag_pipeline(
        self,
        questions: List[str],
        retrieved_passages: List[List[str]],
        relevant_passages: List[List[str]],
    ) -> Dict[str, Any]:
        """
        评估 RAG 管线的检索质量
        
        Args:
            questions: 问题列表
            retrieved_passages: 检索到的段落文本列表
            relevant_passages: 相关段落文本列表
        
        Returns:
            检索指标 dict
        """
        # 文本去重并生成 ID
        all_ids = []
        all_relevant = []
        
        for ret_p, rel_p in zip(retrieved_passages, relevant_passages):
            # 使用文本哈希作为 ID
            ids = [hash(p) for p in ret_p]
            rel_set = {hash(p) for p in rel_p}
            all_ids.append(ids)
            all_relevant.append(rel_set)
        
        return self.evaluate_batch(all_ids, all_relevant)
    
    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """格式化输出检索指标"""
        lines = ["📊 检索评估指标:"]
        
        for key in sorted(metrics.keys()):
            if key.startswith("_") or key in ("num_queries",):
                continue
            val = metrics[key]
            if isinstance(val, float):
                lines.append(f"  {key:<15} = {val:.4f}")
            elif isinstance(val, list):
                lines.append(f"  {key:<15} = {[f'{v:.4f}' for v in val]}")
        
        if "num_queries" in metrics:
            lines.append(f"  {'num_queries':<15} = {metrics['num_queries']}")
        
        return "\n".join(lines)
