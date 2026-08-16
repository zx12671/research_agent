"""
exp2_retrieval.py: 实验二 - 检索性能评估

评估工业知识库的检索质量，独立于 LLM 回答生成。

评估指标:
  - Recall@k: 前 k 个检索结果中相关文档的召回率
  - MRR: Mean Reciprocal Rank
  - NDCG: Normalized Discounted Cumulative Gain
  - Precision@k: 前 k 个检索结果中相关文档的精确率

评测流程:
  1. 加载 IndustryBench 测试题
  2. 使用 IndustrialRetriever 检索相关文档
  3. 将 knowledge_text 作为相关性标注 (ground truth)
  4. 计算检索指标

输出:
  - JSON 详细结果
  - Markdown 格式报告
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Set, Tuple
from collections import defaultdict

# ===== 路径配置 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')

for p in [PROJECT_ROOT, LINS_MAIN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 导入检索评估器
from eval_scripts.industrial_linkeval.retrieval_evaluator import (
    RetrievalEvaluator, RetrievalScoreResult
)


# ============================================================
# 1. 数据加载
# ============================================================

def load_retrieval_data(csv_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """加载检索评估数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": row.get('id', '').strip(),
                "question": row.get('question', '').strip(),
                "knowledge_text": row.get('knowledge_text', '').strip(),
                "capability": row.get('capability', '').strip(),
                "industry_primary": row.get('industry_primary', '').strip(),
                "difficulty": row.get('difficulty', '').strip().lower(),
            })

    if num_samples:
        samples = samples[:num_samples]

    print(f"[INFO] 加载了 {len(samples)} 个检索样本")
    return samples


# ============================================================
# 2. 工业知识检索器（基于嵌入相似度）
# ============================================================

class IndustrialRetriever:
    """
    工业知识检索器
    使用 BGE 嵌入模型计算问题和知识库文档的相似度
    """

    def __init__(self, model_name: str = "BGE", topk: int = 10):
        self.topk = topk
        self._encoder = None
        self._load_encoder(model_name)

    def _load_encoder(self, model_name: str):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            model_map = {
                "BGE": "BAAI/bge-large-zh-v1.5",
                "text2vec": "shibing624/text2vec-base-chinese",
                "all-MiniLM": "all-MiniLM-L6-v2",
            }
            model_path = model_map.get(model_name, model_map["BGE"])
            self._encoder = SentenceTransformer(model_path, device="cpu")
            print(f"[INFO] 嵌入模型已加载: {model_path}")
        except ImportError:
            print("[WARN] sentence-transformers 未安装，回退到词重叠检索")
            self._encoder = None

    def retrieve(self, question: str, knowledge_chunks: List[str], k: int = None) -> List[Tuple[int, float]]:
        """
        检索与问题最相关的文档块

        Args:
            question: 查询问题
            knowledge_chunks: 知识文档块列表
            k: 返回结果数

        Returns:
            [(index, score), ...] 按相关性降序
        """
        if k is None:
            k = self.topk
        k = min(k, len(knowledge_chunks))

        if self._encoder is None:
            # 回退：基于词重叠的 Jaccard 相似度
            scores = []
            for chunk in knowledge_chunks:
                q_words = set(question.lower().split())
                c_words = set(chunk.lower().split())
                intersection = q_words & c_words
                union = q_words | c_words
                score = len(intersection) / max(len(union), 1)
                scores.append(score)
        else:
            # BGE 嵌入相似度
            q_emb = self._encoder.encode(question, normalize_embeddings=True)
            chunk_embs = self._encoder.encode(knowledge_chunks, normalize_embeddings=True)
            scores = np.dot(chunk_embs, q_emb).tolist()

        # 按分数排序，返回 topk
        scored = [(i, s) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def chunk_knowledge(self, text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
        """
        将 knowledge_text 切分为文档块作为检索候选

        Args:
            text: 原始知识文本
            chunk_size: 每个块的字符数
            overlap: 块之间重叠字符数

        Returns:
            文档块列表
        """
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # 在句尾切分
            if end < len(text):
                # 找最近的句号或换行
                for sep in ['。', '. ', '!', '？', '\n']:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else len(text)

        return chunks


# ============================================================
# 3. 实验二核心函数
# ============================================================

def run_retrieval_experiment(
    num_samples: int = 100,
    topk: int = 10,
    chunk_size: int = 200,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行检索性能评估实验

    Args:
        num_samples: 评估样本数
        topk: 检索返回 top-k 结果
        chunk_size: 知识块大小
        output_dir: 输出目录
        run_id: 运行ID

    Returns:
        summary dict
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments', f'exp2_retrieval_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"📊 实验二: 检索性能评估 | 样本数={num_samples} | topk={topk}")
    print("=" * 70)

    # 1. 加载数据
    csv_path = os.path.join(DATA_DIR, 'huggingface_dataset.csv')
    samples = load_retrieval_data(csv_path, num_samples)

    # 2. 初始化检索器
    print(f"\n[初始化] 检索器: BGE | chunk_size={chunk_size}")
    retriever = IndustrialRetriever(model_name="BGE", topk=topk)
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

    # 3. 运行检索评估
    all_recalls = {1: [], 3: [], 5: [], 10: []}
    all_mrr = []
    all_ndcg = []
    all_precisions = {1: [], 3: [], 5: [], 10: []}

    detailed_results = []

    for idx, sample in enumerate(samples):
        question = sample["question"]
        knowledge_text = sample["knowledge_text"]

        # 3a. 将 knowledge_text 切块作为检索候选
        chunks = retriever.chunk_knowledge(knowledge_text, chunk_size=chunk_size)

        if len(chunks) == 0:
            continue

        # 3b. 检索 top-k 结果
        retrieved = retriever.retrieve(question, chunks, k=topk)
        retrieved_indices = [i for i, _ in retrieved]

        # 3c. 构建相关性标注
        # 假设 knowledge_text 切成的所有块都是相关的（作为整体知识）
        relevant_indices = set(range(len(chunks)))

        # 3d. 计算检索指标 (单样本)
        result = evaluator.evaluate(
            retrieved_ids=[retrieved_indices],
            relevant_ids=[relevant_indices],
        )

        # 记录
        for kv in [1, 3, 5, 10]:
            ridx = [1, 3, 5, 10].index(kv) if kv in [1, 3, 5, 10] else -1
            if ridx >= 0 and ridx < len(result.recall_at_k):
                all_recalls[kv].append(result.recall_at_k[ridx])
            if ridx >= 0 and ridx < len(result.precision_at_k):
                all_precisions[kv].append(result.precision_at_k[ridx])

        all_mrr.append(result.mrr)
        all_ndcg.append(result.ndcg)

        detailed_results.append({
            "sample_id": sample["id"],
            "num_chunks": len(chunks),
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "mrr": result.mrr,
            "ndcg": result.ndcg,
            "top_scores": [float(s) for _, s in retrieved[:5]],
        })

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(samples)}] 完成 | 当前 MRR={np.mean(all_mrr):.4f}")

    # 4. 汇总统计
    print(f"\n{'─' * 70}")
    print("📈 检索指标汇总")

    avg_recalls = {k: np.mean(v) if v else 0 for k, v in all_recalls.items()}
    avg_precisions = {k: np.mean(v) if v else 0 for k, v in all_precisions.items()}
    avg_mrr = np.mean(all_mrr) if all_mrr else 0
    avg_ndcg = np.mean(all_ndcg) if all_ndcg else 0

    stats = {
        "run_id": run_id,
        "num_samples": len(detailed_results),
        "topk": topk,
        "chunk_size": chunk_size,
        "timestamp": datetime.now().isoformat(),
        "avg_recall_at_1": avg_recalls[1],
        "avg_recall_at_3": avg_recalls[3],
        "avg_recall_at_5": avg_recalls[5],
        "avg_recall_at_10": avg_recalls[10],
        "avg_precision_at_1": avg_precisions[1],
        "avg_precision_at_3": avg_precisions[3],
        "avg_precision_at_5": avg_precisions[5],
        "avg_precision_at_10": avg_precisions[10],
        "avg_mrr": avg_mrr,
        "avg_ndcg": avg_ndcg,
    }

    print(f"  Recall@1:  {avg_recalls[1]:.4f}")
    print(f"  Recall@3:  {avg_recalls[3]:.4f}")
    print(f"  Recall@5:  {avg_recalls[5]:.4f}")
    print(f"  Recall@10: {avg_recalls[10]:.4f}")
    print(f"  Precision@1:  {avg_precisions[1]:.4f}")
    print(f"  Precision@3:  {avg_precisions[3]:.4f}")
    print(f"  Precision@5:  {avg_precisions[5]:.4f}")
    print(f"  Precision@10: {avg_precisions[10]:.4f}")
    print(f"  MRR:  {avg_mrr:.4f}")
    print(f"  NDCG: {avg_ndcg:.4f}")

    # 5. 保存结果
    output = {
        "experiment": "exp2_retrieval",
        "config": {"num_samples": num_samples, "topk": topk, "chunk_size": chunk_size},
        "stats": stats,
        "detailed_results": detailed_results,
    }

    json_path = os.path.join(output_dir, "exp2_retrieval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] 结果已保存至: {json_path}")

    # 保存 Markdown 报告
    md_path = os.path.join(output_dir, "exp2_retrieval_report.md")
    _save_md_report(md_path, stats)
    print(f"[SAVED] 报告已保存至: {md_path}")

    return output


def _save_md_report(path: str, stats: Dict):
    """保存 Markdown 格式报告"""
    lines = []
    lines.append(f"# 实验二: 检索性能评估报告")
    lines.append(f"")
    lines.append(f"**样本数**: {stats['num_samples']}")
    lines.append(f"**topk**: {stats['topk']}")
    lines.append(f"**chunk_size**: {stats['chunk_size']}")
    lines.append(f"**时间戳**: {stats['timestamp']}")
    lines.append(f"")
    lines.append(f"## 总体检索指标")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| Recall@1 | {stats['avg_recall_at_1']:.4f} |")
    lines.append(f"| Recall@3 | {stats['avg_recall_at_3']:.4f} |")
    lines.append(f"| Recall@5 | {stats['avg_recall_at_5']:.4f} |")
    lines.append(f"| Recall@10 | {stats['avg_recall_at_10']:.4f} |")
    lines.append(f"| Precision@1 | {stats['avg_precision_at_1']:.4f} |")
    lines.append(f"| Precision@3 | {stats['avg_precision_at_3']:.4f} |")
    lines.append(f"| Precision@5 | {stats['avg_precision_at_5']:.4f} |")
    lines.append(f"| Precision@10 | {stats['avg_precision_at_10']:.4f} |")
    lines.append(f"| MRR | {stats['avg_mrr']:.4f} |")
    lines.append(f"| NDCG | {stats['avg_ndcg']:.4f} |")
    lines.append(f"")

    # 可视化 Recall 趋势
    lines.append(f"## Recall@k 趋势")
    lines.append(f"")
    recall_trend = [stats[f'avg_recall_at_{k}'] for k in [1, 3, 5, 10]]
    max_val = max(recall_trend) if recall_trend else 1
    for k, val in zip([1, 3, 5, 10], recall_trend):
        bar_len = int(val / max_val * 30) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  k={k:<3} | {bar} {val:.4f}")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"*由 LINS-Industrial 实验管线自动生成*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验二: 检索性能评估")
    parser.add_argument("--num_samples", type=int, default=50, help="评估样本数")
    parser.add_argument("--topk", type=int, default=10, help="检索返回 top-k")
    parser.add_argument("--chunk_size", type=int, default=200, help="知识块大小")
    args = parser.parse_args()

    result = run_retrieval_experiment(
        num_samples=args.num_samples,
        topk=args.topk,
        chunk_size=args.chunk_size,
    )
