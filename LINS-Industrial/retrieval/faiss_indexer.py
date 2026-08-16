"""
faiss_indexer.py: FAISS 向量索引构建与检索

使用 FAISS 构建高效的向量索引，支持：
  - IndexFlatIP: 精确内积检索 (brute force)
  - IndexIVFFlat: 近似检索 (IVF, 更快)
  - IndexHNSWFlat: 近似检索 (HNSW, 更准)
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Install with: pip install faiss-cpu")


class FaissIndexer:
    """
    FAISS 向量索引构建与检索。
    
    支持三种索引类型:
    - "flat": 精确检索，速度最慢但最准
    - "ivf": 近似检索，速度快但精度略低
    - "hnsw": 近似检索，HNSW 算法精度高
    """
    
    def __init__(self, dim: int, index_type: str = "flat", metric: str = "ip"):
        """
        Args:
            dim: 向量维度
            index_type: 索引类型 (flat / ivf / hnsw)
            metric: 距离度量 (ip=内积, l2=欧氏距离)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install: pip install faiss-cpu")
        
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.chunk_ids: List[str] = []  # 索引到 chunk_id 的映射
    
    def build(self, embeddings: np.ndarray, chunk_ids: List[str]) -> 'FaissIndexer':
        """
        构建 FAISS 索引。
        
        Args:
            embeddings: (n, dim) numpy 数组
            chunk_ids: 对应每个向量的 chunk_id
        """
        n = embeddings.shape[0]
        assert embeddings.shape[1] == self.dim, f"dim mismatch: {embeddings.shape[1]} != {self.dim}"
        assert n == len(chunk_ids), f"count mismatch: {n} != {len(chunk_ids)}"
        
        self.chunk_ids = chunk_ids
        
        # 确保向量是 float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # 选择距离度量
        if self.metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        # 构建索引
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dim) if metric_type == faiss.METRIC_INNER_PRODUCT \
                        else faiss.IndexFlatL2(self.dim)
            self.index.add(embeddings)
        elif self.index_type == "ivf":
            nlist = min(int(np.sqrt(n)), 100)  # IVF 聚类中心数
            quantizer = faiss.IndexFlatIP(self.dim) if metric_type == faiss.METRIC_INNER_PRODUCT \
                       else faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, metric_type)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(nlist, 10)  # 搜索时探测的聚类数
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32)  # HNSW 默认 M=32
            if self.metric == "ip":
                # HNSW 默认用 L2，通过转换实现内积检索
                faiss.vector_to_arrays(self.index)
            self.index.add(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"✅ FAISS 索引构建完成")
        logger.info(f"   类型: {self.index_type} | 维度: {self.dim} | 向量数: {n}")
        
        return self
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        搜索最近邻。
        
        Args:
            query_embedding: (dim,) 或 (1, dim) 查询向量
            k: 返回 top-k 结果数
        
        Returns:
            (chunk_ids, scores): 匹配的块 ID 和相似度分数
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # 确保 k 不超过索引大小
        k = min(k, len(self.chunk_ids))
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        result_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append(self.chunk_ids[idx])
                result_scores.append(float(score))
        
        return results, result_scores
    
    def save(self, path: str):
        """保存索引和 chunk_ids 映射"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        faiss.write_index(self.index, path)
        
        # 保存 chunk_ids 映射
        meta_path = path + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'chunk_ids': self.chunk_ids,
                'dim': self.dim,
                'index_type': self.index_type,
                'metric': self.metric,
            }, f, ensure_ascii=False)
        
        logger.info(f"✅ 索引已保存: {path}")
    
    @staticmethod
    def load(path: str) -> 'FaissIndexer':
        """加载索引"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required")
        
        meta_path = path + '.meta.json'
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        indexer = FaissIndexer(dim=meta['dim'], index_type=meta['index_type'], metric=meta['metric'])
        indexer.index = faiss.read_index(path)
        indexer.chunk_ids = meta['chunk_ids']
        
        logger.info(f"✅ 索引已加载: {path}")
        logger.info(f"   向量数: {len(indexer.chunk_ids)}")
        
        return indexer


# ============================================================
# CLI: 构建索引
# ============================================================

def build_index(embeddings_path: str = None, index_type: str = "flat",
                output_dir: str = None) -> str:
    """
    根据嵌入文件构建 FAISS 索引。
    
    Args:
        embeddings_path: .npy 嵌入文件路径
        index_type: 索引类型
        output_dir: 输出目录 (默认 knowledge_corpus/index)
    
    Returns:
        index_path: 索引文件路径
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if embeddings_path is None:
        # 查找最新的嵌入文件
        emb_dir = os.path.join(project_root, 'knowledge_corpus', 'embeddings')
        if not os.path.exists(emb_dir):
            logger.error(f"嵌入目录不存在: {emb_dir}")
            return ""
        npy_files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
        if not npy_files:
            logger.error(f"未找到 .npy 嵌入文件: {emb_dir}")
            return ""
        embeddings_path = os.path.join(emb_dir, sorted(npy_files)[-1])
    
    logger.info(f"加载嵌入: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    # 加载对应的 meta 文件获取 chunk_ids
    meta_path = embeddings_path.replace('.npy', '_meta.jsonl')
    chunk_ids = []
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunk_ids.append(data['chunk_id'])
    
    if not chunk_ids:
        chunk_ids = [f"chunk_{i}" for i in range(len(embeddings))]
    
    if output_dir is None:
        output_dir = os.path.join(project_root, 'knowledge_corpus', 'index')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 从文件名推断维度
    emb_name = os.path.splitext(os.path.basename(embeddings_path))[0]
    index_path = os.path.join(output_dir, f'{emb_name}_{index_type}.faiss')
    
    indexer = FaissIndexer(dim=embeddings.shape[1], index_type=index_type)
    indexer.build(embeddings, chunk_ids)
    indexer.save(index_path)
    
    return index_path


if __name__ == "__main__":
    import sys
    build_index()
