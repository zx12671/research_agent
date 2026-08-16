"""
embedder.py: 工业文本嵌入生成器

将切分后的文本块转换为向量嵌入，用于语义检索。
支持多种嵌入模型：BGE、text2vec 等。
"""
import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """嵌入记录"""
    chunk_id: str
    doc_id: str
    content: str
    embedding: List[float]  # 向量
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingGenerator:
    """
    嵌入生成器。
    
    支持多种模型:
    - "bge-small": BAAI/bge-small-zh-v1.5 (512维，实测)
    - "bge-base": BAAI/bge-base-zh-v1.5 (768维)
    - "text2vec": text2vec-base-chinese (768维)
    - "m3e": m3e-base (768维)

    注: bge-small-zh-v1.5 经 sentence-transformers 实测输出 512 维
        (config[dim] 曾误标 384，已修正；.dim 始终优先取模型真实维度)。
    """
    
    def __init__(self, model_name: str = "bge-small", device: str = "cpu", 
                 normalize: bool = True, batch_size: int = 64):
        """
        Args:
            model_name: 嵌入模型名称
            device: 运行设备 (cpu / cuda)
            normalize: 是否 L2 归一化
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        
        # 模型配置
        self.model_configs = {
            "bge-small": {
                "huggingface": "BAAI/bge-small-zh-v1.5",
                "dim": 512,
                "max_seq_length": 512,
            },
            "bge-base": {
                "huggingface": "BAAI/bge-base-zh-v1.5",
                "dim": 768,
                "max_seq_length": 512,
            },
            "bge-large": {
                "huggingface": "BAAI/bge-large-zh-v1.5",
                "dim": 1024,
                "max_seq_length": 512,
            },
            "text2vec": {
                "huggingface": "shibing624/text2vec-base-chinese",
                "dim": 768,
                "max_seq_length": 512,
            },
            "m3e": {
                "huggingface": "moka-ai/m3e-base",
                "dim": 768,
                "max_seq_length": 512,
            },
        }
        
        if model_name not in self.model_configs:
            logger.warning(f"Unknown model '{model_name}', using bge-small config")
            self.model_name = "bge-small"
        
        self.config = self.model_configs[self.model_name]
        self._model = None
        self._tokenizer = None
    
    @property
    def dim(self) -> int:
        """
        嵌入维度。
        
        优先取模型真实的输出维度（已加载时），避免硬编码与模型实际不一致；
        模型尚未加载时回退到 config["dim"]（已在初始化时按实测值修正）。
        """
        if self._model is not None:
            try:
                return self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return self.config["dim"]
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            model_id = self.config["huggingface"]
            logger.info(f"Loading model: {model_id}")
            self._model = SentenceTransformer(model_id, device=self.device)
            logger.info(f"Model loaded: {model_id} (dim={self.dim}, device={self.device})")
        except ImportError:
            logger.error("Please install sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        编码文本列表为嵌入向量。
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
        
        Returns:
            embeddings: numpy array, shape (n, dim)
        """
        self._load_model()
        
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        编码单个查询。
        prefix 方式: BGE 模型建议为查询添加指令前缀
        """
        if self.model_name.startswith("bge"):
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.encode([query], show_progress=False)[0]
    
    def encode_queries_batch(self, queries: List[str]) -> np.ndarray:
        """
        批量编码多个查询（用于多查询检索增强）。
        
        Args:
            queries: 查询文本列表
            
        Returns:
            embeddings: (n, dim) numpy array
        """
        return self.encode(queries, show_progress=False)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        获取嵌入模型的统计信息和配置。
        
        Returns:
            config dict
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dim,
            "max_seq_length": self.config.get("max_seq_length", 512),
            "device": self.device,
            "normalize": self.normalize,
        }

    
    def embed_chunks(self, chunks_path: str, output_path: str = None, 
                     chunks: List = None) -> Tuple[str, np.ndarray]:
        """
        对切分后的块进行批量嵌入。
        
        Args:
            chunks_path: chunks JSONL 文件路径
            output_path: 输出嵌入文件路径 (npy)
            chunks: 可选的预加载 chunks 列表
        
        Returns:
            (output_path, embeddings)
        """
        if output_path is None:
            base = os.path.splitext(chunks_path)[0]
            output_path = f"{base}_embeddings.npy"
        
        if chunks is None:
            from retrieval.chunker import load_chunks
            chunks = load_chunks(chunks_path)
        
        texts = [c.content for c in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks with {self.model_name}...")
        embeddings = self.encode(texts)
        
        # 同时保存嵌入和元数据
        meta_path = output_path.replace('.npy', '_meta.jsonl')
        with open(meta_path, 'w', encoding='utf-8') as f:
            for chunk, emb in zip(chunks, embeddings):
                record = EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content[:200],  # 只存前200字符做标识
                    embedding=emb.tolist(),
                    metadata=chunk.metadata,
                )
                f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
        
        # 保存 numpy 嵌入
        np.save(output_path, embeddings)
        
        logger.info(f"✅ 嵌入完成: {output_path}")
        logger.info(f"   形状: {embeddings.shape}")
        logger.info(f"   模型: {self.model_name} (dim={self.dim})")
        
        return output_path, embeddings


def create_embeddings(chunks_path: str = None, model_name: str = "bge-small",
                      device: str = "cpu") -> str:
    """
    一键创建嵌入（CLI 入口）。
    
    Args:
        chunks_path: chunks JSONL 路径
        model_name: 嵌入模型
        device: 设备
    
    Returns:
        embeddings_path: 嵌入文件路径
    """
    if chunks_path is None:
        chunks_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'knowledge_corpus', 'chunks', 'industrial_chunks.jsonl'
        )
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'knowledge_corpus', 'embeddings',
                                f'industrial_embeddings_{model_name}.npy')
    
    generator = EmbeddingGenerator(model_name=model_name, device=device)
    emb_path, _ = generator.embed_chunks(chunks_path, output_path)
    
    return emb_path


if __name__ == "__main__":
    import sys
    create_embeddings()
