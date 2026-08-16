"""
retrieval: 工业知识检索模块

提供完整的 RAG 检索 pipeline:
1. corpus_builder: 从原始 PDF/TXT 文档提取结构化语料库
2. chunker: 文档切分为检索块
3. embedder: 文本嵌入生成 (BGE/text2vec 等)
4. faiss_indexer: FAISS 向量索引构建与检索
5. retriever: 统一检索接口
"""

from .retriever import (
    OpenDomainRetriever, KEDExtractor, RAGPipeline,
    RetrievedChunk, RetrievalResult, quick_retrieve, demo,
)
from .corpus_builder import IndustrialCorpusBuilder, build_corpus
from .chunker import TextChunker, chunk_corpus, load_chunks, DocumentChunk
from .embedder import EmbeddingGenerator, create_embeddings
from .faiss_indexer import FaissIndexer, build_index

__all__ = [
    "OpenDomainRetriever", "KEDExtractor", "RAGPipeline",
    "RetrievedChunk", "RetrievalResult", "quick_retrieve", "demo",
    "IndustrialCorpusBuilder", "build_corpus",
    "TextChunker", "chunk_corpus", "load_chunks", "DocumentChunk",
    "EmbeddingGenerator", "create_embeddings",
    "FaissIndexer", "build_index",

]
