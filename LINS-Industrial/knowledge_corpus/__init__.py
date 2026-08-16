"""
knowledge_corpus: 工业知识语料库

目录结构:
  sources/       - 原始语料 (JSONL)
  chunks/        - 切分后的文本块 (JSONL)
  embeddings/    - 向量嵌入 (npy + meta JSONL)
  index/         - FAISS 索引 (.faiss + meta JSON)
"""
