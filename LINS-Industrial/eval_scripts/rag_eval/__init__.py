"""
rag_eval: 基于 RAG 检索增强生成的工业知识评测

使用路线:
  1. OpenDomainRetriever / IndustrialRetriever 从知识库检索相关文档
  2. 构建 Prompt (上下文 + 问题)
  3. DeepSeek / OpenAI / 本地 LLM (Ollama) 生成答案
  4. 计算评测指标 (Accuracy, EM, F1, etc.)
"""

from .eval_rag import RAGEvaluator, load_industrybench_data

__all__ = ["RAGEvaluator", "load_industrybench_data"]


