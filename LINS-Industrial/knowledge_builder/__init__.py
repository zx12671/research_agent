"""
knowledge_builder: IndustryBench 工业知识库构建管线

从 IndustryBench 数据集提取、去重、切分工业文档，
保留完整元数据 (source, capability, industry, document_id)，
并构建 FAISS 向量索引，支持增量添加新文档。

使用示例:
    from knowledge_builder import IndustryKnowledgeBuilder
    
    builder = IndustryKnowledgeBuilder()
    builder.load_industrybench("data/industrybench/huggingface_dataset.csv")
    builder.deduplicate()
    builder.chunk_all(chunk_size=512)
    builder.build_embeddings(model_name="bge-small")
    builder.build_index(index_type="flat")
    builder.save_manifest()
"""

from .builder import IndustryKnowledgeBuilder, build_knowledge_base

__all__ = ["IndustryKnowledgeBuilder", "build_knowledge_base"]
