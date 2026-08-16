"""
builder.py: IndustryKnowledgeBuilder — 工业知识库构建器

从 IndustryBench 数据集加载文档，执行去重、切分、
嵌入索引构建，并保留完整元数据溯源。

管线流程:
    load_industrybench() → deduplicate() → chunk_all()
    → build_embeddings() → build_index() → save_manifest()

增量添加:
    builder2 = IndustryKnowledgeBuilder()
    builder2.load_industrybench("new_data.csv")
    # ... 过程同上
    builder2.incremental_add(existing_builder)  # 合并去重
"""
import os
import json
import csv
import hashlib
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class IndustryDocument:
    """工业知识文档（去重前的原始文档）"""
    document_id: str               # 文档唯一 ID (由内容 hash 生成)
    source: str                    # 来源: "IndustryBench" / "FactoryWave" / ...
    capability: str                # 工业能力分类
    industry: str                  # 所属行业
    knowledge_text: str            # 知识文本内容
    metadata: Dict = field(default_factory=dict)  # 扩展元数据（难度、领域等）

    def __hash__(self):
        return hash(self.document_id)

    def __eq__(self, other):
        return isinstance(other, IndustryDocument) and self.document_id == other.document_id


@dataclass
class KnowledgeChunk:
    """切分后的知识块"""
    chunk_id: str                  # 块唯一 ID
    document_id: str               # 所属文档 ID
    source: str                    # 来源
    capability: str                # 能力分类
    industry: str                  # 行业
    content: str                   # 块文本
    chunk_index: int               # 块序号
    total_chunks: int              # 总块数


@dataclass
class KnowledgeManifest:
    """知识库清单：记录所有文档和块的元数据"""
    version: str = "1.0"
    total_documents: int = 0
    total_chunks: int = 0
    documents: List[Dict] = field(default_factory=list)   # 文档摘要
    sources: Dict[str, int] = field(default_factory=dict) # 来源统计
    industries: Dict[str, int] = field(default_factory=dict) # 行业统计
    capabilities: Dict[str, int] = field(default_factory=dict) # 能力统计
    embedding_model: str = ""
    index_type: str = ""
    index_path: str = ""


# ============================================================
# 2. IndustryKnowledgeBuilder
# ============================================================

class IndustryKnowledgeBuilder:
    """
    IndustryBench 知识库构建器。

    工作流程:
        1. load_industrybench()  — 加载原始数据集
        2. deduplicate()         — 基于 knowledge_text 去重
        3. chunk_all()           — 切分文档为文本块
        4. build_embeddings()    — 生成向量嵌入
        5. build_index()         — 构建 FAISS 索引
        6. save_manifest()       — 保存元数据清单
    """

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root
        
        self.documents: List[IndustryDocument] = []   # 原始文档
        self.document_map: Dict[str, IndustryDocument] = {}  # document_id → doc
        self.chunks: List[KnowledgeChunk] = []         # 切分后的块
        self.manifest = KnowledgeManifest()
        
        # 输出目录
        self.chunks_dir = os.path.join(project_root, "knowledge_corpus", "chunks")
        self.embeddings_dir = os.path.join(project_root, "knowledge_corpus", "embeddings")
        self.index_dir = os.path.join(project_root, "knowledge_corpus", "index")
        
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 2.1 加载数据
    # ----------------------------------------------------------

    def load_industrybench(self, csv_path: str) -> "IndustryKnowledgeBuilder":
        """
        从 IndustryBench CSV 加载数据集。
        
        CSV 列: id, question, answer, difficulty, _format,
                industry_primary, capability, knowledge_text, ...
        
        Args:
            csv_path: 相对于 project_root 或绝对路径
        
        Returns:
            self (链式调用)
        """
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.project_root, csv_path)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"IndustryBench CSV not found: {csv_path}")
        
        logger.info(f"加载 IndustryBench 数据集: {csv_path}")
        
        count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                knowledge_text = row.get("knowledge_text", "").strip()
                if not knowledge_text:
                    continue  # 跳过空文档
                
                # 用文本内容 hash 生成 document_id
                doc_id = hashlib.md5(knowledge_text.encode()).hexdigest()[:16]
                
                doc = IndustryDocument(
                    document_id=doc_id,
                    source="IndustryBench",
                    capability=row.get("capability", "").strip(),
                    industry=row.get("industry_primary", "").strip(),
                    knowledge_text=knowledge_text,
                    metadata={
                        "question_id": row.get("id", ""),
                        "difficulty": row.get("difficulty", ""),
                        "format": row.get("_format", ""),
                        "domain": row.get("domain", ""),
                    }
                )
                self.documents.append(doc)
                count += 1
        
        logger.info(f"  加载完成: {count} 条文档（含重复）")
        return self

    def add_documents(self, documents: List[IndustryDocument]) -> "IndustryKnowledgeBuilder":
        """
        手动添加文档列表（用于非 IndustryBench 来源）。
        
        Args:
            documents: IndustryDocument 列表
        
        Returns:
            self (链式调用)
        """
        for doc in documents:
            if not doc.document_id:
                doc.document_id = hashlib.md5(doc.knowledge_text.encode()).hexdigest()[:16]
            self.documents.append(doc)
        
        logger.info(f"  手动添加: {len(documents)} 条文档")
        return self

    # ----------------------------------------------------------
    # 2.2 去重
    # ----------------------------------------------------------

    def deduplicate(self) -> "IndustryKnowledgeBuilder":
        """
        基于 document_id (knowledge_text hash) 去重。
        保留第一个出现的文档，合并其 metadata。
        
        Returns:
            self (链式调用)
        """
        seen: Set[str] = set()
        unique_docs: List[IndustryDocument] = []
        
        for doc in self.documents:
            if doc.document_id not in seen:
                seen.add(doc.document_id)
                unique_docs.append(doc)
            else:
                # 合并 metadata（保留非空值）
                existing = self.document_map.get(doc.document_id)
                if existing and doc.metadata:
                    for k, v in doc.metadata.items():
                        if v and k not in existing.metadata:
                            existing.metadata[k] = v
        
        dup_count = len(self.documents) - len(unique_docs)
        self.documents = unique_docs
        
        # 重建 document_map
        self.document_map = {doc.document_id: doc for doc in self.documents}
        
        logger.info(f"  去重完成: {len(self.documents)} 条（去重 {dup_count} 条）")
        return self

    # ----------------------------------------------------------
    # 2.3 切分
    # ----------------------------------------------------------

    def chunk_all(self, chunk_size: int = 512, chunk_overlap: int = 64,
                  strategy: str = "paragraph") -> "IndustryKnowledgeBuilder":
        """
        将所有文档切分为文本块。
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠（字符数）
            strategy: 切分策略 ("paragraph" / "fixed_size" / "recursive")
        
        Returns:
            self (链式调用)
        """
        if not self.documents:
            logger.warning("没有文档可供切分，请先调用 load_industrybench()")
            return self
        
        from retrieval.chunker import TextChunker, DocumentChunk
        
        # 将 IndustryDocument 适配为 chunker 接受的格式
        class _AdaptedDoc:
            def __init__(self, doc: IndustryDocument):
                self.doc_id = doc.document_id
                self.source = doc.source
                self.source_name = f"{doc.industry}/{doc.capability}"
                self.title = f"[{doc.industry}] {doc.capability}"
                self.content = doc.knowledge_text
                self.metadata = {
                    "capability": doc.capability,
                    "industry": doc.industry,
                    **doc.metadata,
                }
        
        adapted_docs = [_AdaptedDoc(doc) for doc in self.documents]
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
        
        total = 0
        self.chunks = []
        for adoc in adapted_docs:
            result_chunks = chunker.chunk_document(adoc)
            for c in result_chunks:
                chunk = KnowledgeChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.doc_id,
                    source=c.source,
                    capability=c.metadata.get("capability", ""),
                    industry=c.metadata.get("industry", ""),
                    content=c.content,
                    chunk_index=c.chunk_index,
                    total_chunks=c.total_chunks,
                )
                self.chunks.append(chunk)
                total += 1
        
        logger.info(f"  切分完成: {len(self.documents)} 篇 → {len(self.chunks)} 块")
        logger.info(f"  策略: {strategy} | chunk_size={chunk_size} | overlap={chunk_overlap}")
        
        return self

    def save_chunks(self, path: str = None) -> str:
        """
        保存切分后的块到 JSONL 文件。
        
        Args:
            path: 输出路径（默认 knowledge_corpus/chunks/industrybench_chunks.jsonl）
        
        Returns:
            文件路径
        """
        if path is None:
            path = os.path.join(self.chunks_dir, "industrybench_chunks.jsonl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
        
        logger.info(f"  块已保存: {path} ({len(self.chunks)} 块)")
        return path

    # ----------------------------------------------------------
    # 2.4 嵌入索引
    # ----------------------------------------------------------

    def build_embeddings(self, model_name: str = "bge-small",
                         device: str = "cpu") -> "IndustryKnowledgeBuilder":
        """
        为所有文本块生成向量嵌入。
        
        Args:
            model_name: 嵌入模型名称
            device: 运行设备 (cpu / cuda)
        
        Returns:
            self (链式调用)
        """
        if not self.chunks:
            logger.warning("没有块可供嵌入，请先调用 chunk_all()")
            return self
        
        from retrieval.embedder import EmbeddingGenerator
        
        generator = EmbeddingGenerator(model_name=model_name, device=device)
        
        texts = [c.content for c in self.chunks]
        logger.info(f"  生成嵌入: {len(texts)} 个块 × {generator.dim} 维")
        
        embeddings = generator.encode(texts)
        
        # 保存嵌入文件
        emb_path = os.path.join(self.embeddings_dir, f"industrybench_{model_name}.npy")
        import numpy as np
        np.save(emb_path, embeddings)
        
        # 保存嵌入元数据（每行: chunk_id + embedding 前 4 个值）
        meta_path = emb_path.replace('.npy', '_meta.jsonl')
        with open(meta_path, 'w', encoding='utf-8') as f:
            for chunk, emb in zip(self.chunks, embeddings):
                record = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "capability": chunk.capability,
                    "industry": chunk.industry,
                    "embedding_preview": emb[:4].tolist(),  # 仅预览
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        self.manifest.embedding_model = model_name
        self.embeddings = embeddings  # 暂存
        
        logger.info(f"  嵌入完成: {emb_path}")
        logger.info(f"  形状: {embeddings.shape}")
        
        return self

    def build_index(self, index_type: str = "flat", metric: str = "ip") -> "IndustryKnowledgeBuilder":
        """
        构建 FAISS 向量索引。
        
        Args:
            index_type: 索引类型 (flat / ivf / hnsw)
            metric: 距离度量 (ip=内积, l2=欧氏距离)
        
        Returns:
            self (链式调用)
        """
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            raise RuntimeError("请先调用 build_embeddings()")
        
        from retrieval.faiss_indexer import FaissIndexer
        
        dim = self.embeddings.shape[1]
        chunk_ids = [c.chunk_id for c in self.chunks]
        
        indexer = FaissIndexer(dim=dim, index_type=index_type, metric=metric)
        indexer.build(self.embeddings, chunk_ids)
        
        # 保存索引
        model_name = self.manifest.embedding_model or "unknown"
        index_path = os.path.join(self.index_dir, f"industrybench_{model_name}_{index_type}.faiss")
        indexer.save(index_path)
        
        self.indexer = indexer
        self.manifest.index_type = index_type
        self.manifest.index_path = index_path
        
        logger.info(f"  索引构建完成: {index_path}")
        logger.info(f"  类型: {index_type} | 向量数: {len(chunk_ids)}")
        
        return self

    # ----------------------------------------------------------
    # 2.5 清单与统计
    # ----------------------------------------------------------

    def save_manifest(self, path: str = None) -> str:
        """
        保存知识库元数据清单。
        
        Args:
            path: 输出路径（默认 knowledge_corpus/manifest.json）
        
        Returns:
            文件路径
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(self.index_dir),
                "manifest.json"
            )
        
        # 统计
        sources = {}
        industries = {}
        capabilities = {}
        for doc in self.documents:
            sources[doc.source] = sources.get(doc.source, 0) + 1
            industries[doc.industry] = industries.get(doc.industry, 0) + 1
            capabilities[doc.capability] = capabilities.get(doc.capability, 0) + 1
        
        # 文档摘要（不含完整文本）
        doc_summaries = []
        for doc in self.documents:
            doc_summaries.append({
                "document_id": doc.document_id,
                "source": doc.source,
                "industry": doc.industry,
                "capability": doc.capability,
                "text_length": len(doc.knowledge_text),
                "metadata": doc.metadata,
            })
        
        self.manifest.total_documents = len(self.documents)
        self.manifest.total_chunks = len(self.chunks)
        self.manifest.documents = doc_summaries
        self.manifest.sources = sources
        self.manifest.industries = industries
        self.manifest.capabilities = capabilities
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.manifest), f, ensure_ascii=False, indent=2)
        
        logger.info(f"  清单已保存: {path}")
        self.print_stats()
        
        return path

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("  📊 工业知识库统计")
        print("=" * 60)
        print(f"  文档总数: {len(self.documents)}")
        print(f"  切分块数: {len(self.chunks)}")
        print(f"  来源分布: {dict(self.manifest.sources)}")
        print(f"  行业分布: {dict(self.manifest.industries)}")
        print(f"  能力分布: {dict(self.manifest.capabilities)}")
        if self.manifest.embedding_model:
            print(f"  嵌入模型: {self.manifest.embedding_model}")
        if self.manifest.index_type:
            print(f"  索引类型: {self.manifest.index_type}")
        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # 2.6 增量添加
    # ----------------------------------------------------------

    def incremental_add(self, existing: "IndustryKnowledgeBuilder",
                        merge_strategy: str = "skip") -> "IndustryKnowledgeBuilder":
        """
        将当前 builder 的文档增量合并到 existing builder 中。
        
        Args:
            existing: 已存在的 IndustryKnowledgeBuilder 实例
            merge_strategy: 合并策略
                - "skip": 跳过已存在的文档（默认）
                - "replace": 用新文档替换旧的
        
        Returns:
            merged_builder: 合并后的新 builder
        """
        merged = IndustryKnowledgeBuilder(project_root=self.project_root)
        
        # 合并文档（先加旧的，再加新的）
        existing_ids = {doc.document_id for doc in existing.documents}
        
        # 加入旧文档
        merged.documents = list(existing.documents)
        
        # 加入新文档
        added = 0
        skipped = 0
        replaced = 0
        for doc in self.documents:
            if doc.document_id not in existing_ids:
                merged.documents.append(doc)
                added += 1
            elif merge_strategy == "replace":
                # 替换同 ID 文档
                for i, edoc in enumerate(merged.documents):
                    if edoc.document_id == doc.document_id:
                        merged.documents[i] = doc
                        replaced += 1
                        break
            else:
                skipped += 1
        
        # 重建 document_map
        merged.document_map = {doc.document_id: doc for doc in merged.documents}
        
        logger.info(f"  增量合并完成: +{added} 新 / 替换{replaced} / 跳过{skipped}")
        
        return merged

    # ----------------------------------------------------------
    # 2.7 导出与工具
    # ----------------------------------------------------------

    def export_corpus_jsonl(self, path: str = None) -> str:
        """
        导出为标准语料库 JSONL 格式（兼容 retrieval/ 模块）。
        
        Args:
            path: 输出路径
        
        Returns:
            文件路径
        """
        if path is None:
            path = os.path.join(
                self.project_root, "knowledge_corpus", "sources",
                "industrybench_corpus.jsonl"
            )
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        from retrieval.corpus_builder import IndustrialDocument as CorpusDoc
        
        with open(path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                cd = CorpusDoc(
                    doc_id=doc.document_id,
                    source="IndustryBench",
                    source_name=f"IndustryBench/{doc.industry}",
                    title=f"[{doc.industry}] {doc.capability}",
                    content=doc.knowledge_text,
                    metadata={
                        "capability": doc.capability,
                        "industry": doc.industry,
                        **doc.metadata,
                    }
                )
                f.write(json.dumps(asdict(cd), ensure_ascii=False) + '\n')
        
        logger.info(f"  语料已导出: {path} ({len(self.documents)} 篇)")
        return path


# ============================================================
# 3. 一键构建函数
# ============================================================

def build_knowledge_base(csv_path: str = "data/industrybench/huggingface_dataset.csv",
                         chunk_size: int = 512,
                         chunk_strategy: str = "paragraph",
                         embedding_model: str = "bge-small",
                         index_type: str = "flat",
                         device: str = "cpu") -> IndustryKnowledgeBuilder:
    """
    一键构建工业知识库：加载 → 去重 → 切分 → 嵌入 → 索引。
    
    Args:
        csv_path: IndustryBench CSV 路径
        chunk_size: 切分大小
        chunk_strategy: 切分策略
        embedding_model: 嵌入模型
        index_type: FAISS 索引类型
        device: 运行设备
    
    Returns:
        构建完成的 IndustryKnowledgeBuilder 实例
    """
    logger.info("=" * 60)
    logger.info("  🏗️  工业知识库构建开始")
    logger.info("=" * 60)
    
    builder = IndustryKnowledgeBuilder()
    
    # Step 1: 加载数据
    logger.info("[1/5] 加载 IndustryBench 数据集...")
    builder.load_industrybench(csv_path)
    
    # Step 2: 去重
    logger.info("[2/5] 去重...")
    builder.deduplicate()
    
    # Step 3: 切分
    logger.info(f"[3/5] 切分文档 (strategy={chunk_strategy}, size={chunk_size})...")
    builder.chunk_all(chunk_size=chunk_size, strategy=chunk_strategy)
    builder.save_chunks()
    
    # Step 4: 嵌入
    logger.info(f"[4/5] 生成嵌入向量 (model={embedding_model})...")
    builder.build_embeddings(model_name=embedding_model, device=device)
    
    # Step 5: 索引
    logger.info(f"[5/5] 构建 FAISS 索引 (type={index_type})...")
    builder.build_index(index_type=index_type)
    
    # 保存清单
    builder.save_manifest()
    
    # 导出为兼容格式
    builder.export_corpus_jsonl()
    
    logger.info("=" * 60)
    logger.info("  ✅  工业知识库构建完成")
    logger.info("=" * 60)
    
    return builder


# ============================================================
# 4. CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="工业知识库构建工具")
    parser.add_argument("--csv", default="data/industrybench/huggingface_dataset.csv",
                        help="IndustryBench CSV 路径")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="切分大小（字符数）")
    parser.add_argument("--chunk-strategy", default="paragraph",
                        choices=["paragraph", "fixed_size", "recursive"],
                        help="切分策略")
    parser.add_argument("--embedding-model", default="bge-small",
                        help="嵌入模型名称")
    parser.add_argument("--index-type", default="flat",
                        choices=["flat", "ivf", "hnsw"],
                        help="FAISS 索引类型")
    parser.add_argument("--device", default="cpu",
                        help="运行设备 (cpu / cuda)")
    
    args = parser.parse_args()
    
    build_knowledge_base(
        csv_path=args.csv,
        chunk_size=args.chunk_size,
        chunk_strategy=args.chunk_strategy,
        embedding_model=args.embedding_model,
        index_type=args.index_type,
        device=args.device,
    )
