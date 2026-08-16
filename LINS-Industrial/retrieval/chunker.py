"""
chunker.py: 工业文档切分器

将长文档切分为适合检索的短文本块 (chunks)。
支持多种切分策略：按长度、按段落、按语义边界。
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class DocumentChunk:
    """文档切分后的一个块"""
    chunk_id: str           # 块 ID
    doc_id: str             # 所属文档 ID
    source: str             # 来源类型
    source_name: str        # 来源名称
    title: str              # 文档标题
    content: str            # 块内容
    chunk_index: int        # 块序号 (从 0 开始)
    total_chunks: int       # 总块数
    metadata: Dict = None   # 额外元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# 2. 切分策略
# ============================================================

class TextChunker:
    """
    文档切分器。
    
    支持策略:
    - "paragraph": 按段落切分 (默认)
    - "fixed_size": 按固定长度切分 (带 overlap)
    - "recursive": 递归切分 (优先段落 -> 句子)
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, strategy: str = "paragraph"):
        """
        Args:
            chunk_size: 块大小 (字符数), 仅 fixed_size 和 recursive 模式使用
            chunk_overlap: 块重叠 (字符数)
            strategy: 切分策略
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk_document(self, doc) -> List[DocumentChunk]:
        """将一篇文档切分为多个块"""
        if self.strategy == "paragraph":
            chunks = self._chunk_by_paragraph(doc)
        elif self.strategy == "fixed_size":
            chunks = self._chunk_fixed_size(doc)
        elif self.strategy == "recursive":
            chunks = self._chunk_recursive(doc)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 生成 chunk_id
        for i, chunk in enumerate(chunks):
            unique = f"{chunk.doc_id}_{i}_{chunk.content[:20]}"
            chunk.chunk_id = hashlib.md5(unique.encode()).hexdigest()[:12]
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_paragraph(self, doc) -> List[DocumentChunk]:
        """按段落切分：每个段落为一个块"""
        paragraphs = [p.strip() for p in doc.content.split('\n\n') if p.strip()]
        # 如果段落太长，进一步切分
        chunks = []
        for pi, para in enumerate(paragraphs):
            if len(para) <= self.chunk_size * 2:
                chunks.append(DocumentChunk(
                    chunk_id="",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    source_name=doc.source_name,
                    title=doc.title,
                    content=para,
                    chunk_index=pi,
                    total_chunks=0,
                    metadata={**doc.metadata, "paragraph_index": pi}
                ))
            else:
                # 长段落按 fixed_size 再切分
                sub_chunks = self._split_text(para, doc)
                chunks.extend(sub_chunks)
        return chunks
    
    def _chunk_fixed_size(self, doc) -> List[DocumentChunk]:
        """按固定长度切分"""
        return self._split_text(doc.content, doc)
    
    def _chunk_recursive(self, doc) -> List[DocumentChunk]:
        """递归切分：先按段落，段落太长再按句子"""
        paragraphs = [p.strip() for p in doc.content.split('\n\n') if p.strip()]
        chunks = []
        for pi, para in enumerate(paragraphs):
            if len(para) <= self.chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id="",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    source_name=doc.source_name,
                    title=doc.title,
                    content=para,
                    chunk_index=pi,
                    total_chunks=0,
                    metadata={**doc.metadata, "paragraph_index": pi}
                ))
            else:
                # 按句子切分
                import re
                sentences = re.split(r'(?<=[。！？.!?])\s*', para)
                buffer = ""
                for sent in sentences:
                    if len(buffer) + len(sent) <= self.chunk_size:
                        buffer += sent
                    else:
                        if buffer:
                            chunks.append(DocumentChunk(
                                chunk_id="",
                                doc_id=doc.doc_id,
                                source=doc.source,
                                source_name=doc.source_name,
                                title=doc.title,
                                content=buffer,
                                chunk_index=len(chunks),
                                total_chunks=0,
                                metadata={**doc.metadata, "paragraph_index": pi}
                            ))
                        buffer = sent
                if buffer:
                    chunks.append(DocumentChunk(
                        chunk_id="",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        source_name=doc.source_name,
                        title=doc.title,
                        content=buffer,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        metadata={**doc.metadata, "paragraph_index": pi}
                    ))
        return chunks
    
    def _split_text(self, text: str, doc) -> List[DocumentChunk]:
        """按固定长度切分文本"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(DocumentChunk(
                chunk_id="",
                doc_id=doc.doc_id,
                source=doc.source,
                source_name=doc.source_name,
                title=doc.title,
                content=chunk_text,
                chunk_index=idx,
                total_chunks=0,
                metadata={**doc.metadata}
            ))
            idx += 1
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks


# ============================================================
# 3. 批处理接口
# ============================================================

def chunk_corpus(corpus_path: str, output_path: str = None,
                 chunk_size: int = 512, chunk_overlap: int = 64,
                 strategy: str = "paragraph") -> str:
    """
    对语料库进行切分。
    
    Args:
        corpus_path: 语料库 JSONL 路径
        output_path: 输出 JSONL 路径
        chunk_size: 块大小
        chunk_overlap: 块重叠
        strategy: 切分策略
    
    Returns:
        output_path: 输出路径
    """
    if output_path is None:
        base = os.path.splitext(corpus_path)[0]
        output_path = f"{base}_chunks.jsonl"
    
    # 加载语料库
    from retrieval.corpus_builder import IndustrialCorpusBuilder
    docs = IndustrialCorpusBuilder.load(corpus_path)
    
    # 切分
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, strategy=strategy)
    total_chunks = 0
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            chunks = chunker.chunk_document(doc)
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
                total_chunks += 1
    
    logger.info(f"✅ 切分完成: {output_path}")
    logger.info(f"   原文档: {len(docs)} 篇 | 切分后: {total_chunks} 块")
    logger.info(f"   策略: {strategy} | chunk_size={chunk_size} | overlap={chunk_overlap}")
    
    return output_path


def load_chunks(chunks_path: str) -> List[DocumentChunk]:
    """加载切分后的块"""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunks.append(DocumentChunk(**data))
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def get_chunk_stats(chunks_path: str) -> Dict:
    """获取切分统计信息"""
    chunks = load_chunks(chunks_path)
    if not chunks:
        return {}
    
    lengths = [len(c.content) for c in chunks]
    sources = {}
    for c in chunks:
        sources[c.source] = sources.get(c.source, 0) + 1
    
    return {
        "total_chunks": len(chunks),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "sources": sources,
    }


if __name__ == "__main__":
    import sys
    
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'knowledge_corpus', 'sources', 'industrial_corpus.jsonl'
    )
    
    if not os.path.exists(corpus_path):
        print(f"[ERROR] 请先运行 corpus_builder.py 构建语料库: {corpus_path}")
        sys.exit(1)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'knowledge_corpus', 'chunks', 'industrial_chunks.jsonl'
    )
    
    chunk_corpus(corpus_path, output_path, chunk_size=512, strategy="paragraph")
