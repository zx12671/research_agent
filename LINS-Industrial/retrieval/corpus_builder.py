"""
corpus_builder.py: 工业知识语料库构建器

从多种知识源提取工业文档，构建统一的工业语料库：
  - IndustryBench knowledge_text
  - 技术手册 (technical manuals)
  - ISO/GB 标准 (standards)
  - 维护手册 (maintenance manuals)
  - 其他工业文档

输出: JSONL 文件，每条记录为一个独立文档
"""
import os
import csv
import json
import glob
import logging
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class IndustrialDocument:
    """一个独立的工业知识文档"""
    doc_id: str                     # 唯一文档 ID
    source: str                     # 来源类型 (industrybench / technical_manual / standard / maintenance / other)
    source_name: str                # 来源名称 (如 "IndustryBench-Aerospace-1")
    title: str                      # 文档标题
    content: str                    # 文档正文
    metadata: Dict = None           # 额外元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# 2. 从 IndustryBench 提取 knowledge_text
# ============================================================

def extract_from_industrybench(csv_path: str) -> List[IndustrialDocument]:
    """
    从 IndustryBench 数据集中提取所有 knowledge_text 作为独立文档。
    
    每个 QA 样本的 knowledge_text 被视为一篇独立的工业知识文档。
    这是构建工业语料库的核心步骤。
    """
    if not os.path.exists(csv_path):
        logger.warning(f"IndustryBench CSV not found: {csv_path}")
        return []
    
    documents = []
    seen_texts = set()  # 去重
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            knowledge_text = row.get('knowledge_text', '').strip()
            if not knowledge_text or knowledge_text in seen_texts:
                continue
            seen_texts.add(knowledge_text)
            
            doc_id = f"industrybench_{row.get('id', 'unknown')}"
            industry = row.get('industry_primary', 'unknown').strip()
            domain = row.get('domain', '').strip()
            capability = row.get('capability', '').strip()
            
            # 取前 50 字作为标题
            title = knowledge_text[:50].replace('\n', ' ').strip()
            
            doc = IndustrialDocument(
                doc_id=doc_id,
                source="industrybench",
                source_name=f"IndustryBench-{industry}",
                title=title,
                content=knowledge_text,
                metadata={
                    "industry": industry,
                    "domain": domain,
                    "capability": capability,
                    "qa_id": row.get('id', ''),
                    "original_question": row.get('question', '')[:100],
                }
            )
            documents.append(doc)
    
    logger.info(f"[IndustryBench] Extracted {len(documents)} unique documents")
    return documents


# ============================================================
# 3. 从文本文件导入文档
# ============================================================

def extract_from_txt_file(filepath: str, source_type: str, source_name: str = None) -> List[IndustrialDocument]:
    """
    从纯文本文件导入工业文档。
    每个非空行段被视为一个独立文档。
    
    Args:
        filepath: 文本文件路径
        source_type: 来源类型 (technical_manual / standard / maintenance / other)
        source_name: 来源名称 (默认用文件名)
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 按空行分割为段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    basename = source_name or os.path.splitext(os.path.basename(filepath))[0]
    documents = []
    
    for idx, para in enumerate(paragraphs):
        doc = IndustrialDocument(
            doc_id=f"{source_type}_{basename}_{idx}",
            source=source_type,
            source_name=basename,
            title=para[:50].replace('\n', ' ').strip(),
            content=para,
            metadata={"paragraph_index": idx, "file": os.path.basename(filepath)}
        )
        documents.append(doc)
    
    logger.info(f"[{source_type}] Imported {len(documents)} documents from {filepath}")
    return documents


def extract_from_jsonl(filepath: str, source_type: str, source_name: str = None) -> List[IndustrialDocument]:
    """从 JSONL 文件导入文档（兼容未来扩展）"""
    if not os.path.exists(filepath):
        logger.warning(f"JSONL not found: {filepath}")
        return []
    
    documents = []
    basename = source_name or os.path.splitext(os.path.basename(filepath))[0]
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs = obj.get('documents', [obj])  # 支持单条或批量
            for d in docs:
                doc = IndustrialDocument(
                    doc_id=d.get('doc_id', f"{source_type}_{len(documents)}"),
                    source=source_type,
                    source_name=d.get('source_name', basename),
                    title=d.get('title', ''),
                    content=d.get('content', ''),
                    metadata=d.get('metadata', {})
                )
                documents.append(doc)
    
    logger.info(f"[{source_type}] Loaded {len(documents)} documents from {filepath}")
    return documents


# ============================================================
# 4. 语料库构建器（统一入口）
# ============================================================

class IndustrialCorpusBuilder:
    """
    统一工业知识语料库构建器。
    
    使用方法:
        builder = IndustrialCorpusBuilder()
        builder.add_industrybench("path/to/huggingface_dataset.csv")
        builder.add_txt_files("path/to/manuals/", "technical_manual")
        builder.add_txt_files("path/to/standards/", "standard")
        builder.save("path/to/output.jsonl")
    """
    
    def __init__(self):
        self.documents: List[IndustrialDocument] = []
        self._doc_ids = set()  # 去重
    
    def add_industrybench(self, csv_path: str) -> 'IndustrialCorpusBuilder':
        """添加 IndustryBench 知识源"""
        docs = extract_from_industrybench(csv_path)
        self._add_docs(docs)
        return self
    
    def add_txt_files(self, directory: str, source_type: str, pattern: str = "*.txt") -> 'IndustrialCorpusBuilder':
        """添加目录下所有文本文件"""
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return self
        
        files = glob.glob(os.path.join(directory, pattern))
        for fp in sorted(files):
            docs = extract_from_txt_file(fp, source_type)
            self._add_docs(docs)
        
        logger.info(f"[{source_type}] Total: {len(files)} files loaded")
        return self
    
    def add_jsonl_files(self, directory: str, source_type: str, pattern: str = "*.jsonl") -> 'IndustrialCorpusBuilder':
        """添加 JSONL 文件"""
        if not os.path.exists(directory):
            return self
        files = glob.glob(os.path.join(directory, pattern))
        for fp in sorted(files):
            docs = extract_from_jsonl(fp, source_type)
            self._add_docs(docs)
        return self
    
    def add_manual_document(self, content: str, title: str, source_type: str = "other",
                            source_name: str = "manual") -> 'IndustrialCorpusBuilder':
        """手动添加一篇文档"""
        doc = IndustrialDocument(
            doc_id=f"manual_{len(self.documents)}",
            source=source_type,
            source_name=source_name,
            title=title,
            content=content,
        )
        self._add_docs([doc])
        return self
    
    def _add_docs(self, docs: List[IndustrialDocument]):
        """添加文档（去重）"""
        for doc in docs:
            if doc.doc_id not in self._doc_ids:
                self._doc_ids.add(doc.doc_id)
                self.documents.append(doc)
    
    def save(self, output_path: str) -> str:
        """
        保存语料库为 JSONL 文件。
        
        格式: 每行一个 JSON 对象
        {"doc_id": "...", "source": "...", "source_name": "...", 
         "title": "...", "content": "...", "metadata": {...}}
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + '\n')
        
        logger.info(f"✅ 语料库已保存: {output_path}")
        logger.info(f"   总文档数: {len(self.documents)}")
        
        # 统计各来源
        sources = {}
        for doc in self.documents:
            sources[doc.source] = sources.get(doc.source, 0) + 1
        for src, cnt in sorted(sources.items()):
            logger.info(f"   [{src}] {cnt} 篇")
        
        return output_path
    
    @staticmethod
    def load(corpus_path: str) -> List[IndustrialDocument]:
        """加载 JSONL 格式的语料库"""
        docs = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    docs.append(IndustrialDocument(**data))
        logger.info(f"Loaded {len(docs)} documents from {corpus_path}")
        return docs
    
    @staticmethod
    def get_summary(corpus_path: str) -> Dict:
        """获取语料库统计摘要"""
        docs = IndustrialCorpusBuilder.load(corpus_path)
        sources = {}
        total_chars = 0
        for doc in docs:
            sources[doc.source] = sources.get(doc.source, 0) + 1
            total_chars += len(doc.content)
        return {
            "total_documents": len(docs),
            "total_characters": total_chars,
            "avg_chars_per_doc": total_chars / max(len(docs), 1),
            "sources": sources,
        }


# ============================================================
# 5. CLI 入口
# ============================================================

def build_corpus(project_root: str = None, output_path: str = None):
    """
    构建完整工业语料库。
    
    Args:
        project_root: LINS-Industrial 根目录
        output_path: 输出 JSONL 路径
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_path is None:
        output_path = os.path.join(project_root, 'knowledge_corpus', 'sources', 'industrial_corpus.jsonl')
    
    builder = IndustrialCorpusBuilder()
    
    # 1. IndustryBench 知识源
    csv_path = os.path.join(project_root, 'data', 'industrybench', 'huggingface_dataset.csv')
    builder.add_industrybench(csv_path)
    
    # 2. 已有的 industry_kb.txt
    kb_txt = os.path.join(project_root, 'data', 'industry_kb', 'industry_kb.txt')
    if os.path.exists(kb_txt):
        builder.add_txt_files(os.path.dirname(kb_txt), "industry_kb", "*.txt")
    
    # 3. 技术手册目录 (如果存在)
    manuals_dir = os.path.join(project_root, 'data', 'technical_manuals')
    builder.add_txt_files(manuals_dir, "technical_manual")
    
    # 4. 标准目录 (如果存在)
    standards_dir = os.path.join(project_root, 'data', 'standards')
    builder.add_txt_files(standards_dir, "standard")
    
    # 保存
    builder.save(output_path)
    
    # 打印摘要
    summary = IndustrialCorpusBuilder.get_summary(output_path)
    print("\n" + "=" * 60)
    print("📊 工业知识语料库构建完成")
    print("=" * 60)
    print(f"  📁 路径: {output_path}")
    print(f"  📄 文档数: {summary['total_documents']}")
    print(f"  📝 总字符: {summary['total_characters']:,}")
    print(f"  📏 平均长度: {summary['avg_chars_per_doc']:.0f} 字符")
    print(f"\n  📚 来源分布:")
    for src, cnt in sorted(summary['sources'].items()):
        print(f"    {src:<20s}: {cnt:>5d} 篇")
    
    return output_path


if __name__ == "__main__":
    build_corpus()
