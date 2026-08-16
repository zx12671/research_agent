"""
tech_manual_qa_source.py: TechManualQA 知识源适配器

从技术手册问答数据集加载文档，适配为 KnowledgeSource 接口。
支持 CSV / JSONL 两种格式。

数据格式期望:
    CSV:   id, question, answer, knowledge_text, industry, capability, ...
    JSONL: {"id": ..., "question": ..., "answer": ..., "knowledge_text": ..., "industry": ..., "capability": ...}
"""

import os
import csv
import json
import glob
import hashlib
import logging
from typing import List, Optional, Dict, Any

from .base_source import KnowledgeSource
from .unified_document import UnifiedDocument

logger = logging.getLogger(__name__)


class TechManualQASource(KnowledgeSource):
    """
    技术手册问答 (TechManualQA) 知识源。

    从包含技术问答对的文件中加载文档，每条记录包含
    问题、答案和知识文本，适用于工业设备选型、维护等技术场景。
    """

    source_name = "TechManualQA"

    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            data_path: CSV/JSONL 文件路径，或包含此类文件的目录路径。
            config:
                - recursive: bool — 是否递归扫描子目录（默认 True）
                - file_pattern: str — 文件通配模式（默认 "*.csv" / "*.jsonl"）
        """
        super().__init__(config)
        self.data_path = os.path.abspath(data_path)
        self._documents: List[UnifiedDocument] = []

    # ----------------------------------------------------------
    # 核心方法
    # ----------------------------------------------------------

    def discover(self) -> List[UnifiedDocument]:
        if self._documents:
            return self._documents.copy()

        if not os.path.exists(self.data_path):
            logger.warning(f"TechManualQA path not found: {self.data_path}")
            return []

        if os.path.isfile(self.data_path):
            # 单个文件
            self._load_file(self.data_path)
        else:
            # 目录：扫描匹配的文件
            recursive = self.config.get("recursive", True)
            for pattern in ["*.csv", "*.jsonl"]:
                files = glob.glob(
                    os.path.join(self.data_path, "**", pattern) if recursive
                    else os.path.join(self.data_path, pattern),
                    recursive=recursive,
                )
                for fp in sorted(files):
                    self._load_file(fp)

        logger.info(f"  → 发现 {len(self._documents)} 条 TechManualQA 文档")
        return self._documents.copy()

    def load(self, document_id: str) -> Optional[UnifiedDocument]:
        if not self._documents:
            self.discover()
        for doc in self._documents:
            if doc.document_id == document_id:
                return doc
        return None

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _load_file(self, filepath: str):
        """加载单个文件"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            self._load_csv(filepath)
        elif ext == ".jsonl":
            self._load_jsonl(filepath)
        else:
            logger.debug(f"跳过不支持的文件格式: {filepath}")

    def _load_csv(self, filepath: str):
        """从 CSV 加载文档"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc = self._row_to_doc(row, basename)
                if doc:
                    self._documents.append(doc)

        logger.debug(f"  [CSV] {filepath}: {len(self._documents)} 条")

    def _load_jsonl(self, filepath: str):
        """从 JSONL 加载文档"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc = self._row_to_doc(row, basename)
                if doc:
                    self._documents.append(doc)

        logger.debug(f"  [JSONL] {filepath}: {len(self._documents)} 条")

    def _row_to_doc(self, row: Dict[str, str], source_name: str) -> Optional[UnifiedDocument]:
        """将一行数据转换为 UnifiedDocument"""
        knowledge_text = (
            row.get("knowledge_text", "")
            or row.get("text", "")
            or row.get("content", "")
        ).strip()
        if not knowledge_text:
            return None

        doc_id = row.get("id", "").strip()
        if not doc_id:
            doc_id = hashlib.md5(knowledge_text.encode()).hexdigest()[:16]

        industry = row.get("industry", "").strip()
        capability = row.get("capability", "").strip()
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()

        title = question[:60] if question else f"{source_name}-{doc_id}"

        return UnifiedDocument(
            document_id=f"techmanualqa_{doc_id}",
            title=title,
            text=knowledge_text,
            source=self.source_name,
            industry=industry,
            capability=capability,
            metadata={
                "question": question,
                "answer": answer,
                "difficulty": row.get("difficulty", "").strip(),
                "source_file": source_name,
            },
        )
