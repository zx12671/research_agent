"""
industrybench_source.py: IndustryBench 知识源适配器

将 IndustryBench 数据集 (huggingface_dataset.csv) 适配为 KnowledgeSource 接口，
输出 UnifiedDocument 统一格式。

CSV 字段映射:
    id               → document_id
    knowledge_text   → text
    industry_primary → industry
    capability       → capability
    question         → metadata["question"]
    answer           → metadata["answer"]
    difficulty       → metadata["difficulty"]
    domain           → metadata["domain"]
"""

import os
import csv
import hashlib
import logging
from typing import List, Optional, Dict, Any

from .base_source import KnowledgeSource
from .unified_document import UnifiedDocument

logger = logging.getLogger(__name__)


class IndustryBenchSource(KnowledgeSource):
    """
    IndustryBench 数据集知识源。

    读取 IndustryBench CSV 文件，将其中的每条记录转换为 UnifiedDocument。
    每行的 knowledge_text 作为文档正文，附加上行业、能力分类和难度等元数据。
    """

    source_name = "IndustryBench"

    def __init__(self, csv_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            csv_path: IndustryBench CSV 文件路径（绝对或相对于 project_root）
            config:   额外配置
        """
        super().__init__(config)
        self.csv_path = os.path.abspath(csv_path)
        self._documents: List[UnifiedDocument] = []

    # ----------------------------------------------------------
    # 核心方法
    # ----------------------------------------------------------

    def discover(self) -> List[UnifiedDocument]:
        """
        扫描 CSV 并返回所有文档摘要。
        注意：对于 IndustryBench，discover 返回的就是完整文档
              （因为 knowledge_text 就在 CSV 中，无需额外加载）。
        """
        if self._documents:
            return self._documents.copy()

        if not os.path.exists(self.csv_path):
            logger.warning(f"IndustryBench CSV not found: {self.csv_path}")
            return []

        logger.info(f"加载 IndustryBench 数据集: {self.csv_path}")

        seen_ids = set()
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                knowledge_text = row.get("knowledge_text", "").strip()
                if not knowledge_text:
                    continue

                doc_id = row.get("id", "").strip()
                if not doc_id:
                    # 如果没有 id 列，用 MD5 生成
                    doc_id = hashlib.md5(knowledge_text.encode()).hexdigest()[:16]

                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                industry = row.get("industry_primary", "").strip()
                capability = row.get("capability", "").strip()

                doc = UnifiedDocument(
                    document_id=f"industrybench_{doc_id}",
                    title=f"[{industry}] {capability}" if industry and capability else f"IndustryBench-{doc_id}",
                    text=knowledge_text,
                    source=self.source_name,
                    industry=industry,
                    capability=capability,
                    metadata={
                        "question": row.get("question", "").strip(),
                        "answer": row.get("answer", "").strip(),
                        "difficulty": row.get("difficulty", "").strip(),
                        "format": row.get("_format", "").strip(),
                        "domain": row.get("domain", "").strip(),
                        "question_en": row.get("question_en", "").strip(),
                        "answer_en": row.get("answer_en", "").strip(),
                    },
                )
                self._documents.append(doc)

        logger.info(f"  → 发现 {len(self._documents)} 条 IndustryBench 文档")
        return self._documents.copy()

    def load(self, document_id: str) -> Optional[UnifiedDocument]:
        """
        按前缀 ID 查找文档（格式: "industrybench_{id}" 或 纯数字 id）。
        """
        if not self._documents:
            self.discover()

        # 支持多种匹配格式
        for doc in self._documents:
            if doc.document_id == document_id:
                return doc
            # 也支持不带前缀的纯 ID
            if document_id.isdigit() and doc.document_id == f"industrybench_{document_id}":
                return doc

        return None

    def stats(self) -> Dict[str, Any]:
        """增强统计：加入难度分布和格式分布"""
        base_stats = super().stats()
        docs = self.discover()

        difficulties = {}
        formats = {}
        for d in docs:
            diff = d.metadata.get("difficulty", "unknown")
            fmt = d.metadata.get("format", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1
            formats[fmt] = formats.get(fmt, 0) + 1

        base_stats["difficulties"] = difficulties
        base_stats["formats"] = formats
        return base_stats
