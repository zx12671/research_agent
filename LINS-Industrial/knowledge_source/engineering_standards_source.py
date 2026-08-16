"""
engineering_standards_source.py: Engineering Standards 知识源适配器

从工程标准文档（国标 GB、ISO、行业标准等）中加载内容，
适配为 KnowledgeSource 接口。

支持的文件格式:
    - .txt (纯文本标准文档)
    - .csv / .jsonl (结构化标准条目)
    - .pdf (带 fallback 的 PDF 标准文档)

支持从标准编号（GB/T XXXXX—XXXX）自动提取元数据。
"""

import os
import re
import csv
import json
import glob
import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple

from .base_source import KnowledgeSource
from .unified_document import UnifiedDocument

logger = logging.getLogger(__name__)

# 标准编号正则：GB/T 12345—2021 或 GB 12345-2021 或 ISO 12345:2021
STANDARD_NUM_RE = re.compile(
    r"(?P<prefix>[A-Z]+(?:/[A-Z]+)?)[\s\-–—]*(?P<num>\d+[\w\-.]*)[\s\-–—:：]*(?P<year>\d{4})?"
)


class EngineeringStandardsSource(KnowledgeSource):
    """
    工程标准知识源。

    从工程标准文档中提取内容，自动识别标准编号、行业分类，
    输出结构化的 UnifiedDocument。

    支持的加载方式:
        - 单个标准文件（.txt / .csv / .jsonl / .pdf）
        - 包含标准文件的目录
    """

    source_name = "Engineering_Standard"

    # 标准编号 → 行业映射
    STANDARD_INDUSTRY_MAP = {
        "GB/T": "通用",
        "GB": "通用",
        "ISO": "国际标准",
        "IEC": "电工电器电力",
        "JB/T": "机械五金加工",
        "JB": "机械五金加工",
        "DL/T": "电工电器电力",
        "DL": "电工电器电力",
        "YB/T": "冶金钢铁矿产",
        "YB": "冶金钢铁矿产",
        "HG/T": "化工",
        "HG": "化工",
        "SH": "石油化工",
        "SY/T": "石油天然气",
        "SY": "石油天然气",
        "QC/T": "汽车制造",
        "QC": "汽车制造",
        "MT/T": "煤矿安全",
        "MT": "煤矿安全",
        "NB/T": "能源",
        "NB": "能源",
        "SJ/T": "电子仪表传感",
        "SJ": "电子仪表传感",
        "LY/T": "林业",
        "LY": "林业",
        "FZ/T": "纺织",
        "FZ": "纺织",
    }

    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            data_path: 标准文件路径或目录路径。
            config:
                - recursive: bool — 是否递归扫描子目录（默认 True）
                - industry: str — 若未自动识别时的默认行业
                - extract_metadata: bool — 是否尝试从文本提取标准号等元数据（默认 True）
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
            logger.warning(f"标准文档路径不存在: {self.data_path}")
            return []

        if os.path.isfile(self.data_path):
            self._load_file(self.data_path)
        else:
            recursive = self.config.get("recursive", True)
            for pattern in ["*.txt", "*.csv", "*.jsonl", "*.pdf"]:
                files = glob.glob(
                    os.path.join(self.data_path, "**" if recursive else "", pattern),
                    recursive=recursive,
                )
                for fp in sorted(files):
                    self._load_file(fp)

        logger.info(f"  → 发现 {len(self._documents)} 条工程标准文档")
        return self._documents.copy()

    def load(self, document_id: str) -> Optional[UnifiedDocument]:
        if not self._documents:
            self.discover()
        for doc in self._documents:
            if doc.document_id == document_id:
                return doc
        return None

    # ----------------------------------------------------------
    # 文件加载
    # ----------------------------------------------------------

    def _load_file(self, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            self._load_csv(filepath)
        elif ext == ".jsonl":
            self._load_jsonl(filepath)
        elif ext == ".txt":
            self._load_txt(filepath)
        elif ext == ".pdf":
            self._load_pdf(filepath)

    def _load_txt(self, filepath: str):
        """从纯文本文件加载标准文档"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return

        # 提取元数据
        metadata = self._extract_metadata(text, basename)
        doc_id = f"std_{metadata.get('standard_num', basename).replace('/', '_').replace(' ', '')}"
        industry = self._infer_industry(metadata.get("standard_num", ""))
        title = metadata.get("title") or basename

        self._documents.append(
            UnifiedDocument(
                document_id=doc_id,
                title=title,
                text=text,
                source=self.source_name,
                industry=industry,
                capability="标准规范与术语",
                metadata=metadata,
            )
        )

    def _load_csv(self, filepath: str):
        """从 CSV 加载标准条目"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    row.get("content", "")
                    or row.get("text", "")
                    or row.get("knowledge_text", "")
                ).strip()
                if not text:
                    continue

                std_num = row.get("standard_num", "").strip()
                doc_id = (
                    f"std_{std_num.replace('/', '_')}"
                    if std_num
                    else f"std_{basename}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                )

                industry = row.get("industry", "") or self._infer_industry(std_num)
                title = row.get("title", "") or row.get("name", "") or f"{std_num} {row.get('clause', '')}"

                self._documents.append(
                    UnifiedDocument(
                        document_id=doc_id,
                        title=title.strip() or basename,
                        text=text,
                        source=self.source_name,
                        industry=industry,
                        capability="标准规范与术语",
                        metadata={
                            "standard_num": std_num,
                            "clause": row.get("clause", "").strip(),
                            "description": row.get("description", "").strip(),
                            "source_file": basename,
                        },
                    )
                )

    def _load_jsonl(self, filepath: str):
        """从 JSONL 加载标准条目"""
        basename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = (
                    data.get("content", "")
                    or data.get("text", "")
                    or data.get("knowledge_text", "")
                ).strip()
                if not text:
                    continue
                std_num = data.get("standard_num", "").strip()
                doc_id = (
                    f"std_{std_num.replace('/', '_')}"
                    if std_num
                    else f"std_{basename}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                )
                industry = data.get("industry", "") or self._infer_industry(std_num)
                title = data.get("title", "") or data.get("name", "") or f"{std_num} {data.get('clause', '')}"

                self._documents.append(
                    UnifiedDocument(
                        document_id=doc_id,
                        title=title.strip() or basename,
                        text=text,
                        source=self.source_name,
                        industry=industry,
                        capability="标准规范与术语",
                        metadata={k: v for k, v in data.items() if k not in ("text", "content", "knowledge_text")},
                    )
                )

    def _load_pdf(self, filepath: str):
        """从 PDF 加载标准文档（委托给 PDFManualSource 处理后适配）"""
        from .pdf_manual_source import PDFManualSource

        pdf_source = PDFManualSource(
            filepath,
            config={
                "page_mode": "merge",
                "industry_from_path": True,
                "capability": "标准规范与术语",
            },
        )
        pdf_docs = pdf_source.discover()
        for pdf_doc in pdf_docs:
            # 提取标准编号
            metadata = self._extract_metadata(pdf_doc.text, pdf_doc.title)
            std_num = metadata.get("standard_num", "")
            if std_num:
                pdf_doc.document_id = f"std_{std_num.replace('/', '_').replace(' ', '')}"
                pdf_doc.industry = pdf_doc.industry or self._infer_industry(std_num)
                pdf_doc.metadata.update(metadata)
            pdf_doc.source = self.source_name
            pdf_doc.capability = pdf_doc.capability or "标准规范与术语"
            self._documents.append(pdf_doc)

    # ----------------------------------------------------------
    # 元数据提取
    # ----------------------------------------------------------

    def _extract_metadata(self, text: str, fallback_title: str) -> Dict[str, Any]:
        """
        从文本中提取标准元数据。

        尝试匹配:
            - 标准编号: GB/T 12345—2021, ISO 12345:2021
            - 标准标题: 常在编号后的首行
            - 发布年份
        """
        metadata: Dict[str, Any] = {}

        # 1. 查找标准编号
        match = STANDARD_NUM_RE.search(text)
        if match:
            prefix = match.group("prefix")
            num = match.group("num")
            year = match.group("year")
            metadata["standard_num"] = f"{prefix} {num}" + (f"—{year}" if year else "")
            metadata["prefix"] = prefix
            metadata["year"] = year

        # 2. 尝试提取标题（编号后的第一行）
        if match:
            lines = text.split("\n")
            for line in lines[:20]:  # 只在前20行搜索
                line = line.strip()
                if line and "标准" in line and len(line) > 10:
                    metadata["title"] = line
                    break

        metadata.setdefault("title", fallback_title)
        return metadata

    def _infer_industry(self, standard_num: str) -> str:
        """根据标准编号前缀推断行业"""
        if not standard_num:
            return self.config.get("industry", "其他")

        for prefix, industry in self.STANDARD_INDUSTRY_MAP.items():
            if standard_num.startswith(prefix):
                return industry

        return self.config.get("industry", "其他")
