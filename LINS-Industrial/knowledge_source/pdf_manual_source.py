"""
pdf_manual_source.py: PDF Manuals 知识源适配器

从 PDF 技术手册文件中提取文本并转换为 UnifiedDocument。
支持单文件或多文件目录加载。

依赖库: PyMuPDF (fitz) 或 pdfplumber
（如未安装，会 fallback 到纯文本文件处理）
"""

import os
import glob
import logging
from typing import List, Optional, Dict, Any, Iterator

from .base_source import KnowledgeSource
from .unified_document import UnifiedDocument

logger = logging.getLogger(__name__)

# 尝试导入 PDF 解析库
try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    import pdfplumber

    _HAS_PDFPLUMBER = False  # 只在 PyMuPDF 不可用时使用
except ImportError:
    _HAS_PDFPLUMBER = False


class PDFManualSource(KnowledgeSource):
    """
    PDF 技术手册知识源。

    从 PDF 文件中提取文本内容，支持按页切分或整篇合并，
    自动识别行业/能力元数据（从文件名/目录结构推断）。
    """

    source_name = "PDF_Manual"

    def __init__(self, data_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            data_path: PDF 文件路径，或包含 PDF 文件的目录路径。
            config:
                - page_mode: str
                    "merge"（默认）— 整篇 PDF 作为一个文档
                    "per_page"       — 每页作为一个独立文档
                - recursive: bool — 是否递归扫描子目录（默认 True）
                - extract_tables: bool — 是否提取表格文本（默认 False）
                - industry_from_path: bool — 从目录名推断 industry（默认 True）
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
            logger.warning(f"PDF 路径不存在: {self.data_path}")
            return []

        if os.path.isfile(self.data_path):
            self._process_pdf(self.data_path)
        else:
            recursive = self.config.get("recursive", True)
            pattern = os.path.join(
                self.data_path, "**" if recursive else "", "*.pdf"
            )
            files = sorted(glob.glob(pattern, recursive=recursive))
            for fp in files:
                self._process_pdf(fp)

        logger.info(f"  → 发现 {len(self._documents)} 条 PDF 文档")
        return self._documents.copy()

    def load(self, document_id: str) -> Optional[UnifiedDocument]:
        if not self._documents:
            self.discover()
        for doc in self._documents:
            if doc.document_id == document_id:
                return doc
        return None

    # ----------------------------------------------------------
    # PDF 处理
    # ----------------------------------------------------------

    def _process_pdf(self, filepath: str):
        """
        处理单个 PDF 文件，提取文本并生成 UnifiedDocument。

        如果无法解析 PDF（依赖缺失或文件损坏），
        会尝试读取同名的 .txt fallback 文件。
        """
        basename = os.path.splitext(os.path.basename(filepath))[0]
        industry = self._infer_industry(filepath)
        capability = self.config.get("capability", "技术手册")

        page_mode = self.config.get("page_mode", "merge")

        if _HAS_PYMUPDF:
            self._process_with_pymupdf(filepath, basename, industry, capability, page_mode)
        else:
            # fallback: 尝试读取同名 .txt 文件
            txt_path = filepath.replace(".pdf", ".txt")
            if os.path.exists(txt_path):
                self._process_txt_fallback(txt_path, basename, industry, capability)
            else:
                logger.warning(
                    f"PDF 解析库不可用，且未找到 fallback 文本文件: {txt_path}"
                )

    def _process_with_pymupdf(
        self,
        filepath: str,
        basename: str,
        industry: str,
        capability: str,
        page_mode: str,
    ):
        """使用 PyMuPDF 解析 PDF"""
        try:
            doc = fitz.open(filepath)

            if page_mode == "per_page":
                # 每页为一个独立文档
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if not text:
                        continue

                    page_id = f"pdf_{basename}_p{page_num + 1:04d}"
                    self._documents.append(
                        UnifiedDocument(
                            document_id=page_id,
                            title=f"{basename} - 第 {page_num + 1} 页",
                            text=text,
                            source=self.source_name,
                            industry=industry,
                            capability=capability,
                            metadata={
                                "file": os.path.basename(filepath),
                                "page": page_num + 1,
                                "total_pages": len(doc),
                            },
                        )
                    )
            else:
                # 整篇合并为一个文档
                full_text = "\n".join(
                    page.get_text().strip() for page in doc if page.get_text().strip()
                )
                if full_text:
                    self._documents.append(
                        UnifiedDocument(
                            document_id=f"pdf_{basename}",
                            title=basename,
                            text=full_text,
                            source=self.source_name,
                            industry=industry,
                            capability=capability,
                            metadata={
                                "file": os.path.basename(filepath),
                                "pages": len(doc),
                            },
                        )
                    )

            doc.close()

        except Exception as e:
            logger.error(f"  解析 PDF 失败 [{filepath}]: {e}")

    def _process_txt_fallback(
        self, filepath: str, basename: str, industry: str, capability: str
    ):
        """从文本文件读取内容（PDF 解析不可用时的 fallback）"""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            self._documents.append(
                UnifiedDocument(
                    document_id=f"pdf_{basename}",
                    title=basename,
                    text=text,
                    source=self.source_name,
                    industry=industry,
                    capability=capability,
                    metadata={"file": os.path.basename(filepath), "fallback": True},
                )
            )

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def _infer_industry(self, filepath: str) -> str:
        """
        从文件路径中推断所属行业。

        如果启用了 industry_from_path，尝试从目录结构中提取行业名。
        否则使用配置文件中的默认值。

        例如: data/pdf_manuals/冶金钢铁矿产/xxx.pdf → "冶金钢铁矿产"
        """
        if not self.config.get("industry_from_path", True):
            return self.config.get("industry", "")

        rel_path = os.path.relpath(filepath, self.data_path)
        parts = rel_path.replace("\\", "/").split("/")

        if len(parts) > 1:
            # 一级子目录名作为行业名
            candidate = parts[-2]
            # 排除常见非行业目录名
            non_industry = {"pdf", "pdfs", "manual", "manuals", "data", "docs"}
            if candidate.lower() not in non_industry:
                return candidate

        return self.config.get("industry", "其他")
