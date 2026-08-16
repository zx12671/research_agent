"""
knowledge_source: 工业知识源抽象层

为所有工业知识源提供统一的接口，支持多种来源类型：
- IndustryBench 数据集
- TechManualQA 技术手册问答
- PDF Manuals PDF 技术手册
- Engineering Standards 工程标准
- 未来扩展（如 FactoryWave、OncoKB 等）

每个知识源均输出 UnifiedDocument 统一格式。
"""

from .unified_document import UnifiedDocument
from .base_source import KnowledgeSource
from .industrybench_source import IndustryBenchSource
from .tech_manual_qa_source import TechManualQASource
from .pdf_manual_source import PDFManualSource
from .engineering_standards_source import EngineeringStandardsSource
from .source_registry import SourceRegistry

__all__ = [
    "UnifiedDocument",
    "KnowledgeSource",
    "IndustryBenchSource",
    "TechManualQASource",
    "PDFManualSource",
    "EngineeringStandardsSource",
    "SourceRegistry",
]
