"""
unified_document.py: 统一文档格式

所有 KnowledgeSource 实现都必须输出此格式。
这是知识源抽象层的核心数据结构。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any


@dataclass
class UnifiedDocument:
    """
    统一文档格式 —— 所有知识源的标准化输出。

    Fields:
        document_id: 文档唯一标识符
        title:       文档标题
        text:        文档正文内容
        source:      来源类型标识（如 "IndustryBench", "TechManualQA", "PDF_Manual", "Engineering_Standard"）
        industry:    所属行业（如 "冶金钢铁矿产", "电工电器电力"）
        capability:  能力分类（如 "选型与替代", "故障诊断与排查"）
        metadata:    扩展元数据字典（可包含难度、领域、原始路径等）
    """
    document_id: str
    title: str
    text: str
    source: str
    industry: str = ""
    capability: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（适合序列化为 JSON）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedDocument":
        """从字典恢复 UnifiedDocument"""
        return cls(**data)

    @property
    def summary(self) -> str:
        """简短摘要（用于日志和调试）"""
        return (
            f"[{self.source}] {self.document_id} | "
            f"industry={self.industry} | "
            f"capability={self.capability} | "
            f"text_len={len(self.text)}"
        )

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"<UnifiedDocument {self.document_id[:20]}... from {self.source}>"
