"""
base_source.py: KnowledgeSource 抽象基类

所有知识源实现都必须继承自此类，并实现：
- discover():    发现/列出可用文档
- load(doc_id):  按 ID 加载单个文档
- load_all():    批量加载所有文档
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .unified_document import UnifiedDocument


class KnowledgeSource(ABC):
    """
    知识源抽象基类。

    每个知识源实现都是一个适配器（Adapter），将外部数据格式
    转换为 UnifiedDocument 统一格式。

    子类必须设置的属性:
        source_name: str — 来源标识（如 "IndustryBench"）
    """

    source_name: str = "unknown"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 知识源配置字典（路径、参数等）
        """
        self.config = config or {}
        self._cache: Dict[str, UnifiedDocument] = {}

    # ----------------------------------------------------------
    # 必须实现的抽象方法
    # ----------------------------------------------------------

    @abstractmethod
    def discover(self) -> List[UnifiedDocument]:
        """
        发现该知识源下的所有可用文档。

        Returns:
            文档列表（仅元数据，text 字段可能为空或摘要）
            可通过 load() 获取完整内容。
        """
        ...

    @abstractmethod
    def load(self, document_id: str) -> Optional[UnifiedDocument]:
        """
        按 ID 加载单个文档的完整内容。

        Args:
            document_id: 文档唯一 ID

        Returns:
            UnifiedDocument 或 None（未找到时）
        """
        ...

    # ----------------------------------------------------------
    # 可选覆写的方法
    # ----------------------------------------------------------

    def load_all(self) -> List[UnifiedDocument]:
        """
        批量加载所有文档（默认实现：discover → load 逐个加载）。

        子类可覆写此方法以实现更高效的批量加载。
        """
        discovered = self.discover()
        results = []
        for doc in discovered:
            full = self.load(doc.document_id)
            if full is not None:
                results.append(full)
            else:
                # discover 返回的可能已包含完整内容
                results.append(doc)
        return results

    def search(self, query: str, top_k: int = 10) -> List[UnifiedDocument]:
        """
        在知识源中搜索相关文档。

        基类提供简单的关键词匹配（按标题/文本包含查询词过滤）。
        子类可覆写为更高级的语义搜索。

        Args:
            query: 搜索查询
            top_k: 返回的最大结果数

        Returns:
            匹配的文档列表
        """
        if not self._cache:
            self._cache = {doc.document_id: doc for doc in self.load_all()}

        query_lower = query.lower()
        matched = []
        for doc in self._cache.values():
            if query_lower in doc.title.lower() or query_lower in doc.text.lower():
                matched.append(doc)
        # 按相关度排序：标题匹配优先
        matched.sort(key=lambda d: (
            0 if query_lower in d.title.lower() else 1,
        ))
        return matched[:top_k]

    def stats(self) -> Dict[str, Any]:
        """
        返回该知识源的统计信息。

        Returns:
            {"total_documents": N, "total_chars": N, "industries": {...}, ...}
        """
        docs = self.discover()
        industries = {}
        capabilities = {}
        total_chars = sum(len(d.text) for d in docs)

        for d in docs:
            ind = d.industry or "unknown"
            cap = d.capability or "unknown"
            industries[ind] = industries.get(ind, 0) + 1
            capabilities[cap] = capabilities.get(cap, 0) + 1

        return {
            "source": self.source_name,
            "total_documents": len(docs),
            "total_characters": total_chars,
            "industries": industries,
            "capabilities": capabilities,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} source='{self.source_name}'>"
