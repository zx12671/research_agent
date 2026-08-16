"""
source_registry.py: 知识源注册表

集中管理所有已注册的 KnowledgeSource 实现。
支持按名称查询、按行业过滤、批量操作所有知识源。
"""

import logging
from typing import Dict, List, Optional, Type

from .base_source import KnowledgeSource
from .unified_document import UnifiedDocument

logger = logging.getLogger(__name__)


class SourceRegistry:
    """
    知识源注册表。

    管理多个 KnowledgeSource 实例，提供统一的查询、统计接口。

    用法:
        registry = SourceRegistry()
        registry.register("industrybench", IndustryBenchSource(csv_path))
        registry.register("techmanual", TechManualQASource(data_path))

        all_docs = registry.load_all()
        ib_stats = registry.get_source("industrybench").stats()
    """

    def __init__(self):
        self._sources: Dict[str, KnowledgeSource] = {}

    # ----------------------------------------------------------
    # 注册 / 注销
    # ----------------------------------------------------------

    def register(self, name: str, source: KnowledgeSource) -> None:
        """
        注册一个知识源。

        Args:
            name:   知识源唯一名称（用于后续引用）
            source: KnowledgeSource 实例
        """
        if name in self._sources:
            logger.warning(f"知识源 '{name}' 已存在，将被覆盖")
        self._sources[name] = source
        logger.info(f"注册知识源: {name} → {source}")

    def register_by_class(
        self,
        name: str,
        source_class: Type[KnowledgeSource],
        *args,
        **kwargs,
    ) -> KnowledgeSource:
        """
        按类名注册（自动实例化）。

        Args:
            name:          注册名称
            source_class:  知识源类
            *args, **kwargs: 传入构造函数的参数

        Returns:
            创建的知识源实例
        """
        instance = source_class(*args, **kwargs)
        self.register(name, instance)
        return instance

    def unregister(self, name: str) -> bool:
        """注销指定知识源。"""
        if name in self._sources:
            del self._sources[name]
            logger.info(f"注销知识源: {name}")
            return True
        return False

    def clear(self) -> None:
        """注销所有知识源。"""
        self._sources.clear()
        logger.info("已清除所有知识源")

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_source(self, name: str) -> Optional[KnowledgeSource]:
        """按名称获取知识源实例。"""
        return self._sources.get(name)

    def list_sources(self) -> List[str]:
        """返回所有已注册的知识源名称列表。"""
        return list(self._sources.keys())

    def get_source_names_and_types(self) -> Dict[str, str]:
        """返回 {名称: source_name} 映射。"""
        return {name: src.source_name for name, src in self._sources.items()}

    # ----------------------------------------------------------
    # 批量操作
    # ----------------------------------------------------------

    def discover_all(self) -> Dict[str, List[UnifiedDocument]]:
        """
        在所有已注册知识源上调用 discover()。

        Returns:
            {source_name: [documents]}
        """
        return {name: src.discover() for name, src in self._sources.items()}

    def load_all(self) -> List[UnifiedDocument]:
        """
        从所有知识源加载全部文档。

        Returns:
            所有文档的扁平列表
        """
        all_docs = []
        for name, src in self._sources.items():
            try:
                docs = src.load_all()
                logger.info(f"  [{name}] 加载 {len(docs)} 篇文档")
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"  [{name}] 加载失败: {e}")
        return all_docs

    def search_all(self, query: str, top_k_per_source: int = 5) -> List[UnifiedDocument]:
        """
        在所有知识源中搜索。

        Args:
            query:              搜索查询
            top_k_per_source:   每个知识源返回的最大结果数

        Returns:
            合并后的搜索结果列表
        """
        results = []
        for name, src in self._sources.items():
            try:
                hits = src.search(query, top_k=top_k_per_source)
                results.extend(hits)
            except Exception as e:
                logger.error(f"  [{name}] 搜索失败: {e}")
        return results

    # ----------------------------------------------------------
    # 过滤
    # ----------------------------------------------------------

    def filter_by_industry(self, industry: str) -> List[UnifiedDocument]:
        """
        按行业筛选所有文档。

        Args:
            industry: 行业名称（如 "冶金钢铁矿产"）

        Returns:
            匹配行业的所有文档
        """
        all_docs = self.load_all()
        return [d for d in all_docs if d.industry == industry]

    def filter_by_capability(self, capability: str) -> List[UnifiedDocument]:
        """按能力分类筛选文档。"""
        all_docs = self.load_all()
        return [d for d in all_docs if d.capability == capability]

    def filter_by_source(self, source_name: str) -> List[UnifiedDocument]:
        """按来源类型筛选文档。"""
        all_docs = self.load_all()
        return [d for d in all_docs if d.source == source_name]

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------

    def global_stats(self) -> Dict[str, object]:
        """
        返回所有知识源的全局统计信息。

        Returns:
            {
                "total_sources": N,
                "total_documents": N,
                "total_characters": N,
                "by_source": {source_name: count},
                "by_industry": {industry: count},
                "by_capability": {capability: count},
            }
        """
        all_docs = self.load_all()

        by_source = {}
        by_industry = {}
        by_capability = {}
        total_chars = 0

        for d in all_docs:
            by_source[d.source] = by_source.get(d.source, 0) + 1
            by_industry[d.industry or "unknown"] = by_industry.get(d.industry or "unknown", 0) + 1
            by_capability[d.capability or "unknown"] = by_capability.get(d.capability or "unknown", 0) + 1
            total_chars += len(d.text)

        return {
            "total_sources": len(self._sources),
            "total_documents": len(all_docs),
            "total_characters": total_chars,
            "by_source": by_source,
            "by_industry": by_industry,
            "by_capability": by_capability,
        }

    def __repr__(self) -> str:
        return f"<SourceRegistry sources={list(self._sources.keys())}>"

    def __len__(self) -> int:
        return len(self._sources)
