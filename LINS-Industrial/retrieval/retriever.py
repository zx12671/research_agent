"""
retriever.py: 开放域工业知识检索器 (Open-Domain Retriever)

直接运行: cd LINS-Industrial && python retrieval/retriever.py
或用:    python -m retrieval.retriever

Pipeline (第一阶段):
    Question
       ↓
    KED (Keyword Extraction & Decomposition)
       ↓
    Embedding Search (over full knowledge base)
       ↓
    Top-k Chunks (configurable k)
       ↓
    Evidence Aggregation
       ↓
    Evidence (输出给 LINS 的 context)

注意:
    IndustrialRetriever 只负责检索，不负责推理/生成。
    检索到的 evidence 应作为 context 传入 LINS (MAIRAG, retrieval=False)，
    由 LINS 负责 Multi-Agent Reasoning + Citation + Answer Generation。
    不要将检索结果直接喂给普通 LLM — 那样会失去 LINS 的 Citation 能力。

核心改进:
  - 开放域检索：不再假设"一个问题对应一篇文档"
  - 从全量知识库中检索 top-k 最相关文本块
  - 支持 KED 关键词提取增强检索精度
  - 支持 Recall@k 评估
  - 完全可配置的 pipeline

使用示例:
    # 快速加载 + 检索
    from retrieval.retriever import OpenDomainRetriever
    from model.model_LINS import LINS
    
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()
    
    # 第一阶段: 检索证据
    result = retriever.retrieve("冶金轧机的过载保护需要什么配置?", k=5)
    evidence = result.get_context()
    
    # 第二阶段: LINS 推理 (retrieval=False, 只用传入的 evidence)
    lins = LINS(LLM_name='deepseek-chat', ...)
    response, urls, passages, history, sub_questions = lins.MAIRAG(
        question="冶金轧机的过载保护需要什么配置?",
        context=evidence,          # 传入外部检索结果
        retrieval=False            # 不调用 LINS 自己的检索器
    )
"""

import sys
import os

# ===== 路径修复：确保项目根目录在 sys.path 上 =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. 数据结构
# ============================================================

@dataclass
class RetrievedChunk:
    """单个检索结果块"""
    chunk_id: str
    document_id: str = ""
    source: str = ""
    capability: str = ""
    industry: str = ""
    content: str = ""
    score: float = 0.0
    rank: int = 0              # 排名 1..k
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "capability": self.capability,
            "industry": self.industry,
            "content": self.content,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass
class RetrievalResult:
    """一次检索的完整结果"""
    query: str
    query_expanded: str = ""       # KED 扩展后的查询
    chunks: List[RetrievedChunk] = field(default_factory=list)
    top_k: int = 5
    timing_ms: float = 0.0
    retrieval_failed: bool = False      # 检索阶段是否失败（避免与"正常检索到0个"混淆）
    retrieval_failed_reason: str = ""   # 失败原因（便于排查是 embedding 失败还是真无命中）
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "query_expanded": self.query_expanded,
            "num_results": len(self.chunks),
            "top_k": self.top_k,
            "timing_ms": round(self.timing_ms, 1),
            "retrieval_failed": self.retrieval_failed,
            "retrieval_failed_reason": self.retrieval_failed_reason,
            "results": [c.to_dict() for c in self.chunks],
        }

    
    def get_context(self, separator: str = "\n---\n") -> str:
        """将所有结果拼接为上下文文本"""
        parts = []
        for c in self.chunks:
            parts.append(f"[{c.rank}] (score={c.score:.3f}) [{c.industry}/{c.capability}]\n{c.content}")
        return separator.join(parts) if parts else ""


# ============================================================
# 2. KED: 关键词提取与分解
# ============================================================

class KEDExtractor:
    """
    关键词提取与分解 (Keyword Extraction & Decomposition).
    
    从问题中提取关键技术术语，构建扩展查询，
    用于提升嵌入检索的召回率。
    """
    
    # 工业领域常见的关键技术模式
    TECH_PATTERNS = [
        r'[A-Za-z0-9]+[-][A-Za-z0-9]+',           # SIMOREG-6RA70, DC-Master
        r'[A-Z]{2,}(?:\d+)?',                       # CNC, PLC, SIMOREG, DC
        r'\d+[\.]*\d*\s*[a-zA-Zμnmk°%#x/]+',       # 6RA70, 3000A, 2200kW
        r'[\u4e00-\u9fff]{2,}(?:装置|系统|设备|器|机|仪|剂|液|料|法)',  # 装置/系统/设备
        r'[\u4e00-\u9fff]{2,}(?:原理|方法|技术|工艺|流程|参数|指标)',     # 原理/方法/技术
        r'[\u4e00-\u9fff]{2,}(?:保护|控制|调节|驱动|检测|测试|诊断|维护)', # 保护/控制/调节
    ]
    
    def __init__(self, enable_ked: bool = True, max_keywords: int = 10):
        """
        Args:
            enable_ked: 是否启用 KED 扩展
            max_keywords: 最大提取关键词数
        """
        self.enable_ked = enable_ked
        self.max_keywords = max_keywords
    
    def extract(self, question: str) -> List[str]:
        """
        从问题中提取关键技术关键词。
        
        Args:
            question: 原始问题
        
        Returns:
            keywords: 提取的关键词列表
        """
        keywords = set()
        
        for pattern in self.TECH_PATTERNS:
            matches = re.findall(pattern, question)
            for m in matches:
                m = m.strip()
                if len(m) >= 2:
                    keywords.add(m)
        
        # 按长度排序，保留最长的（最具体）
        sorted_kw = sorted(keywords, key=len, reverse=True)
        return sorted_kw[:self.max_keywords]
    
    def expand_query(self, question: str) -> str:
        """
        扩展查询：原始问题 + 关键词组合。
        
        Returns:
            expanded_query: 扩展后的查询文本
        """
        if not self.enable_ked:
            return question
        
        keywords = self.extract(question)
        if not keywords:
            return question
        
        # 构建扩展查询：原问题 + 关键词连接
        keyword_str = " ".join(keywords[:5])
        expanded = f"{question} {keyword_str}"
        
        return expanded
    
    def decompose(self, question: str) -> List[str]:
        """
        将复杂问题分解为多个子查询。
        
        Args:
            question: 复杂问题
        
        Returns:
            sub_queries: 子查询列表
        """
        # 按"、"和"，"分割复合问题
        parts = re.split(r'[，、]', question)
        
        # 过滤过短片段
        sub_queries = [p.strip() for p in parts if len(p.strip()) >= 4]
        
        if len(sub_queries) <= 1:
            return [question]
        
        return sub_queries


# ============================================================
# 2.5 run2 两步式检索：相关性评估器抽象（可注入，不依赖具体 LLM）
# ============================================================

class RelevanceRanker:
    """run2 二阶段「AI 自主相关性评估」的抽象基类。

    职责：仅凭【问题 + 候选块内容】对每个候选块给出相关性/适配度分数（越高越适配），
    判定**不得引入标注答案 / GT / knowledge_text**（规避"直接读答案"的嫌疑）。

    实现方只需覆写 `score(query, chunks) -> {chunk_id: 0~1 分数}`。
    典型实现：LLM 打分（真实 DeepSeek，见探针脚本）；兜底：嵌入余弦 / 词法 jt。
    """

    def score(self, query: str, chunks) -> dict:
        """返回 {chunk_id: 相关性分数}，分数 ∈ [0,1]，越大越适配问题。"""
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


def _jt_grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def chinese_jt(a, b):
    """汉字 bigram Jaccard(IoU) —— 供文档级回捞作词法相关性过滤，与仓库 recall 口径一致。"""
    ga, gb = _jt_grams(a), _jt_grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


# ============================================================
# 3. 开放域检索器
# ============================================================

class OpenDomainRetriever:
    """
    开放域工业知识检索器。
    
    完整的检索 pipeline:
        Question → KED → Embedding → Top-k Search → Aggregation → Results
    
    支持:
      - 从 manifest 自动加载
      - KED 关键词扩展
      - 可配置 top-k
      - Recall@k 评估
      - 证据聚合
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.project_root = project_root
        
        self.indexer = None            # FAISS 索引
        self.chunks: Dict[str, Dict] = {}  # chunk_id → chunk_data (包含完整元数据)
        self.embedder = None           # 嵌入生成器 (延迟加载)
        self.embedding_model = "bge-small"
        self.ked = KEDExtractor()      # KED 模块
    
    # ----------------------------------------------------------
    # 3.1 加载索引
    # ----------------------------------------------------------
    
    def load(self, index_path: str, chunks_path: str = None,
             chunks: List[Dict] = None) -> "OpenDomainRetriever":
        """
        加载 FAISS 索引和块元数据。
        
        Args:
            index_path: .faiss 索引文件路径
            chunks_path: chunks JSONL 文件路径 (二选一)
            chunks: 预加载的块列表 (二选一)
        
        Returns:
            self
        """
        from retrieval.faiss_indexer import FaissIndexer
        self.indexer = FaissIndexer.load(index_path)
        
        # 从文件名推断模型名
        basename = os.path.basename(index_path)
        for model in ["bge-small", "bge-base", "text2vec", "m3e"]:
            if model in basename:
                self.embedding_model = model
                break
        
        # 加载 chunks
        if chunks_path is not None:
            self._load_chunks_from_file(chunks_path)
        elif chunks is not None:
            for c in chunks:
                cid = c.get("chunk_id") or c.get("chunk_id", "")
                self.chunks[cid] = c
        
        logger.info(f"✅ 开放域检索器就绪")
        logger.info(f"   索引: {index_path} ({len(self.indexer.chunk_ids)} 向量)")
        logger.info(f"   块: {len(self.chunks)} 个")
        logger.info(f"   模型: {self.embedding_model}")
        
        return self
    
    def load_from_manifest(self, manifest_path: str = None,
                           index_path: str = None,
                           chunks_path: str = None) -> "OpenDomainRetriever":
        """
        从 manifest.json 自动推断加载路径。
        
        Args:
            manifest_path: manifest.json 路径 (默认 knowledge_corpus/manifest.json)
            index_path: 显式指定索引路径 (可选)
            chunks_path: 显式指定 chunks 路径 (可选)
        
        Returns:
            self
        """
        if manifest_path is None:
            manifest_path = os.path.join(
                self.project_root, "knowledge_corpus", "manifest.json"
            )
        
        if not os.path.exists(manifest_path):
            # 没有 manifest，自动查找最新索引
            logger.warning(f"Manifest 不存在: {manifest_path}")
            return self._auto_discover(index_path, chunks_path)
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # 索引路径
        if index_path is None:
            index_path = manifest.get("index_path", "")
            if not index_path or not os.path.exists(index_path):
                index_path = self._find_latest_index()
        
        # chunks 路径
        if chunks_path is None:
            chunks_path = os.path.join(
                self.project_root, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl"
            )
            if not os.path.exists(chunks_path):
                chunks_path = self._find_latest_chunks()
        
        if index_path:
            self.load(index_path, chunks_path)
        else:
            raise FileNotFoundError("未找到 FAISS 索引文件，请先运行 knowledge_builder")
        
        return self
    
    def _auto_discover(self, index_path: str = None,
                       chunks_path: str = None) -> "OpenDomainRetriever":
        """自动发现最新的索引和 chunks 文件"""
        index_dir = os.path.join(self.project_root, "knowledge_corpus", "index")
        chunks_dir = os.path.join(self.project_root, "knowledge_corpus", "chunks")
        
        if index_path is None and os.path.exists(index_dir):
            faiss_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
            if faiss_files:
                index_path = os.path.join(index_dir, sorted(faiss_files)[-1])
        
        if chunks_path is None and os.path.exists(chunks_dir):
            jsonl_files = [f for f in os.listdir(chunks_dir) if f.endswith('.jsonl')]
            if jsonl_files:
                chunks_path = os.path.join(chunks_dir, sorted(jsonl_files)[-1])
        
        if index_path and os.path.exists(index_path):
            self.load(index_path, chunks_path)
        else:
            raise FileNotFoundError("未找到 FAISS 索引，请先运行 knowledge_builder")
        
        return self
    
    def _find_latest_index(self) -> Optional[str]:
        """查找最新的索引文件"""
        index_dir = os.path.join(self.project_root, "knowledge_corpus", "index")
        if os.path.exists(index_dir):
            faiss_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
            if faiss_files:
                return os.path.join(index_dir, sorted(faiss_files)[-1])
        return None
    
    def _find_latest_chunks(self) -> Optional[str]:
        """查找最新的 chunks 文件"""
        chunks_dir = os.path.join(self.project_root, "knowledge_corpus", "chunks")
        if os.path.exists(chunks_dir):
            jsonl_files = [f for f in os.listdir(chunks_dir) if f.endswith('.jsonl')]
            if jsonl_files:
                return os.path.join(chunks_dir, sorted(jsonl_files)[-1])
        return None
    
    def _load_chunks_from_file(self, chunks_path: str):
        """从 JSONL 文件加载块元数据"""
        if not os.path.exists(chunks_path):
            logger.warning(f"Chunks 文件不存在: {chunks_path}")
            return
        
        count = 0
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cid = data.get("chunk_id", "")
                    self.chunks[cid] = data
                    count += 1
        
        logger.info(f"  已加载 {count} 个块元数据")
    
    # ----------------------------------------------------------
    # 3.2 检索
    # ----------------------------------------------------------
    
    def retrieve(self, query: str, k: int = 10, use_ked: bool = True,
                 min_score: float = None) -> RetrievalResult:
        """
        执行检索。
        
        Pipeline: Question → KED → Embedding → FAISS Search → Top-k
        
        Args:
            query: 查询文本
            k: 返回 top-k 结果
            use_ked: 是否使用 KED 扩展
            min_score: 最低分数阈值
        
        Returns:
            RetrievalResult 对象
        """
        import time
        start_time = time.time()
        
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")
        
        result = RetrievalResult(query=query, top_k=k)
        
        # Step 1: KED 查询扩展
        if use_ked:
            expanded_query = self.ked.expand_query(query)
            result.query_expanded = expanded_query
        else:
            expanded_query = query
        
        # Step 2: 嵌入查询
        # [FIX] 诊断 + FAIL_SAFE + FALLBACK：
        #   长文本 embedding 失败会导致 query 向量含 NaN，FAISS.search(NaN) 返回空
        #   -> evidence 为空 -> 下游（Organizer/Solver）误判"没命中"。
        #   这里按序降级，绝不静默返回空：
        #     (a) 优先用 KED 扩展后的 query 做 dense；
        #     (b) 若 embedding 非法，改回【原始 query】再走一次 dense（防 KED 长文本是根因）；
        #     (c) 若仍非法，fallback 到【BM25 稀疏】检索（完全不依赖 embedding）。
        #   无论走哪条路，只要检索成功，evidence 就喂给下游；仅当所有路都失败才置 retrieval_failed。
        query_emb = self._encode_query(expanded_query)
        emb_ok = bool(np.all(np.isfinite(query_emb)))
        used_fallback = False
        fallback_detail = ""
        if not emb_ok:
            nan_ratio = float(np.mean(~np.isfinite(query_emb)))
            logger.warning(
                f"[Retriever·ORGANIZER-DIAG] query embedding 含非法值 "
                f"(NaN/Inf 占比={nan_ratio:.2f}, query_len={len(expanded_query)})。"
                f"尝试降级：先回退到原始 query 做 dense。"
            )
            # (b) 回退到【原始 query】再试一次 dense
            try:
                raw_emb = self._encode_query(query)
                if bool(np.all(np.isfinite(raw_emb))):
                    query_emb = raw_emb
                    used_fallback = True
                    fallback_detail = "fallback=raw_query_dense"
                    logger.warning(
                        f"[Retriever·ORGANIZER-DIAG] 降级成功：原始 query 可编码(left {nan_ratio:.2f})，"
                        f"改用它检索。"
                    )
                else:
                    raise ValueError("raw query embedding also NaN/Inf")
            except Exception as _e:
                # (c) fallback 到 BM25 稀疏（无需 embedding）
                sparse = self.bm25_search(query, k=k)
                if sparse:
                    used_fallback = True
                    fallback_detail = "fallback=bm25_sparse"
                    logger.warning(
                        f"[Retriever·ORGANIZER-DIAG] dense 不可用，回退到 BM25 稀疏检索，"
                        f"返回 {len(sparse)} 条。"
                    )
                    for rank, (cid, sc) in enumerate(sparse):
                        rc = self._chunk_to_obj(cid, float(sc), rank + 1)
                        rc.score = 1.0 / (60 + (rank + 1))  # RRF 风格分数，仅供排序
                        result.chunks.append(rc)
                    result.retrieval_failed = False
                    result.retrieval_failed_reason = ""
                    result.timing_ms = (time.time() - start_time) * 1000
                    return result
                # 所有路都失败：留痕返回空
                logger.warning(
                    f"[Retriever·ORGANIZER-DIAG] dense(raw+expanded) 与 BM25 均失败，返回空并留痕。"
                )
                result.retrieval_failed = True
                result.retrieval_failed_reason = "query_embedding_invalid(NaN)+no_fallback"
                result.timing_ms = (time.time() - start_time) * 1000
                return result


        # Step 3: FAISS 搜索
        chunk_ids, scores = self.indexer.search(query_emb, k=k)
        
        # Step 4: 组装结果
        for rank, (cid, score) in enumerate(zip(chunk_ids, scores)):

            if min_score is not None and score < min_score:
                continue
            
            chunk_data = self.chunks.get(cid, {})
            
            rc = RetrievedChunk(
                chunk_id=cid,
                document_id=chunk_data.get("document_id", chunk_data.get("doc_id", "")),
                source=chunk_data.get("source", ""),
                capability=chunk_data.get("capability", ""),
                industry=chunk_data.get("industry", ""),
                content=chunk_data.get("content", ""),
                score=float(score),
                rank=rank + 1,
            )
            result.chunks.append(rc)
        
        result.timing_ms = (time.time() - start_time) * 1000
        
        return result
    
    def retrieve_with_context(self, query: str, k: int = 10,
                              use_ked: bool = True) -> Tuple[RetrievalResult, str]:
        """
        检索并返回拼接后的上下文文本（用于 LLM prompt）。
        
        Args:
            query: 查询文本
            k: top-k
            use_ked: 是否使用 KED
        
        Returns:
            (result, context_text)
        """
        result = self.retrieve(query, k=k, use_ked=use_ked)
        context = result.get_context()
        return result, context
    
    def multi_query_retrieve(self, query: str, k: int = 10,
                             use_ked: bool = True,
                             strategy: str = "fusion") -> RetrievalResult:
        """
        多查询检索：使用 KED 分解问题后分别检索，再融合结果。
        
        Args:
            query: 查询文本
            k: 最终 top-k
            use_ked: 是否使用 KED
            strategy: 融合策略 ("fusion"=加权融合, "union"=并集)
        
        Returns:
            融合后的 RetrievalResult
        """
        import time
        start_time = time.time()
        
        # Step 1: 分解问题
        sub_queries = self.ked.decompose(query)
        
        # 如果只有一个子查询，退化为普通检索
        if len(sub_queries) <= 1:
            return self.retrieve(query, k=k, use_ked=use_ked)
        
        # Step 2: 分别检索
        #   改动(20260808·路径2落地)：把【原始完整 query】的 dense 结果作为权重最高成员
        #   放在 all_results[0]——_fusion_merge 的 weight=len(results)-rank_weight 会给
        #   第 0 个成员最高权重，从而保证原查询的高相关结果不会被低价值子查询挤出 top-n，
        #   消除 multi 相对 single 的负增益（离线模拟 multi_groundfix_sim.json: cov10
        #   24.4%→33.8% 回到并≥single 33.7%）。
        all_results: List[RetrievalResult] = [self.dense_search(query, k=k * 2, use_ked=use_ked)]
        for sq in sub_queries:
            r = self.retrieve(sq, k=k * 2, use_ked=use_ked)  # 检索更多以融合
            all_results.append(r)
        
        # Step 3: 融合
        if strategy == "fusion":
            merged = self._fusion_merge(all_results, k)
        else:
            merged = self._union_merge(all_results, k)
        
        merged.query = query
        merged.query_expanded = f"multi_query({len(sub_queries)}): {' | '.join(sub_queries)}"
        merged.timing_ms = (time.time() - start_time) * 1000
        
        return merged
    
    def _encode_query(self, text: str) -> np.ndarray:
        """编码查询文本"""
        from retrieval.embedder import EmbeddingGenerator
        if self.embedder is None:
            self.embedder = EmbeddingGenerator(model_name=self.embedding_model)
        return self.embedder.encode_query(text)

    # ----------------------------------------------------------
    # 3.2b BM25 稀疏检索 (混合检索)
    # ----------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词 for BM25（jieba 精确模式 + 字母数字正则保留型号/标准号/数值）。

        相比旧实现("中文整段 + 2/3字字符滑窗 n-gram")的两点改进：
          1) 中文按【词】切分（jieba），而非字符 n-gram——旧 n-gram 词项在 BM25Okapi
             IDF 下被高频字符组合稀释，且字符级匹配与 query 词边界错位，是 BM25
             在中文语料上"假性失效"的根因之一。
          2) 保留字母数字 token（gb/t 26467 / simoreg-6ra70 / 380v 等），工业型号/标准号
             恰恰是字符 n-gram 覆盖不到的精确命中信号。

        去重保留（同一文本内重复词项不放大 BM25 词频，避免单个词刷榜）。
        """
        text = text.lower()
        tokens: List[str] = []
        # 1) 字母数字 tokens（型号/标准号/数值参数，如 gb/t 26467-2011, 380v）
        for m in re.finditer(r'[a-z0-9][a-z0-9\-/\.\#\%]*(?<![\\.\-\/])', text):
            tok = m.group(0).strip('.-#/')
            # 过滤纯标点碎串与单字符
            if any(ch.isalnum() for ch in tok) and len(tok) >= 2:
                tokens.append(tok)
        # 2) 中文：jieba 精确模式切词后只保留含汉字的词
        for w in self._get_jieba_tokens(text):
            if any('\u4e00' <= ch <= '\u9fff' for ch in w):
                tokens.append(w)
        # 去重（保持首个出现顺序），避免重复词项放大词频
        seen = set()
        dedup = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        return dedup

    def _get_jieba_tokens(self, text: str) -> List[str]:
        """
        用 jieba 精确模式切中文词；纯字母/数字/型号由 _tokenize 的正则单独处理。
        失败（jieba 不可用）时静默返回空 list，保证 BM25 降级为仅字母数字 token，
        不影响 dense 主检索路径。
        """
        try:
            import jieba
        except Exception:
            return []
        return [w for w in jieba.cut(text) if w and w.strip() and len(w.strip()) >= 2]

    def _ensure_bm25(self):
        """惰性构建 BM25 索引。"""
        if getattr(self, '_bm25_index', None) is not None:
            return
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 未安装，BM25 稀疏检索不可用。pip install rank_bm25")
            self._bm25_index = None
            return
        if getattr(self, '_bm25_ids', None) is None:
            self._bm25_built = False
        if getattr(self, '_bm25_built', False):
            return
        bm25_built = False
        try:
            tokenized = []
            valid_ids: List[str] = []
            for cid, chunk_data in self.chunks.items():
                content = chunk_data.get("content", "")
                if not content:
                    continue
                t = self._tokenize(content)
                if t:
                    tokenized.append(t)
                    valid_ids.append(cid)
            if not tokenized:
                self._bm25_index = None
                self._bm25_ids = []
                self._bm25_built = True
                return
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_ids = valid_ids
            self._bm25_built = True
            logger.info(f"  已构建 BM25 索引 ({len(valid_ids)} 篇chunk)")
        except Exception as e:
            logger.warning(f"BM25 构建失败: {e}")
            self._bm25_index = None
            self._bm25_ids = []
            self._bm25_built = True

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 稀疏检索。

        Returns:
            [(chunk_id, score), ...] 按得分降序
        """
        self._ensure_bm25()
        if self._bm25_index is None or not self._bm25_ids:
            return []
        q = self._tokenize(query)
        if not q:
            return []
        try:
            scores = self._bm25_index.get_scores(q)
            order = np.argsort(scores)[::-1]
            out = []
            for idx in order:
                if scores[idx] <= 0.0:
                    continue
                if idx < len(self._bm25_ids):
                    out.append((self._bm25_ids[idx], float(scores[idx])))
                if len(out) >= k:
                    break
            return out
        except Exception as e:
            logger.warning(f"BM25 检索失败: {e}")
            return []

    def _chunk_to_obj(self, cid: str, score: float, rank: int) -> RetrievedChunk:
        """根据 chunk_id 和分数构造 RetrievedChunk 对象"""
        chunk_data = self.chunks.get(cid, {})
        return RetrievedChunk(
            chunk_id=cid,
            document_id=chunk_data.get("document_id", chunk_data.get("doc_id", "")),
            source=chunk_data.get("source", ""),
            capability=chunk_data.get("capability", ""),
            industry=chunk_data.get("industry", ""),
            content=chunk_data.get("content", ""),
            score=float(score),
            rank=rank,
        )

    def dense_search(self, query: str, k: int = 10, use_ked: bool = True) -> RetrievalResult:
        """纯 Dense(FAISS) 检索，返回 RetrievalResult。"""
        import time
        st = time.time()
        exp = self.ked.expand_query(query) if use_ked else query
        qemb = self._encode_query(exp)
        # 与 retrieve() 一致：embedding 非法时先回退到原始 query，再回退 BM25，绝不静默返回空
        if not bool(np.all(np.isfinite(qemb))):
            logger.warning(
                f"[Retriever·ORGANIZER-DIAG] dense_search 扩展 query embedding 含非法值 "
                f"(query_len={len(exp)})。先回退到原始 query 重试 dense。"
            )
            try:
                raw = self._encode_query(query)
                if bool(np.all(np.isfinite(raw))):
                    qemb = raw
                    logger.warning("[Retriever·ORGANIZER-DIAG] dense_search 降级成功：用原始 query 检索。")
                else:
                    raise ValueError("raw embedding NaN/Inf")
            except Exception:
                sparse = self.bm25_search(query, k=k)
                if sparse:
                    logger.warning(
                        f"[Retriever·ORGANIZER-DIAG] dense_search dense 不可用，回退 BM25 稀疏检索，"
                        f"返回 {len(sparse)} 条。"
                    )
                    res = RetrievalResult(query=query, query_expanded=exp, top_k=k)
                    for rank, (cid, sc) in enumerate(sparse):
                        rc = self._chunk_to_obj(cid, float(sc), rank + 1)
                        rc.score = 1.0 / (60 + (rank + 1))
                        res.chunks.append(rc)
                    res.timing_ms = (time.time() - st) * 1000
                    return res
                logger.warning("[Retriever·ORGANIZER-DIAG] dense_search dense+BM25 均失败，返回空并留痕。")
                res = RetrievalResult(query=query, query_expanded=exp, top_k=k)
                res.retrieval_failed = True
                res.retrieval_failed_reason = "query_embedding_invalid(NaN)+no_fallback"
                res.timing_ms = (time.time() - st) * 1000
                return res
        cids, scores = self.indexer.search(qemb, k=k)
        res = RetrievalResult(query=query, query_expanded=exp, top_k=k)


        for rank, (cid, sc) in enumerate(zip(cids, scores)):
            res.chunks.append(self._chunk_to_obj(cid, float(sc), rank + 1))
        res.timing_ms = (time.time() - st) * 1000
        return res

    def hybrid_retrieve(self, query: str, k: int = 10, use_ked: bool = True,
                        use_multi_query: bool = True,
                        use_sparse: bool = True,
                        sparse_pool: int = 50,
                        dense_weight: float = 1.0,
                        sparse_weight: float = 0.2) -> RetrievalResult:
        """
        混合检索 (Multi-Query Dense + BM25 Sparse → RRF Fusion)。

        Pipeline:
            Query
              ├─(可选 multi_query) KED decompose → 每个子查询 dense
              └─(可选 sparse) BM25 稀疏检索
                     ↓ RRF 融合
            Top-k

        Args:
            query: 查询
            k: 最终 top-k
            use_ked: 是否使用 KED 扩展
            use_multi_query: 是否用多查询拆分 + 融合 dense 子结果
            use_sparse: 是否融合 BM25 稀疏结果
            sparse_pool: 稀疏阶段候选池大小 (融合前先截断)
            dense_weight / sparse_weight: RRF 权重系数
        """
        import time
        st = time.time()
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")

        # ── 1) Dense 检索 ──
        #   改动(20260808·路径2落地)：无论 use_multi_query 与否，dense 侧【始终保底原始
        #   完整 query】并给其更高 RRF 权重（dense_weight*1.2），避免被低价值子查询稀释，
        #   与离线模拟 multi_groundfix_sim.json(原query权重2:子查询1, cov10 回 33.8%)一致。
        sub = [query]
        if use_multi_query:
            dec = self.ked.decompose(query)
            if len(dec) > 1:
                sub = dec
        dense_results = [self.dense_search(query, k=max(k, 20), use_ked=use_ked)]
        dense_weights = [dense_weight * 1.2 if dense_weight != 1.0 else 1.2]
        seen = {query}
        for sq in sub:
            if sq.strip() == query.strip() or sq in seen:
                continue
            seen.add(sq)
            dense_results.append(self.dense_search(sq, k=max(k, 20), use_ked=use_ked))
            dense_weights.append(dense_weight if dense_weight != 1.0 else 1.0)
        # RRF 融合多个 dense 子结果
        pooled_dense = self._rrf_merge_lists(
            [r.chunks for r in dense_results], k=max(k, 20),
            weights=dense_weights,
        )

        # ── 2) BM25 稀疏检索 ──
        if use_sparse and BM25_AVAILABLE:
            sparse = self.bm25_search(query, k=max(sparse_pool, k))
            sparse_pool_ranked = []
            for rank, (cid, sc) in enumerate(sparse):
                sparse_pool_ranked.append((cid, 1.0 / (60 + (rank + 1))))
        else:
            sparse_pool_ranked = []

        # ── 3) RRF 融合 Dense + Sparse ──
        # pooled_dense: list of RetrievedChunk (已带 rank)
        dense_rrf = {c.chunk_id: dense_weight / (60 + c.rank) for c in pooled_dense}
        for cid, sc in sparse_pool_ranked:
            dense_rrf[cid] = dense_rrf.get(cid, 0.0) + sparse_weight * sc

        # 为每个 chunk_id 取对象
        ranking = sorted(dense_rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = RetrievalResult(query=query, top_k=k)
        merged.query_expanded = f"hybrid(multi={use_multi_query},sparse={bool(use_sparse and BM25_AVAILABLE)})"
        for i, (cid, rrf) in enumerate(ranking):
            merged.chunks.append(self._chunk_to_obj(cid, rrf, i + 1))
        merged.timing_ms = (time.time() - st) * 1000
        return merged

    @staticmethod
    def _rrf_merge_lists(chunk_lists: List[List['RetrievedChunk']],
                         k: int, weights: Optional[List[float]] = None) -> List['RetrievedChunk']:
        """将多个 (ranked) chunk 列表做 RRF 融合，返回按分数降序的 top-k 列表。"""
        if weights is None:
            weights = [1.0] * len(chunk_lists)
        acc: Dict[str, float] = {}
        obj: Dict[str, RetrievedChunk] = {}
        for w, clist in zip(weights, chunk_lists):
            for c in clist:
                acc[c.chunk_id] = acc.get(c.chunk_id, 0.0) + w / (60 + c.rank)
                if c.chunk_id not in obj:
                    obj[c.chunk_id] = c
        order = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for i, (cid, s) in enumerate(order):
            c = obj[cid]
            c.score = s
            c.rank = i + 1
            out.append(c)
        return out

    def two_stage_retrieve(self, query: str, k: int = 10, pool_k: int = 60,
                           ranker: "RelevanceRanker" = None, n_anchor: int = 3,
                           backfill_th: float = 0.02, backfill_max: int = 50,
                           w_dense: float = 1.0, w_rel: float = 1.0,
                           w_anchor: float = 0.5) -> "RetrievalResult":
        """run2 两步式检索（EF 改进版，两步法）。

        第一步（初步检索）：现有 dense 检索取宽候选池 pool（`pool_k`）。
        第二步（AI 相关性评估 + 文档级反向补全）：
          A. `ranker.score(query, pool.chunks)` 仅凭【问题+候选块】给相关/适配分（不碰 GT/答案）；
          B. 取相关性最高的 `n_anchor` 个锚块 → 用其 `document_id` 反查**同一源文档**下其余块
             （自 `self.chunks` 按 document_id 归组），并用 `chinese_jt(query, 块) >= backfill_th`
             过滤（沿用 EF 口径避免灌噪音）；
          C. 融合排序输出 top-k：`score = dense_RRF + w_rel*rel + w_anchor*(锚/回捞bonus)`。

        Args:
            ranker: 二阶段相关性评估器（必须注入；见 RelevanceRanker）。None 时退化为纯 dense_search。
            pool_k: 第一步候选池宽度（越大 LLM 打分越多，成本越高）。
            n_anchor: 取前多少个高相关块作为文档级回捞锚。
            backfill_th: 回捞块与 query 的词法 jt(IoU) 阈值，低于则丢弃（防噪声）。
            backfill_max: 回捞块数上限（防超大文档刷屏）。
            w_dense / w_rel / w_anchor: 融合权重。

        Returns:
            标准 RetrievalResult（`.chunks` 为 RetrievedChunk），与 single/multi/hybrid 同构。
        """
        import time
        st = time.time()
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")

        # ---- 第一步：初步检索（现有模型，宽候选池） ----
        pool = self.dense_search(query, k=pool_k, use_ked=True)
        pool_chunks = list(pool.chunks)
        if not pool_chunks:
            return pool

        # ---- 第二步.A：AI 相关性评估（仅问题+候选块，绝不引入 GT/答案） ----
        if ranker is None:
            ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        else:
            try:
                ranked = ranker.score(query, pool_chunks) or {}
            except Exception as _e:
                logger.warning(f"[run2] ranker.score 失败: {_e!r}；退化为纯初检 top-k")
                ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}

        if ranked and any(isinstance(v, (int, float)) for v in ranked.values()):
            mx = max((v for v in ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
            ranked = {cid: (float(v) / mx) for cid, v in ranked.items()}

        # ---- 第二步.B：取高相关锚块 → 文档级反向补全 ----
        by_rel = sorted(pool_chunks, key=lambda c: ranked.get(c.chunk_id, 0.0), reverse=True)
        anchors = by_rel[:n_anchor]
        anchor_doc_ids = {c.document_id for c in anchors if c.document_id}
        pool_ids = {c.chunk_id for c in pool_chunks}
        backfilled = set()
        num_backfilled = 0
        if anchor_doc_ids:
            for cid, chunk_data in self.chunks.items():
                doc = chunk_data.get("document_id") or chunk_data.get("doc_id") or ""
                if doc not in anchor_doc_ids:
                    continue
                content = chunk_data.get("content", "")
                if not content or cid in pool_ids:
                    continue
                if backfill_th > 0 and chinese_jt(query, content) < backfill_th:
                    continue
                backfilled.add(cid)
                num_backfilled += 1
                if num_backfilled >= backfill_max:
                    break

        # ---- 第二步.C：融合排序 top-k ----
        s: Dict[str, float] = {}
        obj: Dict[str, RetrievedChunk] = {}
        for c in pool_chunks:
            rrf = w_dense / (60 + c.rank)
            rel = ranked.get(c.chunk_id, 0.0) * w_rel
            anchor_bonus = w_anchor if (c.document_id in anchor_doc_ids) else 0.0
            s[c.chunk_id] = rrf + rel + anchor_bonus
            obj[c.chunk_id] = c
        for cid in backfilled:
            rrf = w_dense / (60 + pool_k)
            rel = ranked.get(cid, 0.0) * w_rel
            s[cid] = rrf + rel + w_anchor
            obj[cid] = self._chunk_to_obj(cid, 0.0, 1)

        order = sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = RetrievalResult(query=query, top_k=k)
        merged.query_expanded = f"run2(pool={pool_k},anchor={n_anchor},backfill={len(backfilled)})"
        for i, (cid, sc) in enumerate(order):
            c = obj[cid]
            c.score = sc
            c.rank = i + 1
            merged.chunks.append(c)
        merged.timing_ms = (time.time() - st) * 1000
        return merged

    def two_stage_retrieve_v2(self, query: str, k: int = 10, pool_k: int = 60,
                              ranker: "RelevanceRanker" = None, n_anchor: int = 3,
                              backfill_th: float = 0.02, backfill_max: int = 50,
                              w_dense: float = 1.0, w_rel: float = 1.0,
                              w_anchor: float = 0.5) -> "RetrievalResult":
        """run2 两步式检索 v2：修复 v1 中「回捞块 rel 恒为 0」缺陷的方案 B。

        与 v1 (`two_stage_retrieve`) 的差异只在执行顺序与打分对象：
          v1: 第一步捞池 → A 只给【池块】打分(ranked) → B 回捞池外块 → C 融合（回捞块
              `ranked.get(cid)` 必然缺失 → rel 恒 0 → 回捞块进不了 top-k）。
          v2: 第一步捞池 → B 先回捞池外块 → 把【池块 + 回捞块】合并为一组候选，
              统一 `ranker.score(query, 全体候选)` 给每个候选（含回捞块）真实 rel
              → C 融合（回捞块也带 rel 参与竞争）。

        代价：ranker 需打分 50+N 块（N=回捞数），成本略升；换来回捞块可被有效排序。

        返回与 v1/v1-single 等完全同构的 RetrievalResult（`.chunks` 为 RetrievedChunk）。
        """
        import time
        st = time.time()
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")

        # ---- 第一步：初步检索（现有模型，宽候选池） ----
        pool = self.dense_search(query, k=pool_k, use_ked=True)
        pool_chunks = list(pool.chunks)
        if not pool_chunks:
            return pool

        # ---- 第二步.B（提前）：先确定锚文档并从池外反向回捞（与 v1 同逻辑） ----
        pool_ids = {c.chunk_id for c in pool_chunks}
        # 锚判定需要 rel；v2 用【池块】先出一版 rel 选锚（回捞对象定位用）
        pre_ranked = {}
        if ranker is None:
            pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        else:
            try:
                pre_ranked = ranker.score(query, pool_chunks) or {}
            except Exception as _e:
                logger.warning(f"[run2v2] 池块预打分失败: {_e!r}；退化为纯初检序选锚")
                pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        pmx = max((v for v in pre_ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
        pre_ranked = {cid: (float(v) / pmx) for cid, v in pre_ranked.items()}

        by_rel = sorted(pool_chunks, key=lambda c: pre_ranked.get(c.chunk_id, 0.0), reverse=True)
        anchors = by_rel[:n_anchor]
        anchor_doc_ids = {c.document_id for c in anchors if c.document_id}
        backfilled = set()
        if anchor_doc_ids:
            for cid, chunk_data in self.chunks.items():
                doc = chunk_data.get("document_id") or chunk_data.get("doc_id") or ""
                if doc not in anchor_doc_ids:
                    continue
                content = chunk_data.get("content", "")
                if not content or cid in pool_ids:
                    continue
                if backfill_th > 0 and chinese_jt(query, content) < backfill_th:
                    continue
                backfilled.add(cid)
                if len(backfilled) >= backfill_max:
                    break

        # ---- 合并候选：池块 + 回捞块（回捞块转成 RetrievedChunk 以便 ranker 打分） ----
        cand = list(pool_chunks)
        for cid in backfilled:
            cand.append(self._chunk_to_obj(cid, 0.0, 1))

        # ---- 第二步.A：统一对【全体候选】打分（含回捞块 → rel 不再为 0） ----
        if ranker is None:
            ranked = {c.chunk_id: (len(cand) - i) for i, c in enumerate(cand)}
        else:
            try:
                ranked = ranker.score(query, cand) or {}
            except Exception as _e:
                logger.warning(f"[run2v2] 全体候选打分失败: {_e!r}；退化为纯初检序")
                ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        if ranked and any(isinstance(v, (int, float)) for v in ranked.values()):
            mx = max((v for v in ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
            ranked = {cid: (float(v) / mx) for cid, v in ranked.items()}

        # ---- 第二步.C：融合排序 top-k ----
        # 回捞块此时在 ranked 里有真实 rel；同时保留锚 bonus + 池内块的 dense RRF。
        s: Dict[str, float] = {}
        obj: Dict[str, RetrievedChunk] = {}
        for c in cand:
            is_backfill = c.chunk_id in backfilled
            rrf = (w_dense / (60 + pool_k)) if is_backfill else (w_dense / (60 + c.rank))
            rel = ranked.get(c.chunk_id, 0.0) * w_rel
            bonus = w_anchor if (c.document_id in anchor_doc_ids) else 0.0
            # 回捞块天然属于锚文档(bonus=wanchor)，与 v1 一致；池块按其是否锚文档计。
            s[c.chunk_id] = rrf + rel + bonus
            obj[c.chunk_id] = c

        order = sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = RetrievalResult(query=query, top_k=k)
        merged.query_expanded = (
            f"run2v2(pool={pool_k},anchor={n_anchor},backfill={len(backfilled)},"
            f"cand={len(cand)})"
        )
        n_bf_in = 0
        for i, (cid, sc) in enumerate(order):
            c = obj[cid]
            c.score = sc
            c.rank = i + 1
            merged.chunks.append(c)
            if cid in backfilled:
                n_bf_in += 1
        merged.query_expanded = (
            f"run2v2(pool={pool_k},anchor={n_anchor},backfill={len(backfilled)},"
            f"bf_in_topk={n_bf_in})"
        )
        merged.timing_ms = (time.time() - st) * 1000
        return merged

    def two_stage_retrieve_v3(self, query: str, k: int = 10, pool_k: int = 60,
                              ranker: "RelevanceRanker" = None, n_anchor: int = 3,
                              backfill_th: float = 0.02, backfill_max: int = 50,
                              w_dense: float = 1.0, w_rel: float = 1.0,
                              w_anchor: float = 0.5) -> "RetrievalResult":
        """run2 三步检索 v3：锚文档【按 document_id 去重取 n_anchor 篇】扩大回捞范围。

        v2 的缺陷：`by_rel[:3]` 取 top-3 块作锚，但 dense 天然把同源文档多块排前面，
        top-3 块常来自**同一篇文档** → 真正追溯的文档只有 1~2 篇，回捞范围过窄、
        押注单文档，错则全错（实证 id=184 锚文档仅 1 篇）。

        v3 改动（仅"选锚"一步）：按 rel 降序遍历池块，**按 document_id 去重**收集前
        `n_anchor` 篇不同文档作为回捞范围；其余（回捞、全体候选统一打分、融合 top-k）
        与 v2 完全一致。

        参数 n_anchor 语义 = 追溯的【文档篇数】（非块数）。
        """
        import time
        st = time.time()
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")

        pool = self.dense_search(query, k=pool_k, use_ked=True)
        pool_chunks = list(pool.chunks)
        if not pool_chunks:
            return pool

        pool_ids = {c.chunk_id for c in pool_chunks}
        pre_ranked = {}
        if ranker is None:
            pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        else:
            try:
                pre_ranked = ranker.score(query, pool_chunks) or {}
            except Exception as _e:
                logger.warning(f"[run2v3] 池块预打分失败: {_e!r}；退化为纯初检序选锚")
                pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        pmx = max((v for v in pre_ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
        pre_ranked = {cid: (float(v) / pmx) for cid, v in pre_ranked.items()}

        # ---- 关键差异：锚文档按 document_id 去重取前 n_anchor 篇 ----
        by_rel = sorted(pool_chunks, key=lambda c: pre_ranked.get(c.chunk_id, 0.0), reverse=True)
        anchor_doc_ids = set()
        for c in by_rel:
            d = c.document_id
            if d:
                anchor_doc_ids.add(d)
            if len(anchor_doc_ids) >= n_anchor:
                break
        del pre_ranked, by_rel

        backfilled = set()
        if anchor_doc_ids:
            for cid, chunk_data in self.chunks.items():
                doc = chunk_data.get("document_id") or chunk_data.get("doc_id") or ""
                if doc not in anchor_doc_ids:
                    continue
                content = chunk_data.get("content", "")
                if not content or cid in pool_ids:
                    continue
                if backfill_th > 0 and chinese_jt(query, content) < backfill_th:
                    continue
                backfilled.add(cid)
                if len(backfilled) >= backfill_max:
                    break

        cand = list(pool_chunks)
        for cid in backfilled:
            cand.append(self._chunk_to_obj(cid, 0.0, 1))

        if ranker is None:
            ranked = {c.chunk_id: (len(cand) - i) for i, c in enumerate(cand)}
        else:
            try:
                ranked = ranker.score(query, cand) or {}
            except Exception as _e:
                logger.warning(f"[run2v3] 全体候选打分失败: {_e!r}；退化为纯初检序")
                ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        if ranked and any(isinstance(v, (int, float)) for v in ranked.values()):
            mx = max((v for v in ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
            ranked = {cid: (float(v) / mx) for cid, v in ranked.items()}

        s: Dict[str, float] = {}
        obj: Dict[str, RetrievedChunk] = {}
        for c in cand:
            is_backfill = c.chunk_id in backfilled
            rrf = (w_dense / (60 + pool_k)) if is_backfill else (w_dense / (60 + c.rank))
            rel = ranked.get(c.chunk_id, 0.0) * w_rel
            bonus = w_anchor if (c.document_id in anchor_doc_ids) else 0.0
            s[c.chunk_id] = rrf + rel + bonus
            obj[c.chunk_id] = c

        order = sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = RetrievalResult(query=query, top_k=k)
        n_bf_in = 0
        for i, (cid, sc) in enumerate(order):
            c = obj[cid]
            c.score = sc
            c.rank = i + 1
            merged.chunks.append(c)
            if cid in backfilled:
                n_bf_in += 1
        merged.query_expanded = (
            f"run2v3(pool={pool_k},anchor_docs={len(anchor_doc_ids)},"
            f"backfill={len(backfilled)},bf_in_topk={n_bf_in})"
        )
        merged.timing_ms = (time.time() - st) * 1000
        return merged

    def two_stage_retrieve_v4(self, query: str, k: int = 10, pool_k: int = 60,
                              ranker: "RelevanceRanker" = None, n_anchor: int = 3,
                              backfill_th: float = 0.02, backfill_max: int = 50,
                              w_dense: float = 1.0, w_rel: float = 1.0,
                              w_anchor: float = 0.5) -> "RetrievalResult":
        """run2 检索 v4：去重到 n_anchor 篇文档回捞，但 anchor_bonus 只给第 1 篇。

        针对 v3 的负优化（21 题 LLM: cov@10 −1.8pp / good% −4.8pp）打补丁：
          v3 把 3 篇文档的【全体块】都加 anchor_bonus(+0.5) → 低相关文档靠 bonus 硬挤
          top10、稀释真正相关块 → 主口径大跌。
          v4 把「回捞范围」与「排序加权」解耦：
            - 范围：仍按 document_id 去重取前 n_anchor 篇回捞（广撒网，保补全面）；
            - 权重：只有 rel 最高【第 1 篇】文档的块才吃 anchor_bonus；第 2/3 篇的块
              只靠真实 rel 分与池块公平竞争（择优录取，不稀释 top10）。
          形态 = recall 阶段广、precision 阶段严（hierarchical）。

        参数 n_anchor 语义 = 追溯的【文档篇数】（非块数）。
        """
        import time
        st = time.time()
        if self.indexer is None:
            raise RuntimeError("索引未加载。请先调用 load() 或 load_from_manifest()")

        pool = self.dense_search(query, k=pool_k, use_ked=True)
        pool_chunks = list(pool.chunks)
        if not pool_chunks:
            return pool

        pool_ids = {c.chunk_id for c in pool_chunks}
        pre_ranked = {}
        if ranker is None:
            pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        else:
            try:
                pre_ranked = ranker.score(query, pool_chunks) or {}
            except Exception as _e:
                logger.warning(f"[run2v4] 池块预打分失败: {_e!r}；退化为纯初检序选锚")
                pre_ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        pmx = max((v for v in pre_ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
        pre_ranked = {cid: (float(v) / pmx) for cid, v in pre_ranked.items()}

        # ---- 选锚：按 document_id 去重取前 n_anchor 篇，且记录第 1 篇(最相关) ----
        by_rel = sorted(pool_chunks, key=lambda c: pre_ranked.get(c.chunk_id, 0.0), reverse=True)
        ordered_docs: List[str] = []
        seen = set()
        for c in by_rel:
            d = c.document_id
            if d and d not in seen:
                seen.add(d)
                ordered_docs.append(d)
            if len(ordered_docs) >= n_anchor:
                break
        anchor_doc_ids = set(ordered_docs)
        top_doc = ordered_docs[0] if ordered_docs else None
        del pre_ranked, by_rel, seen

        # ---- 回捞：锚文档(3篇)的池外块，jt>=th 过滤 ----
        backfilled = set()
        if anchor_doc_ids:
            for cid, chunk_data in self.chunks.items():
                doc = chunk_data.get("document_id") or chunk_data.get("doc_id") or ""
                if doc not in anchor_doc_ids:
                    continue
                content = chunk_data.get("content", "")
                if not content or cid in pool_ids:
                    continue
                if backfill_th > 0 and chinese_jt(query, content) < backfill_th:
                    continue
                backfilled.add(cid)
                if len(backfilled) >= backfill_max:
                    break

        cand = list(pool_chunks)
        for cid in backfilled:
            cand.append(self._chunk_to_obj(cid, 0.0, 1))

        if ranker is None:
            ranked = {c.chunk_id: (len(cand) - i) for i, c in enumerate(cand)}
        else:
            try:
                ranked = ranker.score(query, cand) or {}
            except Exception as _e:
                logger.warning(f"[run2v4] 全体候选打分失败: {_e!r}；退化为纯初检序")
                ranked = {c.chunk_id: (pool_k - i) for i, c in enumerate(pool_chunks)}
        if ranked and any(isinstance(v, (int, float)) for v in ranked.values()):
            mx = max((v for v in ranked.values() if isinstance(v, (int, float))), default=1.0) or 1.0
            ranked = {cid: (float(v) / mx) for cid, v in ranked.items()}

        # ---- 融合（v4 关键：bonus 只给 top_doc 的块） ----
        s: Dict[str, float] = {}
        obj: Dict[str, RetrievedChunk] = {}
        for c in cand:
            is_backfill = c.chunk_id in backfilled
            rrf = (w_dense / (60 + pool_k)) if is_backfill else (w_dense / (60 + c.rank))
            rel = ranked.get(c.chunk_id, 0.0) * w_rel
            bonus = w_anchor if (top_doc and c.document_id == top_doc) else 0.0
            s[c.chunk_id] = rrf + rel + bonus
            obj[c.chunk_id] = c

        order = sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = RetrievalResult(query=query, top_k=k)
        n_bf_in = 0
        for i, (cid, sc) in enumerate(order):
            c = obj[cid]
            c.score = sc
            c.rank = i + 1
            merged.chunks.append(c)
            if cid in backfilled:
                n_bf_in += 1
        merged.query_expanded = (
            f"run2v4(pool={pool_k},anchor_docs={len(anchor_doc_ids)},"
            f"top_doc_only_bonus=1,backfill={len(backfilled)},bf_in_topk={n_bf_in})"
        )
        merged.timing_ms = (time.time() - st) * 1000
        return merged

    @staticmethod
    def _fusion_merge(results: List[RetrievalResult], k: int) -> "RetrievalResult":

        """加权融合合并 (基于 Reciprocal Rank Fusion)"""
        rrf_score: Dict[str, Tuple[float, RetrievedChunk]] = {}
        
        for rank_weight, r in enumerate(results):
            weight = len(results) - rank_weight  # 越早的查询权重越高
            for chunk in r.chunks:
                if chunk.chunk_id not in rrf_score:
                    rrf_score[chunk.chunk_id] = [0.0, chunk]
                # RRF 分数
                rrf_score[chunk.chunk_id][0] += weight / (60 + chunk.rank)
        
        # 按 RRF 分数排序
        sorted_chunks = sorted(
            rrf_score.values(),
            key=lambda x: x[0],
            reverse=True
        )[:k]
        
        merged = RetrievalResult(query="", top_k=k)
        for i, (score, chunk) in enumerate(sorted_chunks):
            chunk.score = score
            chunk.rank = i + 1
            merged.chunks.append(chunk)
        
        return merged
    
    @staticmethod
    def _union_merge(results: List[RetrievalResult], k: int) -> RetrievalResult:
        """并集合并 (去重后按最高分排序)"""
        seen: Dict[str, RetrievedChunk] = {}
        
        for r in results:
            for chunk in r.chunks:
                if chunk.chunk_id not in seen:
                    seen[chunk.chunk_id] = chunk
                else:
                    # 保留最高分
                    if chunk.score > seen[chunk.chunk_id].score:
                        seen[chunk.chunk_id] = chunk
        
        sorted_chunks = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:k]
        
        merged = RetrievalResult(query="", top_k=k)
        for i, chunk in enumerate(sorted_chunks):
            chunk.rank = i + 1
            merged.chunks.append(chunk)
        
        return merged
    
    # ----------------------------------------------------------
    # 3.3 证据聚合
    # ----------------------------------------------------------
    
    def aggregate_evidence(self, result: RetrievalResult,
                           strategy: str = "dedup_by_doc") -> RetrievalResult:
        """
        证据聚合：对检索结果进行后处理。
        
        Args:
            result: 原始检索结果
            strategy: 聚合策略
                - "dedup_by_doc": 同一文档的块合并，取最高分
                - "dedup_by_content": 内容高度相似的块去重
                - "rerank_by_industry": 按行业相关性重排序
        
        Returns:
            聚合后的结果
        """
        if strategy == "dedup_by_doc":
            return self._dedup_by_document(result)
        elif strategy == "dedup_by_content":
            return self._dedup_by_content(result)
        elif strategy == "rerank_by_industry":
            return result  # 需要 query 的行业信息
        else:
            return result
    
    @staticmethod
    def _dedup_by_document(result: RetrievalResult) -> RetrievalResult:
        """同一文档的多块合并（保留最高分）"""
        seen_docs: Dict[str, RetrievedChunk] = {}
        
        for chunk in result.chunks:
            doc_key = chunk.document_id or chunk.chunk_id
            if doc_key not in seen_docs:
                seen_docs[doc_key] = chunk
            elif chunk.score > seen_docs[doc_key].score:
                seen_docs[doc_key] = chunk
        
        sorted_chunks = sorted(seen_docs.values(), key=lambda x: x.score, reverse=True)
        
        aggregated = RetrievalResult(
            query=result.query,
            query_expanded=result.query_expanded,
            top_k=result.top_k,
        )
        for i, chunk in enumerate(sorted_chunks):
            chunk.rank = i + 1
            aggregated.chunks.append(chunk)
        
        return aggregated
    
    @staticmethod
    def _dedup_by_content(result: RetrievalResult, threshold: float = 0.85) -> RetrievalResult:
        """基于内容相似度的去重（用 Jaccard 词重叠）"""
        unique: List[RetrievedChunk] = []
        
        def jaccard_similarity(a: str, b: str) -> float:
            set_a = set(a[:200])
            set_b = set(b[:200])
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)
        
        for chunk in result.chunks:
            is_dup = False
            for existing in unique:
                if jaccard_similarity(chunk.content, existing.content) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(chunk)
        
        aggregated = RetrievalResult(
            query=result.query,
            query_expanded=result.query_expanded,
            top_k=result.top_k,
        )
        for i, chunk in enumerate(unique):
            chunk.rank = i + 1
            aggregated.chunks.append(chunk)
        
        return aggregated
    
    # ----------------------------------------------------------
    # 3.4 Recall@k 评估
    # ----------------------------------------------------------
    
    def evaluate_recall(self, query: str, relevant_ids: Set[str],
                        k_values: List[int] = None) -> Dict[str, float]:
        """
        评估 Recall@k。
        
        Args:
            query: 查询文本
            relevant_ids: 相关文档的 document_id 集合 (ground truth)
            k_values: 要评估的 k 值列表, 默认 [1, 3, 5, 10, 20]
        
        Returns:
            recall_results: { "recall@1": 0.5, "recall@3": 0.8, ... }
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        result = self.retrieve(query, k=max(k_values))
        
        retrieved_ids = set()
        for c in result.chunks:
            if c.document_id:
                retrieved_ids.add(c.document_id)
        
        if not relevant_ids:
            return {f"recall@{k}": 0.0 for k in k_values}
        
        total_relevant = len(relevant_ids)
        recall_results = {}
        
        for k in k_values:
            # 取前 k 个 chunk，看命中了多少相关文档
            top_k_ids = set()
            for c in result.chunks[:k]:
                if c.document_id:
                    top_k_ids.add(c.document_id)
            
            hits = len(top_k_ids & relevant_ids)
            recall_results[f"recall@{k}"] = hits / total_relevant if total_relevant > 0 else 0.0
        
        return recall_results
    
    def batch_evaluate_recall(self, queries_and_relevant: List[Tuple[str, Set[str]]],
                              k_values: List[int] = None) -> Dict[str, float]:
        """
        批量评估 Recall@k。
        
        Args:
            queries_and_relevant: [(query, relevant_ids), ...]
            k_values: 要评估的 k 值
        
        Returns:
            avg_recall: { "recall@k": avg_score }
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        accum = {f"recall@{k}": [] for k in k_values}
        
        for query, relevant_ids in queries_and_relevant:
            recalls = self.evaluate_recall(query, relevant_ids, k_values)
            for k_str, v in recalls.items():
                accum[k_str].append(v)
        
        avg_results = {}
        for k_str, values in accum.items():
            avg_results[k_str] = sum(values) / len(values) if values else 0.0
        
        return avg_results


# ============================================================
# 4. 带 Generator 的端到端 RAG
# ============================================================

class RAGPipeline:
    """
    端到端 RAG Pipeline。
    
    Pipeline:
        Question → Retriever (KED + Search + Aggregation)
                 → Context Construction
                 → Generator (LLM)
                 → Final Answer
    """
    
    def __init__(self, retriever: OpenDomainRetriever = None,
                 llm_name: str = "deepseek-chat",
                 api_key: str = None):
        self.retriever = retriever or OpenDomainRetriever()
        self.llm_name = llm_name
        self.api_key = api_key
        self.llm = None  # 延迟加载
    
    def answer(self, question: str, k: int = 10, use_ked: bool = True,
               system_prompt: str = None) -> Dict:
        """
        回答一个问题 (Retrieve + Generate)。
        
        Args:
            question: 问题
            k: 检索 top-k
            use_ked: 是否使用 KED
            system_prompt: 自定义 system prompt
        
        Returns:
            {"question": ..., "context": ..., "answer": ..., "sources": [...]}
        """
        # Step 1: 检索
        result, context = self.retriever.retrieve_with_context(
            question, k=k, use_ked=use_ked
        )
        
        # Step 2: 构建 prompt
        if system_prompt is None:
            system_prompt = (
                "你是一个工业领域知识问答助手。请根据以下检索到的参考资料回答问题。\n"
                "在回答中标注引用编号如 [1], [2] 以标明信息来源。\n"
                "如果参考资料不足以回答问题，请如实说明。"
            )
        
        prompt = f"{system_prompt}\n\n"
        prompt += f"**检索到的参考资料**:\n{context}\n\n"
        prompt += f"**问题**: {question}\n**回答**:"
        
        # Step 3: 生成回答
        answer_text = self._generate(prompt)
        
        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "answer": answer_text,
            "sources": [
                {"chunk_id": c.chunk_id, "industry": c.industry,
                 "capability": c.capability, "score": c.score}
                for c in result.chunks[:5]
            ],
            "num_sources": len(result.chunks),
        }
    
    def _generate(self, prompt: str) -> str:
        """调用 LLM 生成回答"""
        # 简版直接返回 prompt (实际使用时需要替换为真正的 LLM 调用)
        # 用户可根据需要接入 LINS-main 或其他 LLM
        logger.info(f"Generator 准备就绪 (LLM={self.llm_name})")
        logger.info(f"Prompt 长度: {len(prompt)} 字符")
        return f"[使用 {self.llm_name} 生成回答]"


# ============================================================
# 5. 快速启动函数
# ============================================================

def quick_retrieve(query: str, k: int = 5, use_ked: bool = True,
                   manifest_path: str = None) -> RetrievalResult:
    """
    快速检索: 一行命令完成开放域检索。
    
    Args:
        query: 查询文本
        k: top-k
        use_ked: 是否使用 KED
        manifest_path: manifest.json 路径
    
    Returns:
        RetrievalResult
    """
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest(manifest_path)
    return retriever.retrieve(query, k=k, use_ked=use_ked)


def demo():
    """演示开放域检索"""
    print("=" * 70)
    print("🔍 LINS-Industrial 开放域检索 Demo")
    print("  从全量工业知识库检索 top-k 相关文本块")
    print("=" * 70)
    
    retriever = OpenDomainRetriever()
    try:
        retriever.load_from_manifest()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("请先运行 knowledge_builder 构建知识库")
        return
    
    queries = [
        "冶金轧机过载保护",
        "SIMOREG DC Master 6RA70 并联运行需要什么配置？",
        "轴承座的游隙如何调整？有哪些注意事项？",
        "PLC 控制系统的安全要求",
        "不锈钢焊接工艺参数选择",
    ]
    
    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"❓ 查询: {query}")
        
        # 带 KED
        result = retriever.retrieve(query, k=3, use_ked=True)
        
        print(f"  扩展查询: {result.query_expanded}")
        print(f"  耗时: {result.timing_ms:.1f}ms")
        print(f"  Top-3 结果:")
        
        for i, chunk in enumerate(result.chunks):
            print(f"\n    [{i+1}] score={chunk.score:.4f}")
            print(f"        行业: {chunk.industry} | 能力: {chunk.capability}")
            print(f"        来源: {chunk.source}")
            content_preview = chunk.content[:120].replace('\n', ' ')
            print(f"        内容: {content_preview}{'...' if len(chunk.content) > 120 else ''}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = quick_retrieve(query, k=5)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        demo()

