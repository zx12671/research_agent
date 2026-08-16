"""
organizer.py: EvidenceOrganizer — LIGHTWEIGHT version (FIXED)

Key Fix (Problem ②):
  - REMOVED relevance re-scoring (was overwriting FAISS semantic ranking!)
  - REMOVED top-k filtering (was discarding potentially useful docs)
  - Now only does: 1) deduplication (keep content clean) + 2) lightweight grouping
  - FAISS original score/ranking is PRESERVED throughout

Old behavior (BROKEN):
  FAISS returns top-10 → Organizer re-scores by keyword → keeps only 5-8 → drops semantic info
New behavior (FIXED):
  FAISS returns top-10 → Organizer deduplicates → groups without reordering → all docs preserved
"""

import re
import math
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .task_types import OrganizedEvidence, ExecutionDirective, TaskType

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EvidenceOrganizer:
    """
    Lightweight EvidenceOrganizer — NO relevance re-scoring, NO top-k filtering.
    
    Processing pipeline:
        1. Content de-duplication (near-dup removal)
        2. Lightweight grouping (by topic / comparison / symptom / etc.)
        3. No relevance re-scoring → FAISS original ranking PRESERVED
        4. No document dropping → all documents preserved after dedup
    
    Key fix: REMOVED _compute_relevance and top-k filtering.
    These were overwriting FAISS semantic ranking and discarding useful docs.
    
    Usage:
        directive = plan.get_instruction("organization")
        organized = organizer.execute(directive, raw_documents, question=question)
    """

    # ─── Task → max groups mapping ───
    TASK_MAX_GROUPS: Dict[str, int] = {
        "selection": 2,               # only "标准" + "选型"
        "comparison": 1,              # only the compared pair
        "diagnosis": 2,               # "故障/原因" + "方案"
        "procedure": 2,               # "步骤" + "注意事项"
        "calculation": 2,             # "公式" + "参数"
        "standard_interpretation": 2, # "标准" + "说明"
        "explanation": 2,
        "general": 3,
    }

    # ─── Task → preferred group name prefixes ───
    TASK_GROUP_PRIORITY: Dict[str, List[str]] = {
        "selection": ["选型", "标准", "selection", "standard", "specification"],
        "comparison": ["comparison", "比较", "对比"],
        "diagnosis": ["故障", "诊断", "failure", "诊断/原因", "cause", "解决方案", "solution"],
        "procedure": ["步骤", "流程", "procedure", "preparation", "注意事项"],
        "calculation": ["公式", "formula", "参数", "parameter"],
        "standard_interpretation": ["标准", "规范", "standard", "regulation"],
        "explanation": ["原理", "principle", "说明", "explanation"],
        "general": [],
    }

    def __init__(self):
        """Initialize the EvidenceOrganizer v2."""
        self.action_handlers = {
            "group_by_comparison": self._group_by_comparison,
            "group_by_symptom": self._group_by_symptom,
            "group_by_candidate": self._group_by_candidate,
            "group_by_formula": self._group_by_formula,
            "group_chronological": self._group_chronological,
            "group_by_topic": self._group_by_topic,
            "pass_through": self._pass_through,
        }

    def execute(
        self,
        directive: Optional[ExecutionDirective],
        documents: List[Any],
        question: str = "",
        task_type: Optional[str] = None,
        rerank: bool = False,
        rerank_weight_lex: float = 0.7,
        rerank_weight_score: float = 0.3,
        truncate: bool = False,
        max_chunks: Optional[int] = None,
        max_chars: int = 6000,
        signal: str = "lex",
        w_cap: float = 0.3,
    ) -> OrganizedEvidence:
        """
        Execute organization — lightweight version.

        Steps:
            1) Deduplicate documents (near-content removal)
            2) Optionally rerank deduped docs by query-relevance (L2-RERANK)
            3) Optionally truncate low-relevance / long docs (L2-TRUNCATE)
            4) Group deduped docs by task-aware strategy
            5) Limit groups per task type

        CRITICAL: By DEFAULT (rerank=False, truncate=False) there is NO relevance
        re-scoring, NO top-k filtering — FAISS original ranking is preserved and
        all documents kept after dedup (unchanged original behavior).
        `rerank`/`truncate` are OPT-IN for the L2 rerank+truncate experiment; when
        enabled they reorder by query relevance and optionally cut long/irrelevant
        docs to keep only the strongest evidence at the top.

        Args:
            directive: Organization directive from ExecutionGraph
            documents: Raw retrieved documents (preserve FAISS order)
            question: Original question (for grouping hints AND rerank signal)
            task_type: Task type string (e.g., "selection", "diagnosis")
            rerank: If True, reorder docs by combined (lexical+score) query relevance.
            rerank_weight_lex: weight of lexical 2-gram overlap in rerank score.
            rerank_weight_score: weight of FAISS retrieval score in rerank score.
            truncate: If True, drop low-relevance/long docs to fit max_chars/max_chunks.
            max_chunks: keep at most this many docs (None=no doc-count cap).
            max_chars: keep docs until cumulative char length exceeds this (0/None=off).
            signal: rerank relevance signal — "lex"(default, lexical) or "cap"(semantic+capability prior).
            w_cap: weight of capability-alignment term when signal=="cap".

        Returns:
            OrganizedEvidence with deduped + grouped docs (no filtering by default)
        """
        if not documents:
            logger.warning("[Organizer] No documents to organize.")
            return OrganizedEvidence(documents=[], groups={},
                                     original_count=0, organized_count=0,
                                     organization_method="empty")

        original_count = len(documents)

        # Step 1: Deduplicate only (NO relevance re-scoring, NO top-k filter)
        deduped, dedup_removed = self._deduplicate(documents)
        dedup_count = original_count - len(deduped)

        logger.info(f"[Organizer] Input: {original_count} docs → After dedup: {len(deduped)} "
                    f"(removed {dedup_count} duplicates)")

        # ── Step 1.5 (OPT-IN): L2 query-relevance rerank + truncate ──────────
        # This is the L2-RERANK/TRUNCATE experiment hook. OFF by default to keep the
        # "group = presentation, never a filter" recall-safe contract untouched.
        # When enabled:
        #   - rerank=True : order docs by combined lexical(2-gram w/ query) + score,
        #                   so query-relevant evidence moves to the TOP of get_context().
        #   - truncate=True: drop low-relevance / long docs until fits max_chars/max_chunks,
        #                   cutting the noise that dilutes the prompt (DISTRACT).
        if rerank or truncate:
            deduped = self._rerank_truncate(
                deduped, question,
                rerank=rerank,
                w_lex=rerank_weight_lex, w_score=rerank_weight_score,
                truncate=truncate,
                max_chunks=max_chunks, max_chars=max_chars,
                signal=signal, w_cap=w_cap,
            )

        # Step 2: Group deduped docs (preserving original FAISS order within groups)
        if not directive or directive.action not in self.action_handlers:
            handler = self._group_by_topic
        else:
            handler = self.action_handlers[directive.action]

        params = directive.params.copy() if directive and directive.params else {}
        params["_task_type"] = task_type or "general"

        result = handler(deduped, question, params)

        # Step 3: NOTE — Group LIMITING IS DISABLED (paper stage decision).
        #
        # "Group = Presentation, NOT Filtering":
        #   Group headers only decide the DISPLAY ORDER of chunks; they must
        #   never decide WHICH chunks are kept. In IndustryBench the corpus is
        #   an Industrial Manual — a single "group" may map to a whole manual
        #   section (Safety / Inspection / Parameter). Trimming it would be
        #   deleting a chapter.
        #
        #   Therefore `_select_top_groups` is intentionally NOT called. Every
        #   deduplicated original chunk stays in `result.documents`, and
        #   `OrganizedEvidence.get_context()` emits the FULL `self.documents`
        #   list (chunks in any pruned/uncategorized view fall through to the
        #   "other evidence" header). Evidence Fidelity = 100%.
        #
        #   The group view itself is also kept intact (all groups retained) so
        #   it remains purely presentational.
        task_type_str = task_type or "general"

        # Update metadata (organized_count = all deduped docs, NOT filtered)
        result.original_count = original_count
        result.organized_count = len(deduped)
        result.organization_method = f"{result.organization_method}_lightweight"

        return result


    # ============================================================
    # Step 1: De-duplication
    # ============================================================

    # ─── De-duplication heuristic thresholds ─────────────────────────────
    # Problem ④ fix (Recall-Safe):
    #   The old `> 0.80` threshold treated highly similar *standard/parameter*
    #   chunks (e.g. GB/T specifications, tables) as duplicates and silently
    #   DROPPED the second chunk. If the key fact ("GB/T 20476", "65℃") existed
    #   only in the dropped chunk, it never reached the final prompt — even
    #   though the Retriever had hit it. That was a genuine context-loss bug.
    #   New strategy is intentionally conservative: only drop a chunk when it
    #   is *virtually character-identical* (sim >= 0.97). Near-duplicates that
    #   are merely similar are KEPT so no original fact is lost.
    DEDUP_DROP_THRESHOLD = 0.97

    def _deduplicate(self, documents: List[Any]) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """Remove near-duplicate documents based on content overlap.
        
        Recall-Safe (Problem ④):
            Only chunks that are *virtually identical* (similarity >= 0.97) are
            dropped, so standard/parameter chunks that merely overlap in wording
            are preserved — the Retriever's hits reach the final context intact.

        Returns:
            Tuple of (unique_docs, removed_with_reason) where removed_with_reason
            is a list of (doc, reason_string) tuples.
        """
        unique: List[Any] = []
        seen_texts: List[str] = []
        removed: List[Tuple[Any, str]] = []

        for doc in documents:
            text = self._get_doc_text(doc).strip()
            if not text:
                removed.append((doc, "empty_content"))
                continue

            is_dup = False
            dup_similarity = 0.0
            for existing in seen_texts:
                similarity = self._jaccard_similarity(text, existing)
                if similarity >= self.DEDUP_DROP_THRESHOLD:  # virtually identical
                    is_dup = True
                    dup_similarity = similarity
                    break

            if not is_dup:
                unique.append(doc)
                seen_texts.append(text)
            else:
                removed.append((doc, f"duplicate(sim={dup_similarity:.2f})"))

        return unique, removed

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Compute character-level Jaccard similarity between two texts.

        Uses a slightly larger sample (500 chars) so that distinct paragraphs
        of a long standard/parameter chunk are not misjudged as duplicates
        based on a short, shared header alone.
        """
        set_a = set(a[:500].lower())
        set_b = set(b[:500].lower())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ============================================================
    # L2-RERANK/TRUNCATE (OPT-IN experiment) — query-relevance steering
    # ============================================================
    # 目的(9.6)：真实链路 L2 稀释是唯一真实病灶(纯度 43%, 56.6% 无关字符进 prompt)。
    # 该钩子在不改 retriever/pipeline 的前提下，在去重后、分组前把"与 query 相关"的块
    # 顶到 get_context() 最前，并可选砍掉低相关/超长块以降低稀释。
    # 默认关闭(显式传 rerank=True/truncate=True 才生效)，不改变原有全量保留行为。

    @staticmethod
    def _char_2grams(s: str):
        s = str(s).replace(" ", "").lower()
        return {s[i:i + 2] for i in range(max(0, len(s) - 1))} if s else set()

    def _lexical_overlap(self, doc_text: str, query: str) -> float:
        """Dice(2-gram overlap) between doc text and query —— 轻量 query 相关信号。

        免去外部重排模型：只统计 query 的字符 bigram 有多少出现在该块。
        """
        g_doc = self._char_2grams(doc_text)
        g_q = self._char_2grams(query)
        if not g_doc or not g_q:
            return 0.0
        return len(g_doc & g_q) / len(g_q)  # query覆盖率 -> 越相关越接近1

    def _rerank_truncate(
        self,
        documents: List[Any],
        question: str,
        rerank: bool = True,
        w_lex: float = 0.7,
        w_score: float = 0.3,
        truncate: bool = False,
        max_chunks: Optional[int] = None,
        max_chars: int = 6000,
        signal: str = "lex",
        w_cap: float = 0.3,
    ) -> List[Any]:
        """
        Reorder (and optionally trim) deduped docs by query relevance.

        signal:
          - "lex" (default, backward-compatible):
               score = w_lex*lexical_overlap(query, doc) + w_score*norm(FAISS score)
          - "cap": capability/semantic-led —— 语义分 + capability 先验对齐 + 少量词法:
               score = w_score*norm(FAISS score) + w_cap*cap_aligned + w_lex*lex
               cap_aligned = 1 若块 capability ∈ 题目期望能力(FMT2CAP/CAP_SUB 推断)。
               目的：避免"词法重排奖励题干复述块"，改用"同能力域 + 语义近"顶置真答案源块。
        rerank=True : 按分降序(相关块置顶)。
        truncate=True:从低分端砍，直到 块数<=max_chunks 且 累积字符<=max_chars。
        返回重排后(可能裁剪)的列表；顺序即 get_context() 输出顺序。
        """
        if not documents:
            return documents

        expect = self._expect_capabilities(question) if signal == "cap" else []

        scored = []
        for idx, doc in enumerate(documents):
            text = self._get_doc_text(doc)
            lex = self._lexical_overlap(text, question)
            raw_score = float(getattr(doc, "score", 0.0) or 0.0)
            norm_score = max(0.0, min(raw_score / 1.0, 1.0))  # FAISS score~0..1
            if signal == "cap":
                cap = (getattr(doc, "capability", "") or "")
                cap_aligned = 1.0 if (expect and cap in expect) else 0.0
                combined = w_score * norm_score + w_cap * cap_aligned + w_lex * lex
            else:  # lex (default)
                combined = w_lex * lex + w_score * norm_score
            scored.append((doc, combined, len(text), idx))

        # 重排: 综合分降序; tie 保持原 FAISS 位次(稳定)
        if rerank:
            scored.sort(key=lambda t: (t[1], -t[3]), reverse=True)

        kept = []
        cum_chars = 0
        if truncate:
            for doc, combined, nchars, _idx in scored:
                if max_chunks is not None and len(kept) >= max_chunks:
                    break
                if max_chars and max_chars > 0 and cum_chars + nchars > max_chars:
                    break
                kept.append(doc)
                cum_chars += nchars
            before = len(documents)
            if len(kept) < before:
                logger.info(f"[Organizer] L2-TRUNCATE: {before} → {len(kept)} docs "
                            f"(cum {cum_chars} chars ≤ {max_chars})")
        else:
            kept = [d for d, _c, _n, _i in scored]

        # 便捷字段：为后续探针留痕(不改变类结构)
        self._last_l2_meta = {
            "rerank": rerank, "truncate": truncate, "signal": signal,
            "n_in": len(documents), "n_out": len(kept),
            "avg_lex": sum(self._lexical_overlap(self._get_doc_text(d), question)
                           for d in kept) / len(kept) if kept else 0.0,
        }
        return kept

    # ── capability 先验（signal="cap" 用）─────────────────────────────
    # 复用既有已验证口径：按题型(format) + task 关键词推断"题目期望 capability"，
    # 作为证据层先验，把同能力域的答案源块顶置(而非词法复述块)。自包含，不依赖 probe 文件。
    FMT2CAP = {
        "calculation": ["工程计算与估算"],
        "multiplechoice": ["选型与替代"],
        "fillblank": ["标准规范与术语", "工艺原理与参数影响"],
        "qa": [],
    }
    CAP_SUB = [
        (re.compile(r"选型|选择|推荐|哪个|对比|差异|区别|更适合|适用"), "选型与替代"),
        (re.compile(r"标准|规范|GB|IEC|条款|要求|限值|规定"), "标准规范与术语"),
        (re.compile(r"故障|排查|诊断|失效|过压|烧毁|报警|污染"), "故障诊断与排查"),
        (re.compile(r"计算|多少|容量|压降|电流|电压|加热时间|公式|功率"), "工程计算与估算"),
        (re.compile(r"原理|概念|工作原理|是什么|如何|过程|机制"), "工艺原理与参数影响"),
    ]

    def _expect_capabilities(self, question: str) -> List[str]:
        """由 format+task 关键词推断期望 capability（证据层先验，非硬标签）。"""
        q = question or ""
        ql = q.lower()
        for fmt, caps in self.FMT2CAP.items():
            if fmt in ql and caps:
                return caps
        for pat, cap in self.CAP_SUB:
            if pat.search(q):
                return [cap]
        return []


    # ============================================================
    # Step 2: Relevance Scoring
    # ============================================================

    # ⚠️ DEPRECATED: _compute_relevance and _extract_keywords are preserved
    #    for reference only. They are NOT called by execute() anymore.
    #    Removal of relevance re-scoring was the KEY FIX for Problem ②.
    #    These methods overwrote FAISS semantic ranking with keyword-based
    #    scoring, causing recall loss. Keeping them as documentation.

    def _compute_relevance(
        self, documents: List[Any], question: str
    ) -> List[Tuple[Any, float]]:
        """
        [DEPRECATED — not called by execute()]
        
        Compute relevance score for each document vs the question.
        
        ⚠️ REMOVED: This method was overwriting FAISS semantic ranking
           with keyword-based scoring. It caused recall loss.
           
        Scoring method (preserved for reference):
            - keyword_overlap: fraction of question keywords found in doc
            - industry_match: bonus if doc industry matches question topic
            - title_match: bonus if key terms appear in capability/source
        
        Returns:
            List of (document, score) sorted descending.
        """
        question_lower = question.lower()
        question_keywords = self._extract_keywords(question)

        # Determine the main topic(s) from question
        main_topics = set()
        for kw in question_keywords:
            main_topics.add(kw.lower())

        scored = []
        for doc in documents:
            text = self._get_doc_text(doc).lower()
            meta = self._get_doc_metadata(doc)
            capability = meta.get("capability", "").lower()
            industry = meta.get("industry", "").lower()
            rank = int(meta.get("rank", 10))
            retrieval_score = float(meta.get("score", 0.0))

            # Keyword overlap: what fraction of question keywords appear in doc text?
            if question_keywords:
                hits = sum(1 for kw in question_keywords if kw in text)
                keyword_score = hits / len(question_keywords)
            else:
                keyword_score = 0.0

            # Industry/capability alignment
            alignment_bonus = 0.0
            for topic in main_topics:
                if topic in industry or topic in capability:
                    alignment_bonus += 0.15
                # Check if any question keyword is in the capability field
                for kw in question_keywords:
                    if kw in capability or kw in industry:
                        alignment_bonus += 0.10
                        break

            # Boost documents whose capability matches expected task
            # e.g., for "选择HDMI" question, documents with "选型" capability get bonus
            question_keywords_lower = [kw.lower() for kw in question_keywords]
            task_relevant_kw = ["选择", "选型", "规格", "标准", "selection",
                                "specification", "standard", "comparison",
                                "比较", "对比", "故障", "诊断", "流程", "步骤"]
            for rk in task_relevant_kw:
                if rk in capability:
                    alignment_bonus += 0.10
                    break

            # Normalize retrieval score to 0-1 range (assume typical max=1.0)
            norm_retrieval = min(retrieval_score / 1.0, 1.0)

            # Combined score
            score = (
                0.35 * keyword_score
                + 0.25 * alignment_bonus
                + 0.40 * norm_retrieval
            )

            scored.append((doc, score))

        return scored

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "but",
            "if", "or", "because", "as", "until", "while", "of", "at",
            "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "should", "now", "这", "那", "的", "了", "在", "是",
            "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上",
            "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "什么", "怎么", "如何", "哪个", "哪些",
        }

        # Tokenize: split by non-alphanumeric (preserve Chinese + English)
        tokens = re.findall(r'[a-zA-Z0-9]+(?:[-][a-zA-Z0-9]+)?|[\u4e00-\u9fff]+', text.lower())

        # Filter
        keywords = []
        for t in tokens:
            t_stripped = t.strip()
            if t_stripped and t_stripped not in stopwords and len(t_stripped) >= 2:
                keywords.append(t_stripped)

        return list(set(keywords))

    # ============================================================
    # Step 5: Task-aware Group Selection
    # ============================================================

    def _select_top_groups(
        self,
        groups: Dict[str, List[Any]],
        task_type: str,
        max_groups: int,
    ) -> Dict[str, List[Any]]:
        """
        Select the top N groups most relevant to the task type.
        
        Priority order:
            1. Groups whose name matches TASK_GROUP_PRIORITY for this task
            2. Groups with most documents
            3. Groups with highest average retrieval score
        
        Args:
            groups: Current group dict
            task_type: Task type string
            max_groups: Maximum groups to keep
        
        Returns:
            Filtered group dict with at most max_groups entries.
        """
        if len(groups) <= max_groups:
            return groups

        priority_prefixes = self.TASK_GROUP_PRIORITY.get(task_type, [])

        def group_priority(item: Tuple[str, List[Any]]) -> Tuple[int, int, float]:
            group_name, docs = item
            name_lower = group_name.lower()

            # Priority 0: name matches one of the preferred prefixes
            for i, prefix in enumerate(priority_prefixes):
                if prefix.lower() in name_lower:
                    return (0, i, sum(self._get_doc_score(d) for d in docs))

            # Priority 1: has documents
            if docs:
                avg_score = sum(self._get_doc_score(d) for d in docs) / len(docs)
                return (1, -len(docs), avg_score)

            return (2, 0, 0.0)

        sorted_groups = sorted(groups.items(), key=group_priority)
        selected = dict(sorted_groups[:max_groups])

        removed_groups = [g for g in groups if g not in selected]
        if removed_groups:
            logger.info(f"[Organizer] 📊 Group修剪: removed {len(removed_groups)} groups "
                        f"({', '.join(removed_groups)}) → kept {len(selected)}")

        return selected

    def _get_doc_score(self, doc: Any) -> float:
        """Get retrieval score from a document."""
        return float(getattr(doc, "score", 0.0))

    # ============================================================
    # OrganizedEvidence Builder (overridden from v1)
    # ============================================================

    def _make_organized(
        self,
        documents: List[Any],
        groups: Dict[str, List[Any]],
        method: str,
    ) -> OrganizedEvidence:
        """
        Create an OrganizedEvidence instance.
        
        NOTE: documents is the FILTERED list (v2 change).
              groups reflect only the kept documents.
        """
        all_docs = list(documents)
        return OrganizedEvidence(
            documents=all_docs,
            groups=groups,
            original_count=0,  # will be set by execute()
            organized_count=len(all_docs),
            organization_method=method,
        )

    def _get_doc_text(self, doc: Any) -> str:
        """Get text content from a document regardless of its type."""
        return getattr(doc, "content", str(doc))

    def _get_doc_metadata(self, doc: Any) -> Dict[str, str]:
        """Get metadata from a document."""
        return {
            "industry": getattr(doc, "industry", ""),
            "capability": getattr(doc, "capability", ""),
            "category": getattr(doc, "category", ""),
            "rank": str(getattr(doc, "rank", 0)),
            "score": str(round(getattr(doc, "score", 0.0), 3)),
        }

    # -------------------------------------------------------
    # Handler Implementations — ONLY add section headers
    # -------------------------------------------------------

    def _group_by_comparison(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Group evidence under comparison entity headers.
        """
        params = params or {}
        grouped: Dict[str, List[Any]] = defaultdict(list)
        compared_objects = self._extract_comparison_entities(question)

        if compared_objects:
            for doc in documents:
                text = self._get_doc_text(doc).lower()
                assigned = False
                for obj in compared_objects:
                    if obj.lower() in text:
                        grouped[f"comparison: {obj}"].append(doc)
                        assigned = True
                        break
                if not assigned:
                    grouped["other"].append(doc)
        else:
            for doc in documents:
                key = self._get_doc_metadata(doc).get("capability", "evidence")
                grouped[key].append(doc)

        groups = dict(sorted(grouped.items()))
        return self._make_organized(documents, groups, "comparison")

    def _group_by_symptom(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Group evidence under symptom/cause headers.
        """
        params = params or {}
        grouped: Dict[str, List[Any]] = defaultdict(list)
        symptoms = self._extract_symptoms(question)

        if symptoms:
            for doc in documents:
                text = self._get_doc_text(doc).lower()
                assigned = False
                for symptom in symptoms:
                    if symptom.lower() in text:
                        grouped[f"symptom: {symptom}"].append(doc)
                        assigned = True
                        break
                if not assigned:
                    grouped["causes/remediation"].append(doc)
        else:
            for doc in documents:
                industry = self._get_doc_metadata(doc).get("industry", "evidence")
                grouped[industry].append(doc)

        groups = dict(sorted(grouped.items()))
        return self._make_organized(documents, groups, "symptom")

    def _group_by_candidate(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Group evidence under candidate option headers.
        """
        params = params or {}
        grouped: Dict[str, List[Any]] = defaultdict(list)
        candidates = self._extract_candidates(question)

        if candidates:
            for doc in documents:
                text = self._get_doc_text(doc).lower()
                assigned = False
                for candidate in candidates:
                    if candidate.lower() in text:
                        grouped[f"candidate: {candidate}"].append(doc)
                        assigned = True
                        break
                if not assigned:
                    grouped["evaluation_criteria"].append(doc)
        else:
            for doc in documents:
                capability = self._get_doc_metadata(doc).get("capability", "options")
                grouped[capability].append(doc)

        groups = dict(sorted(grouped.items()))
        return self._make_organized(documents, groups, "candidate")

    def _group_by_formula(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Add formula/parameter section headers.
        """
        params = params or {}
        groups: Dict[str, List[Any]] = {
            "formulas": [],
            "parameters": [],
            "context": [],
        }

        formula_pattern = re.compile(
            r"[A-Za-z]+\s*[=:]\s*[A-Za-z0-9_+\-*/^()\s.]+|"
            r"formula|equation|calculate|compute|using\s+the\s+formula",
            re.IGNORECASE,
        )

        for doc in documents:
            text = self._get_doc_text(doc)
            if formula_pattern.search(text):
                groups["formulas"].append(doc)
            elif any(
                kw in text.lower()
                for kw in ["parameter", "constant", "value", "specification"]
            ):
                groups["parameters"].append(doc)
            else:
                groups["context"].append(doc)

        groups = {k: v for k, v in groups.items() if v}
        return self._make_organized(documents, groups, "formula")

    def _group_chronological(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Add chronological phase headers.
        """
        params = params or {}
        groups: Dict[str, List[Any]] = defaultdict(list)

        for doc in documents:
            text = self._get_doc_text(doc)
            step_match = re.search(r"(?:step|phase)\s*(\d+)", text, re.IGNORECASE)
            if step_match:
                step_num = int(step_match.group(1))
                if step_num <= 5:
                    key = "initial_steps"
                elif step_num <= 10:
                    key = "intermediate_steps"
                else:
                    key = "final_steps"
            else:
                if any(kw in text.lower() for kw in ["first", "prerequisite", "prepare"]):
                    key = "preparation"
                elif any(kw in text.lower() for kw in ["then", "next", "after"]):
                    key = "execution"
                elif any(kw in text.lower() for kw in ["finally", "result", "verify"]):
                    key = "verification"
                else:
                    key = "context"

            groups[key].append(doc)

        groups = dict(groups)
        return self._make_organized(documents, groups, "chronological")

    def _group_by_topic(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        Default topic-based grouping with industry/capability headers.
        """
        params = params or {}
        grouped: Dict[str, List[Any]] = defaultdict(list)

        for doc in documents:
            industry = self._get_doc_metadata(doc).get("industry", "general")
            capability = self._get_doc_metadata(doc).get("capability", "")
            key = f"{industry}/{capability}" if capability else industry
            grouped[key].append(doc)

        groups = dict(sorted(grouped.items()))
        return self._make_organized(documents, groups, "topic")

    def _pass_through(
        self,
        documents: List[Any],
        question: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> OrganizedEvidence:
        """
        No grouping - pass through as-is. No headers added.
        """
        return self._make_organized(documents, {}, "none")

    # -------------------------------------------------------
    # Entity extraction helpers
    # -------------------------------------------------------

    def _extract_comparison_entities(self, question: str) -> List[str]:
        """Extract entities being compared from the question."""
        entities = []
        between_match = re.search(
            r"(?:compare|difference|similarit|between)\s+(.+?)(?:\s+and\s+|\s+vs\.?\s+)(.+)",
            question,
            re.IGNORECASE,
        )
        if between_match:
            entities.append(between_match.group(1).strip())
            entities.append(between_match.group(2).strip().rstrip("?").strip())

        vs_match = re.search(r"(.+?)\s+vs\.?\s+(.+)", question, re.IGNORECASE)
        if vs_match:
            entities.append(vs_match.group(1).strip())
            entities.append(vs_match.group(2).strip().rstrip("?").strip())

        or_match = re.search(r"(.+?)\s+or\s+(.+)", question, re.IGNORECASE)
        if or_match:
            entities.append(or_match.group(1).strip())
            entities.append(or_match.group(2).strip().rstrip("?").strip())

        return list(set(entities))

    def _extract_symptoms(self, question: str) -> List[str]:
        """Extract symptom/problem descriptions from the question."""
        symptoms = []
        patterns = [
            r"(?:symptom|issue|problem|fault|error)\s*(?::|is|are)?\s*(.+?)(?:[?.]|$)",
            r"(?:what|cause|why).+?(?:of|for)\s+(.+?)(?:[?.]|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for m in matches:
                symptom = m.strip().rstrip("?").strip()
                if symptom and len(symptom.split()) <= 10:
                    symptoms.append(symptom)
        return symptoms[:5]

    def _extract_candidates(self, question: str) -> List[str]:
        """Extract candidate names from the question."""
        candidates = []
        patterns = [
            r"(?:choose|select|which|recommend|best)(?:\s+\w+){0,3}\s+(.+?)(?:[?.]|$)",
            r"(?:between|among)\s+(.+?)(?:\s+and\s+|\s*[?.]|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for m in matches:
                candidate = m.strip().rstrip("?").strip()
                if candidate:
                    candidates.append(candidate)
        return candidates
class RerankTruncOrganizer(EvidenceOrganizer):
    """生产级 Organizer 变体：默认开启 L2 词法重排 + 低分端截断（信号='lex'）。

    背景：S4→S5 组织消融真实 30 题全量（base→本组织器）avg 1.70→1.867（+0.167），
    升6/降1/平23。增益来自把词法相关证据顶置（signal='lex':
    score = 0.7*lex(query,doc) + 0.3*norm(FAISS)）并把低相关尾部噪声截断。
    检索侧不动；仅替换此组件即"生产直接开 L2 组织"，可单点回退 EvidenceOrganizer。
    """

    def execute(
        self,
        directive,
        documents,
        question="",
        task_type=None,
        rerank=True,
        truncate=True,
        signal="lex",
        max_chars=6000,
        max_chunks=None,
        rerank_weight_lex=0.7,
        rerank_weight_score=0.3,
    ):
        return super().execute(
            directive=directive,
            documents=documents,
            question=question,
            task_type=task_type,
            rerank=rerank,
            truncate=truncate,
            signal=signal,
            max_chars=max_chars,
            max_chunks=max_chunks,
            rerank_weight_lex=rerank_weight_lex,
            rerank_weight_score=rerank_weight_score,
        )

