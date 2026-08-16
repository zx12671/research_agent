"""
pipeline.py: AdaptiveAgenticPipeline v2 with ExecutionGraph and GraphExecutor.

REDESIGNED v2 (ExecutionGraph):
    The pipeline now executes a directed graph (ExecutionGraph) instead of a
    flat plan. The GraphExecutor walks the graph, dispatching each typed node
    to the appropriate module.

    Architecture v2:
        Question
           ↓
        TaskAnalyzer ──→ TaskAnalysis
           ↓
        StrategyPlanner ──→ ExecutionGraph (workflow DAG)
           ↓
        ┌──────────────────────────────────────┐
        │          GraphExecutor                │
        │  Walks graph, dispatches by node type │
        │                                       │
        │  retrieve_1 ──→ organize_1 ──→ ...    │
        │    ↓                                   │
        │  decide (branch)                       │
        │    ├─ yes → reason_1 ──→ verify_1      │
        │    └─ no  → retrieve_2 ──→ merge_1     │
        └──────────────────────────────────────┘
           ↓
        DeepSeek (final answer generation)
           ↓
        Answer

    Key features:
        - True workflow graph with conditional branching
        - Multi-hop retrieval
        - Verification loops
        - Decision nodes that dynamically route execution
        - Each adaptive component independently measurable
"""

import logging
import re
from typing import Any, Dict, List, Optional

from agentic.task_types import (
    TaskType,
    TaskAnalysis,
    GraphNode,
    ExecutionGraph,
    ExecutionDirective,
    OrganizedEvidence,
)
from agentic.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class GraphExecutor:
    """
    Executes an ExecutionGraph by walking nodes and dispatching to modules.

    The executor maintains a runtime context that tracks:
        - Current context (accumulated evidence + reasoning state)
        - Node execution history
        - Branch decisions
        - Loop counters (to prevent infinite loops)

    Execution Algorithm:
        1. Start at entry point nodes
        2. For each node in topological order:
            a. Dispatch to appropriate handler based on node.type
            b. If node.decide: evaluate condition, follow branch
            c. If node.verify: check verdict, follow pass/fail
            d. Accumulate results in context
        3. Return final context (answer) from the end node's predecessor
    """

    def __init__(
        self,
        retriever_module: Any,
        organizer_module: Any,
        solver_module: Any,
        llm_client: Any,
        max_loop_iterations: int = 3,
        semantic_rewriter: Any = None,
    ):
        """
        Initialize the GraphExecutor.

        Args:
            retriever_module: The retrieval module (e.g., Retriever)
            organizer_module: The evidence organizer (EvidenceOrganizer)
            solver_module: The task solver (TaskSolver)
            llm_client: LLM client for reasoning and decisions
            max_loop_iterations: Max iterations for verification loops
            semantic_rewriter: Optional SemanticQueryRewriter (agentic/query_rewriter.py).
                When provided AND a retrieve node sets `semantic_mv=True`, the node
                uses LLM semantic multi-view rewriting instead of lexical comma-split:
                rewrite(question, task_analysis) -> views; each view runs hybrid_retrieve
                (dense+BM25jieba, sparse 只补不扰); results fused via _fusion_merge with the
                original query kept as the first/highest-weight view (recall 保底).
                Default None => production behavior unchanged (comma-split / single / hybrid).
        """
        self.retriever = retriever_module
        self.organizer = organizer_module
        self.solver = solver_module
        self.llm = llm_client
        self.max_loop_iterations = max_loop_iterations
        self.semantic_rewriter = semantic_rewriter
        self.prompt_builder = PromptBuilder()
        # How many questions actually reached a SECOND retrieval hop.
        # Used to measure the "Second Retrieval Trigger Rate" (triggers / N).
        self.second_retrieval_count = 0

    def execute(
        self,
        graph: ExecutionGraph,
        question: str,
        retriever_kwargs: Optional[Dict] = None,
        pre_retrieved_evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an ExecutionGraph end-to-end.

        Args:
            graph: The ExecutionGraph to execute
            question: The original question
            retriever_kwargs: Additional kwargs for the retriever
            pre_retrieved_evidence: If provided, skip all retrieve nodes and use
                this externally-retrieved evidence instead. This makes the
                retriever a Stable Knowledge Interface — always retrieve(question, k=10),
                independent of the Agent's analysis/planning.

        Returns:
            Dict with execution results including final answer
        """
        self.second_retrieval_count = 0
        context = ExecutionContext(question=question)
        retriever_kwargs = retriever_kwargs or {}

        # If pre-retrieved evidence is provided, inject it into context
        # so that organize/reason nodes can use it directly,
        # and retrieve nodes will be skipped.
        if pre_retrieved_evidence is not None:
            context.last_retrieval_results = list(pre_retrieved_evidence)
            context.evidence_cache["__external_retrieval__"] = list(pre_retrieved_evidence)
            logger.info(
                f"[GraphExecutor] Using pre-retrieved evidence: "
                f"{len(pre_retrieved_evidence)} documents (skip internal retrieve)"
            )

        # Get topological order
        ordered_nodes = graph.topological_sort()
        logger.info(
            f"Executing graph with {len(ordered_nodes)} nodes: "
            f"{[n.id for n in ordered_nodes]}"
        )

        # Track loop iterations per node pair
        loop_counters: Dict[str, int] = {}

        # Walk the graph
        visited = set()
        node_queue = list(graph.entry_points)

        while node_queue:
            node_id = node_queue.pop(0)

            # Prevent infinite loops
            if node_id in loop_counters:
                loop_counters[node_id] += 1
                if loop_counters[node_id] > self.max_loop_iterations:
                    logger.warning(
                        f"Max iterations ({self.max_loop_iterations}) "
                        f"reached for node {node_id}, forcing exit"
                    )
                    continue
            else:
                loop_counters[node_id] = 0

            if node_id in visited:
                continue
            visited.add(node_id)

            node = graph.get_node(node_id)
            if node is None:
                logger.warning(f"Node '{node_id}' not found in graph, skipping")
                continue

            # Execute node based on type
            try:
                self._execute_node(
                    node=node,
                    context=context,
                    graph=graph,
                    retriever_kwargs=retriever_kwargs,
                )
            except Exception as e:
                logger.error(f"Error executing node {node_id}: {e}")
                if node.fallback:
                    context.add_log(
                        "fallback",
                        node_id,
                        f"Error: {e}, falling back to {node.fallback}",
                    )
                    node_queue.append(node.fallback)
                continue

            # Determine next nodes
            next_nodes = self._get_next_nodes(node, context)

            # Add next nodes to queue (prepend for depth-first-ish)
            for next_id in reversed(next_nodes):
                if next_id not in visited:
                    node_queue.insert(0, next_id)

        # After graph execution, produce final answer
        final_result = self._produce_final_answer(question, context, graph)

        final_result["execution_graph"] = graph.to_dict()
        final_result["execution_nodes"] = context.execution_log
        final_result["node_count"] = len(context.execution_log)
        final_result["second_retrieval_triggers"] = self.second_retrieval_count
        return final_result

    def _execute_node(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Dispatch a single node to the appropriate handler."""
        dispatch = {
            "retrieve": self._handle_retrieve,
            "organize": self._handle_organize,
            "reason": self._handle_reason,
            "decide": self._handle_decide,
            "merge": self._handle_merge,
            "verify": self._handle_verify,
            "end": self._handle_end,
        }

        handler = dispatch.get(node.type)
        if handler:
            handler(node, context, graph, retriever_kwargs)
        else:
            logger.warning(f"Unknown node type: {node.type}")

    def _build_evidence_guided_query(
        self,
        question: str,
        context: "ExecutionContext",
    ) -> str:
        """
        Build an Evidence-guided follow-up query from the ALREADY-HIT chunks.

        WHY (user report): the first retrieval DID hit the relevant standard
        fragment, but the follow-up query was built from the reasoner's lossy
        summary (accumulated_context) — the very step that dropped keys like
        "GB/T 20476 / 65℃". Building the query from the ORIGINAL chunk text
        (instead of a summary) preserves those discriminating keys.

        Extraction targets (deterministic, no LLM):
          - standard / regulation numbers: GB/T 20476, GB 150, ISO 9001, JB/T,
            IEC 60034, DL/T ...
          - numeric parameter-value pairs near units: 65℃, 0.5 MPa, 380V ...
        """
        # --- 1) Gather ORIGINAL chunk content (not reasoner summary) ---
        originals: List[str] = []
        for evid_list in context.evidence_cache.values():
            if not isinstance(evid_list, list):
                continue
            for doc in evid_list:
                content = getattr(doc, "content", "")
                if content:
                    originals.append(str(content))
        original_text = "\n".join(originals)

        tokens: List[str] = []

        # --- 2) Standard / regulation numbers (question + chunks) ---
        std_pattern = re.compile(
            r"\b(?:(?:GB|GB/T|JB/T|JG/T|DL/T|NB/T|QB/T|YY/T|HJ|AQ|TSG)\s*/?\s*"
            r"\d+(?:\.\d+)*|ISO\s*\d+(?:\.\d+)*|IEC\s*\d+(?:-\d+)*|EN\s*\d+)\b",
            re.IGNORECASE,
        )
        for src_name, src in (("question", question), ("chunks", original_text)):
            for m in std_pattern.finditer(src):
                norm = re.sub(r"\s+", "", m.group(0)).upper()
                if norm not in tokens:
                    tokens.append(norm)

        # --- 3) Numeric value + unit parameter pairs (chunks only) ---
        # Heat/pressure/voltage/etc: a number immediately followed by a unit.
        param_unit_pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*"
            r"(℃|°C|°F|K|MPa|kPa|Pa|bar|kV|V|kW|kW?|W|mm|cm|m|Hz|rpm|%|mol/L|mg/L|g/L)",
            re.IGNORECASE,
        )
        # Common Chinese parameter keywords we want to co-express.
        param_keywords = [
            "温度", "压力", "电压", "电流", "频率", "直径", "厚度", "浓度",
            "质量", "速度", "时间", "加热", "保温", "冷却", "退火", "正火",
            "淬火", "回火", "屈服强度", "抗拉强度", "硬度", "标准", "GB/T",
        ]
        if any(kw in question for kw in param_keywords) or "℃" in question or "°C" in question:
            for m in param_unit_pattern.finditer(original_text):
                value = f"{m.group(1)}{m.group(2)}".strip()
                if value not in tokens:
                    tokens.append(value)
            # keep (value, unit) pairs bounded — avoid flooding query
            tokens = tokens[:8]

        # --- 4) Always keep the original question as the semantic anchor ---
        anchor = question.strip()
        if len(anchor) <= 600:
            return anchor + (" ; evidence: " + " ; ".join(tokens) if tokens else "")

        return question

    def _handle_retrieve(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute a retrieve node — supports single query and multi_query."""
        params = node.params
        retrieve_k = params.get("retrieve_k", 10)
        use_multi_query = params.get("multi_query", False)
        use_fusion = params.get("use_fusion", False)
        use_ked = params.get("use_ked", True)
        source_priority = params.get("source_priority", "standard")
        min_score = params.get("min_score")
        # Hybrid (dense + BM25 → RRF) 检索开关：单查询即可启用，不依赖 multi_query。
        # sparse_pool / dense_weight / sparse_weight 透传给 retriever.hybrid_retrieve。
        use_hybrid = params.get("hybrid", False)
        hybrid_sparse_pool = params.get("hybrid_sparse_pool", 50)
        hybrid_dense_weight = params.get("hybrid_dense_weight", 1.0)
        hybrid_sparse_weight = params.get("hybrid_sparse_weight", 0.2)

        # [PRODUCTION] 语义级多视角改写开关（P1 主杠杆，默认关）。
        # 当为 True 且 executor 注入了 semantic_rewriter 时，本 retrieve 节点改用
        # SemanticQueryRewriter.rewrite(question, task_analysis) 生成语义互补视角，
        # 各视角走 hybrid_retrieve(dense+BM25jiebasparse) 再 RRF 融合 —— 取代机械切逗号。
        semantic_mv = params.get("semantic_mv", False)
        mv_max_views = int(params.get("mv_max_views", 4))
        mv_sparse_weight = float(params.get("mv_sparse_weight", 0.2))

        logger.info(
            f"[{node.id}] Retrieving k={retrieve_k}, "
            f"multi_query={use_multi_query}, fusion={use_fusion}, "
            f"ked={use_ked}, hybrid={use_hybrid}"
        )

        # Detect a SECOND (or later) retrieval hop so we can:
        #   (a) count the Second Retrieval Trigger Rate;
        #   (b) route it through Evidence-guided query building.
        is_followup = (
            node.params.get("targeted", False)
            or node.params.get("evidence_guided", False)
            or node.id not in ("retrieve_1", "retrieve")
        )
        if "retrieve_1" in context.evidence_cache and is_followup:
            self.second_retrieval_count += 1

        # ===== 20260808 S2-P0 SHORTCUT: 主检索短路外部证据 =====
        # Stable Knowledge Interface（run() 注入 pre_retrieved_evidence）的设计意图是
        # "外部已 retrieve(question, k=10)，Agent 内部 retrieve_1 不必重跑"。但原实现仅把
        # 证据写入 context，_handle_retrieve 从不短路 → 60/60 次 retrieve_1 真实重跑（冗余
        # 二次检索，见 docs/s2_planner_cost_audit.md）。
        # 修复：对【主检索节点】（retrieve_1 / retrieve，即非 followup）检测 __external_retrieval__
        # 已存在时，直接把外部证据登记为本节点输出并 return，不调用 retriever。
        # 注意：followup 检索（retrieve_2 等 evidence_guided/targeted）【不】短路——它是
        # verify fail 后按需补证据的第二跳，跳过会丢针对性补充检索、损害 recall。
        _ext = context.evidence_cache.get("__external_retrieval__")
        if _ext and not is_followup:
            evidence = _ext if isinstance(_ext, list) else [_ext]
            # 对齐原 retrieve_k 语义：主检索节点按 k 取外部证据前缀（缺失时取全部）
            _k = int(params.get("retrieve_k", 10))
            if len(evidence) > _k:
                evidence = evidence[:_k]
            context.add_evidence(node.id, evidence)
            context.last_retrieval_results = evidence
            context.add_log(
                "retrieve",
                node.id,
                f"SHORTCUT (Stable Knowledge Interface): {len(evidence)} docs from "
                f"__external_retrieval__, skipped internal retrieval ({retrieve_k})",
            )
            logger.info(
                f"[{node.id}] SHORTCUT: using pre-retrieved evidence "
                f"({len(evidence)} docs), skipped internal retrieval"
            )
            logger.info("=" * 80)
            logger.info(f"[DEBUG][SHORTCUT] Retrieve Node: {node.id}")
            for i, doc in enumerate(evidence[:3]):
                content = getattr(doc, "content", str(doc))
                score = getattr(doc, "score", 0.0)
                logger.info(
                    f"[Doc {i}] score={score:.3f} len={len(content)} {content[:120]}"
                )
            logger.info(
                f"[DEBUG] Evidence keys in context after retrieve: "
                f"{list(context.evidence_cache.keys())}"
            )
            logger.info("=" * 80)
            return

        # Build retrieval kwargs
        retrieval_kwargs = {
            **retriever_kwargs,
            "k": retrieve_k,
            "use_ked": use_ked,
        }
        if min_score is not None:
            retrieval_kwargs["min_score"] = min_score

        question_to_use = context.question

        # ===== Evidence-guided follow-up retrieval =====
        # NEVER build the 2nd query from the reasoner's LOSSY summary. Build it
        # from the ORIGINAL hit chunks' discriminating keys (standard number,
        # numeric parameter like 65℃). This is the fix for "Retriever hit it
        # but the Agent dropped it": the key survives because we read it from
        # the chunk verbatim, not from a compressed summary.
        if params.get("evidence_guided", False) or params.get("targeted", False):
            question_to_use = self._build_evidence_guided_query(
                context.question, context
            )
            logger.info(
                f"[{node.id}] Evidence-guided follow-up query "
                f"(len={len(question_to_use)}): {question_to_use[:180]}..."
            )

        # ==== SEMANTIC MULTI-VIEW retrieval (dense+BM25 per view → RRF fusion) ====
        # P1 主杠杆：语义级多视角改写。由 LLM(DeepSeek) 依 task/expected_evidence 生成语义互补
        # 子视角，各视角独立跑 hybrid_retrieve（BM25jieba 稀疏臂"只补不扰"，sweet spot 参数见
        # _ab_bm25_jieba_grid 结论），再 RRF 融合。resolve 语义鸿沟 hard-miss（机械切逗号救不了
        # 的同义改写/术语替换/标准号关联），同时视图内【原 query 首保底+RRF 加权】不稀释主信号。
        # 优先级最高：节点显式 semantic_mv=True（默认 False）才进；未注入 rewriter 时静默降级到
        # 后续标准分派，生产行为零回归。
        if semantic_mv and getattr(self, "semantic_rewriter", None) is not None:
            try:
                views = self.semantic_rewriter.rewrite(
                    question_to_use, getattr(graph, "task_analysis", None)
                ) or [question_to_use]
                views = [v for v in views if (v or "").strip()] or [question_to_use]
                logger.info(
                    f"[{node.id}] SEMANTIC-MV rewrite → {len(views)} views "
                    f"(task={getattr(getattr(graph, 'task_analysis', None), 'task', '?')}): "
                    f"{[v[:40] for v in views[:mv_max_views]]}"
                )
                per_view = []
                for v in views[:mv_max_views]:
                    try:
                        vr = self.retriever.hybrid_retrieve(
                            v,
                            k=retrieve_k,
                            use_ked=use_ked,
                            use_multi_query=False,
                            use_sparse=True,
                            sparse_pool=hybrid_sparse_pool,
                            dense_weight=hybrid_dense_weight,
                            sparse_weight=mv_sparse_weight,
                        )
                        per_view.append(vr)
                    except Exception as e:  # 单视角失败不阻断整路
                        logger.warning(f"[{node.id}] view '{v[:30]}…' 检索失败: {e}")
                if per_view:
                    results = self.retriever._fusion_merge(per_view, retrieve_k)
                    results.query = question_to_use
                    self._last_semantic_mv_views = views
                else:
                    results = self.retriever.retrieve(question_to_use, **retrieval_kwargs)
            except Exception as e:
                logger.warning(
                    f"[{node.id}] SEMANTIC-MV 改写异常，降级标准检索: {e}"
                )
                results = self.retriever.retrieve(question_to_use, **retrieval_kwargs)

        # ==== HYBRID retrieval (dense + BM25 → RRF) when configured ====
        # Single-query hybrid: 对原始 query 跑 dense(KED) + BM25 稀疏，再做 RRF 融合，
        # 用精确 token 命中(BM25) 补强纯语义 dense 的弱项（capability/型号/标准号）。
        # 独立于 multi_query；`hybrid_pure` 为 True 时也强制单查询，不做子查询分解。
        if use_hybrid and hasattr(self.retriever, 'hybrid_retrieve'):
            logger.info(
                f"[{node.id}] Using hybrid_retrieve (k={retrieve_k}, "
                f"multi_query=False, sparse_pool={hybrid_sparse_pool}, "
                f"dense_w={hybrid_dense_weight}, sparse_w={hybrid_sparse_weight})"
            )
            results = self.retriever.hybrid_retrieve(
                question_to_use,
                k=retrieve_k,
                use_ked=use_ked,
                use_multi_query=False,
                use_sparse=True,
                sparse_pool=hybrid_sparse_pool,
                dense_weight=hybrid_dense_weight,
                sparse_weight=hybrid_sparse_weight,
            )
        elif use_multi_query and hasattr(self.retriever, 'multi_query_retrieve'):
            logger.info(
                f"[{node.id}] Using multi_query_retrieve (k={retrieve_k}, "
                f"fusion={use_fusion})"
            )
            results = self.retriever.multi_query_retrieve(
                question_to_use,
                k=retrieve_k,
                use_ked=use_ked,
                strategy="fusion" if use_fusion else "union",
            )
        else:
            # Standard single-query retrieval
            if use_ked and hasattr(self.retriever, 'ked') and self.retriever.ked:
                # Even for single query, KED expansion helps recall
                expanded = self.retriever.ked.expand_query(question_to_use)
                if expanded != question_to_use:
                    logger.info(
                        f"[{node.id}] KED query expansion: "
                        f"'{question_to_use[:80]}...' → '{expanded[:80]}...'"
                    )

            results = self.retriever.retrieve(
                question_to_use,
                **retrieval_kwargs,
            )

        # ==== [DIAG-ASSERT] 上游即空? 排除 Organizer 丢失 ====
        # 直击核心疑问：Retriever 到底有没有命中？
        # 在 Organizer 介入之前，把 retriever 返回的证据数量忠实记下来。
        # - retrieval_failed=True  => dense+fallback 全链失败，query embedding 非法（Retriever 有责）
        # - 有 chunks 且 failed=False => Retriever 命中，后续证据丢失发生在 Organizer/PromptBuilder
        # - 无 chunks 且 failed=False => Retriever 真的 0 命中（真没检索到）
        if hasattr(results, "retrieval_failed"):
            _rf = getattr(results, "retrieval_failed", False)
            _reason = getattr(results, "retrieval_failed_reason", "")
            _n = len(getattr(results, "chunks", []))
            if _rf:
                logger.warning(
                    f"[{node.id}·DIAGNOSE] Retriever 失败留痕 retrieval_failed=True, "
                    f"reason={_reason!r}. 上游 evidence={_n} 条（可能为 BM25 fallback）。"
                )
            else:
                logger.info(
                    f"[{node.id}·DIAGNOSE] Retriever 正常返回 {_n} 条 (failed=False, reason='')。"
                    f" Organizer【收到】{_n} 条原始证据 —— 后续若最终答案丢关键信息，责任在 "
                    f"Organizer/PromptBuilder，非 Retriever。"
                )
        elif isinstance(results, (list, tuple)):
            logger.info(f"[{node.id}·DIAGNOSE] retriever 返回 {len(results)} 条证据 (list)。")

        # ==== ADAPTATION LAYER: Convert RetrievalResult to List[RetrievedChunk] ====
        # retriever.retrieve() returns a RetrievalResult object with .chunks (List[RetrievedChunk])
        # Organizer expects List[Any] where each item has .content, .score, .rank, etc.
        if hasattr(results, "chunks"):
            evidence = list(results.chunks)
        elif isinstance(results, list):
            evidence = results
        else:
            evidence = []

        context.add_evidence(node.id, evidence)
        context.last_retrieval_results = evidence

        context.add_log(
            "retrieve",
            node.id,
            f"Retrieved {len(evidence) if hasattr(evidence, '__len__') else '?'} documents",
        )

        # ===== [DEBUG] Retriever Output =====
        logger.info("=" * 80)
        logger.info(f"[DEBUG] Retrieve Node: {node.id}")
        docs = evidence if isinstance(evidence, list) else []
        logger.info(f"Retrieved docs: {len(docs)}")
        for i, doc in enumerate(docs[:3]):
            content = getattr(doc, "content", str(doc))
            score = getattr(doc, "score", 0.0)
            logger.info(
                f"[Doc {i}] score={score:.3f} len={len(content)} {content[:200]}"
            )
        logger.info(f"[DEBUG] Evidence keys in context after retrieve: {list(context.evidence_cache.keys())}")
        if node.id in context.evidence_cache:
            cached = context.evidence_cache[node.id]
            logger.info(f"context.evidence['{node.id}'] count={len(cached) if hasattr(cached, '__len__') else '?'}")
        logger.info("=" * 80)
        # ===== END DEBUG =====

    def _handle_organize(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute an organize node using the new ExecutionDirective interface."""
        params = node.params
        organize_by = params.get("organize_by", "topic")

        logger.info(f"[{node.id}] Organizing evidence by {organize_by}")

        # Get evidence from context
        evidence = self._get_current_evidence(context)

        if not evidence:
            logger.warning(f"[{node.id}] No evidence to organize")
            context.add_log("organize", node.id, "No evidence to organize")
            return

        # ===== [DEBUG] Organize Input =====
        logger.info("=" * 80)
        logger.info(f"[DEBUG] Organize Node: {node.id}")
        logger.info(f"Organize Input Docs: {len(evidence)}")
        for i, doc in enumerate(evidence[:3]):
            content = getattr(doc, "content", str(doc))
            score = getattr(doc, "score", 0.0)
            logger.info(
                f"[Input Doc {i}] score={score:.3f} len={len(content)} {content[:200]}"
            )
        logger.info("=" * 80)
        # ===== END DEBUG =====

        # Build an ExecutionDirective from the node params for the organizer
        directive = ExecutionDirective(
            module="organization",
            action=f"group_by_{organize_by}",
            params=params,
        )

        # Determine task_type from graph
        task_type = graph.task.value if hasattr(graph, 'task') and graph.task else "general"

        # Organize the evidence using the new interface (v2: with task_type)
        #
        # ── 件① 提纯档显式开关（organize_panel）──────────────────────────────
        # 背景：生产装配了 RerankTruncOrganizer（其 execute 默认 rerank=True/
        # truncate=True → L2 词法顶置 + 低分截断），而此处原不传参，全部吃组织器
        # 自身默认。为了让"提纯档"可显式开启/关闭做 A/B（对照 EvidenceOrganizer），
        # 这里支持从 node.params["organize_panel"] 读显式面板（dict，键与
        # EvidenceOrganizer.execute 的 rerank/truncate/signal/max_chars/max_chunks/
        # rerank_weight_lex/rerank_weight_score 一致）。
        #   organize_panel 缺失/None → 保持现状（走 organizer 自身默认，不改变行为）；
        #   organize_panel = {"rerank": False, "truncate": False} → 显式关闭提纯档
        #     （等价 EvidenceOrganizer 无提纯），供 A/B baseline 使用；
        #   organize_panel = {"rerank": True, "truncate": True, "max_chars": N} →
        #     显式打开/调参提纯档。
        _organize_kwargs: Dict[str, Any] = {}
        _panel = params.get("organize_panel")
        if isinstance(_panel, dict):
            for _k in ("rerank", "truncate", "signal", "max_chars", "max_chunks",
                       "rerank_weight_lex", "rerank_weight_score", "w_cap"):
                if _k in _panel:
                    _organize_kwargs[_k] = _panel[_k]
        organized = self.organizer.execute(
            directive=directive,
            documents=evidence,
            question=context.question,
            task_type=task_type,
            **_organize_kwargs,
        )

        context.last_organized = organized
        context.organized_evidence[node.id] = organized

        groups = getattr(organized, "groups", {})

        # ===== [DEBUG] Organize Output =====
        logger.info("=" * 80)
        logger.info(f"[DEBUG] Organize Output - {node.id}")
        logger.info(f"Groups ({len(groups)}):")
        for gname, gdocs in groups.items():
            logger.info(f"  {gname}: {len(gdocs)} docs")
        evidence_str_debug = organized.get_context() if hasattr(organized, 'get_context') else ""
        logger.info(f"Formatted Evidence Length: {len(evidence_str_debug)} chars")
        logger.info("=" * 80)
        # ===== END DEBUG =====

        context.add_log(
            "organize",
            node.id,
            f"Organized into {len(groups)} groups: {list(groups.keys())}",
        )

    def _handle_reason(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute a reason node using Standard Prompt + PromptBuilder prompt_extra."""
        params = node.params
        reasoning_type = params.get("reasoning_type", "general")

        logger.info(
            f"[{node.id}] Reasoning type={reasoning_type}, "
            f"using Standard Prompt + PromptBuilder"
        )

        # Get organized evidence
        evidence = self._get_current_evidence(context)
        organized = context.last_organized

        # Build evidence string
        evidence_str = ""
        if organized:
            evidence_str = organized.get_context()
        elif evidence:
            evidence_str = self._evidence_to_string(evidence)

        # ===== [DEBUG] Reason: Evidence before PromptBuilder =====
        logger.info("=" * 80)
        logger.info(f"[DEBUG] Reason Node: {node.id}")
        logger.info(f"Evidence Length = {len(evidence_str)}")
        logger.info(f"Evidence preview (first 1000 chars):")
        logger.info(evidence_str[:1000])
        logger.info("=" * 80)
        # ===== END DEBUG =====

        # ===== PromptBuilder: generate task-aware instruction =====
        task_analysis = graph.task_analysis
        instruction = self.prompt_builder.build(
            task_analysis=task_analysis,
            graph=graph,
            evidence_context=evidence_str,
        )

        # ===== USE STANDARD PROMPT (always "general") + prompt_extra (already includes output_format) =====
        from agentic.prompts import format_prompt

        full_prompt = format_prompt(
            task_key="general",
            question=context.question,
            evidence=evidence_str,
            prompt_extra=instruction.prompt_extra,
        )

        # ===== [DEBUG] Reason: Final Prompt =====
        logger.info("=" * 80)
        logger.info(f"[DEBUG] Prompt Length = {len(full_prompt)}")
        logger.info(f"Prompt preview (first 1500 chars):")
        logger.info(full_prompt[:1500])
        logger.info("=" * 80)
        # ===== END DEBUG =====

        # Split into system and user parts
        system_end = full_prompt.find("\n\n##")
        system_part = full_prompt[:system_end].strip()
        user_part = full_prompt[system_end:].strip()

        # Direct LLM call (skip solver)
        if hasattr(self.llm.chat, "completions"):
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part},
                ],
                temperature=0.3,
                max_tokens=2048,
                timeout=60,
            )
            answer = response.choices[0].message.content.strip()
        else:
            response = self.llm.chat(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_part},
                    {"role": "user", "content": user_part},
                ],
                temperature=0.3,
                max_tokens=2048,
                timeout=60,
            )
            answer = response.choices[0].message.content.strip()

        # Store reasoning result
        context.last_reasoning_result = str(answer)
        context.reasoning_results[node.id] = str(answer)
        context.accumulated_context = str(answer)

        context.add_log(
            "reason",
            node.id,
            f"Reasoning completed (Standard Prompt + PromptBuilder), "
            f"answer length: {len(str(answer))}",
        )

    def _wrap_as_organized(self, evidence: List[Any]) -> OrganizedEvidence:
        """Wrap raw evidence as an OrganizedEvidence for the solver."""
        from agentic.task_types import OrganizedEvidence
        return OrganizedEvidence(
            documents=evidence,
            groups={"all": evidence},
            original_count=len(evidence),
            organized_count=len(evidence),
            organization_method="pass_through",
        )

    def _handle_decide(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute a decide node — evaluate condition and choose branch."""
        params = node.params
        condition = params.get("condition", "default")
        threshold = params.get("threshold", 0.5)

        logger.info(f"[{node.id}] Evaluating condition: {condition}")

        # Evaluate the condition
        decision = self._evaluate_condition(
            condition=condition,
            threshold=threshold,
            context=context,
            graph=graph,
        )

        # Store the decision for get_next_nodes
        context.last_decision = decision
        context.decisions[node.id] = decision

        context.add_log(
            "decide",
            node.id,
            f"Decision: {decision}, "
            f"branches: {dict(node.branches)}",
        )

    def _handle_merge(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute a merge node — combine multiple evidence streams."""
        params = node.params
        merge_strategy = params.get("merge_strategy", "concatenate")

        logger.info(f"[{node.id}] Merging with strategy: {merge_strategy}")

        # Collect all evidence from context
        all_evidence = []
        for evid_list in context.evidence_cache.values():
            if isinstance(evid_list, list):
                all_evidence.extend(evid_list)

        # ===== RECALL-SAFE MERGE (fix) =====
        # Do NOT collapse accumulated_context into the reasoner's LOSSY summary.
        # The Retriever HIT the answer in the original chunks; the Organizer's
        # get_context() keeps every original chunk verbatim. Preserve ALL of it
        # so the second-hop reasoning + final answer can still see them. The
        # reasoner summaries are appended ONLY as supplementary reference.
        evidence_parts: List[str] = []

        # 1) Organizer's full original context (deduplicated, verbatim chunks)
        if context.last_organized is not None and hasattr(
            context.last_organized, "get_context"
        ):
            org_ctx = context.last_organized.get_context()
            if org_ctx:
                evidence_parts.append(
                    "<ORGANIZED_EVIDENCE (original chunks, verbatim)>\n"
                    f"{org_ctx}\n</ORGANIZED_EVIDENCE>"
                )

        # ===== 20260808 S5-P0 修复：merge 去重（消除同 chunk 二次并入膨胀）=====
        # 原实现把 organizer 全量 get_context() + evidence_cache 全部 raw chunks 拼接，
        # 而 organizer 已原样保留第一跳全部去重后 chunk → 同一 chunk 内容输出两遍，
        # accumulated_context 膨胀 ≈2.63×（results/s5_merge_inflate.json，100% 样本>=1.5×）。
        # 修复：raw 阶段按 chunk_id 跳过“已被 organized 覆盖”的 chunk，只保留第二跳新增 chunk。
        # Recall-Safe：organizer 已含第一跳全部去重后 chunk（0 丢失）→ 跳过不丢任何证据；
        # 第二跳新增（retrieve_2 未含于 first-hop organized）的 chunk 仍被保留。
        _covered_ids: set = set()
        if (
            context.last_organized is not None
            and hasattr(context.last_organized, "get_all_documents")
        ):
            try:
                _covered_ids = {
                    getattr(_d, "chunk_id", "") or ""
                    for _d in context.last_organized.get_all_documents()
                }
            except Exception:
                _covered_ids = set()


        # 2) All raw evidence_cache chunks (covers every retrieval hop)
        raw_parts: List[str] = []
        for evid_list in context.evidence_cache.values():
            if not isinstance(evid_list, list):
                continue
            for doc in evid_list:
                content = getattr(doc, "content", "")
                _cid = getattr(doc, "chunk_id", "") or ""
                # S5-P0: 跳过已被 organized 覆盖(chunk_id 命中)的 chunk → 消除同 chunk 二次并入膨胀；
                # 非 organized 覆盖(如第二跳新增)的 chunk 仍保留 → 不丢证据 (Recall-Safe).
                if _cid and _cid in _covered_ids:
                    continue
                if content:
                    raw_parts.append(str(content))
        if raw_parts:
            evidence_parts.append(
                "<RAW_EVIDENCE_CHUNKS (verbatim)>\n"
                + "\n---\n".join(raw_parts)
                + "\n</RAW_EVIDENCE_CHUNKS>"
            )

        # 3) Reasoner summaries — supplementary ONLY
        reasoning_texts = []
        for res in context.reasoning_results.values():
            if res:
                reasoning_texts.append(res)

        base = "\n\n".join(evidence_parts) if evidence_parts else ""
        if reasoning_texts:
            base += (
                "\n\n<REASONER_SUMMARIES (references only; NOT authoritative "
                "replacement for the evidence above)>\n"
                + "\n\n".join(reasoning_texts)
                + "\n</REASONER_SUMMARIES>"
            )
        context.accumulated_context = base

        context.add_log(
            "merge",
            node.id,
            f"Merged {len(all_evidence)} evidence items, "
            f"{len(reasoning_texts)} reasoning results",
        )

    def _handle_verify(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute a verify node — check answer quality."""
        params = node.params
        criteria = params.get("criteria", "answer_quality")

        logger.info(f"[{node.id}] Verifying with criteria: {criteria}")

        # Perform verification using LLM
        verdict = self._verify_answer(
            criteria=criteria,
            question=context.question,
            answer=context.last_reasoning_result or "",
            evidence_count=len(self._get_current_evidence(context)),
        )

        context.last_verdict = verdict
        context.verdicts[node.id] = verdict

        context.add_log(
            "verify",
            node.id,
            f"Verdict: {verdict}",
        )

    def _handle_end(
        self,
        node: GraphNode,
        context: "ExecutionContext",
        graph: ExecutionGraph,
        retriever_kwargs: Dict,
    ) -> None:
        """Execute an end node — terminal."""
        logger.info(f"[{node.id}] Graph execution complete")
        context.execution_complete = True
        context.add_log("end", node.id, "Execution complete")

    def _get_next_nodes(
        self, node: GraphNode, context: "ExecutionContext"
    ) -> List[str]:
        """
        Determine successor nodes based on node type and runtime context.

        - Normal nodes: return next[]
        - decide nodes: evaluate branches{}
        - verify nodes: evaluate branches{}
        - end nodes: return empty
        """
        if node.type == "end":
            return []

        if node.type == "decide":
            decision = context.last_decision
            # Try exact match first
            if decision in node.branches:
                return [node.branches[decision]]
            # Try "yes"/"no" mapping
            if isinstance(decision, bool):
                if decision and "yes" in node.branches:
                    return [node.branches["yes"]]
                if not decision and "no" in node.branches:
                    return [node.branches["no"]]
            if decision is True and "sufficient" in node.branches:
                return [node.branches["sufficient"]]
            if decision is False and "insufficient" in node.branches:
                return [node.branches["insufficient"]]
            # Fallback: take first branch
            if node.branches:
                return [list(node.branches.values())[0]]
            return node.next

        if node.type == "verify":
            verdict = context.last_verdict
            if verdict is True or verdict == "pass":
                if "pass" in node.branches:
                    return [node.branches["pass"]]
            else:
                if "fail" in node.branches:
                    return [node.branches["fail"]]
            # Fallback
            if node.branches:
                return [list(node.branches.values())[0]]
            return node.next

        # Default: follow next[]
        return list(node.next)

    def _evaluate_condition(
        self,
        condition: str,
        threshold: float,
        context: "ExecutionContext",
        graph: ExecutionGraph,
    ) -> str:
        """
        Evaluate a decision condition using LLM.

        Returns the branch key (e.g., "sufficient", "insufficient").
        """
        evidence = self._get_current_evidence(context)
        evidence_str = self._evidence_to_string(evidence)
        evidence_preview = evidence_str[:1500] if len(evidence_str) > 1500 else evidence_str

        prompt = f"""Evaluate the following condition for a RAG system:

CONDITION: {condition}
THRESHOLD: {threshold}

QUESTION: {context.question}

EVIDENCE AVAILABLE:
{evidence_preview}

Respond with ONLY the branch key that applies:
- If evidence is sufficient: "sufficient"
- If evidence is insufficient: "insufficient"
- If yes: "yes"
- If no: "no"
"""

        try:
            # Use chat.completions.create() API (OpenAI-compatible)
            if hasattr(self.llm.chat, "completions"):
                response = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=32,
                    timeout=30,
                )
                decision = response.choices[0].message.content.strip().lower()
            else:
                response = self.llm.chat(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=32,
                    timeout=30,
                )
                decision = response.choices[0].message.content.strip().lower()

            # Normalize
            for valid_key in ["sufficient", "insufficient", "yes", "no"]:
                if valid_key in decision:
                    return valid_key
            return "sufficient"  # Default: proceed
        except Exception:
            return "sufficient"

    def _verify_answer(
        self,
        criteria: str,
        question: str,
        answer: str,
        evidence_count: int,
    ) -> bool:
        """
        Verify the quality of an answer.

        Returns True (pass) or False (fail).
        """
        if not answer or len(answer.strip()) < 10:
            return False
        if evidence_count < 1:
            return False

        prompt = f"""Verify the following answer quality:

CRITERIA: {criteria}

QUESTION: {question}
ANSWER: {answer[:2000]}
EVIDENCE CHUNKS: {evidence_count}

Respond with ONLY:
- "pass" if the answer is complete, well-supported, and directly addresses the question
- "fail" if the answer is incomplete, lacks evidence, or is not well-supported
"""

        try:
            # Use chat.completions.create() API (OpenAI-compatible)
            if hasattr(self.llm.chat, "completions"):
                response = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=32,
                    timeout=30,
                )
                verdict = response.choices[0].message.content.strip().lower()
            else:
                response = self.llm.chat(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=32,
                    timeout=30,
                )
                verdict = response.choices[0].message.content.strip().lower()
            return "pass" in verdict
        except Exception as e:
            logger.warning(f"Verification LLM call failed: {e}")
            return evidence_count >= 2  # Default: pass if we have evidence


    def _produce_final_answer(
        self,
        question: str,
        context: "ExecutionContext",
        graph: ExecutionGraph,
    ) -> Dict[str, Any]:
        """
        Produce the final answer after graph execution.

        If reasoning was already done, use the last reasoning result.
        Otherwise, do a final LLM call.
        """
        reasoning_result = context.last_reasoning_result
        evidence = self._get_current_evidence(context)
        organized = context.last_organized

        # Build evidence string
        evidence_str = ""
        if organized:
            evidence_str = organized.get_context()
        elif evidence:
            evidence_str = self._evidence_to_string(evidence)

        # ===== RECALL-SAFE FINAL ANSWER (fix for dropped truly-hit chunks) =====
        # Do NOT short-circuit on the reasoner's lossy summary. The reasoner
        # summary may have concluded "insufficient evidence" even though the
        # Retriever HIT the answer and the Organizer's full original context
        # (evidence_str) still contains it verbatim. ALWAYS regenerate the final
        # answer from the FULL original chunk text; attach the reasoner summary
        # only as supplementary reference, never as the exclusive basis.
        if reasoning_result and len(reasoning_result) > 50:
            evidence_str = (
                f"{evidence_str}\n\n"
                f"<REASONER_SUMMARY_REFERENCE (attached verbatim; NOT the answer. "
                f"Ground the answer in the evidence above, not this summary)>\n"
                f"{reasoning_result}\n</REASONER_SUMMARY_REFERENCE>"
            )

        # Use Standard Prompt + PromptBuilder for consistent final answer generation
        task_analysis = graph.task_analysis
        from agentic.prompts import format_prompt

        # Build PromptBuilder instruction (if we have task_analysis)
        instruction_extra = ""
        if task_analysis:
            instruction = self.prompt_builder.build(
                task_analysis=task_analysis,
                graph=graph,
                evidence_context=evidence_str,
            )
            instruction_extra = instruction.prompt_extra

        # IMPORTANT (Recall-Safe): pass the FULL evidence, not a [:4000] slice.
        # Slicing here would drop chunks ranked below the character budget even
        # though the Retriever had hit them — the exact failure the user reported.
        # Organizer's get_context() already returns every deduplicated original
        # chunk verbatim; we must not re-drop them at the final answer stage.
        full_prompt = format_prompt(
            task_key="general",
            question=question,
            evidence=evidence_str if evidence_str else "No evidence available.",
            prompt_extra=instruction_extra,
        )
        prompt = full_prompt


        try:
            # Use chat.completions.create() API (OpenAI-compatible)
            if hasattr(self.llm.chat, "completions"):
                response = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=60,
                )
                answer = response.choices[0].message.content.strip()
            else:
                response = self.llm.chat(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=60,
                )
                answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Final answer generation failed: {e}")
            answer = reasoning_result or f"Error generating final answer: {e}"


        return {
            "answer": answer,
            "question": question,
            "task": graph.task.value if graph.task else "general",
            "evidence": evidence_str[:500] if evidence_str else "",
            "evidence_str": evidence_str if evidence_str else "",
            "num_evidence": len(evidence) if evidence else 0,
            "group_count": len(organized.groups) if organized and hasattr(organized, 'groups') else 0,
            "document_count": sum(len(g) for g in organized.groups.values()) if organized and hasattr(organized, 'groups') else (len(evidence) if evidence else 0),
            "evidence_length": len(evidence_str) if evidence_str else 0,
            "prompt_length": len(prompt) if prompt else 0,
        }

    def _get_current_evidence(self, context: "ExecutionContext") -> List[Any]:
        """Get the current evidence from context (most recent first)."""
        # Try organized evidence first
        if context.last_organized:
            return context.last_organized.get_all_documents()

        # Try evidence cache
        if context.evidence_cache:
            for node_id in reversed(list(context.evidence_cache.keys())):
                evid = context.evidence_cache[node_id]
                if evid:
                    return evid if isinstance(evid, list) else [evid]

        # Try last retrieval results
        if context.last_retrieval_results:
            return (
                context.last_retrieval_results
                if isinstance(context.last_retrieval_results, list)
                else [context.last_retrieval_results]
            )

        return []

    def _evidence_to_string(self, evidence: List[Any], max_chars: int = 3000) -> str:
        """Convert evidence list to a context string."""
        parts = []
        char_count = 0

        for i, doc in enumerate(evidence):
            content = getattr(doc, "content", str(doc))
            score = getattr(doc, "score", 0.0)
            citation = getattr(doc, "citation", f"[{i+1}]")

            chunk = f"{citation} (score={score:.3f})\n{content}\n"
            if char_count + len(chunk) > max_chars:
                parts.append(f"{citation} (truncated, ...)")
                break

            parts.append(chunk)
            char_count += len(chunk)

        return "\n---\n".join(parts)


class ExecutionContext:
    """
    Runtime context maintained during graph execution.

    Tracks evidence, reasoning results, decisions, and execution history.
    """

    def __init__(self, question: str):
        self.question = question
        self.evidence_cache: Dict[str, List[Any]] = {}
        self.organized_evidence: Dict[str, Any] = {}
        self.reasoning_results: Dict[str, str] = {}
        self.decisions: Dict[str, str] = {}
        self.verdicts: Dict[str, bool] = {}
        self.execution_log: List[Dict] = []
        self.execution_complete: bool = False

        # Runtime state
        self.last_retrieval_results: Optional[List[Any]] = None
        self.last_organized: Optional[Any] = None
        self.last_reasoning_result: Optional[str] = None
        self.last_decision: Optional[str] = None
        self.last_verdict: Optional[bool] = None
        self.accumulated_context: str = ""

    def add_evidence(self, node_id: str, evidence: Any) -> None:
        """Cache evidence from a retrieve node."""
        if isinstance(evidence, list):
            self.evidence_cache[node_id] = evidence
        else:
            self.evidence_cache[node_id] = [evidence]

    def add_log(self, node_type: str, node_id: str, message: str) -> None:
        """Record an execution log entry."""
        self.execution_log.append({
            "type": node_type,
            "node_id": node_id,
            "message": message,
        })
        logger.debug(f"[{node_type}:{node_id}] {message}")


class AdaptiveAgenticPipeline:
    """
    AdaptiveAgenticPipeline v2 — Entry point for the redesigned agentic RAG system.

    Architecture:
        Question → TaskAnalyzer → TaskAnalysis
            ↓
        StrategyPlanner → ExecutionGraph
            ↓
        GraphExecutor (walks graph, dispatches nodes)
            ↓
        Answer

    Attributes:
        analyzer: TaskAnalyzer instance
        planner: StrategyPlanner instance
        executor: GraphExecutor instance
        config: Pipeline configuration
    """

    def __init__(
        self,
        analyzer: Any,
        planner: Any,
        retriever: Any,
        organizer: Any,
        solver: Any,
        llm_client: Any,
        config: Optional[Dict] = None,
        semantic_rewriter: Any = None,
    ):
        """
        Initialize the pipeline.

        Args:
            analyzer: TaskAnalyzer instance
            planner: StrategyPlanner instance
            retriever: Retrieval module
            organizer: EvidenceOrganizer instance
            solver: TaskSolver instance
            llm_client: LLM client
            config: Additional configuration
            semantic_rewriter: Optional SemanticQueryRewriter injected into the executor.
                Enables the per-node `semantic_mv` production retrieve switch (P1 语义多视角).
        """
        self.analyzer = analyzer
        self.planner = planner
        self.executor = GraphExecutor(
            retriever_module=retriever,
            organizer_module=organizer,
            solver_module=solver,
            llm_client=llm_client,
            semantic_rewriter=semantic_rewriter,
        )
        self.config = config or {}

    def run(
        self,
        question: str,
        retriever_kwargs: Optional[Dict] = None,
        pre_retrieved_evidence: Optional[List[Any]] = None,
        format: str = "",
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on a question.

        When pre_retrieved_evidence is provided, the retriever operates as a
        Stable Knowledge Interface — all retrieve nodes are skipped, and the
        externally-retrieved evidence is used directly by organize/reason nodes.

        Args:
            question: The question to answer
            retriever_kwargs: Additional kwargs for the retriever
            pre_retrieved_evidence: Pre-retrieved documents (skip internal retrieve)
            format: The question format / 题型 truth value (e.g. CSV `_format`: 问答题 / QA).
                    Passed straight through to TaskAnalyzer.analyze(question, format=format)
                    so the front-end task determination reads the ground-truth format
                    directly (normalized onto TaskAnalysis.format) instead of falling
                    back to the heuristic. Empty string keeps the existing heuristic path.

        Returns:
            Dict with final answer and execution metadata
        """
        # Step 1: Analyze the task (format truth threaded through when provided)
        task_analysis = self.analyzer.analyze(question, format=format)
        logger.info(
            f"Task Analysis: {task_analysis.task.value} "
            f"(conf={task_analysis.confidence:.2f})"
        )

        # Step 2: Generate execution graph
        graph = self.planner.plan(task_analysis)
        logger.info(
            f"Execution Graph: {len(graph.nodes)} nodes, "
            f"entry: {graph.entry_points}"
        )

        # Step 3: Execute the graph
        result = self.executor.execute(
            graph=graph,
            question=question,
            retriever_kwargs=retriever_kwargs,
            pre_retrieved_evidence=pre_retrieved_evidence,
        )

        # Add metadata
        result["task_analysis"] = task_analysis.to_dict()
        result["pipeline_version"] = "v2-execution-graph"
        result["used_external_retrieval"] = pre_retrieved_evidence is not None
        result["second_retrieval_triggers"] = self.executor.second_retrieval_count

        return result

    def run_with_ablation(
        self,
        question: str,
        ablation_config: Dict[str, bool],
        retriever_kwargs: Optional[Dict] = None,
        pre_retrieved_evidence: Optional[List[Any]] = None,
        format: str = "",
    ) -> Dict[str, Any]:
        """
        Run with specific adaptive components enabled/disabled.

        When pre_retrieved_evidence is provided, the retriever operates as a
        Stable Knowledge Interface — all retrieve nodes are skipped, and the
        externally-retrieved evidence is used directly by organize/reason nodes.

        Args:
            question: The question to answer
            ablation_config: Dict of component booleans, e.g.:
                {"adaptive_retrieval": True, "adaptive_organize": False, ...}
            retriever_kwargs: Additional kwargs for the retriever
            pre_retrieved_evidence: Pre-retrieved documents (skip internal retrieve)
            format: The question format / 题型 truth value (e.g. CSV `_format`).
                    Passed through to TaskAnalyzer.analyze(question, format=format) so the
                    front-end determination reads the ground-truth format directly.

        Returns:
            Dict with answer and ablation metadata
        """
        # Generate full graph (format truth threaded through when provided)
        task_analysis = self.analyzer.analyze(question, format=format)
        graph = self.planner.plan(task_analysis)

        # If adaptive retrieval disabled, force all retrieve nodes to use defaults
        if not ablation_config.get("adaptive_retrieval", True):
            for node in graph.nodes.values():
                if node.type == "retrieve":
                    node.params = {
                        "retrieve_k": 10,
                        "multi_query": False,
                        "use_ked": True,
                        "use_fusion": False,
                    }

        # If adaptive organize disabled, use topic-based organization
        if not ablation_config.get("adaptive_organize", True):
            for node in graph.nodes.values():
                if node.type == "organize":
                    node.params = {
                        "organize_by": "topic",
                        "remove_duplicate": True,
                    }

        # If adaptive reasoning disabled, use simple general reasoning
        if not ablation_config.get("adaptive_reasoning", True):
            for node in graph.nodes.values():
                if node.type == "reason":
                    node.params = {
                        "reasoning_type": "general",
                        "workflow_steps": [
                            "Understand the question",
                            "Synthesize an answer from evidence",
                            "Provide the final answer with citations",
                        ],
                        "requires_citation": True,
                    }

        # If conditional branching disabled, remove decide and verify nodes
        if not ablation_config.get("conditional_branching", True):
            to_remove = []
            for nid, node in graph.nodes.items():
                if node.type in ("decide", "verify"):
                    to_remove.append(nid)
                    # Rewire: connect predecessor to successor
                    for other_node in graph.nodes.values():
                        if node.id in other_node.next:
                            idx = other_node.next.index(node.id)
                            other_node.next[idx:idx+1] = node.next

            # Also remove merge nodes if no branching
            if not ablation_config.get("multi_hop_retrieval", True):
                for nid, node in graph.nodes.items():
                    if node.type == "merge":
                        to_remove.append(nid)
                        for other_node in graph.nodes.values():
                            if node.id in other_node.next:
                                idx = other_node.next.index(node.id)
                                other_node.next[idx:idx+1] = node.next

            for nid in to_remove:
                if nid in graph.nodes:
                    del graph.nodes[nid]

        # Execute the modified graph
        result = self.executor.execute(
            graph=graph,
            question=question,
            retriever_kwargs=retriever_kwargs,
            pre_retrieved_evidence=pre_retrieved_evidence,
        )

        result["task_analysis"] = task_analysis.to_dict()
        result["ablation_config"] = ablation_config
        result["pipeline_version"] = "v2-execution-graph"
        result["used_external_retrieval"] = pre_retrieved_evidence is not None
        result["second_retrieval_triggers"] = self.executor.second_retrieval_count

        return result


def PipelineResult(result) -> Any:
    """
    Factory function creating a PipelineResult-like object.

    Accepts both Dict and SimpleNamespace (or any object with .__dict__).
    Maintains backward compatibility with code that expects a PipelineResult
    object with .answer, .task_analysis, .task attributes.

    Args:
        result: Dict or object with pipeline execution results

    Returns:
        SimpleNamespace with answer, task_analysis, task, etc.
    """
    from types import SimpleNamespace

    # If it's already an object with .answer (e.g. SimpleNamespace), just pass through
    if not isinstance(result, dict):
        if hasattr(result, "answer"):
            return result
        # Try to convert via __dict__ or vars()
        try:
            result = vars(result)
        except TypeError:
            result = {}

    obj = SimpleNamespace()
    obj.answer = result.get("answer", "") if isinstance(result, dict) else ""
    obj.task_analysis = result.get("task_analysis", {}) if isinstance(result, dict) else {}
    obj.task = result.get("task", "general") if isinstance(result, dict) else "general"
    obj.execution_graph = result.get("execution_graph", {}) if isinstance(result, dict) else {}
    obj.execution_nodes = result.get("execution_nodes", []) if isinstance(result, dict) else []
    obj.node_count = result.get("node_count", 0) if isinstance(result, dict) else 0
    obj.result_dict = result
    return obj
