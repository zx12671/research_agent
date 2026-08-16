"""
planner.py：StrategyPlanner v2 —— 生成 ExecutionGraph（执行工作流 DAG，有向无环图）。

重新设计 v2（ExecutionGraph）：
    StrategyPlanner 不再生成一个扁平化的执行计划，而是生成一个真正的
    工作流图（Workflow Graph）。该图可以包含：

        - 顺序执行链：retrieve → organize → reason
          （检索 → 整理 → 推理）

        - 条件分支：decide → (retry | proceed)
          （决策 →（重试｜继续执行））

        - 多跳检索：retrieve → check → retrieve_again
          （检索 → 检查 → 再次检索）

        - 验证循环：reason → verify → (pass | fail)
          （推理 → 验证 →（通过｜失败））

    整体架构：
        TaskAnalysis → StrategyPlanner → ExecutionGraph → GraphExecutor → Answer
        （任务分析 → 策略规划器 → 执行图 → 图执行器 → 最终答案）

    与 v1（ExecutionPlan）的核心区别：

        v1：ExecutionPlan
            使用一个扁平化的独立执行指令列表：
                [retrieval, organization, reasoning]
                （检索、组织、推理）

        v2：ExecutionGraph
            使用由不同类型 GraphNode（图节点）组成的有向执行图：

                retrieve_1 → organize_1 → decide_sufficient
                                              ├─ yes → reason_1 → verify_1 → end
                                              └─ no  → retrieve_2 → merge_1 ─┘

            即：

                第一次检索
                    ↓
                信息整理
                    ↓
                判断信息是否充足
                 ├── 是 → 推理 → 验证 → 结束
                 └── 否 → 第二次检索 → 信息合并 ───────┘
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional

from agentic.task_types import (
    TaskType,
    TaskAnalysis,
    GraphNode,
    ExecutionGraph,
)

logger = logging.getLogger(__name__)


class StrategyPlanner:
    """
    StrategyPlanner v2: Generates an ExecutionGraph from TaskAnalysis.

    Uses an LLM to dynamically generate a workflow graph with typed nodes.
    Falls back to template-based graphs if LLM fails.
    """

    def __init__(self, llm_client: Any, model_name: str = "deepseek-chat",
                 retrieve_hybrid: bool = False,
                 retrieve_semantic_mv: bool = False):
        """
        Initialize the StrategyPlanner.

        Args:
            llm_client: LLM client for dynamic planning
            model_name: Name of the model to use
            retrieve_hybrid: 是否默认对模板图的 retrieve 节点启用 hybrid
                            (dense + BM25 → RRF，单查询)。默认 False。
                            仅影响 _fallback_graph 的默认参数；LLM 自定义图不受影响。
            retrieve_semantic_mv: 是否默认对模板图高级任务的 retrieve 节点启用
                            LLM 语义级多视角改写（P1 主杠杆；需 executor 注入 semantic_rewriter）。
                            默认 False。为 True 时高级任务的 retrieve_1 改走 semantic_mv 分支，
                            并关闭机械切逗号 multi_query（避免叠加稀释）。零侵入现状。
        """
        self.llm = llm_client
        self.model_name = model_name
        self._default_retrieve_hybrid = bool(retrieve_hybrid)
        self._default_retrieve_semantic_mv = bool(retrieve_semantic_mv)

    def plan(self, task_analysis: TaskAnalysis) -> ExecutionGraph:
        """
        Generate an ExecutionGraph from the task analysis.

        Uses LLM to dynamically construct a workflow graph.
        Falls back to template-based graphs on failure.

        Args:
            task_analysis: Rich semantic description from TaskAnalyzer

        Returns:
            ExecutionGraph with typed, connected nodes
        """
        task = task_analysis.task
        question = task_analysis.original_question

        # Try LLM-based graph generation
        try:
            graph_dict = self._llm_generate_graph(task_analysis)
            if graph_dict and self._validate_graph_dict(graph_dict):
                return self._build_execution_graph(graph_dict, task_analysis)
        except Exception as e:
            logger.warning(f"LLM graph generation failed: {e}")

        # Fallback: generate template-based graph
        return self._fallback_graph(task_analysis)

    def _llm_generate_graph(self, task_analysis: TaskAnalysis) -> Optional[Dict]:
        """
        Use LLM to dynamically generate a workflow graph.

        Args:
            task_analysis: The task analysis

        Returns:
            Parsed graph dict or None
        """
        prompt = self._build_graph_prompt(task_analysis)

        try:
            # OpenAI client: llm.chat.completions.create(...)
            if hasattr(self.llm.chat, "completions"):
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                    timeout=60,
                )
                content = response.choices[0].message.content.strip()
            else:
                response = self.llm.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                    timeout=60,
                )
                content = response.choices[0].message.content.strip()
            return self._parse_graph_output(content)
        except Exception as e:
            logger.error(f"LLM graph generation error: {e}")
            return None


    def _build_graph_prompt(self, task_analysis: TaskAnalysis) -> str:
        """
        Build the prompt for LLM graph generation.

        The LLM must output a JSON graph structure with typed nodes
        connected via next[] and branches{}.
        """
        task = task_analysis.task.value
        question = task_analysis.original_question

        return f"""You are designing an adaptive retrieval-augmented generation workflow as a DIRECTED GRAPH.

TASK: {task}
QUESTION: {question}

TASK ANALYSIS:
- Confidence: {task_analysis.confidence:.2f}
- Complexity: {task_analysis.reasoning_complexity}
- Multi-source required: {task_analysis.requires_multi_source}
- Formula required: {task_analysis.requires_formula}
- Step reasoning: {task_analysis.requires_step_reasoning}
- Preferred source: {task_analysis.preferred_source}

NODE TYPES AVAILABLE:
1. "retrieve"  — Retrieves evidence from knowledge base
   params: retrieve_k (int), multi_query (bool), use_fusion (bool), use_ked (bool),
           source_priority (str), min_score (float)

2. "organize"  — Organizes/group evidence
   params: organize_by (str: "comparison"|"symptom"|"candidate"|"formula"|"chronological"|"topic"),
           remove_duplicate (bool)

3. "reason"    — Reasoning step (LLM call)
   params: reasoning_type (str), workflow_steps (list[str]),
           prompt_style (str), requires_citation (bool)

4. "decide"    — Conditional decision (branch)
   params: condition (str), threshold (float)
   branches: {{"yes": "node_id", "no": "node_id"}}

5. "merge"     — Merge multiple evidence streams
   params: merge_strategy (str: "concatenate"|"interleave"|"priority")

6. "verify"    — Verify/validate step
   params: criteria (str)
   branches: {{"pass": "node_id", "fail": "node_id"}}

7. "end"       — Terminal node (no next/branches)

OUTPUT FORMAT (JSON only, no explanation):
```json
{{
  "entry_point": "node_id_of_first_node",
  "nodes": [
    {{
      "id": "retrieve_1",
      "type": "retrieve",
      "action": "semantic_search",
      "params": {{
        "retrieve_k": 15,
        "multi_query": true,
        "use_fusion": true
      }},
      "next": ["organize_1"],
      "description": "Initial retrieval with multi-query expansion"
    }},
    {{
      "id": "organize_1",
      "type": "organize",
      "action": "group_by_comparison",
      "params": {{
        "organize_by": "comparison",
        "remove_duplicate": true
      }},
      "next": ["decide_sufficient"],
      "description": "Group evidence by compared objects"
    }},
    {{
      "id": "decide_sufficient",
      "type": "decide",
      "action": "check_evidence_sufficiency",
      "params": {{
        "condition": "is_evidence_sufficient",
        "threshold": 0.6
      }},
      "branches": {{
        "sufficient": "reason_1",
        "insufficient": "retrieve_2"
      }},
      "description": "Check if we have enough evidence"
    }},
    {{
      "id": "retrieve_2",
      "type": "retrieve",
      "action": "targeted_retrieval",
      "params": {{
        "retrieve_k": 10,
        "multi_query": false,
        "use_fusion": false
      }},
      "next": ["merge_1"],
      "description": "Supplementary retrieval for missing evidence"
    }},
    {{
      "id": "merge_1",
      "type": "merge",
      "action": "merge_evidence_streams",
      "params": {{
        "merge_strategy": "concatenate"
      }},
      "next": ["reason_1"],
      "description": "Merge initial and supplementary evidence"
    }},
    {{
      "id": "reason_1",
      "type": "reason",
      "action": "execute_reasoning_workflow",
      "params": {{
        "reasoning_type": "compare",
        "workflow_steps": [
          "Identify the specific items being compared",
          "Extract evidence for each item",
          "Analyze similarities",
          "Analyze differences",
          "Generate conclusion with recommendation"
        ],
        "requires_citation": true
      }},
      "next": ["verify_1"],
      "description": "Execute comparison reasoning pipeline"
    }},
    {{
      "id": "verify_1",
      "type": "verify",
      "action": "verify_answer_quality",
      "params": {{
        "criteria": "answer_completeness_and_citation"
      }},
      "branches": {{
        "pass": "end",
        "fail": "retrieve_2"
      }},
      "description": "Verify answer quality, retry if insufficient"
    }},
    {{
      "id": "end",
      "type": "end",
      "action": "complete",
      "description": "Terminal node"
    }}
  ]
}}
```

DESIGN RULES:
1. Every graph MUST have at least one retrieve node and one end node
2. The end node type must be "end" (terminal)
3. All node IDs must be unique
4. All next/branches references must point to valid node IDs
5. Each node type must have valid params for that type
6. decide nodes MUST have branches (at least 2)
7. verify nodes MUST have "pass" and "fail" branches
8. Keep the graph DAG (no cycles except verify→retrieve)
9. For simple tasks, use a minimal graph (3-5 nodes)
10. For complex tasks, use richer graphs with decisions and verification

Now design the graph for this {task} task. Respond with ONLY valid JSON."""

    def _parse_graph_output(self, content: str) -> Optional[Dict]:
        """Parse LLM output to extract graph JSON."""
        # Try to extract JSON from markdown code block
        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL
        )
        if json_match:
            content = json_match.group(1).strip()

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object
        brace_match = re.search(r'\{[\s\S]*\}', content)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _validate_graph_dict(self, graph_dict: Dict) -> bool:
        """Validate the LLM-generated graph structure."""
        if "nodes" not in graph_dict or not isinstance(graph_dict["nodes"], list):
            return False
        if len(graph_dict["nodes"]) < 2:
            return False

        node_ids = set()
        has_end = False
        has_retrieve = False

        for node in graph_dict["nodes"]:
            # Check required fields
            if "id" not in node or "type" not in node:
                return False
            nid = node["id"]
            ntype = node["type"]

            # Check unique IDs
            if nid in node_ids:
                return False
            node_ids.add(nid)

            # Check end node
            if ntype == "end":
                has_end = True

            # Check retrieve node
            if ntype == "retrieve":
                has_retrieve = True

            # Check references
            for ref in node.get("next", []):
                if ref not in graph_dict.get("_all_ids", set()):
                    pass  # We'll validate after building full id set

            # Decide nodes must have branches
            if ntype == "decide" and not node.get("branches"):
                return False

            # Verify nodes must have pass/fail
            if ntype == "verify":
                branches = node.get("branches", {})
                if "pass" not in branches or "fail" not in branches:
                    return False

        # Build all_ids for ref validation
        all_ids = set()
        for node in graph_dict["nodes"]:
            all_ids.add(node["id"])
        graph_dict["_all_ids"] = all_ids

        # Validate all references
        for node in graph_dict["nodes"]:
            for ref in node.get("next", []):
                if ref not in all_ids:
                    return False
            for branch_id in node.get("branches", {}).values():
                if branch_id not in all_ids:
                    return False

        return has_end and has_retrieve

    def _build_execution_graph(
        self, graph_dict: Dict, task_analysis: TaskAnalysis
    ) -> ExecutionGraph:
        """
        Build an ExecutionGraph from parsed dict.

        Args:
            graph_dict: Parsed graph JSON from LLM
            task_analysis: Original task analysis

        Returns:
            ExecutionGraph with GraphNodes
        """
        nodes: Dict[str, GraphNode] = {}

        for node_data in graph_dict["nodes"]:
            node = GraphNode(
                id=node_data["id"],
                type=node_data["type"],
                action=node_data.get("action", ""),
                params=node_data.get("params", {}),
                next=node_data.get("next", []),
                branches=node_data.get("branches", {}),
                fallback=node_data.get("fallback"),
                description=node_data.get("description", ""),
            )
            nodes[node.id] = node

        entry_point = graph_dict.get("entry_point", "")
        entry_points = [entry_point] if entry_point else []

        # If no entry point specified, find nodes with no incoming edges
        if not entry_points:
            all_ids = set(nodes.keys())
            has_incoming = set()
            for node in nodes.values():
                for nid in node.next:
                    has_incoming.add(nid)
                for branch_id in node.branches.values():
                    has_incoming.add(branch_id)
                if node.fallback:
                    has_incoming.add(node.fallback)
            entry_points = list(all_ids - has_incoming)
            # Default to first node if no clean entry found
            if not entry_points:
                entry_points = [list(nodes.keys())[0]]

        # Collect task-specific hints from reason nodes
        hints_parts = []
        reason_nodes = [n for n in nodes.values() if n.type == "reason"]
        for rn in reason_nodes[:2]:
            steps = rn.params.get("workflow_steps", [])
            if steps:
                hints_parts.append(f"Reasoning ({rn.id}): " + " → ".join(steps[:5]))
        hints = "\n".join(hints_parts)

        return ExecutionGraph(
            task=task_analysis.task,
            task_analysis=task_analysis,
            nodes=nodes,
            entry_points=entry_points,
            task_specific_hints=hints,
        )

    def _fallback_graph(self, task_analysis: TaskAnalysis) -> ExecutionGraph:
        """
        Generate a template-based ExecutionGraph when LLM fails.

        These are task-specific workflow graphs with conditional
        branching and verification built in.

        Args:
            task_analysis: The task analysis

        Returns:
            A well-structured ExecutionGraph
        """
        task = task_analysis.task

        # --- Build base nodes common to most tasks ---
        base_retrieve_params = {
            "retrieve_k": 10,
            "multi_query": False,
            "use_ked": True,
            "use_fusion": False,
            "source_priority": "standard",
            "hybrid": self._default_retrieve_hybrid,  # dense + BM25 → RRF (单查询)
            "hybrid_sparse_pool": 50,
            "hybrid_dense_weight": 1.0,
            "hybrid_sparse_weight": 0.2,
            # [PRODUCTION] P1 语义级多视角改写（需 executor 注入 semantic_rewriter）：
            # 默认关，随 _default_retrieve_semantic_mv 开启；开启时高级任务 retrieve_1
            # 走 semantic_mv 分支（replace 机械切逗号）。mv 参数与 pipeline 兜底一致。
            "semantic_mv": self._default_retrieve_semantic_mv,
            "mv_max_views": 4,
            "mv_sparse_weight": 0.2,
        }

        base_organize_params = {
            "organize_by": "topic",
            "remove_duplicate": True,
        }

        base_reasoning_steps = [
            "Understand the question",
            "Extract relevant evidence",
            "Synthesize an answer from evidence",
            "Verify completeness and accuracy",
            "Provide the final answer with citations",
        ]

        # --- Task-specific configurations ---
        task_configs = {
            TaskType.COMPARISON: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 15, "multi_query": True, "use_fusion": True},
                "organize": {**base_organize_params, "organize_by": "comparison"},
                "reasoning_type": "compare",
                "steps": [
                    "Identify the specific items being compared",
                    "Extract evidence for each item from the documents",
                    "Analyze similarities between the items",
                    "Analyze differences between the items",
                    "Draw a balanced conclusion with recommendation",
                ],
            },
            TaskType.DIAGNOSIS: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 20, "multi_query": True},
                "organize": {**base_organize_params, "organize_by": "symptom"},
                "reasoning_type": "diagnosis",
                "steps": [
                    "Identify symptoms from the question",
                    "Extract possible root causes from evidence",
                    "Match evidence to each possible cause",
                    "Determine the most likely cause",
                    "Generate remediation recommendation",
                ],
            },
            TaskType.CALCULATION: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 5, "multi_query": False},
                "organize": {**base_organize_params, "organize_by": "formula"},
                "reasoning_type": "calculation",
                "steps": [
                    "Extract variables and parameters from the question",
                    "Identify applicable formulas from evidence",
                    "Perform step-by-step calculation",
                    "Verify units and reasonableness of the result",
                    "State final answer with interpretation",
                ],
            },
            TaskType.SELECTION: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 12, "multi_query": True, "use_fusion": True},
                "organize": {**base_organize_params, "organize_by": "candidate"},
                "reasoning_type": "selection",
                "steps": [
                    "Identify requirements and constraints from the question",
                    "List candidate options from evidence",
                    "Evaluate each candidate against the requirements",
                    "Perform trade-off analysis",
                    "Make recommendation with justification",
                ],
            },
            TaskType.PROCEDURE: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 8, "multi_query": False},
                "organize": {**base_organize_params, "organize_by": "chronological"},
                "reasoning_type": "procedure",
                "steps": [
                    "Identify the procedure or process being described",
                    "Identify prerequisites and safety measures",
                    "Describe each step in the correct sequence",
                    "Highlight critical parameters and quality checks",
                    "Summarize the complete procedure",
                ],
            },
            TaskType.STANDARD_INTERPRETATION: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 10, "multi_query": False},
                "organize": {**base_organize_params, "organize_by": "topic"},
                "reasoning_type": "interpret",
                "steps": [
                    "Identify the applicable standards",
                    "Extract key requirements from evidence",
                    "Interpret each requirement in practical terms",
                    "Note compliance guidance",
                    "Provide practical examples",
                ],
            },
            TaskType.EXPLANATION: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 10, "multi_query": False},
                "organize": {**base_organize_params, "organize_by": "topic"},
                "reasoning_type": "explain",
                "steps": [
                    "Define the concept or principle",
                    "Explain the underlying mechanism",
                    "Provide industrial context and applications",
                    "Give concrete examples",
                    "Summarize key takeaways",
                ],
            },
            TaskType.GENERAL: {
                "retrieve": {**base_retrieve_params, "retrieve_k": 10, "multi_query": False},
                "organize": {**base_organize_params, "organize_by": "topic"},
                "reasoning_type": "general",
                "steps": base_reasoning_steps,
            },
        }

        cfg = task_configs.get(task, task_configs[TaskType.GENERAL])

        # Build the graph nodes
        nodes: Dict[str, GraphNode] = {}

        # Determine whether to use advanced graph (with decide + verify) or simple graph
        # Advanced graph for: comparison, selection, diagnosis, multi-hop, or high confidence
        task_str = task.value if task else "general"
        use_advanced = task_str in ["comparison", "selection", "diagnosis", "standard_interpretation"]

        # ===== Node 1: Retrieve =====
        if use_advanced:
            # For advanced tasks, use broader coverage.
            # 默认：multi_query + fusion（机械切逗号分解，兼容现状）。
            # 若开启 retrieve_semantic_mv：改用 P1 语义级多视角改写（semantic_mv=True），
            # 并【关闭机械 multi_query】避免"语义视角 × 切逗号"叠加稀释原 query 信号
            # （语义改写已含原 query 首保底 + 各视角 semantic 互补，无需再切逗号）。
            advanced_retrieve_params = dict(cfg["retrieve"])
            if self._default_retrieve_semantic_mv:
                advanced_retrieve_params["multi_query"] = False
                advanced_retrieve_params["use_fusion"] = False
                advanced_retrieve_params["semantic_mv"] = True
            else:
                advanced_retrieve_params["multi_query"] = True
                advanced_retrieve_params["use_fusion"] = True
            advanced_retrieve_params["retrieve_k"] = max(cfg["retrieve"].get("retrieve_k", 10), 15)
        else:
            advanced_retrieve_params = cfg["retrieve"]

        nodes["retrieve_1"] = GraphNode(
            id="retrieve_1",
            type="retrieve",
            action="semantic_search",
            params=advanced_retrieve_params if use_advanced else cfg["retrieve"],
            next=["organize_1"],
            description=f"Initial retrieval for {task.value} task",
        )

        # ===== Node 2: Organize =====
        nodes["organize_1"] = GraphNode(
            id="organize_1",
            type="organize",
            action="group_evidence",
            params=cfg["organize"],
            next=["reason_1"] if not use_advanced else ["decide_1"],
            description=f"Task-aware evidence organization by {cfg['organize']['organize_by']}",
        )

        if use_advanced:
            # ===== Node 3: Decide — evidence sufficiency check =====
            nodes["decide_1"] = GraphNode(
                id="decide_1",
                type="decide",
                action="check_evidence_sufficiency",
                params={
                    "condition": "is_evidence_sufficient",
                    "threshold": 0.5,
                },
                branches={
                    "sufficient": "reason_1",
                    "insufficient": "retrieve_2",
                },
                description="Check if evidence is sufficient for answering",
            )

            # ===== Supplementary Retrieve =====
            supplementary_params = dict(advanced_retrieve_params)
            supplementary_params["retrieve_k"] = 10
            if self._default_retrieve_semantic_mv:
                # 语义多视角开启：第二跳 keep semantic_mv，不退回机械切逗号
                supplementary_params["semantic_mv"] = True
                supplementary_params["multi_query"] = False
                supplementary_params["use_fusion"] = False
            else:
                supplementary_params["multi_query"] = True
                supplementary_params["use_fusion"] = True
            nodes["retrieve_2"] = GraphNode(
                id="retrieve_2",
                type="retrieve",
                action="targeted_retrieval",
                params={
                    "retrieve_k": 10,
                    "multi_query": True,
                    "use_fusion": True,
                    "use_ked": True,
                    "source_priority": "standard",
                    "targeted": True,
                    # Evidence-guided follow-up: build the 2nd query from the
                    # ORIGINAL retrieved chunks' discriminating keys (e.g. the
                    # standard number "GB/T 20476", the parameter "65℃"), NOT from
                    # the reasoner's lossy summary. See GraphExecutor._handle_retrieve.
                    "evidence_guided": True,
                },
                next=["merge_1"],
                description="Supplementary retrieval for missing evidence",

            )

            # ===== Merge =====
            nodes["merge_1"] = GraphNode(
                id="merge_1",
                type="merge",
                action="merge_evidence_streams",
                params={"merge_strategy": "concatenate"},
                next=["reason_1"],
                description="Merge initial and supplementary evidence",
            )

            # ===== Reason =====
            nodes["reason_1"] = GraphNode(
                id="reason_1",
                type="reason",
                action="execute_reasoning_workflow",
                params={
                    "reasoning_type": cfg["reasoning_type"],
                    "workflow_steps": cfg["steps"],
                    "requires_citation": True,
                },
                next=["verify_1"],
                description=f"Execute {cfg['reasoning_type']} reasoning workflow",
            )

            # ===== Verify =====
            nodes["verify_1"] = GraphNode(
                id="verify_1",
                type="verify",
                action="verify_answer_quality",
                params={
                    "criteria": "answer_completeness_and_citation",
                },
                branches={
                    "pass": "end",
                    "fail": "retrieve_2",
                },
                description="Verify answer quality, retry if insufficient",
            )

            # ===== End =====
            nodes["end"] = GraphNode(
                id="end",
                type="end",
                action="complete",
                description="Terminal node",
            )

            entry_points = ["retrieve_1"]
        else:
            # Simple graph: retrieve → organize → reason → end
            # ===== Reason =====
            nodes["reason_1"] = GraphNode(
                id="reason_1",
                type="reason",
                action="execute_reasoning_workflow",
                params={
                    "reasoning_type": cfg["reasoning_type"],
                    "workflow_steps": cfg["steps"],
                    "requires_citation": True,
                },
                next=["end"],
                description=f"Execute {cfg['reasoning_type']} reasoning workflow",
            )

            # ===== End =====
            nodes["end"] = GraphNode(
                id="end",
                type="end",
                action="complete",
                description="Terminal node",
            )

            entry_points = ["retrieve_1"]

        hints = f"Reasoning: " + " → ".join(cfg["steps"])

        return ExecutionGraph(
            task=task,
            task_analysis=task_analysis,
            nodes=nodes,
            entry_points=entry_points,
            task_specific_hints=hints,
        )
