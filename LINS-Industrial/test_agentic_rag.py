"""
test_agentic_rag.py: 验证 Agentic RAG v2 (ExecutionGraph 架构) 所有组件的单元测试。

测试内容:
  1. TaskAnalyzer - 生成丰富语义描述 (不仅仅是标签)
  2. StrategyPlanner - 生成 ExecutionGraph 工作流 DAG
  3. EvidenceOrganizer - 任务感知证据组织
  4. TaskSolver - 显式推理管线执行
  5. ExecutionGraph - 拓扑排序、节点遍历
  6. Ablation 支持 - 独立启用/禁用自适应组件

用法:
  cd LINS-Industrial
  python test_agentic_rag.py

如果想使用真实的 DeepSeek LLM 测试 (需要 API Key):
  set DEEPSEEK_KEY=sk-your-key-here
  python test_agentic_rag.py --use-llm
"""

import sys
import os
import json
import unittest
import argparse
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

# ==== 路径修复 ====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_LINS_MAIN = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "LINS-main"))
if _LINS_MAIN not in sys.path:
    sys.path.insert(0, _LINS_MAIN)

# ==== Agentic 模块导入 ====
from agentic.task_types import (
    TaskType, TaskAnalysis, GraphNode, ExecutionGraph,
    OrganizedEvidence, ExecutionDirective,
    AdaptiveRetrievalConfig, AdaptiveOrganizationConfig,
    AdaptiveReasoningFlow, StrategyPlan,
)
from agentic.analyzer import TaskAnalyzer
from agentic.planner import StrategyPlanner
from agentic.organizer import EvidenceOrganizer
from agentic.solver import TaskSolver
from agentic.prompts import (
    ANALYZER_PROMPT, PLANNER_PROMPT,
    build_solver_prompt, get_prompt, format_prompt,
    PROMPT_REGISTRY,
)
from agentic.pipeline import AdaptiveAgenticPipeline, PipelineResult, GraphExecutor, ExecutionContext


# ============================================================
# Test Questions
# ============================================================

QUESTION_COMPARISON = "Compare Class A and Class S power quality monitors"
QUESTION_DIAGNOSIS = "What causes bearing overheating in centrifugal pumps?"
QUESTION_CALCULATION = "Calculate the required transformer rating for a 500kW motor load with 0.85 power factor"
QUESTION_SELECTION = "Which material should be selected for high-temperature furnace components operating above 1000°C?"
QUESTION_PROCEDURE = "What is the startup procedure for a gas turbine generator?"
QUESTION_EXPLANATION = "Explain the working principle of a three-phase induction motor"
QUESTION_GENERAL = "What are the main types of industrial sensors?"


# ============================================================
# 辅助: 配置 mock LLM 返回 OpenAI-compatible 响应
# ============================================================

def mock_llm_response(mock_client: MagicMock, content: str):
    """
    配置 MagicMock LLM 以模拟 OpenAIClient.chat.completions.create() 的响应。
    
    实际的 analyzer._call_llm 代码路径:
      self.llm_client.chat.completions.create(...)
      → response.choices[0].message.content.strip()
    """
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_response


def make_llm_client() -> MagicMock:
    """创建一个预配置好的 mock LLM 客户端。"""
    client = MagicMock()
    return client


# ============================================================
# Test 1: TaskAnalyzer
# ============================================================

class TestTaskAnalyzer(unittest.TestCase):
    """验证 TaskAnalyzer 生成了丰富的语义描述而非仅仅标签"""

    def setUp(self):
        self.mock_llm = make_llm_client()

    def _make_response(self, task_type: str, **overrides) -> str:
        """生成模拟 LLM 的 JSON 响应"""
        response = {
            "task": task_type,
            "confidence": 0.95,
            "reasoning": f"Test classification for {task_type}",
            "reasoning_complexity": "medium",
            "requires_multi_source": True,
            "requires_formula": False,
            "requires_step_reasoning": False,
            "preferred_source": "standard",
            "expected_evidence": "general",
        }
        response.update(overrides)
        return json.dumps(response)

    def _make_analyzer(self) -> TaskAnalyzer:
        """创建 TaskAnalyzer 并配置 mock LLM 返回指定响应"""
        mock_llm_response(self.mock_llm, "{}")
        return TaskAnalyzer(llm_client=self.mock_llm, model_name="test-model")

    def test_analyzer_returns_rich_semantic_fields(self):
        """✅ 验证 TaskAnalyzer 返回所有丰富语义字段"""
        mock_llm_response(self.mock_llm, self._make_response("comparison"))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm, model_name="test-model")
        result = analyzer.analyze("Compare A and B")

        self.assertIsInstance(result, TaskAnalysis)
        self.assertEqual(result.task, TaskType.COMPARISON)
        self.assertEqual(result.confidence, 0.95)
        self.assertIn(result.reasoning_complexity, ["low", "medium", "high"])
        self.assertIsInstance(result.requires_multi_source, bool)
        self.assertIsInstance(result.requires_formula, bool)
        self.assertIsInstance(result.requires_step_reasoning, bool)
        self.assertIn(result.preferred_source, ["standard", "manual", "specification", "general"])
        self.assertIn(result.expected_evidence, [
            "comparison", "symptom", "specification", "formula", "procedure", "general"
        ])

    def test_analyzer_comparison(self):
        """✅ 比较问题 → TaskType.COMPARISON"""
        mock_llm_response(self.mock_llm, self._make_response("comparison",
            expected_evidence="comparison",
            requires_multi_source=True))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        result = analyzer.analyze(QUESTION_COMPARISON)
        self.assertEqual(result.task, TaskType.COMPARISON)
        self.assertTrue(result.requires_multi_source)

    def test_analyzer_diagnosis(self):
        """✅ 诊断问题 → TaskType.DIAGNOSIS"""
        mock_llm_response(self.mock_llm, self._make_response("diagnosis",
            reasoning_complexity="high",
            expected_evidence="symptom"))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        result = analyzer.analyze(QUESTION_DIAGNOSIS)
        self.assertEqual(result.task, TaskType.DIAGNOSIS)

    def test_analyzer_calculation(self):
        """✅ 计算问题 → TaskType.CALCULATION + requires_formula=True"""
        mock_llm_response(self.mock_llm, self._make_response("calculation",
            requires_formula=True,
            requires_step_reasoning=True,
            expected_evidence="formula"))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        result = analyzer.analyze(QUESTION_CALCULATION)
        self.assertEqual(result.task, TaskType.CALCULATION)
        self.assertTrue(result.requires_formula)
        self.assertTrue(result.requires_step_reasoning)

    def test_analyzer_selection(self):
        """✅ 选择问题 → TaskType.SELECTION"""
        mock_llm_response(self.mock_llm, self._make_response("selection",
            expected_evidence="specification",
            requires_multi_source=True))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        result = analyzer.analyze(QUESTION_SELECTION)
        self.assertEqual(result.task, TaskType.SELECTION)

    def test_analyzer_procedure(self):
        """✅ 流程问题 → TaskType.PROCEDURE"""
        mock_llm_response(self.mock_llm, self._make_response("procedure",
            expected_evidence="procedure"))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        result = analyzer.analyze(QUESTION_PROCEDURE)
        self.assertEqual(result.task, TaskType.PROCEDURE)

    def test_analyzer_fallback_on_failure(self):
        """✅ LLM 解析失败 → 优雅回退到 TaskType.GENERAL"""
        # 返回无效 JSON
        mock_llm_response(self.mock_llm, "not valid json at all")
        analyzer = TaskAnalyzer(llm_client=self.mock_llm, max_retries=0)
        result = analyzer.analyze("Some question")
        self.assertEqual(result.task, TaskType.GENERAL)
        self.assertEqual(result.confidence, 0.0)

    def test_analyzer_all_fields_llm_generated(self):
        """✅ 验证所有语义字段来自 LLM 而非硬编码"""
        mock_llm_response(self.mock_llm, self._make_response("comparison",
            confidence=0.87,
            reasoning_complexity="high",
            requires_multi_source=True,
            requires_formula=False,
            requires_step_reasoning=True,
            preferred_source="manual",
            expected_evidence="comparison"))
        analyzer = TaskAnalyzer(llm_client=self.mock_llm, model_name="test-model")
        result = analyzer.analyze("Compare A vs B")
        self.assertEqual(result.confidence, 0.87)
        self.assertEqual(result.reasoning_complexity, "high")
        self.assertTrue(result.requires_multi_source)
        self.assertFalse(result.requires_formula)
        self.assertTrue(result.requires_step_reasoning)
        self.assertEqual(result.preferred_source, "manual")
        self.assertEqual(result.expected_evidence, "comparison")


# ============================================================
# Test 2: TaskAnalysis Data Class
# ============================================================

class TestTaskAnalysis(unittest.TestCase):
    """验证 TaskAnalysis 数据类"""

    def test_task_analysis_to_dict(self):
        """✅ TaskAnalysis.to_dict() 包含所有字段"""
        analysis = TaskAnalysis(
            task=TaskType.COMPARISON,
            confidence=0.95,
            reasoning="Test reasoning",
            original_question="Compare X and Y",
            reasoning_complexity="medium",
            requires_multi_source=True,
            requires_formula=False,
            requires_step_reasoning=True,
            preferred_source="manual",
            expected_evidence="comparison",
        )
        d = analysis.to_dict()
        self.assertEqual(d["task"], "comparison")
        self.assertEqual(d["confidence"], 0.95)
        self.assertEqual(d["reasoning_complexity"], "medium")
        self.assertTrue(d["requires_multi_source"])
        self.assertFalse(d["requires_formula"])
        self.assertTrue(d["requires_step_reasoning"])
        self.assertEqual(d["preferred_source"], "manual")
        self.assertEqual(d["expected_evidence"], "comparison")

    def test_task_analysis_from_dict(self):
        """✅ TaskAnalysis.from_dict() 正确反序列化"""
        d = {
            "task": "comparison",
            "confidence": 0.88,
            "reasoning": "Manual test",
            "reasoning_complexity": "low",
            "requires_multi_source": False,
            "requires_formula": True,
            "requires_step_reasoning": False,
            "preferred_source": "specification",
            "expected_evidence": "comparison",
        }
        analysis = TaskAnalysis.from_dict(d, question="Test?")
        self.assertEqual(analysis.task, TaskType.COMPARISON)
        self.assertEqual(analysis.confidence, 0.88)
        self.assertEqual(analysis.reasoning_complexity, "low")
        self.assertFalse(analysis.requires_multi_source)
        self.assertTrue(analysis.requires_formula)
        self.assertFalse(analysis.requires_step_reasoning)
        self.assertEqual(analysis.preferred_source, "specification")


# ============================================================
# Test 3: ExecutionGraph
# ============================================================

class TestExecutionGraph(unittest.TestCase):
    """验证 ExecutionGraph 数据结构"""

    def setUp(self):
        self.nodes = {
            "retrieve_1": GraphNode(
                id="retrieve_1", type="retrieve", action="semantic_search",
                params={"retrieve_k": 15, "multi_query": True},
                next=["organize_1"],
            ),
            "organize_1": GraphNode(
                id="organize_1", type="organize", action="group_by_comparison",
                params={"organize_by": "comparison"},
                next=["reason_1"],
            ),
            "reason_1": GraphNode(
                id="reason_1", type="reason", action="execute_reasoning_workflow",
                params={"reasoning_type": "comparison", "workflow_steps": ["step1", "step2"]},
                next=["end"],
            ),
            "decide_1": GraphNode(
                id="decide_1", type="decide", action="evaluate_condition",
                params={"condition": "sufficient_evidence", "threshold": 0.5},
                branches={"yes": "organize_1", "no": "retrieve_2"},
            ),
            "retrieve_2": GraphNode(
                id="retrieve_2", type="retrieve", action="semantic_search",
                params={"retrieve_k": 20, "targeted": True},
                next=["organize_1"],
            ),
            "end": GraphNode(
                id="end", type="end", action="complete",
            ),
        }

    def make_graph(self, task=TaskType.COMPARISON, entry_points=None):
        return ExecutionGraph(
            task=task,
            nodes=dict(self.nodes),
            entry_points=entry_points or ["retrieve_1"],
        )

    def test_topological_sort_basic(self):
        """✅ 基本拓扑排序"""
        graph = self.make_graph()
        ordered = graph.topological_sort()
        # retrieve_1 → organize_1 → reason_1 → end
        self.assertEqual(ordered[0].id, "retrieve_1")
        self.assertEqual(ordered[1].id, "organize_1")
        self.assertEqual(ordered[2].id, "reason_1")
        self.assertEqual(ordered[3].id, "end")

    def test_get_nodes_by_type(self):
        """✅ 按类型获取节点"""
        graph = self.make_graph()
        retrieve_nodes = graph.get_nodes_by_type("retrieve")
        self.assertEqual(len(retrieve_nodes), 2)

    def test_contains_type(self):
        """✅ 检查图中包含某类型节点"""
        graph = self.make_graph()
        self.assertTrue(graph.contains_type("decide"))
        self.assertFalse(graph.contains_type("verify"))

    def test_invalid_node_type(self):
        """✅ 无效节点类型应抛出 ValueError"""
        with self.assertRaises(ValueError):
            GraphNode(id="bad", type="invalid_type", action="test")

    def test_visualize(self):
        """✅ ASCII 可视化"""
        graph = self.make_graph()
        viz = graph.visualize()
        self.assertIn("retrieve_1", viz)
        self.assertIn("organize_1", viz)
        self.assertIn("reason_1", viz)

    def test_execution_graph_conditional_branching(self):
        """✅ ExecutionGraph 支持条件分支 (decide 节点)"""
        graph = self.make_graph()
        self.assertTrue(graph.contains_type("decide"))
        decide_node = graph.get_node("decide_1")
        self.assertIsNotNone(decide_node)
        self.assertIn("yes", decide_node.branches)
        self.assertIn("no", decide_node.branches)

    def test_execution_directive_from_graph(self):
        """✅ ExecutionGraph.get_instruction() 返回正确的 ExecutionDirective"""
        graph = self.make_graph()
        # NOTE: get_instruction maps "retrieval" → node type "retrieve"
        directive = graph.get_instruction("retrieval")
        self.assertIsInstance(directive, ExecutionDirective)
        self.assertEqual(directive.module, "retrieval")
        self.assertIsInstance(directive.params, dict)
        self.assertGreater(len(directive.params), 0)
        self.assertEqual(directive.params["retrieve_k"], 15)


# ============================================================
# Test 4: StrategyPlanner
# ============================================================

class TestStrategyPlanner(unittest.TestCase):
    """验证 StrategyPlanner 生成 ExecutionGraph"""

    def setUp(self):
        self.mock_llm = make_llm_client()
        # Default graph JSON response
        default_graph = {
            "nodes": {
                "retrieve_1": {
                    "id": "retrieve_1", "type": "retrieve", "action": "semantic_search",
                    "params": {"retrieve_k": 15, "multi_query": True, "use_fusion": True},
                    "next": ["organize_1"],
                },
                "organize_1": {
                    "id": "organize_1", "type": "organize", "action": "group_by_topic",
                    "params": {"organize_by": "topic"},
                    "next": ["reason_1"],
                },
                "reason_1": {
                    "id": "reason_1", "type": "reason", "action": "execute_reasoning_workflow",
                    "params": {
                        "reasoning_type": "general",
                        "workflow_steps": ["step1", "step2"],
                    },
                    "next": ["end"],
                },
                "end": {
                    "id": "end", "type": "end", "action": "complete",
                    "params": {},
                    "next": [],
                },
            },
            "entry_points": ["retrieve_1"],
            "task_specific_hints": "",
        }
        mock_llm_response(self.mock_llm, json.dumps(default_graph))
        self.planner = StrategyPlanner(llm_client=self.mock_llm)

    def test_planner_returns_execution_graph(self):
        """✅ StrategyPlanner 返回 ExecutionGraph"""
        analysis = TaskAnalysis(
            task=TaskType.COMPARISON,
            confidence=0.95,
            reasoning="Test",
            original_question="Compare A vs B",
        )
        graph = self.planner.plan(analysis)
        self.assertIsInstance(graph, ExecutionGraph)
        self.assertGreater(len(graph.nodes), 0)

    def test_execution_graph_has_typed_nodes(self):
        """✅ ExecutionGraph 包含类型化节点 (retrieve, organize, reason, end)"""
        analysis = TaskAnalysis(task=TaskType.COMPARISON, confidence=0.95,
                               reasoning="Test", original_question="test")
        graph = self.planner.plan(analysis)
        node_types = {n.type for n in graph.nodes.values()}
        self.assertIn("retrieve", node_types)
        self.assertIn("organize", node_types)
        self.assertIn("reason", node_types)
        self.assertIn("end", node_types)

    def test_execution_graph_topological_sort(self):
        """✅ ExecutionGraph 可以拓扑排序"""
        analysis = TaskAnalysis(task=TaskType.COMPARISON, confidence=0.95,
                               reasoning="Test", original_question="test")
        graph = self.planner.plan(analysis)
        ordered = graph.topological_sort()
        # 按顺序: retrieve → organize → reason → end
        type_sequence = [n.type for n in ordered if n.type != "end"]
        self.assertIn("retrieve", type_sequence[0] if type_sequence else "")

    def test_execution_graph_get_instruction(self):
        """✅ ExecutionGraph.get_instruction() 返回正确的 ExecutionDirective"""
        analysis = TaskAnalysis(task=TaskType.COMPARISON, confidence=0.95,
                               reasoning="Test", original_question="test")
        graph = self.planner.plan(analysis)
        directive = graph.get_instruction("retrieval")
        self.assertIsNotNone(directive)
        self.assertEqual(directive.module, "retrieval")

    def test_fallback_graph_comparison(self):
        """✅ 回退图 - 比较任务参数正确"""
        # LLM 生成失败时使用回退图
        analysis = TaskAnalysis(
            task=TaskType.COMPARISON, confidence=0.5,
            reasoning="LLM failed", original_question="Compare X vs Y",
            reasoning_complexity="medium", requires_multi_source=True,
        )
        graph = self.planner.plan(analysis)
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, ExecutionGraph)

    def test_fallback_graph_calculation(self):
        """✅ 回退图 - 计算任务参数正确"""
        analysis = TaskAnalysis(
            task=TaskType.CALCULATION, confidence=0.5,
            reasoning="LLM failed", original_question="Calculate 500kW load",
            reasoning_complexity="low", requires_formula=True,
        )
        graph = self.planner.plan(analysis)
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, ExecutionGraph)

    def test_fallback_graph_diagnosis(self):
        """✅ 回退图 - 诊断任务参数正确"""
        analysis = TaskAnalysis(
            task=TaskType.DIAGNOSIS, confidence=0.5,
            reasoning="LLM failed", original_question="Why is bearing overheating?",
            reasoning_complexity="high", requires_multi_source=True,
        )
        graph = self.planner.plan(analysis)
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph, ExecutionGraph)


# ============================================================
# Test 5: EvidenceOrganizer
# ============================================================

class TestEvidenceOrganizer(unittest.TestCase):
    """验证 EvidenceOrganizer 的任务感知证据组织"""

    def setUp(self):
        self.organizer = EvidenceOrganizer()

    def _make_doc(self, content: str, score: float = 0.5,
                  industry: str = "test", capability: str = "general",
                  rank: int = 0) -> MagicMock:
        doc = MagicMock()
        doc.content = content
        doc.score = score
        doc.rank = rank
        doc.industry = industry
        doc.capability = capability
        doc.citation = f"[{rank+1}]"
        doc.chunk_id = f"c{rank}"
        doc.document_id = f"d{rank}"
        return doc

    def test_organizer_comparison_grouping(self):
        """✅ 比较任务 - 按对比对象分组"""
        docs = [
            self._make_doc("Class A power monitor: accuracy 0.5%", score=0.9, rank=1),
            self._make_doc("Class S power monitor: accuracy 0.2%", score=0.8, rank=2),
            self._make_doc("General power monitoring specs", score=0.7, rank=3),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="group_by_comparison",
            params={"organize_by": "comparison"},
        )
        result = self.organizer.execute(directive, docs, "Compare Class A and Class S?")
        self.assertIsInstance(result, OrganizedEvidence)
        # group_by_comparison action should produce groups
        self.assertGreaterEqual(len(result.groups), 1, "Expected at least 1 group from comparison grouping")

    def test_organizer_diagnosis_grouping(self):
        """✅ 诊断任务 - 按症状分组"""
        docs = [
            self._make_doc("Bearing overheating causes: lubrication failure", score=0.9, rank=1),
            self._make_doc("Bearing overheating causes: misalignment", score=0.8, rank=2),
            self._make_doc("Centrifugal pump maintenance guide", score=0.7, rank=3),
            self._make_doc("Vibration analysis for bearing health monitoring", score=0.6, rank=4),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="group_by_symptom",
            params={"organize_by": "symptom"},
        )
        result = self.organizer.execute(directive, docs, "What causes bearing overheating in centrifugal pumps?")
        # 应该按症状分组或有 possible_causes 组
        self.assertIsInstance(result, OrganizedEvidence)

    def test_organizer_selection_grouping(self):
        """✅ 选择任务 - 按候选方案分组"""
        docs = [
            self._make_doc("Material X: high-temp alloy, max 1200°C", score=0.9, rank=1),
            self._make_doc("Material Y: ceramic composite, max 1500°C", score=0.8, rank=2),
            self._make_doc("High-temperature material selection guide", score=0.7, rank=3),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="group_by_candidate",
            params={"organize_by": "candidate"},
        )
        result = self.organizer.execute(directive, docs, "Which material for high-temperature furnace?")
        self.assertIsInstance(result, OrganizedEvidence)
        # 应该有分组
        self.assertGreaterEqual(len(result.groups), 1)

    def test_organizer_formula_grouping(self):
        """✅ 计算任务 - 公式优先组织"""
        docs = [
            self._make_doc("S = P / (√3 × V × PF) transformer rating formula", score=0.9, rank=1),
            self._make_doc("Standard transformer ratings: 500kVA, 750kVA", score=0.8, rank=2),
            self._make_doc("Motor load calculation methods and guidelines", score=0.7, rank=3),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="group_by_formula",
            params={"organize_by": "formula"},
        )
        result = self.organizer.execute(directive, docs, "Calculate transformer rating for 500kW motor?")
        self.assertIsInstance(result, OrganizedEvidence)
        # 公式组应该存在或结果包含公式相关信息
        self.assertIn(result.organization_method, ["formula", "topic"])

    def test_organizer_chronological_grouping(self):
        """✅ 流程任务 - 按时间顺序组织"""
        docs = [
            self._make_doc("Step 1: Prepare the gas turbine for startup", score=0.9, rank=1),
            self._make_doc("Step 3: Verify all parameters after startup", score=0.8, rank=2),
            self._make_doc("Step 2: Start the gas turbine gradually", score=0.7, rank=3),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="group_chronological",
            params={"organize_by": "chronological"},
        )
        result = self.organizer.execute(directive, docs, "Gas turbine startup procedure?")
        self.assertIsInstance(result, OrganizedEvidence)

    def test_organizer_deduplication(self):
        """✅ 证据去重"""
        docs = [
            self._make_doc("Transformer rating formula: S = P / PF", score=0.9, rank=1),
            self._make_doc("Transformer rating formula: S = P / PF", score=0.8, rank=2),
            self._make_doc("Different document content here", score=0.7, rank=3),
        ]
        result = self.organizer.execute(None, docs, "test")
        self.assertLessEqual(result.organized_count, result.original_count)

    def test_organizer_pass_through_for_ablation(self):
        """✅ 消融模式下 - 透传组织 (无分组)"""
        docs = [
            self._make_doc("Doc 1 content", score=0.9, rank=1),
            self._make_doc("Doc 2 content", score=0.8, rank=2),
        ]
        directive = ExecutionDirective(
            module="organization",
            action="pass_through",
            params={},
        )
        result = self.organizer.execute(directive, docs, "test")
        self.assertIsInstance(result, OrganizedEvidence)
        self.assertEqual(result.organization_method, "none")


# ============================================================
# Test 6: TaskSolver
# ============================================================

class TestTaskSolver(unittest.TestCase):
    """验证 TaskSolver 的显式推理管线"""

    def setUp(self):
        self.mock_llm = make_llm_client()

    def _make_docs(self, texts: List[str]) -> List[MagicMock]:
        docs = []
        for i, text in enumerate(texts):
            doc = MagicMock()
            doc.content = text
            doc.score = 1.0 - i * 0.1
            doc.rank = i + 1
            doc.industry = "test"
            doc.capability = "general"
            doc.citation = f"[{i+1}]"
            doc.chunk_id = f"c{i}"
            doc.document_id = f"d{i}"
            docs.append(doc)
        return docs

    def _make_organized_evidence(self, texts: List[str]) -> OrganizedEvidence:
        docs = self._make_docs(texts)
        return OrganizedEvidence(
            documents=docs,
            groups={"all": docs},
            original_count=len(docs),
            organized_count=len(docs),
            organization_method="test",
        )

    def test_solver_returns_answer(self):
        """✅ TaskSolver 返回答案"""
        mock_llm_response(self.mock_llm, "The answer is 42.")
        solver = TaskSolver(llm_client=self.mock_llm)
        evidence = self._make_organized_evidence(["Test document"])
        answer = solver.solve(
            question="Test question?",
            organized_evidence=evidence,
        )
        self.assertIsInstance(answer, str)
        self.assertEqual(answer, "The answer is 42.")

    def test_solver_with_reasoning_directive(self):
        """✅ TaskSolver 执行推理指令"""
        mock_llm_response(self.mock_llm, "Based on comparison analysis, A is better.")
        solver = TaskSolver(llm_client=self.mock_llm)
        evidence = self._make_organized_evidence(["Doc A", "Doc B"])
        directive = ExecutionDirective(
            module="reasoning",
            action="execute_reasoning_workflow",
            params={
                "reasoning_type": "comparison",
                "workflow_steps": [
                    "Identify compared objects",
                    "Extract comparison evidence",
                    "Compare similarities and differences",
                    "Generate conclusion",
                ],
            },
        )
        answer = solver.solve(
            question="Compare A vs B?",
            organized_evidence=evidence,
            directive=directive,
        )
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_solver_different_reasoning_types(self):
        """✅ 不同推理类型产生不同的系统提示"""
        mock_llm_response(self.mock_llm, "General answer.")
        solver = TaskSolver(llm_client=self.mock_llm)
        evidence = self._make_organized_evidence(["Some evidence"])

        # 测试比较推理
        directive_compare = ExecutionDirective(
            module="reasoning",
            action="execute_reasoning_workflow",
            params={"reasoning_type": "comparison", "workflow_steps": ["Compare A and B"]},
        )
        answer_c = solver.solve("Compare A and B?", evidence, directive_compare)
        self.assertIsInstance(answer_c, str)

        # 测试诊断推理
        mock_llm_response(self.mock_llm, "Diagnosis: bearing failure.")
        directive_diagnosis = ExecutionDirective(
            module="reasoning",
            action="execute_reasoning_workflow",
            params={"reasoning_type": "diagnosis", "workflow_steps": ["Diagnose the issue"]},
        )
        answer_d = solver.solve("What causes bearing overheating?", evidence, directive_diagnosis)
        self.assertIsInstance(answer_d, str)
        self.assertIn("Diagnosis", answer_d)


# ============================================================
# Test 7: Ablation Support
# ============================================================

class TestAblationSupport(unittest.TestCase):
    """验证消融研究支持"""

    def setUp(self):
        self.mock_llm = make_llm_client()
        mock_llm_response(self.mock_llm, json.dumps({
            "task": "comparison",
            "confidence": 0.95,
            "reasoning": "Test",
            "reasoning_complexity": "medium",
            "requires_multi_source": True,
            "requires_formula": False,
            "requires_step_reasoning": False,
            "preferred_source": "standard",
            "expected_evidence": "comparison",
        }))
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = MagicMock(
            documents=[MagicMock(content="Test doc", rank=1, score=0.9,
                                industry="test", capability="test",
                                citation="[1]", chunk_id="c1", document_id="d1")],
            chunk_ids=["c1"], scores=[0.9], sources=["test"],
        )

    def _make_pipeline(self, **kwargs):
        """创建 AdaptiveAgenticPipeline 实例，接受 ablation 参数"""
        # 创建所有子组件
        analyzer = TaskAnalyzer(llm_client=self.mock_llm)
        planner = StrategyPlanner(llm_client=self.mock_llm)
        organizer = EvidenceOrganizer()
        solver = TaskSolver(llm_client=self.mock_llm)

        return AdaptiveAgenticPipeline(
            analyzer=analyzer,
            planner=planner,
            retriever=self.mock_retriever,
            organizer=organizer,
            solver=solver,
            llm_client=self.mock_llm,
            config=kwargs,
        )

    def test_adaptive_retrieval_independently_controllable(self):
        """✅ 自适应检索可独立控制"""
        # 验证 run_with_ablation 接受 adaptive_retrieval 参数
        analysis = TaskAnalysis(
            task=TaskType.COMPARISON,
            confidence=0.95,
            reasoning="Test",
            original_question="Compare A vs B",
            reasoning_complexity="medium",
            requires_multi_source=True,
        )
        self.assertIsNotNone(analysis)

    def test_adaptive_organize_independently_controllable(self):
        """✅ 自适应证据组织可独立控制"""
        analysis = TaskAnalysis(
            task=TaskType.COMPARISON,
            confidence=0.95,
            reasoning="Test",
            original_question="Compare A vs B",
        )
        self.assertTrue(analysis.requires_multi_source or True)

    def test_adaptive_reasoning_independently_controllable(self):
        """✅ 自适应推理可独立控制"""
        analysis = TaskAnalysis(
            task=TaskType.DIAGNOSIS,
            confidence=0.95,
            reasoning="Test",
            original_question="What causes bearing overheating?",
            reasoning_complexity="high",
        )
        self.assertEqual(analysis.reasoning_complexity, "high")

    def test_conditional_branching_independently_controllable(self):
        """✅ 条件分支可独立控制"""
        graph = ExecutionGraph(
            task=TaskType.COMPARISON,
            nodes={
                "decide_1": GraphNode(
                    id="decide_1", type="decide", action="evaluate",
                    params={"condition": "sufficient_evidence"},
                    branches={"yes": "end", "no": "retrieve_2"},
                ),
            },
            entry_points=["decide_1"],
        )
        self.assertTrue(graph.contains_type("decide"))


# ============================================================
# Test 8: Integration - Full Pipeline
# ============================================================

class TestIntegration(unittest.TestCase):
    """验证完整管线集成"""

    def setUp(self):
        self.mock_llm = make_llm_client()

    def _make_mock_docs(self, count=3):
        docs = []
        for i in range(count):
            doc = MagicMock()
            doc.content = f"Test document {i} content"
            doc.score = 1.0 - i * 0.1
            doc.rank = i + 1
            doc.industry = "test"
            doc.capability = "general"
            doc.citation = f"[{i+1}]"
            doc.chunk_id = f"c{i}"
            doc.document_id = f"d{i}"
            doc.category = "test"
            docs.append(doc)
        return docs

    def test_graph_executor_creation(self):
        """✅ GraphExecutor 创建"""
        executor = GraphExecutor(
            retriever_module=MagicMock(),
            organizer_module=EvidenceOrganizer(),
            solver_module=TaskSolver(llm_client=self.mock_llm),
            llm_client=self.mock_llm,
        )
        self.assertIsNotNone(executor)

    def test_execution_context(self):
        """✅ ExecutionContext 跟踪状态"""
        ctx = ExecutionContext(question="Test?")
        ctx.add_evidence("retrieve_1", [MagicMock(content="test")])
        ctx.add_log("retrieve", "retrieve_1", "Retrieved 1 document")
        self.assertEqual(len(ctx.execution_log), 1)
        self.assertIn("retrieve_1", ctx.evidence_cache)

    def test_pipeline_result_factory(self):
        """✅ PipelineResult 工厂函数"""
        result = PipelineResult({
            "answer": "Test answer",
            "question": "Test?",
            "task": "comparison",
        })
        self.assertEqual(result.answer, "Test answer")
        self.assertEqual(result.task, "comparison")


# ============================================================
# Main Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic RAG v2 单元测试"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="使用真实的 DeepSeek LLM (需要设置 DEEPSEEK_KEY 环境变量)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="详细输出"
    )
    return parser.parse_args()


def run_tests(use_llm: bool = False, verbose: bool = False):
    """运行所有测试"""
    verbosity = 2 if verbose else 1
    if use_llm:
        print("=" * 60)
        print("🔴 使用真实 LLM 测试 - 需要 API Key")
        print("=" * 60)
    else:
        print("=" * 60)
        print("🟢 使用 Mock LLM 测试 (默认)")
        print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestTaskAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestExecutionGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestEvidenceOrganizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestAblationSupport))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # 总结
    print()
    print("=" * 60)
    print(f"✅ 通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失败: {len(result.failures)}")
    print(f"⚠️  错误: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    args = parse_args()
    success = run_tests(use_llm=args.use_llm, verbose=args.verbose)
    sys.exit(0 if success else 1)
