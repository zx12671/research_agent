"""
prompt_builder.py: PromptBuilder — 为 Agentic RAG 编排层构造任务感知的 Prompt。

职责:
    接收 TaskAnalysis + ExecutionGraph + OrganizedEvidence，
    输出 PromptInstruction（结构化指令），用于增强标准 RAG prompt。

设计原则:
    - PromptBuilder 不直接调用 LLM
    - PromptBuilder 不生成最终答案
    - 最终答案始终由 exp1_qa Standard RAG prompt + DeepSeek.chat() 生成
    - PromptBuilder 只影响 prompt 中的额外指令和格式

架构:
    TaskAnalyzer → StrategyPlanner → Retriever → EvidenceOrganizer
        ↓                                       ↓
    TaskAnalysis + ExecutionGraph    OrganizedEvidence
        ↓                                       ↓
        └──────────→ PromptBuilder ←────────────┘
                         ↓
                PromptInstruction
                         ↓
            + Standard RAG Prompt
                         ↓
                DeepSeek.chat()
                         ↓
                   Answer
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .task_types import TaskAnalysis, TaskType, ExecutionGraph


# ============================================================
# PromptInstruction — PromptBuilder 的结构化输出
# ============================================================

@dataclass
class PromptInstruction:
    """
    PromptBuilder 的输出：一组结构化指令，用于增强标准 RAG prompt。
    
    Attributes:
        task_type: 分析出的任务类型
        format_instruction: 针对该任务类型的格式指令（如比较表格、步骤编号等）
        reasoning_instruction: 推理方法指令（如逐步推理、引用来源等）
        output_style: 输出风格提示
        task_specific_hints: 额外任务特定提示
        prompt_extra: 要追加到标准 prompt 后的额外内容
        output_format: 偏好的输出格式（table / bullet / step / ""）
    """
    task_type: str = "general"
    format_instruction: str = ""
    reasoning_instruction: str = ""
    output_style: str = ""
    task_specific_hints: str = ""
    prompt_extra: str = ""
    output_format: str = ""


# ============================================================
# Minimal prompt_extra — shared across all task types.
#
# Design rationale (per recall/quality investigation):
# - IndustryBench questions are mostly direct fact/spec lookups
#   (Question -> one sentence -> Answer). Heavy per-task "Reasoning
#   Workflow" (Verify -> Reflect -> Reason, or symptom/compare/calculate
#   step summaries) forces the model to summarise/organise and throws
#   away fine-grained detail (e.g. "5 mL"). 
# - Fix: keep the instruction to a single minimal, scoring-aligned
#   line. Evidence is already first via the general template; do NOT
#   re-inject multi-step workflows here.
# ============================================================
_MINIMAL_EXTRA = (
    "\n基于提供的证据给出你最好的答案。引用 [1], [2] 等来源。"
    "即使证据不完整也要尽量作答，除非完全没有相关证据。"
    "**IMPORTANT: 请务必使用中文回答，不要用英文。**"
)



class PromptBuilder:
    """
    PromptBuilder — 构造任务感知的 prompt 增强指令。

    
    根据 TaskAnalysis 中的语义信息，生成针对不同任务类型的指令，
    这些指令会被追加到标准 RAG prompt 后，增强 LLM 的应答质量。
    
    使用方式:
        builder = PromptBuilder()
        instruction = builder.build(task_analysis, graph, evidence_context)
        # instruction.prompt_extra 追加到标准 RAG prompt
    """
    
    def __init__(self):
        """初始化 PromptBuilder。

        Task 分类维度已中性化（analyzer 恒返回 GENERAL），因此只保留通用
        handler。7 套任务专属 handler（_build_comparison 等）已不再被分发，
        留作废弃代码待清理（见 docs/analyzer_rule_classification_audit.md）。
        """
        self._task_handlers = {
            TaskType.GENERAL: self._build_general,
        }

    
    def build(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph] = None,
        evidence_context: str = "",
    ) -> PromptInstruction:
        """
        构建 PromptInstruction。
        
        Args:
            task_analysis: TaskAnalyzer 输出的任务分析
            graph: StrategyPlanner 生成的 ExecutionGraph（可选）
            evidence_context: 经过 OrganizedEvidence 处理后的证据上下文（可选）
        
        Returns:
            PromptInstruction: 可用于增强标准 RAG prompt 的指令
        """
        handler = self._task_handlers.get(
            task_analysis.task,
            self._build_general,
        )
        return handler(task_analysis, graph, evidence_context)
    
    # ──────────────────────────────────────────────────────────
    # 各任务类型的指令构建
    # ──────────────────────────────────────────────────────────
    
    def _build_comparison(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """比较类问题的指令。"""
        return PromptInstruction(
            task_type="comparison",
            format_instruction=(
                "请以比较结构组织答案：\n"
                "1. 识别被比较的项目。\n"
                "2. 使用表格或结构化列表展示相似点和差异。\n"
                "3. 突出关键区分因素。\n"
                "4. 给出平衡的结论和实用建议。"
            ),
            reasoning_instruction=(
                "首先基于检索到的知识分别分析每个项目，"
                "然后并列比较它们。"
            ),
            output_style="structured",
            task_specific_hints=(
                "使用清晰的比较格式（鼓励表格）。"
                "在适用时引用检索知识中的证据。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="table",

        )
    
    def _build_diagnosis(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """诊断/故障排查类问题的指令。"""
        return PromptInstruction(
            task_type="diagnosis",
            format_instruction=(
                "请以诊断结构组织答案：\n"
                "1. 从问题中识别症状。\n"
                "2. 基于检索知识列出可能的根本原因。\n"
                "3. 按可能性对原因进行排序。\n"
                "4. 推荐纠正措施或后续步骤。"
            ),
            reasoning_instruction=(
                "遵循系统化的诊断方法："
                "症状 → 可能原因 → 证据匹配 → 结论。"
            ),
            output_style="step_by_step",
            task_specific_hints=(
                "清晰地列出症状。"
                "按可能性对可能原因进行排序。"
                "如果适用，包含安全预防措施。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="bullet",

        )
    
    def _build_selection(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """选择/推荐类问题的指令。"""
        return PromptInstruction(
            task_type="selection",
            format_instruction=(
                "请以选择推荐结构组织答案：\n"
                "1. 从问题中识别需求和约束条件。\n"
                "2. 基于检索知识列出候选选项。\n"
                "3. 对照需求评估每个候选方案。\n"
                "4. 给出最终推荐并说明理由。"
            ),
            reasoning_instruction=(
                "系统化地对照评估标准评估每个候选方案。"
                "使用证据支持每项评估。"
            ),
            output_style="structured",
            task_specific_hints=(
                "清晰地说明选择标准。"
                "如果存在多个选项，使用比较矩阵。"
                "用证据证明最终推荐。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="table",

        )
    
    def _build_calculation(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """计算类问题的指令。"""
        return PromptInstruction(
            task_type="calculation",
            format_instruction=(
                "请逐步进行计算：\n"
                "1. 识别所有变量和参数。\n"
                "2. 说明所用公式或方法。\n"
                "3. 展示带单位的计算步骤。\n"
                "4. 给出带正确单位的最终数值结果。\n"
                "5. 验证结果的合理性。"
            ),
            reasoning_instruction=(
                "明确展示所有计算步骤。"
                "每一步都包含单位。"
                "说明所做的任何假设。"
            ),
            output_style="step_by_step",
            task_specific_hints=(
                "先显示公式，然后逐步代入数值。"
                "在整个计算过程中验证单位。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="step",

        )
    
    def _build_standard_interpretation(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """标准/规范解读类问题的指令。"""
        return PromptInstruction(
            task_type="standard_interpretation",
            format_instruction=(
                "请解读标准或规范：\n"
                "1. 从问题中识别相关标准/规范。\n"
                "2. 引用具体条款或章节。\n"
                "3. 解释技术要求。\n"
                "4. 提供实际意义或合规指导。"
            ),
            reasoning_instruction=(
                "引用标准的特定条款。"
                "用实际术语解释技术要求。"
            ),
            output_style="structured",
            task_specific_hints=(
                "引用相关标准条款。"
                "解释要求在实际中的含义。"
                "注意任何合规考虑因素。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="bullet",

        )
    
    def _build_procedure(

        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """流程/步骤类问题的指令。"""
        return PromptInstruction(
            task_type="procedure",
            format_instruction=(
                "请逐步描述流程：\n"
                "1. 列出前提条件或准备步骤。\n"
                "2. 按时间顺序描述每个步骤。\n"
                "3. 注明关键参数、安全检查和质量控制点。\n"
                "4. 如果适用，包括验证步骤。"
            ),
            reasoning_instruction=(
                "遵循时间顺序。"
                "包含具体参数（温度、压力、速度）。"
            ),
            output_style="step_by_step",
            task_specific_hints=(
                "按时间顺序列出步骤。"
                "在相关步骤中包含安全预防措施。"
                "在可用时引用具体参数。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="step",
        )

    
    def _build_explanation(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """解释类问题的指令。"""
        return PromptInstruction(
            task_type="explanation",
            format_instruction=(
                "请以解释结构组织答案：\n"
                "1. 从清晰的定义或简洁的答案开始。\n"
                "2. 解释基本原理。\n"
                "3. 提供背景和实际示例。\n"
                "4. 联系工业应用。"
            ),
            reasoning_instruction=(
                "从基础到实际应用逐步构建。"
                "使用精确的技术术语。"
            ),
            output_style="standard",
            task_specific_hints=(
                "从简洁的定义开始。"
                "使用工业实践中的示例。"
            ),
            prompt_extra=_MINIMAL_EXTRA,
            output_format="",
        )

    
    def _build_general(
        self,
        task_analysis: TaskAnalysis,
        graph: Optional[ExecutionGraph],
        evidence_context: str,
    ) -> PromptInstruction:
        """通用类问题的指令。"""
        return PromptInstruction(
            task_type="general",
            format_instruction=(
                "基于检索到的知识提供清晰简洁的答案。"
            ),
            reasoning_instruction=(
                "基于提供的证据回答。"
                "如果证据不足，说明还需要哪些额外信息。"
            ),
            output_style="standard",
            task_specific_hints="",
            prompt_extra=_MINIMAL_EXTRA,
        )



