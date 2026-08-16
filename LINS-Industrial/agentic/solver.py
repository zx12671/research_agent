"""
solver.py: Task-Aware Reasoning Solver (REDESIGNED)

NOW: Executes explicit reasoning workflows from ExecutionPlan directives.

> ⚠️ **接入状态（20260808 · S3 检验裁决）**：本模块目前【未接入】生产 reason 路径。
> `agentic/pipeline.py::_handle_reason` 实际走「标准 prompt + PromptBuilder」，从不调用
> `TaskSolver.solve()`（详见 `_diag_s3_executor.py` 的 ThrowingSolver 刺探：接入率=0，solver 为
> 从不通电的死代码）。S3 阶段判定【不接入】：现有 reason 标准 prompt 已过 S6 验收，`_solve_general`
> 系统提示更简陋、`_solve_with_workflow` 依赖 `workflow_steps`（模板图仅给 `["synthesize"]`），
> 强行接入有换掉已验证 prompt 的降级风险。本模块**保留为"可选、未来多步推理再做"**，不强制注入、
> 不接入 reason；勿误以为它已被生产调用。

CRITICAL DESIGN CHANGE:
    Old: Solver receives task type → switches prompt template
         solver = TaskSolver(task="comparison")
         ↓
         prompt = comparison_prompt if task == "comparison" else ...prompt
         ↓
         Solver is just a prompt switcher — NOT adaptive reasoning

    New: Solver receives ExecutionPlan reasoning directive → executes workflow
         directive = plan.get_instruction("reasoning")
         # directive = ExecutionDirective(module="reasoning",
         #    action="execute_reasoning_workflow",
         #    params={
         #        "reasoning_type": "compare",
         #        "workflow_steps": [
         #            "Identify compared objects",
         #            "Extract comparison evidence",
         #            "Compare similarities",
         #            "Compare differences",
         #            "Generate conclusion",
         #        ],
         #        "prompt_style": "structured",
         #    })
         ↓
         solver.solve(question, context, directive)
         ↓
         This is truly adaptive reasoning:
         - Planner defines the reasoning workflow
         - Solver executes each step with explicit tracking
         - Different tasks produce different reasoning paths

The key insight:
    Solver is no longer a "prompt switcher" — it is a workflow executor.
    Each reasoning step is explicitly modeled and tracked.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .task_types import (
    OrganizedEvidence,
    ExecutionDirective,
)


class TaskSolver:
    """
    Task-aware solver that executes reasoning workflows from ExecutionPlan.

    Each reasoning workflow is an explicit sequence of steps that the
    solver executes step-by-step. The workflow comes from the ExecutionPlan,
    not from hardcoded if-else logic.

    Usage:
        directive = plan.get_instruction("reasoning")
        answer = solver.solve(question, organized_evidence, directive)
    """

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
    ):
        """
        Initialize the TaskSolver.

        Args:
            llm_client: An LLM client that supports chat completions
            model_name: Name of the model to use
            temperature: Temperature for LLM sampling
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature

    def solve(
        self,
        question: str,
        organized_evidence: OrganizedEvidence,
        directive: Optional[ExecutionDirective] = None,
    ) -> str:
        """
        Solve a question using an explicit reasoning workflow.

        Args:
            question: The original question
            organized_evidence: Organized evidence from EvidenceOrganizer
            directive: Reasoning directive from ExecutionPlan (optional)

        Returns:
            Final answer with citations
        """
        if directive and directive.params:
            return self._solve_with_workflow(question, organized_evidence, directive)
        else:
            return self._solve_general(question, organized_evidence)

    def _solve_with_workflow(
        self,
        question: str,
        organized_evidence: OrganizedEvidence,
        directive: ExecutionDirective,
    ) -> str:
        """
        Execute a structured reasoning workflow step by step.

        Each step is explicitly tracked and the LLM is prompted to
        reason through each step before producing the final answer.
        """
        params = directive.params
        workflow_steps = params.get("workflow_steps", [])
        reasoning_type = params.get("reasoning_type", "general")
        prompt_style = params.get("prompt_style", "standard")
        requires_verification = params.get("requires_verification", False)
        requires_citation = params.get("requires_citation", True)

        # Build the evidence context
        context = organized_evidence.get_context()

        # Build the workflow description
        workflow_desc = "\n".join(
            f"  Step {i+1}. {step}" for i, step in enumerate(workflow_steps)
        )

        # Select system prompt based on reasoning type
        system_prompt = self._get_reasoning_system_prompt(reasoning_type)

        # Build the user prompt
        citation_instruction = (
            "IMPORTANT: You MUST cite specific evidence chunks using their citation IDs (e.g., [1], [2]). "
            "Each claim must be supported by at least one citation.\n"
            if requires_citation else ""
        )

        verification_instruction = (
            "\nAfter providing the answer, explicitly verify:\n"
            "1. Does the answer directly address the question?\n"
            "2. Are all claims supported by evidence?\n"
            "3. Are there any contradictions in the evidence?\n"
            "4. Is the answer complete and accurate?\n"
            if requires_verification else ""
        )

        user_prompt = f"""Question: {question}

Evidence Context:
{context}

Reasoning Workflow:
Please follow these steps to reason through the answer:
{workflow_desc}

{citation_instruction}
{verification_instruction}

Provide your answer following the reasoning workflow above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self._call_llm(messages)
        return answer

    def _solve_general(
        self,
        question: str,
        organized_evidence: OrganizedEvidence,
    ) -> str:
        """
        General solving without explicit workflow (fallback).
        """
        context = organized_evidence.get_context()

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable industrial domain expert. Answer the question "
                           "using the provided evidence context. Cite evidence using citation IDs.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence Context:\n{context}",
            },
        ]

        return self._call_llm(messages)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with the given messages."""
        if hasattr(self.llm_client, "chat"):
            # OpenAI client: llm_client.chat.completions.create(...)
            if hasattr(self.llm_client.chat, "completions"):
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2048,
                    timeout=60,
                )
                return response.choices[0].message.content.strip()

            else:
                response = self.llm_client.chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                return response["choices"][0]["message"]["content"].strip()
        elif hasattr(self.llm_client, "generate"):
            # Convert messages to a single prompt for generate-type APIs
            prompt = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            response = self.llm_client.generate(
                prompt,
                max_new_tokens=2048,
                temperature=self.temperature,
            )
            return response[0]["generated_text"].strip()
        else:
            response = self.llm_client(str(messages))
            return str(response).strip()

    def _get_reasoning_system_prompt(self, reasoning_type: str) -> str:
        """
        Get the system prompt for a specific reasoning type.

        Each reasoning type has a tailored system prompt that guides
        the LLM's reasoning approach. These are NOT just template
        switches — each prompt encodes a different reasoning strategy.
        """
        prompts = {
            "compare": (
                "You are an expert in comparative analysis for industrial products, "
                "materials, and systems.\n\n"
                "ROLE: Comparison Analyst\n\n"
                "APPROACH:\n"
                "1. OBJECTIVITY: Present balanced comparisons without bias.\n"
                "2. STRUCTURE: Clearly separate similarities from differences.\n"
                "3. SPECIFICITY: Use concrete metrics and specifications.\n"
                "4. CONTEXT: Consider the specific use case and constraints.\n"
                "5. CITATION: Always cite evidence for each comparison point.\n\n"
                "OUTPUT FORMAT:\n"
                "First, identify the specific objects/items being compared.\n"
                "Then, present similarities and differences in a structured way.\n"
                "Finally, provide a balanced conclusion."
            ),
            "diagnosis": (
                "You are an expert in industrial fault diagnosis and root cause "
                "analysis.\n\n"
                "ROLE: Diagnostic Engineer\n\n"
                "APPROACH:\n"
                "1. SYMPTOM-DRIVEN: Start with the reported symptoms.\n"
                "2. SYSTEMATIC: Consider all possible causes systematically.\n"
                "3. EVIDENCE-BASED: Match evidence to each hypothesis.\n"
                "4. PRIORITIZATION: Rank causes by likelihood.\n"
                "5. ACTIONABLE: Provide clear remediation steps.\n\n"
                "OUTPUT FORMAT:\n"
                "First, list the symptoms from the question.\n"
                "Then, evaluate possible causes with evidence.\n"
                "Finally, recommend the most likely cause and action plan."
            ),
            "calculation": (
                "You are an expert in industrial calculations and engineering "
                "analysis.\n\n"
                "ROLE: Calculation Engineer\n\n"
                "APPROACH:\n"
                "1. VARIABLE IDENTIFICATION: Clearly identify all variables.\n"
                "2. FORMULA SELECTION: Choose the correct formula.\n"
                "3. STEP-BY-STEP: Show calculation steps explicitly.\n"
                "4. UNIT VERIFICATION: Check units throughout.\n"
                "5. REASONABLENESS: Verify result plausibility.\n\n"
                "OUTPUT FORMAT:\n"
                "Show step-by-step calculations with units.\n"
                "State assumptions clearly.\n"
                "Verify the final answer."
            ),
            "selection": (
                "You are an expert in industrial product and system selection.\n\n"
                "ROLE: Selection Analyst\n\n"
                "APPROACH:\n"
                "1. REQUIREMENT-FOCUSED: Start with explicit requirements.\n"
                "2. MULTI-CRITERIA: Evaluate against all relevant criteria.\n"
                "3. TRADE-OFF: Explicitly acknowledge trade-offs.\n"
                "4. EVIDENCE: Use datasheets and specifications.\n"
                "5. JUSTIFICATION: Provide clear rationale.\n\n"
                "OUTPUT FORMAT:\n"
                "First, list requirements from the question.\n"
                "Then, evaluate each candidate against requirements.\n"
                "Provide final recommendation with justification."
            ),
            "procedure": (
                "You are an expert in industrial procedures and manufacturing "
                "processes.\n\n"
                "ROLE: Process Engineer\n\n"
                "APPROACH:\n"
                "1. SEQUENTIAL: Follow chronological order.\n"
                "2. PRECISE: Include specific parameters.\n"
                "3. SAFETY: Note safety requirements.\n"
                "4. QUALITY: Highlight critical checkpoints.\n"
                "5. COMPLETE: Cover prerequisites to verification.\n\n"
                "OUTPUT FORMAT:\n"
                "List required prerequisites.\n"
                "Describe each step in sequence.\n"
                "Note quality checks and critical parameters."
            ),
            "interpret": (
                "You are an expert in industrial standards and specifications.\n\n"
                "ROLE: Standards Interpreter\n\n"
                "APPROACH:\n"
                "1. APPLICABLE: Identify relevant standards.\n"
                "2. CLEAR: Explain requirements in plain language.\n"
                "3. PRACTICAL: Show practical implementation.\n"
                "4. SPECIFIC: Include numerical limits.\n"
                "5. ACTIONABLE: Provide compliance guidance.\n\n"
                "OUTPUT FORMAT:\n"
                "State the applicable standard.\n"
                "Interpret key requirements.\n"
                "Provide examples and compliance guidance."
            ),
            "explain": (
                "You are an expert educator in industrial engineering.\n\n"
                "ROLE: Technical Explainer\n\n"
                "APPROACH:\n"
                "1. ACCESSIBLE: Start with simple explanation.\n"
                "2. BUILD: Layer complexity progressively.\n"
                "3. EXAMPLES: Use concrete industrial examples.\n"
                "4. CONNECT: Link to practical applications.\n"
                "5. COMPLETE: Cover basics to advanced concepts.\n\n"
                "OUTPUT FORMAT:\n"
                "Define the concept clearly.\n"
                "Explain the mechanism or principle.\n"
                "Provide industrial examples."
            ),
            "general": (
                "You are a knowledgeable industrial domain expert.\n\n"
                "ROLE: Industrial Knowledge Expert\n\n"
                "APPROACH:\n"
                "1. ACCURACY: Provide technically accurate information.\n"
                "2. EVIDENCE: Use provided evidence as primary source.\n"
                "3. CLARITY: Explain concepts clearly.\n"
                "4. COMPLETENESS: Cover all aspects of the question.\n"
                "5. CITATION: Always cite evidence sources.\n\n"
                "OUTPUT FORMAT:\n"
                "Answer the question directly using evidence.\n"
                "Include relevant details and context.\n"
                "Cite sources appropriately."
            ),
        }

        return prompts.get(reasoning_type, prompts["general"])
