"""
analyzer.py: TaskAnalyzer — NEUTRALIZED task-dimension version.

Rationale (audit doc: docs/analyzer_rule_classification_audit.md):
  - Existing A/B (agentic_task_dimension_evidence_prior.md) showed the task
    classification dimension adds NO gain (Solver prompt avg_delta=0) and is
    harmful as an Evidence-priority prior (hit@1/3/5 dropped; prior coverage
    only 35.8%).
  - Rule-based keyword classification merely routed queries into 7 task-
    specific branches (planner task_configs, prompt_builder templates,
    organizer evidence priority) that provide no benefit.
  - Decision: the task dimension is now neutralized at the source. analyze()
    always returns TaskType.GENERAL with neutral defaults, so ALL downstream
    consumers (planner / prompt_builder / organizer) take their general path.
    No query modification, no LLM call, zero latency, deterministic.
"""

import logging
from typing import Any

from .task_types import TaskType, TaskAnalysis, normalize_format

logger = logging.getLogger(__name__)


class TaskAnalyzer:
    """
    Neutralized TaskAnalyzer.

    No keyword classification, no LLM call, no query modification.
    Always returns TaskType.GENERAL with neutral defaults so the downstream
    planner / prompt_builder / organizer all take their general / neutral path.
    """

    def __init__(self, llm_client: Any = None, model_name: str = "deepseek-chat",
                 temperature: float = 0.1, max_retries: int = 2,
                 neutralize_task: bool = False):
        """
        Initialize analyzer.

        NOTE: llm_client is accepted for backward compatibility but NOT used.
        The task-dimension is always neutral (GENERAL), so `neutralize_task`
        is retained solely for API compatibility; the behavior is identical
        regardless of its value.
        """
        self.llm_client = llm_client  # kept for API compat, NOT called
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.neutralize_task = neutralize_task  # retained for API compat

    def analyze(self, question: str, format: str = "") -> TaskAnalysis:
        """
        Analyze a question WITHOUT any LLM call and WITHOUT task classification.

        Always returns a neutral GENERAL TaskAnalysis that preserves the original
        query, so retrieval ranking is not distorted.

        Args:
            question: The raw industrial question
            format: The question format / 题型 truth label (e.g. "问答题", "QA").
                    Stored (normalized) on the returned TaskAnalysis.format; the
                    `task` (8-class reasoning task) field is always GENERAL.

        Returns:
            TaskAnalysis with neutral GENERAL defaults, no LLM call needed.
        """
        # Task dimension is neutralized at the source: always GENERAL.
        task_type = TaskType.GENERAL

        # Normalize the `_format` truth value → standard 题型 enum (QA/FillBlank/...)
        fmt = normalize_format(format, question)

        # Build TaskAnalysis with neutral defaults
        # CRITICAL: All fields use default/neutral values that won't distort retrieval
        return TaskAnalysis(
            task=task_type,
            confidence=0.95,  # high confidence since we're being conservative
            reasoning="Neutralized task dimension (GENERAL / observation-only)",
            original_question=question,
            format=fmt,
            reasoning_complexity="medium",
            requires_multi_source=False,
            requires_formula=False,
            requires_step_reasoning=False,
            preferred_source="standard",
            expected_evidence="general",
        )

    def _call_llm(self, prompt: str) -> str:
        """Not used — kept for API compatibility only."""
        return '{"task": "general", "confidence": 0.95, "reasoning": "neutralized"}'

    def _parse_response(self, response: str) -> dict:
        """Not used — kept for API compatibility only."""
        return {"task": "general", "confidence": 0.95}
