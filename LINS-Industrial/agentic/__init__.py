"""
agentic: Adaptive Agentic RAG for industrial question answering.

This module implements a fully adaptive Agentic RAG architecture where
TaskAnalyzer acts as the true controller of the pipeline.

Architecture:
    Question
        ↓
    TaskAnalyzer (LLM generates rich semantic description)
        ↓
    TaskAnalysis (task, confidence, reasoning_complexity, etc.)
        ↓
    StrategyPlanner (LLM generates ExecutionGraph — a workflow DAG)
        ↓
    GraphExecutor (walks graph, dispatches nodes by type)
        ┌──────────────────────────────────────┐
        │  retrieve_1 → organize_1 → reason_1  │
        │    ↓                                 │
        │  decide (branch)                     │
        │    ├─ yes → verify_1 → end           │
        │    └─ no  → retrieve_2 → merge_1     │
        └──────────────────────────────────────┘
        ↓
    Answer

Key Design:
- TaskAnalyzer is the entry point of adaptive decision making
- Its output (TaskAnalysis) drives every downstream decision
- StrategyPlanner generates ExecutionGraph (workflow DAG)
- Each adaptive component is independently configurable for ablation
- Rich semantic descriptions are LLM-generated, not hardcoded
"""

# Type definitions
from .task_types import (
    # Task taxonomy
    TaskType,
    # Rich analysis output
    TaskAnalysis,
    # Execution graph (workflow DAG) and graph nodes
    GraphNode,
    ExecutionGraph,
    # Organized evidence
    OrganizedEvidence,
    # Execution directive (used by organizer and solver)
    ExecutionDirective,
)

# Pipeline components
from .analyzer import TaskAnalyzer
from .planner import StrategyPlanner
from .organizer import EvidenceOrganizer
from .solver import TaskSolver

# Pipeline orchestration
from .pipeline import AdaptiveAgenticPipeline, PipelineResult, GraphExecutor, ExecutionContext

# Prompt utilities
from .prompts import (
    SYSTEM_PROMPT,
    ANALYZER_PROMPT,
    PLANNER_PROMPT,
    build_solver_prompt,
    get_prompt,
    format_prompt,
    PROMPT_REGISTRY,
)

__all__ = [
    # Types
    "TaskType",
    "TaskAnalysis",
    "GraphNode",
    "ExecutionGraph",
    "OrganizedEvidence",
    "ExecutionDirective",
    # Components
    "TaskAnalyzer",
    "StrategyPlanner",
    "EvidenceOrganizer",
    "TaskSolver",
    # Pipeline
    "AdaptiveAgenticPipeline",
    "PipelineResult",
    "GraphExecutor",
    "ExecutionContext",
    # Prompts
    "SYSTEM_PROMPT",
    "ANALYZER_PROMPT",
    "PLANNER_PROMPT",
    "build_solver_prompt",
    "get_prompt",
    "format_prompt",
    "PROMPT_REGISTRY",
]
