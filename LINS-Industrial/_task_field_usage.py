# -*- coding: utf-8 -*-
"""审计：TaskAnalysis 的 task / format / rich 字段在下游哪些位置被消费。
用于判断 TaskAnalyzer 目前是"行为介入"还是"仅传递(观测)"，从而决定是否可降级为纯观测。
"""
import os

_EV = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "agentic/planner.py",
    "agentic/pipeline.py",
    "agentic/solver.py",
    "agentic/organizer.py",
    "agentic/analyzer.py",
    "agentic/prompts.py",
    "agentic/prompt_builder.py",
]
# 把字段名按出现方式分类扫描，避免内联转义
PATTERNS = [
    "analysis.",
    "analysis[",
    "\.task",
    "task_type",
    "taskType",
    "reasoning_complexity",
    "requires_formula",
    "requires_multi_source",
    "requires_step_reasoning",
    "expected_evidence",
    "preferred_source",
    "\.format",
    "format=",
    "TA",
]


def scan():
    for f in FILES:
        print("=" * 20, f)
        p = os.path.join(_EV, f)
        if not os.path.exists(p):
            print("  (missing)")
            continue
        with open(p, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, raw in enumerate(lines, 1):
            l = raw.rstrip("\n")
            s = l.strip()
            if not s or s.startswith(("#", "'", '"', "*")):
                continue
            if any(pat in l for pat in PATTERNS):
                print(f"{i:4}: {s[:120]}")


if __name__ == "__main__":
    scan()
