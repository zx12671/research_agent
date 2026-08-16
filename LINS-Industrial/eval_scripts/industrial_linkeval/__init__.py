"""
industrial_linkeval: 工业版 Link-Eval 评测框架

自动检测问题格式 (QA / Fill-in-Blank / Multiple Choice)，
按格式应用不同的评估指标，最终输出统一的评测报告。

使用方式:
    from eval_scripts.industrial_linkeval import IndustrialLinkEval

    evaluator = IndustrialLinkEval()
    report = evaluator.evaluate(questions=questions, answers=answers, references=references)
    print(report.to_markdown())
"""

from .format_detector import FormatDetector, QuestionFormat
from .qa_evaluator import QAEvaluator
from .blank_evaluator import BlankEvaluator
from .mc_evaluator import MCEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .linkeval_core import IndustrialLinkEval, EvalReport, EvalSample

__all__ = [
    "IndustrialLinkEval",
    "EvalReport",
    "EvalSample",
    "FormatDetector",
    "QuestionFormat",
    "QAEvaluator",
    "BlankEvaluator",
    "MCEvaluator",
    "RetrievalEvaluator",
]
