"""
_verify_format_analyzer.py — 聚焦验证 task analyzer 的题型(format)改造。

覆盖:
  1. normalize_format 四种标准枚举 + 中文标签 + 空值兜底
  2. infer_format_heuristic (填空/选择/计算/问答 启发式)
  3. TaskAnalyzer.analyze(question, format=...) → TaskAnalysis.format 归一化正确
  4. TaskAnalysis.to_dict / from_dict 的 format 字段往返一致性
"""
import sys

sys.path.insert(0, r"c:\Users\duodu\Desktop\LINS\LINS-Industrial")

from agentic.task_types import (
    QUESTION_FORMATS,
    TaskType,
    TaskAnalysis,
    normalize_format,
    infer_format_heuristic,
)
from agentic.analyzer import TaskAnalyzer

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


print("== 1) QUESTION_FORMATS 枚举 ==")
check("四种标准题型", QUESTION_FORMATS == [
    "QA", "FillBlank", "MultipleChoice", "Calculation"])

print("\n== 2) normalize_format ==")
check("英文原样保留", normalize_format("QA") == "QA")
check("中文 问答题→QA", normalize_format("问答题") == "QA")
check("中文 填空题→FillBlank", normalize_format("填空题") == "FillBlank")
check("中文 选择题→MultipleChoice", normalize_format("选择题") == "MultipleChoice")
check("中文 计算题→Calculation", normalize_format("计算题") == "Calculation")
check("空值→启发式", normalize_format("", "计算某设备的功率？") == "Calculation")
check("None→启发式问答", normalize_format(None, "什么是变频调速？") == "QA")

print("\n== 3) infer_format_heuristic ==")
check("含___ → FillBlank", infer_format_heuristic("请填写____的内容") == "FillBlank")
check("含A.选项 → MultipleChoice",
      infer_format_heuristic("A. 正确 B. 错误 以下哪个说法对？") == "MultipleChoice")
check("含计算词 → Calculation", infer_format_heuristic("请计算该变压器的功率") == "Calculation")
check("默认为 QA", infer_format_heuristic("介绍一下这个设备") == "QA")

print("\n== 4) TaskAnalyzer.analyze(format=) ==")
an = TaskAnalyzer()
res1 = an.analyze("对比 A 与 B 的差异", format="选择题")
check("选择题 format 落位", res1.format == "MultipleChoice", res1.format)
# task (推理任务) 仍走关键词规则，不受 format 影响
check("task 仍为 COMPARISON(关键词覆盖)",
      res1.task == TaskType.COMPARISON, res1.task)
check("原始 question 保留", res1.original_question == "对比 A 与 B 的差异")

res2 = an.analyze("计算功率是多少", format="计算题")
check("计算题 → Calculation", res2.format == "Calculation", res2.format)
check("task 为 CALCULATION", res2.task == TaskType.CALCULATION, res2.task)

res3 = an.analyze("介绍一下原理")
check("未传 format 问答 → QA", res3.format == "QA", res3.format)

print("\n== 5) TaskAnalysis 序列化往返 ==")
d = res1.to_dict()
check("to_dict 含 format", d.get("format") == "MultipleChoice", d)
cls1 = TaskAnalysis.from_dict(d, question=res1.original_question)
check("from_dict format 往返", cls1.format == "MultipleChoice", cls1.format)
check("from_dict 中文标签兼容",
      TaskAnalysis.from_dict({"format": "填空题"}, question="xxx").format == "FillBlank")

print(f"\n===== 结果: {PASS} PASS / {FAIL} FAIL =====")
sys.exit(0 if FAIL == 0 else 1)
