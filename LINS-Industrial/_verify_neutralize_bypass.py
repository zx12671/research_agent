"""免联网验证：neutralize_task 旁路行为（默认 classify 保留 task；neutralize 强制 general）。"""
from agentic import TaskAnalyzer

q = "比较下列两种算法的计算复杂度差异"

a = TaskAnalyzer(neutralize_task=False)
ta = a.analyze(q)
print("DEFAULT task:", ta.task.value)

an = TaskAnalyzer(neutralize_task=True)
tn = an.analyze(q)
print("NEUTRAL task:", tn.task.value)
