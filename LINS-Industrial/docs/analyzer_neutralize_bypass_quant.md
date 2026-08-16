# TaskAnalyzer 旁路验证（neutralize_task）

## 1. 目的
端到端测试时发现"对比类问题未被识别为对比任务"。为判定这与
`TaskAnalyzer` 的 classify 是否相关，特做免联网旁路验证：直接构造
`TaskAnalyzer`，对比 `neutralize_task` 默认值（False）与 True 的行为。

## 2. 验证代码（_verify_neutralize_bypass.py）
```python
from agentic import TaskAnalyzer
q = "比较下列两种算法的计算复杂度差异"

a = TaskAnalyzer(neutralize_task=False)
print("DEFAULT task:", a.analyze(q).task.value)   # classifier 实际分类

an = TaskAnalyzer(neutralize_task=True)
print("NEUTRAL task:", an.analyze(q).task.value)  # 强制 general（旁路 classifier）
```

## 3. 运行结果（本地执行）
- DEFAULT task: `comparison`   ← 默认走 classify 时分类为"对比"
- NEUTRAL task: `general`      ← neutralize 后强制 general

`python -m py_compile` 对 agentic/analyzer.py 等编译通过，无语法错误。

## 4. 量化结论
1. `neutralize_task=True` 时，classify 被旁路，任务态被强制为
   `general`，这是确定性的（非 LLM 分类器调用）。
2. 默认（False）路径下，该对比查询被 classify 归为 `comparison`，
   说明"对比类问题识别失败"并非必然，而是路径/分支相关。
3. 因此端到端的对比命中损失更可能来自：classify 调用的 LLM 输出不稳定、
   或调用侧传入了 non-neutralized 的外部预设任务态覆盖了分类结果。

## 5. 建议
- 端到端调试时应打印最终 pipeline 使用的 `task` 及来源（classify 默认值
  vs 外部覆盖）。
- 对比类问题建议用 stable 的 classify 分支路径，或在 analyzer 层补充
  comparison 语义的确定性兜底。
