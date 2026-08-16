# audit：纯规则分类 / KEYWORD_MAP 是否可删（证据核对）

## 0. 待答问题
1. 纯规则分类（keyword matching，零 LLM、零延迟、确定性、不改写查询）有用吗？用户倾向删除。
2. KEYWORD_MAP 曾被确认"无用"，是否删除？

## 1. 关键证据：task 值是否被下游"真实消费"（代码级核对）

结论：**是**。task 不是 OBSERVATION-ONLY，它在 4 处真实驱动行为：

| 下游 | 消费点（文件:行） | 作用 |
|---|---|---|
| planner | planner.py:96/481 `task=task_analysis.task`；599 `cfg=task_configs.get(task, general)`；507~591 每类 task_config | task 选择建图/检索配置 |
| prompt_builder | prompt_builder.py:101~108 派发表；129 `task_analysis.task` | 选 7 套"专门 prompt"模板 |
| pipeline | pipeline.py:502~505 `task_type=graph.task.value` → `organize(...,task_type=)` | 把 task 下传 organizer |
| organizer | organizer.py:384 `TASK_GROUP_PRIORITY.get(task_type)` | 设证据组优先级 |

因此"task 分类"并非装饰性字段。

## 2. 已有 A/B 结论（agentic_task_dimension_evidence_prior.md）
- task→Solver 专门 prompt：avg_delta=0（无增益）。
- task/format→capability 先验重排：前提覆盖仅 35.8%（39/109）；hit@1/3/5 显著下降（有害）。
- 处置共识：task 分类不作为 Solver/检索先验信号，纯观测用途。

## 3. 两个问题的分析

### 问题一：纯规则分类本身有用吗？
- **优点**：零成本、零延迟、确定性、不改写查询——无害。
- **缺点**：它驱动上述 7 类专门分支（planner task_configs / 7 套模板 / 证据优先级），而这些分支 A/B 证明无增益甚至有害（因为分类先验覆盖低、且 KED+dense 已把真值排进窗口）。
- **量化立场**：keyword 命中后"分流到专门模板"的收益 ≈ 0；唯一不伤害的路径是"落到 GENERAL"。因此纯规则分类的**边际价值 ≈ 0**。
- **裁决**：可删除，但必须连坐下游 7 类专门分支一起收敛到 general，否则单删 analyzer 会留下不受控的分类残留。

### 问题二：KEYWORD_MAP 是否删除？
- 既有确认"无用"指的是"用 task 分类做强信号"无增益——**不是**"字段没被消费"。
- KEYWORD_MAP 的作用仅是给 analyzer 内部 task 赋值（comparison/diagnosis/…）；真正有害的是下游拿这 7 类去选模板/起重排先验。
- 保留 KEYWORD_MAP + neutralize=True（强制 general）= 分类被旁路，observation-only；删除 KEYWORD_MAP 且 analyze 直接返回 GENERAL = 行为等价、代码更少。
- **裁决**：可删，等价且更简洁；删除后需确保所有下游（planner/prompt_builder/organizer）默认兜底都是 general。

## 4. 建议的删除方案（安全收敛）
1. `analyzer._classify_by_keywords` 及其 KEYWORD_MAP 删除；`analyze()` 直接返回 GENERAL 中立默认（保留 format 归一化，题型仍有价值）。
2. `planner.task_configs` 中 7 类配置合并为 general 默认（或保留但 map 全指 GENERAL）。
3. `prompt_builder` 的 7 套专门 `_build_*` 保留或删；若保留需保证 template 选择恒为 general，避免死代码误导。
4. `organizer.TASK_GROUP_PRIORITY` 收敛到 general 默认集。
5. 删除/旁路后跑一遍 `_ab_evidence_prior_rerank.py` 与 exp1 回归，确认 hit 与分数不劣于现状。

## 5. 一句话结论
> 纯规则 keyword 分类"命中的 7 类专门分支"无增益且可能有害，KEYWORD_MAP 与专门模板/优先集一起收敛到 general 是安全且更简的处置；但**不能只删 analyzer**，必须连坐下游对 task 的消费，并做回归验证。
