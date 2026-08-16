# S3 图执行阶段检验报告 —— 判定【通过：reason 现实执行，TaskSolver 不接入】

> 状态：检验完成 · **未改生产分派（不接入 TaskSolver）** · 可复现留痕
> 生成：20260808 · 对标 `docs/stagewise_debug_plan.md` ##S3
> 脚本：`_diag_s3_executor.py`（真实 faiss + MockLLM + ThrowingSolver 刺探，N=12，seed=7）
> 产物：`results/_diag_s3_executor.json`

---

## 一、检验项与证据

### 1) 节点实际命中率（reason/decide/verify）
走完整 `AdaptiveAgenticPipeline.run`（TaskAnalyzer→StrategyPlanner→GraphExecutor），
逐题统计各类型节点真实执行的次数（MockLLM 记录调用点）：

```
节点命中率/题  = retrieve 1.0 · organize 1.0 · reason 1.0 · end 1.0
LLM 调用点/题  = planner 1.0 · reason 2.0
```
（简单图无 decide/verify/merge，属预期——它们仅 advanced 图触发；reason 节点**真实执行**，走标准 prompt+PromptBuilder。）

### 2) `TaskSolver` 接入率（死代码检验）—— 决定性问题
向 `GraphExecutor` 注入 **`ThrowingSolver`**（任何方法被调即抛异常）。12 题全链路跑完：

```
TaskSolver 刺探被触碰?  False
TaskSolver 接入率      = 0.0
```
且静态核实 `agentic/pipeline.py` 中 `self.solver` **仅出现在 `__init__` 赋值（L104）**，
`_handle_reason`/`_handle_decide`/`_handle_verify`/`_handle_merge` 等全部处理器**从不引用**它。

→ **铁证：`TaskSolver.solve()` 从未被生产调用，为「被注入但从不通电」的死代码。**

### 3) evidence→prompt 传递（S5 已覆盖，非本阶段重复）
organize/get_context 保真、0 缺失、merge 膨胀已于 20260808 修复（见 S5 / `s5_merge_inflate.json`）。

---

## 二、裁决（满足 S3"存在明确方案=接入或移除"）

**选择「不接入 TaskSolver」。** 依据：

1. 现有 reason 标准 prompt + PromptBuilder **已过 S6 验收**（评分公平、低分主因归 S4 召回，
   端到端瓶颈不在 reason prompt）——接入 solver 不能解决当前主要失分点；
2. 对比劣化风险：`TaskSolver._solve_general` 系统提示仅"industrial domain expert"，**无输出格式
   约束**（比标准 prompt 更简陋）；`_solve_with_workflow` 依赖 `directive.params.workflow_steps`，
   而模板图 reason 节点仅给 `["synthesize"]`（见 planner）——强行接入=换掉已验证的标准 prompt，
   存在降低答案可评分性的风险；
3. 收益小：当前 task 恒 general（analyzer 中性化），task-specific workflow 无可触发分支，
   solver 的多步推理优势无法体现。

**处置**：
- `TaskSolver` **保留**为"可选、未来多步推理再做"模块；在 `solver.py` 顶部加「未接入」状态标注
  （防误用/误维护）；
- **不强制注入、不改 reason 分派**（生产零改动，与本报告基线一致）。

---

## 三、判定

> **S3 = 通过**：明确方案已确认（不接入 TaskSolver + 加未接入标注）；merge 项已修复达标；
> prompt 长度受控。**不再列为阻塞项。**

## 复现
```bash
python _diag_s3_executor.py --max-q 12 --seed 7    # → results/_diag_s3_executor.json
```
