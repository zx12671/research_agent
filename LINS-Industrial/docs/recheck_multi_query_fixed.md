# 核查：multi_query（切逗号）负增益在 agentic 里是否真的解决？

> 触发：验证 stagewise 声称"multi_query 负增益已解决"是否与 agentic 真实执行一致。
> 结论：**部分解决——"消除负增益"靠"原 query 保底+提权"真实落地，但 multi 仅为"中性偏微负"；真正稳定为正的是 hybrid；语义级改写(SemanticQueryRewriter)未接入生产**。文档与实现有关联但存在"宣称解决≠净正收益"的落差。

---

## 一、文档声称了什么（stagewise_debug_plan.md S4 段）

- L67："**20260808 路径2 已落地：原查询保底+提权后 multi/hybrid 负增益消除**（见 retriever.py）"
- L60 / L77："路径2 落地后 multi/hybrid 分别回 **33.4% / 36.0%**，负增益被消除"
- 关联脚本产物：`results/s4_recall/multi_groundfix_sim.json`、`multi_generalize_check.json`、`multi_gate_calib.json`

## 二、代码层面：路径2 确实落地（当前磁盘版本）

1. **`retriever.py::multi_query_retrieve`（L584-589）**：把【原始完整 query】的 dense 结果作为 `all_results[0]`（`_fusion_merge` 给它最高权重），子查询只做补充 → 抵消"切逗号稀释原 query"。
2. **`retriever.py::hybrid_retrieve`（L824-826）**：dense 侧**始终保底原始 query 并提权 1.2**，`use_multi_query=False`（不切逗号）。
3. **`pipeline.py::_handle_retrieve`**：hybrid 分支强制 `use_multi_query=False`（L390）。

## 三、验证产物存在且已跑（非只停留在注释）

`multi_generalize_check.json`（3 seed × 42 题 = 126 行）：

| seed | single | multi | hybrid | multi≥single题占比 | hybrid≥single题占比 |
|---|---|---|---|---|---|
| 7 | 30.5% | 30.5% | **32.5%** | 90% | 95% |
| 11 | 33.3% | 31.9% | **34.9%** | 83% | 93% |
| 42 | 28.2% | 28.4% | **30.0%** | 79% | 90% |
| 全量 Δ(multi−single) | pos=23 / neg=20 / zero=83 | | | | |

`multi_gate_calib.json`：165 个子查询中，有效(useful) top1 中位 0.749 vs 无效 0.652，p25 0.702 vs 0.603 —— **分布高度重叠，相似度门控可分性差**。

## 四、精确结论（文档"宣称解决" vs 真实执行）

### A. "消除了 multi 相对于 single 的负增益" —— ✅ 基本属实，但要看口径
- 改造前（20260807）：multi cov@10 **24.1%**（明确劣于 single 33.7%，−9.6pp）；hybrid 30.0%。
- 改造后：multi 回 **~31.9~33.4%**（seed11 仍 −1.31%，seed7 ≈0，seed42 +0.21%）。
- **准确说法**：路径2 把 multi 从"大幅负增益(−9.6pp)"修到"中性偏微负(约 0±0.5%)"——**"大幅负增益"消除属实**；但 **multi 并未变成净正收益**（seed11 仍负）。

### B. agentic 真实执行 —— 确曾大量走 multi_query，且现在仍是可选通路
- recheck 日志 `_recheck_err.txt` 中 `multi_query=True` 出现 **55 次**，`Using multi_query_retrieve` 频繁出现。
- 触发来源：`planner.py` `task_configs` 里 `TaskType.COMPARISON` 设 `multi_query: True`（L517）、`retrieve_k:15`；base 默认 `multi_query: False`（L491）。
- → **agentic 里是否走切逗号 multi，取决于 planner 给的是哪个 task 模板**；COMPARISON 等模板仍会触发。它现在走的是**已修复(原query保底)**的 `multi_query_retrieve`，非旧版。

### C. 真正稳定为正的是 hybrid（负增益的"解药"其实是它）
- `hybrid_gain_beyond_single` 三 seed 全部为正：+1.9% / +1.7% / +1.9%，题占比 90~95%。
- → 若要"比 single 更好"的检索增强，**默认开 hybrid（单查询 dense+BM25）比开 multi_query 更有据**。

### D. 仍未解决的残留：语义级多视角改写**未接入生产**
- `SemanticQueryRewriter`（`agentic/query_rewriter.py`，P1 推荐的"语义级多视角"）**仅存在于实验脚本**（`_ab_mv_query_rewrite.py`、`_exp1_agentic_rag_opt.py`）；
- **`pipeline.py` / `planner.py` 均未 import / 引用**（`Select-String` 无命中）；
- 自身文档 `agentic_mv_query_rewrite_ab.md` L83 明言"生产代码零改动"。
- → 切逗号(`ked.decompose`) 仍是生产唯一 multi 通道；语义鸿沟型 hard-miss（VDRM 类）**切逗号救不了**（`agentic_query_vision_ab_result.md` §3.3 亦如此判断）。

---

## 五、给决策者的一句话
- **"负增益消除"的实现是真的（原 query 保底代码已落地、有 3-seed 产物）**，但只能算"把 multi 从明显劣化修到中性"；
- **真正能带来正向提升的是 hybrid(dense+BM25)**（三 seed 稳定 +1.7~1.9%）；
- **要治本（语义鸿沟 hard-miss）需把 `SemanticQueryRewriter` 从实验接入生产**——目前它只躺在脚本里，stagewise 若标"已解决"需补充这一条未闭环。

## 复现
```bash
cd LINS-Industrial
python _verify_multi_generalization.py              # 3-seed 通用性验证 → multi_generalize_check.json
python _calibrate_multi_gate.py                     # 子查询质量门控标定 → multi_gate_calib.json
python _sim_multi_groundfix.py                      # 离线"原 query 保底"模拟 → multi_groundfix_sim.json
```
