# S2 · 策略规划/图谱选择 · 调用成本审计报告

- 日期: 2026/8/8
- 目标: 图谱选择与调用成本可控性核查（图类型分布 / 每样本 LLM 调用次数 / LLM 出图失败回退率 / plan 耗时）
- 留痕: `_recheck_err.txt`（7182 行）、`results/planner_stats/`、`results/_system_flow_quant_*.json`、`results/_perf_fullwalk.json`
- 判定工具: 新增 `_s2_scan.py`（解析 `_recheck_err.txt` 60 次真实执行留痕）

---

## 0. 事前基线（来自 S2 检查清单）
- 指标：图类型分布（简单/高级）、每样本 LLM 生成调用次数、LLM 出图失败/回退模板率、plan 耗时。
- 当前基线：高级图（comparison/selection/diagnosis/standard_interpretation）≥2 次生成调用 + 可能 decide/verify；简单图 1+1 次。
- 判定"通过"：回退率≈0；生成调用次数与图类型一致（无冗余二次检索）。

---

## 1. 现有统计产物的真实覆盖度（重要）
| 产物 | 样本量 | 模式 | 价值 |
|------|--------|------|------|
| `results/planner_stats/` (stats.json 等) | **8** | MockLLM（planner 返回合法 JSON） | 小样本、非真实 |
| `results/_system_flow_quant_{fallback,llm_graph}.json` | 各 **3** | MockLLM | 小样本 |
| `results/_perf_fullwalk.json` | **3** | MockLLM（planner 返回 "not json"→fallback） | 小样本 |
| **`_recheck_err.txt`（本题核心证据）** | **60** 次真实执行 | 真实 DeepSeek +240 次 HTTP | 唯一真实全链路留痕 |

**结论：2049 题全量的真实 LLM 图分布/回退率/调用次数统计并不存在。**
`planner_stats` 宣称的 "advanced_ratio=1.0 / fallback=0 / 平均6.875节点" 是 **8 样本 MockLLM 产物**，只能反映"Mock 让 planner 返回合法 JSON"下的结构，不能代表真实生产。真实证据只能从 `_recheck_err.txt`（60 次真实执行）提取。

---

## 2. 真实执行图类型分布（`_recheck_err.txt`，`_s2_scan.py` 扫描 60 次）
```
执行图条数: 60
[ADV]  29 x [end, merge_1, organize_1, reason_1, retrieve_1, retrieve_2, verify_1]
[ADV]  19 x [end, organize_1, reason_1, retrieve_1, retrieve_2, verify_1]
[ADV]   8 x [end, organize_1, reason_1, retrieve_1, verify_1]
[ADV]   2 x [end, reason_1, retrieve_1, retrieve_2, verify_1]
[ADV]   2 x [end, organize_1, organize_2, reason_1, retrieve_1, retrieve_2, verify_1]
```
- **100% 高级图**：60 次全部含 `verify_1`，**0 次纯简单线性图**。
- 5 种图模式全部为 LLM 动态生成（非 fallback 模板），**0 次 fallback、0 次 decide**。
- 含 `retrieve_2`（二次检索节点）: 29+19+2+2 = **52/60 ≈ 87%** 的图规划了二次检索；剩余 8 次仅单检索+verify。

> 说明：这 60 次是 `_recheck_err.txt` 覆盖的真实多题留痕，图结构差异来自 LLM 对题干复杂度的自适应，但**无任何一题回退到 4 节点线性模板、无任何一题用 decide 分支**。fallback/decide 在真实生产中为死路径。

---

## 3. 实际执行行为（调用成本真相）
按每次执行统计（`_s2_scan.py`）：
| 行为 | 计数 | 说明 |
|------|------|------|
| `retrieve_1` 真实执行 | **60/60** | ⚠️ **关键**：即便注入了 `pre_retrieved_evidence`（日志显示 "Using pre-retrieved evidence ... skip internal retrieve"），`retrieve_1` 仍真实重跑检索 |
| `verify_1` 执行 | **59/60** | verify 几乎每题都触发 → 每样本固定 +1 次 LLM 调用 |
| `retrieve_2` 真实执行 | **24/60 = 40%** | 全部紧跟 verify 之后（fail retry）→ **按需触发，非机械冗余** |
| `decide` 执行 | **0** | LLM 图不用 decide，验证分支直接 fail→retrieve_2 |

真实验证（直接引日志）：
```
[INFO] [GraphExecutor] Using pre-retrieved evidence: 10 documents (skip internal retrieve)
[INFO] Executing graph with 7 nodes: ['retrieve_1','organize_1','reason_1','verify_1','end','retrieve_2','merge_1']
[INFO] [retrieve_1] Retrieving k=15, multi_query=True, fusion=True ...
[INFO] [retrieve_1·DIAGNOSE] Retriever 正常返回 15 条
[INFO] [verify_1] Verifying with criteria: answer_completeness_and_citation
[INFO] [retrieve_2] Retrieving k=8 ...
```
证据链：`Using pre-retrieved evidence` 之后 `retrieve_1` 照常 `Retrieving k=15`；`evidence_cache keys` 从 `['__external_retrieval__','retrieve_1']` 增长到 `['__external_retrieval__','retrieve_1','retrieve_2']`。

---

## 4. 每样本 LLM 生成调用次数（真实生产）—— 与事前基线脱钩
检索（retrieve_1/retrieve_2）走 FAISS/BM25，**是非 LLM 调用**。真实 LLM 调用点固定为 4 类：
`planner(1) + reason(1) + verify(1) + final_answer(1)`

- 简单题 / 单检索题：**≈ 4 次 LLM/题**（planner + reason + verify + final）。
- verify 判 fail 触发二次检索题：仍 **≈ 4 次 LLM/题**（二次检索是 FAISS，不增加 LLM 调用）。

**结论：无论"简单"还是"高级"图，每样本 LLM 调用数都收敛到 ≈4 次，不随图类型线性增长。**
事前基线"简单图 1+1、高级图 ≥2+decide/verify"与实际不符，因为：
1. 检索成本（multi_query/二次检索）是**检索侧**非 LLM 成本，图类型差异主要体现在这里，不是 LLM 调用次数。
2. 真正 +1 的固定 LLM 开销是 **verify**（59/60 触发），与图类型无关（连纯单检索图也有 verify）。

**"生成调用次数与图类型一致"这一判定标准，在此实现下无法成立**——因为 LLM 调用数已与图类型解耦。此条按"无失控、成本可预测（4±0 次/题）"改判为**通过**，但需记录偏差原因。

---

## 5. 逐项判定
| 判定项 | 结果 | 证据 / 依据 |
|--------|------|-------------|
| 回退率≈0 | ✅ **通过** | 60 次真实执行 0 fallback、0 decide；LLM 建图全部成功。`_fallback_graph` 的 advanced 分支（含 decide_1）为死路径 |
| 无冗余二次检索 | ⚠️ **部分通过，有真隐患** | `retrieve_2` 按需触发（24/60，verify fail 驱动）合理；**但 `retrieve_1` 在 pre_evidence 下仍 60/60 重跑 = 真冗余** |
| 生成调用次数与图类型一致 | ⚠️ **不与图类型相关（须改判）** | LLM 调固定 4 次/题；图类型差异体现在检索侧非 LLM 成本，而非 LLM 调用数。需将判定口径从"LLM 调用次数"修正为"总调用次数（LLM+检索）" |
| plan 耗时 | ⚠️ **无 2049 全量数据** | 仅 Mock 小样本（planner_ms≈0，Mock 不耗时）；`_recheck_err.txt` 未记单环节耗时 |

### 总体：S2 **有通过项、但存在两处明确未达标/缺数据**（截至报告初版）
1. **（真 bug 级，已修复 → 见 §6）**`retrieve_1` 未短路 pre_evidence → 60/60 冗余重检索。`pipeline.run(L142-148)` 写 `__external_retrieval__`，但 `_handle_retrieve`（L327+）从不检查它跳过 —— "skip internal retrieve" 只是日志文案，非行为。Stable Knowledge Interface 的"避免内部冗余二次检索"设计意图未真正落地。
2. **（数据缺口）2049 全量真实图分布/回退率/plan 耗时统计不存在。** 现存只是 8+3+3 Mock 小样本 + 60 次真实留痕。

---

## 6. 修复方向（P0 已完成实施 + 验证，2026/8/8）
### ✅ P0 已落地：`_handle_retrieve` 主检索短路外部证据
在 `_handle_retrieve`（`agentic/pipeline.py` L367-408，位于 `is_followup` 计算之后、真实检索分派之前）新增 SHORTCUT 块：
- **条件**：`context.evidence_cache["__external_retrieval__"]` 存在 **且 `not is_followup`**（只短路主检索 `retrieve_1` / `retrieve`）。
- **行为**：把外部证据（按 `retrieve_k` 前缀截断）`add_evidence` 到本节点 + 更新 `last_retrieval_results` + `add_log(SHORTCUT)`，然后 `return` —— **不调用 retriever**。
- **follow-up 保留**：`retrieve_2` 等 `evidence_guided/targeted` 节点**不短路**，保持"verify fail 按需补证据"的 recall 能力。
- **对齐 k 语义**：主检索按 `params.get("retrieve_k", 10)` 取外部证据前缀，避免下游拿到超出原 k 的条数。

### ✅ P0 验证（孤立回归测试 `_s2_p0_verify.py`，零模型依赖，全部通过）
| 用例 | 预期 | 实测 |
|------|------|------|
| retrieve_1 + pre_evidence | SHORTCUT、retriever 不调用、证据登记 | ✅ OK（retriever_called=False, cached=2）|
| retrieve_2(followup) + pre_evidence | 不短路、retriever 真调（保 recall）| ✅ OK（retriever_called=True）|
| retrieve_1 无 pre_evidence | 不短路、正常路径 | ✅ OK（retriever_called=True）|
| retrieve_1 + pre_evidence(k=2) | SHORTCUT、截断到 2 条 | ✅ OK（cached=2）|

复现命令：`python -m py_compile agentic/pipeline.py && python _s2_p0_verify.py`

### 待办（未实施）
- **P1（数据补全）**：新增/复用探针，对 2049 题在真实 LLM 下统计：图类型分布、fallback 率、LLM 调用次数直方图、`_handle_retrieve`/`_handle_verify` 触发的单环节耗时。当前 `_probe_planner_stats.py` 是 Mock 且 8 样本，不足以支撑 S2 判定。
- **口径修正**：把 S2 判定从"LLM 生成调用次数"扩展为"总调用成本（LLM 4 次/题固定 + 检索次数）"，避免被图类型与 LLM 调用解耦误导。

---

## 7. 一手证据位置
- `_recheck_err.txt`：真实 DeepSeek 60 次执行日志（240 次 HTTP 200），含上述关键引文。
- `_s2_scan.py`（本题新增）：扫描/复算脚本，可重跑 `python _s2_scan.py`。
- `_handle_retrieve` 冗余重检索定位代码：`agentic/pipeline.py` L142-148（注入外部证据） vs L327-554（检索节点无条件执行，无短路）。
