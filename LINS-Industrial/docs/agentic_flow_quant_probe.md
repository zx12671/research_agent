# Agentic 执行流水线确定性全链路量化探针（`_system_flow_quant.py`）

> 本文是对 `docs/agentic_quant_end_to_end.md` 的**低成本复验补充**：
> 既有文档以「代码精读 + 需 DEEPSEEK_KEY 的在线实验」为主；本文新增一个**不访问 LLM API 的确定性探针**，
> 仅用真实 faiss 检索 + MockLLM，即可对 analyzer → planner → retriever → organizer → executor 全链路逐环节量化、可复现。
>
> 覆盖工具：`LINS-Industrial/_system_flow_quant.py`
> 产物：`results/_system_flow_quant_{fallback,llm_graph}.json`（rows + aggregated）

---

## 1. 动机与设计

既有文档结论（与本探针一致）：
- `TaskAnalyzer` 已中性化，`task` 恒 `GENERAL`，无 LLM；
- `StrategyPlanner` LLM 出图失败回退 4 节点模板；GENERAL 走简单图 `retrieve→organize→reason→end`；
- `_handle_reason` 跳过 solver、`format_prompt("general")` 直连 LLM；
- 最终答案由 `_produce_final_answer` 从完整证据再生成一次（第 2 次 general LLM 调用）。

本探针把这些**用一次可复现运行全部实测出来**，并把「LLM 调用点与次数」这类运行时细节量化。

**运行**（需真实 faiss 索引，已存在于 `knowledge_corpus/index/`）：
```powershell
cd LINS-Industrial
python _system_flow_quant.py
```

**MockLLM 能力**（`_system_flow_quant.py` 的 `MockLLM` 类）：
- OpenAI 兼容 `.chat.completions.create()` / `.chat()` 接口；
- 按 prompt 特征自动识别调用用途 `planner / decide / verify / reason`；
- 记录每次调用的 `kind / prompt_len / max_tokens`；
- `planner_mode=fallback` → 返回坏 JSON，强制走模板图；`planner_mode=llm_graph` → 返回合法 GENERAL JSON。

---

## 2. 全链路实测数据（35 题 × 2 模式，零 LLM 成本）

样本：`results/probe_evidence_forward_e2e_35.json`（7 能力 × 5 题）。
两模式（fallback 图 / LLM 生成图）结果一致，合并列示。

### 环节 1 — TaskAnalyzer（统计）
| 指标 | 实测 | 结论 |
|---|---|---|
| task=GENERAL 占比 | **35/35 = 1.0** | 完全中性化 |
| confidence 是否恒定 | 全相等 | 走中性默认（0.95） |
| 布尔标签均值为真/题 | **0.0** | requires_multi_source/formula/step_reasoning 全 False |
| format 分布 | QA 33, Calculation 2 | 直读 35 题池标注，非 LLM 判定 |
| LLM 调用 | **0** | 无 |

### 环节 2 — StrategyPlanner（统计，两模式一致）
| 指标 | 实测 |
|---|---|
| 平均节点数 | **4** |
| 节点类型分布 | retrieve, organize, reason, end（各 1） |
| retrieve_2 / decide / verify 出现率 | **0 / 0 / 0** |
| all_same_topology | True（同源题图结构完全相同） |

> 中和化连锁再次确认：GENERAL 恒落简单图，**无二跳检索、无 decide、无 verify**。

### 环节 3 — OpenDomainRetriever（真实 faiss，bge-small-zh，55095 块）
| 指标 | 实测 |
|---|---|
| 每题命中块数 | 恒 10 / 10 |
| 命中均值 score | **0.652**（min 平均 0.622） |
| 平均检索耗时 | **约 15–17 ms** |
| 检索失败 | 0 / 35 |

> 与既有文档「检索是主瓶颈」一致：score 均值不高、长尾低分块（0.62 档）进入 top-10，
> 正是「top-10 混入同题材相关但判定无用的块 → 稀释」现象（见 §9.6/9.8 纯度分析）的检索层根源。

### 环节 4 — EvidenceOrganizer（去重 + topic 分组 + get_context）
| 指标 | 实测 |
|---|---|
| 平均输入块数（去掉重） | 10 |
| 平均去重后块数 | **9.91**（平均仅丢 0.086 块） |
| 平均去重率 | **0.9%** |
| 平均分组数 | **4.9**（按 industry/capability） |
| 平均 get_context 字符量 | **~1735 字符/题**（喂给 reason 的证据规模） |

> 印证组织是 recall-safe：Jaccard≥0.97 才去重，几乎不丢块；上下文规模 ~1.7K 字符。

### 环节 5 — GraphExecutor（走完整 DAG，MockLLM）
| 指标 | 实测 |
|---|---|
| 实际执行节点序 | retrieve_1 → organize_1 → reason_1 → end（4 节点） |
| 每题 LLM 调用 | **2**（均为 general 模板，max_tokens=2048） |
| 这 2 次构成 | **① reason 节点中间推理 ② `_produce_final_answer` 最终答案**（第 2 次即定分答案） |
| 每个 reason prompt 长 | ~3200 字符（证据 ~1735 + 系统指令） |
| second_retrieval_triggers | 0 / 题 |
| 平均 executor 耗时 | ~18–20 ms |

> **重要归因**：simple 图 `reason` 节点在探测里表现为 2 次「general 模板」LLM 调用，
> **不是 bug，而是设计**——一次是 `_handle_reason` 中间推理（其 prompt 长度 3204），
> 一次是 `_produce_final_answer` 最终答案（3206），后者才是分数真正决定者（与既有文档 §8.1 第 2 点一致）。

---

## 3. 与既有文档结论的交叉验证

| 既有文档断言 | 本探针实测 | 一致 |
|---|---|---|
| task 恒 GENERAL，无 LLM | task 1.0 / LLM 0 | ✅ |
| 简单图 4 节点、无 decide/verify/二跳 | 4 节点 / 0 / 0 / 0 | ✅ |
| reason 跳过 solver、直连 general 模板 | reason 直连（2 次含最终答案） | ✅ |
| 组织 recall-safe、证据 ~K 级 | 去重率 0.9%、ctx ~1735 字符 | ✅ |
| 检索是主瓶颈 / top-10 混入长尾 | score 均值 0.652、min 0.622 | ✅ |
| 最终答案 = `_produce_final_answer`（第 2 次 LLM） | 第 2 次 general 调用即定分答案 | ✅ |

---

## 4. 一句话结论

`_system_flow_quant.py` 提供了一条**不花钱、秒级、可复现**的验证路径，
完整重现了既有文档的端到端画像：链路 `Q → analyze(0 LLM) → plan(4节点图) → retrieve(10块,0.65) → organize(≈不丢,1.7K字符) → execute(2次general LLM调用: 中间推理+最终答案) → 评分`。
任何对 analyzer/planner/organizer/executor 的改动，都可以先跑本探针看链路是否被破坏、各环节输入输出是否漂移，再决定是否上在线 API 实验。

---

## 5. 真实 LLM / Mock 可选开关（`--real`）

为兼顾「廉价回归」与「真实准确性」，探针新增 `build_llm()` + 计数包装 `CountingLLM`：

| 模式 | 触发 | 量化能力 |
|---|---|---|
| MockLLM（默认） | 无参数 / 无 key | 结构、调用点、证据流（不烧钱、确定性） |
| CountingLLM（真实） | `--real` 或 `SYSTEM_FLOW_REAL=1` 且存在 key | 真实端到端准确率 + 真实规划图/分支 |

Key 来源：环境变量 `SYSTEM_FLOW_REAL_API_KEY`（优先）→ `experiments.config.DEEPSEEK_KEY`（回退）。
**无可用 key 时自动回退 MockLLM，绝不意外烧钱。**

```powershell
python _system_flow_quant.py --mode fallback --max-q 5   # 零成本，仅 5 题
python _system_flow_quant.py --real --max-q 5            # 真实 LLM，建议限题(烧 token)
```

**`CountingLLM` 设计**：包装真实 OpenAI 客户端，暴露 `chat.completions.create()`（pipeline 走真实路径），
同时复用 `MockLLM._detect_kind` 记录每次调用的 `kind/prompt_len/max_tokens`——真实模式下同样能拿到调用点与 prompt 规模。

### 5.1 真实模式实测（1 题冒烟，fallback 名但真实规划）
| 项 | 值 |
|---|---|
| answer_is_mock | **False（真实 DeepSeek 文本）** |
| 真实规划图节点序 | retrieve_1 → organize_1 → **decide_1** → reason_1 → **verify_1** → end（6 节点） |
| 真实 LLM 调用序列 | **decide → reason → verify → reason**（4 次） |
| answer 开头 | "根据检索到的证据，电磨（直磨机）在高速使用小磨头作业时可能引发以下问题及相应的安…" |

> **关键发现（Mock 掩盖的真实行为）**：即使 `task=GENERAL`、`--mode fallback`，
> 真实 planner（deepseek）仍会给简单 FACT 题自主生成**带 decide+verify 的高级图**（6 节点、4 次 LLM）。
> MockLLM 固定回退 4 节点图，**测不到这一真实分支**。这正是 `--real` 开关的增量价值——
> 它能暴露"真实规划器在 task 中性化后仍自主加分支"这一行为（提示：简单题也可能无谓多消耗 2 次 LLM token）。

### 5.2 使用建议
- **回归/CI**：默认 Mock 路径（快、零成本），验证链路不被改动破坏。
- **准确率/分支行为**：`--real` 上真实 API；建议 `--max-q` 限题控制 token。
- **两者互补**：先 Mock 确认结构不漂移，再 `--real` 测真实得分与规划分支。

