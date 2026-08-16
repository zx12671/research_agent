# Agentic 系统端到端量化分析（任务分析 → 结果输出）

> 目标：把 LINS-Industrial Agentic RAG 从「问题进来」到「答案/指标出去」的每一条职能链，
> 逐一对应到代码路径，给出每一阶段的实测量化度量与归因。所有数字均来自仓库内已实际运行的实验产物。
>
> 覆盖组件：`agentic/pipeline.py`(`AdaptiveAgenticPipeline` / `GraphExecutor`)、
> `agentic/analyzer.py`、`agentic/planner.py`、`retrieval/retriever.py`、`agentic/organizer.py`、
> `metrics/industrybench_scorer.py`、`eval_scripts/.../retrieval_evaluator.py`。

---

## 0. 总体调用链与数据流

```
CSV(2049) → question
  ↓ 阶段1  TaskAnalyzer.analyze() ────── TaskAnalysis (task 恒 GENERAL，无 LLM；format 直读标注)
  ↓ 阶段2  StrategyPlanner.plan() ────── ExecutionGraph (LLM 出图，失败回退模板)
  ↓ 阶段3  GraphExecutor.execute() ───── ExecutionContext 状态机
  ↓ 阶段4  retrieve 节点 ─────────────── OpenDomainRetriever.hybrid_retrieve
  ↓ 阶段5  organize / merge 节点 ─────── EvidenceOrganizer 去重/分组/get_context
  ↓ 阶段6  reason / decide / verify ──── PromptBuilder.build → format_prompt → LLM
  ↓ 阶段7  _produce_final_answer() ───── Standard Prompt + PromptBuilder → LLM → answer
  ↓ 阶段8  评分 / 指标 ─────────────── RuleBasedScorer + RetrievalEvaluator → report/metrics
```

关键事实（代码实测）：**Analyzer 不调用 LLM（已中和化，task 恒 GENERAL，format 直读标注）**、
**reason 节点未走 TaskSolver.solve()**、
**外部证据预注入不强制跳过 retrieve**（`9.1/9.3` 见 `agentic_framework_and_runtime.md` §）。

---

## 阶段 1 — 任务分析（TaskAnalyzer）

> **重要口径澄清（回应"task 为何有 73%/48% 的疑问"）**：当前 `analyzer.py` 是**已中和化**版本，
> 它的 `analyze()` **恒返回 `task=GENERAL`，不做关键词分类、不调 LLM**；只有 `format`（题型）
> 通过 `normalize_format(format, question)` **直接读取传入的 CSV 真实标注 `_format` → 100% 保真**。
> 因此**对当前实现而言**：`task` 恒为 general（"准确率"口径不适用）、`format` 划分 = 100%（读标注，非推断）。

**现状（当前代码实现，实测）**
- 入口 `AdaptiveAgenticPipeline.run()` → `analyzer.analyze(question, format=_format)`。
- `task = TaskType.GENERAL` 恒定；`format` 由 `normalize_format(_format, question)` 归一化后直存真值（100%）。
- 语义字段（`requires_multi_source` 等）用保守中性默认值，全部不扭曲检索。
- `neutralize_task` 仅作 API 兼容参数，不再影响行为。
- 复现现状：`python _verify_neutralize_bypass.py`（default 与 neutralize 均输出 general）；
  `python _eval_task_analyzer.py` → 现重跑 41 题 task 全判 general（准确率崩到 4.88%）、2049 题 100% 落 general。

**现状 vs 历史口径（区分两个完全不同的"task 数字"）**
| 口径 | task 预测结果 | 备注 |
|---|---|---|
| 当前中和化 analyzer | **恒 general**；`format` 直读=100% | `analyzer.py:68-72` + `_verify_neutralize_bypass.py` 实测 |
| 旧关键词分类 analyzer（历史评测，已不生效） | 41 题手标 8 类判定 30 对=**73.17%**；2049 题落 general **48.12%** | 仅存于 `task_analyzer_classification_eval.json`（旧版 `_eval_task_analyzer.py` 产物） |

**旧版关键词分类评测的具体数字（仅供回溯，勿误解为当前行为）**
| 旧逻辑指标 | 值 |
|---|---|
| 手工金标准总体准确率 | **73.17%**（30/41） |
| comparison / diagnosis / calculation | 各 5~6/6，最稳 |
| explanation | 3/8（被 diagnosis/procedure/calculation 抢占） |
| standard_interpretation | 1/5（对题号写法不识别） |
| 真实 2049 题落到 general | **986 = 48.12%** |
| 有效分类率 | 51.88% |

**旧版归因（历史，现已剔除）：** 先到先得 + 朴素子串 → explanation 被误伤、标准题号漏网、英文词形不全。
> 结论：`task` 维度因信息量不足已在中和化中被剔除（`docs/analyzer_rule_classification_audit.md`），
> 当前真正承载前端划分信号的是 **`format`（读标注）** 而非 `task`；旧 73%/48% 数字不再代表现状。

---

## 阶段 2 — 策略规划（StrategyPlanner）

> **规划器 v2 专项探针实测**（`python LINS-Industrial/_probe_planner_stats.py --n 8 --scorer rule` →
> `results/planner_stats/{stats.json, per_sample.json, planner_stats_report.md}`）

| 指标 | 实测值 | 含义/归因 |
|---|---|---|
| LLM 高级图占比 | **8/8 = 100%** | 该 8 题全部输出含 `decide/verify/二次检索` 的 6~8 节点图 |
| fallback 模板(4 节点线性)占比 | **0/8 = 0%** | `_fallback_graph()` 在此数据上几乎不触发（LLM 从不失败/不降级） |
| 平均规划节点数 | **6.9**（exec 日志 5.5） | 规划含 verify/二次检索；执行侧被证据短路/轻量化 |
| `has_conditional_branching` | 8/8 = 1.0 | 图普遍声明分支 |
| `has_multi_hop_retrieval` | 8/8 = 1.0 | 图普遍声明多跳（`retrieve_2 evidence_guided`） |
| `has_verification` / verify 节点 | 8/8 = 1.0 | verify 声称在规划层 100% 真实存在 |
| 二次检索(规划 retrieve 节点>1) | 8/8 = 1.0（avg retrieve 节点=2） | 全样本规划均含 `retrieve_2` |
| advanced 组 avg_adjusted | 1.25 → 无 SV 违规 | 高级图组得分方向为正 |

> 诚实备注：执行日志平均仅 5.5 节点 < 规划 6.9，因为 `retrieve_1` 会被已注入的外部证据短路跳过、
> `verify` 可能被轻量化执行——"规划声称聪明"与"实际执行努力"存在差异（差距需在阶段 3/4 继续量化）。

**实测函数与执行流程**
- `planner.plan(task_analysis)`：先尝试 LLM 返回 JSON 图，失败/校验失败走 `_fallback_graph()` 模板。
- 简单图（calculation/procedure/explanation/general）：`retrieve→organize→reason→end`。
- 高级图（comparison/selection/diagnosis/standard_interpretation）：含 `decide/verify` 分支与二次检索 `retrieve_2(targeted=True, evidence_guided=True)`。

> 中和化连锁：当前 `task` 恒为 GENERAL，所以规划端 task 驱动的"高级 vs 简单图选择"将由
> task=GENERAL 落入模板匹配的 general 分支（template selection 依 task 字段），LLM 建图仍可用
> （`_probe_planner_stats` 实测 8/8 出图）；`format` 直读标注仍参与 PromptBuilder/答题结构定制。

**量化**
- 简单图：≥2 次生成类 LLM 调用（reason + 最终答案）。
- 高级图：reason + 最终答案 + 可选 decide/verify = ≥2~4 次 LLM 调用。
- 本质是**模板选择**，不是新的语义理解（task 由关键词触发 → 图由 task 触发）。

---

## 阶段 3 — 图执行（GraphExecutor，代码精读）

**调度流程（`execute()`，pipeline.py:104-213）**
- 建 `ExecutionContext(question)`；`node_queue = graph.entry_points`；`while nq:` pop → 防循环 `loop_counters`（`max_loop_iterations` 超限强制退出）→ `visited` 去重。
- `topological_sort()` 仅用于日志展示；实际用 `get_next_nodes()` 后向补队（`for nid in reversed(next): q.insert(0,nid)` → 近似 DFS）。
- 节点分发 `_execute_node()` 的 `dispatch` 表：retrieve→`_handle_retrieve` / organize / reason / decide / merge / verify / end。
- 节点异常 → 有 `fallback` 则入队、否则 `continue`；所有节点**无论类型都会被执行**（含 retrieve）。

**关键状态流（`ExecutionContext`）**
- `evidence_cache[node_id]`：每跳检索结果按节点 id 存。
- `last_organized` / `last_retrieval_results` / `reasoning_results` / `decisions` / `verdicts` / `accumulated_context` / `execution_log`。

**pre_retrieved_evidence 的"软跳过"语义（修正幻觉）**
- `execute()` 入口 `if pre_retrieved_evidence is not None:` 仅把外部证据写入 `last_retrieval_results` 与 `evidence_cache["__external_retrieval__"]`（pipeline.py:133-135），**并未改变调度循环**。
- `_handle_retrieve`（311-）**没有 skip 分支**，故提供外部证据下 retrieve 节点**仍会真实调用 retriever**（产生额外检索/LLM 调用）；docstring 中 "all retrieve nodes are skipped" 是**注释与实际不一致**，下游实际看到的是外部注入证据，但检索仍在跑。

**`_handle_reason`（560-638，跳过 solver 的直接 LLM）**
- `PromptBuilder.build(task_analysis, graph, evidence_context)` → `format_prompt(task_key="general", prompt_extra=...)`。
- 按 `\n\n##` 切 system/user 两段，`chat.completions.create(model="deepseek-chat", temp=0.3, max_tokens=2048)`。
- **模板恒为 "general"**（task 恒 GENERAL，type 不参与 prompt task_key），task 语义只通过 PromptBuilder 的 prompt_extra 注入。
- 结果写 `last_reasoning_result` + `reasoning_results[node.id]` + `accumulated_context`。
- 注意：`_handle_reason` 并没有实际调用注入的 `TaskSolver.solve()`；solver 仅在建图/入口语义上被保留。

**`_handle_merge`（684-759，recall-safe）**
- 拼接顺序：① Organizer 全量原始 verbatim 上下文 → ② `evidence_cache` 全部 raw chunks → ③ 仅追加 `reasoning_results` 作 `<REASONER_SUMMARIES (references only; NOT authoritative)>`。
- 明确"不要把 accumulated_context 坍缩成 LOSSY summary"——修复"Retriever hit 但 Agent 丢掉"。

**`_handle_decide`（651-683）**：`_evaluate_condition(condition, threshold, context, graph)` 写 `last_decision` + `decisions[node.id]`。

**`_handle_verify`（761-780+）**：`_verify_answer(criteria, question, answer, evidence_count)`。

**`reason` 的第二跳查询构建（`_build_evidence_guided_query`，239-309）**
- 从**原始 chunk 文本**（非推理摘要）确定性抽取 `GB/T 20476 / 65℃` 等判别键 + 原始问题作语义锚，无 LLM。
- 触发条件：`params.get("evidence_guided") or params.get("targeted")`；二次跳计数 `second_retrieval_count`。

**第二次检索触发（`_handle_retrieve` 333-369）**
- `is_followup` = `targeted/evidence_guided` 或节点 id ≠ `retrieve_1/retrieve`；`if "retrieve_1" in evidence_cache and is_followup:` → `second_retrieval_count += 1`。
- 检索分派：`multi_query_retrieve`（ked.decompose 拆子查询+fusion/union）或单查询 + 可选 KED 展开；`retrieval_kwargs={"k", "use_ked", "min_score"}`。
- 内置 `retrieval_failed` 诊断留痕：dense+fallback 全链失败 vs 真 0 命中 vs Retriever 命中但后续丢失——用于归因。


---

## 阶段 4 — 检索（Retriever，实测主瓶颈）

**实测函数与执行流程**
- `retrieve` 节点读 `retrieve_k/multi_query/use_ked/use_fusion/min_score`；多查询走 `multi_query_retrieve`；二次检索走 `_build_evidence_guided_query()`。
- 接口：`OpenDomainRetriever.hybrid_retrieve(query, k)` → `.chunks`（A/B 实验与矩阵均已复用验证）。

**量化（端到端语义检索指标）**
| 运行 | 样本 | Semantic Hit@1 | Hit@10 | MRR | NDCG |
|---|---|---|---|---|---|
| exp1_agentic_rag_fix | 35 | 0.457 | 0.51 | 0.47 | 0.769 |
| exp1_agentic_rag_20260801_223628 | 20 | **0.550** | 0.650 | **0.567** | **0.895** |

**正交性结论**：format×capability 矩阵实测 → 召回主轴是 **capability**，与 format 基本无关。
> **检索是召回瓶颈**：Top-1 命中不足一半，最终答案对不对高度依赖证据是否落在参考内容里。

---

## 阶段 5 — 证据组织（EvidenceOrganizer）

**实测函数与执行流程**
- `organize` 节点：`ExecutionDirective(module="organization")` → `EvidenceOrganizer.execute()`。
- 实际：按 chunk id / 内容相似度去重；按 comparison/symptom/candidate/formula/chronological/topic 分组；**保留全部去重后文档与原始顺序**；`get_context()` 输出带 citation 的上下文。
- 当前主路径**不做二次相关性过滤**，分组主要影响组织与展示顺序。

**量化注意**
- `merge` 同时并入 organizer 上下文与 `evidence_cache` 原始 chunk → **prompt 可能膨胀**（看 `evidence_length` 与 `prompt_length`）。
- `organize` 决定 `evidence_cache` 是否保留完整 chunk → 影响最终答案召回安全。

---

## 阶段 6 — 推理 / 判断 / 验证

- `reason`：`PromptBuilder.build()` + `format_prompt("general")` + LLM → `last_reasoning_result`。
- `decide`：`_evaluate_condition()` 把 question + ≤1500字符证据预览给 LLM，返回 `sufficient/insufficient/yes/no`；失败默认 `sufficient`。
- `verify`：本地硬检查 + LLM pass/fail（见阶段 3）。

**量化**：见阶段 4 报告的按能力分层 —— 推理/答题质量依赖证据质量（证据差→rule 分低）。

---

## 阶段 7 — 最终答案生成（`_produce_final_answer`）

**实测函数与执行流程**
- 优先 `OrganizedEvidence.get_context()`；有推理结果则放入 `REASONER_SUMMARY_REFERENCE`（仅参考）。
- `format_prompt(task_key="general")` + `PromptBuilder.build(task_analysis=...)` 指令；**传完整证据，不切片**（召回安全）。
- `chat.completions.create(model="deepseek-chat", temp=0.3, max_tokens=2048)`。
- 返回 `answer/evidence_str/num_evidence/group_count/prompt_length/...` + `pipeline_version` 元数据。

**A/B 答题贴合度量化（format 定制 vs 统一，`_ab_format_first.py`，检索同源）**
| 指标 | A(统一) | B(format) | Δ |
|---|---|---|---|
| rule_score (0-3) | 2.688 | 2.562 | **-0.125** |
| 关键词覆盖 cov% | 65.6 | 62.4 | -3.2pt |
| 实体覆盖 ent_cov% | 25.0 | 25.0 | +0.0 |
| 结构命中 struct% | 50.0 | 75.0 | **+25.0pt** |
| 冗余引导语 verb% | 0.0 | 0.0 | +0.0 |

> format 定制不提升内容分（检索同源下 Δrule≈0），唯一实质正收益是**结构**（选择题正确输出 A/B）。

---

## 阶段 8 — 评分与指标输出（Scorer / Evaluator）

- 回答质量：`RuleBasedScorer.rule_based_score(0-3)` / `compute_coverage` / `compute_entity_coverage`；可选 LLM judge + SV 安全违规检查。
- 检索质量：`RetrievalEvaluator` → recall@k / hit@k / MRR / NDCG + 语义 `semantic_hit@k/semantic_mrr/semantic_ndcg`（词/粒度 jaccard≥阈值）。
- 评分是**确定、无 LLM** 的规则覆盖分（用于量化），另有 option judge。

**端到端总体量化（多批交叉验证）**
| 运行 | 样本 | avg rule/3 | Hit@1 | NDCG | 耗时/样本 |
|---|---|---|---|---|---|
| exp1_agentic_rag_fix | 35 | 1.71 | 0.457 | 0.769 | 9.1s |
| exp1_agentic_rag_20260801_223628 | 20 | **1.55** | 0.550 | 0.895 | **18.9s** |

按能力（agentic 20 样本）：标准规范与术语 0.50、安全合规与风险控制 1.33、选型与替代 1.50 偏弱；质量计量与检测 3.00。SV 违规 0。

---

## §8.1 黑箱拆解：`mean_score_ef` 的完整生成链路（逐节点，含代码行）

> 目的：把「题目 → mean_score_ef」的运行时黑箱逐层打开。以 `_e2e_agentic_ef_35_pairs.py` 的 `ef` 条件为例，
> 一条问题从进引擎到出 0-3 分，实际经历了 **1 次外部检索 + 5 次以上 LLM 调用**。

```
question ──────────────┐
                       │ _e2e_agentic_ef_35_pairs.main() L117
                       ▼
┌─[1] EFAgenticRAGEngine.answer(q, retrieval_k=10)
│      exp1_agentic_rag.answer() L471  (EF 覆写 _init_external_retriever → EFExternalRetriever)
│        │
│        ▼
│    ExternalRetriever.retrieve(q, k=10)   ← Stable Knowledge Interface（固定 top10）
│      _ab_conditional_ef_35.EFExternalRetriever.retrieve L115
│        ├─ super().retrieve(q, k) → 生产 OpenDomainRetriever 混合检索取 top10（use_ked）
│        └─ _backfill(result, q) L123：对 top10 每档回捞同文档全部块
│              jt(块,query)≥0.02 过滤、每档封顶 MAX_PER_DOC=50、最多 200 块、citation 打 [*ef]
│        → retrieval_result.documents = top10 + 回捞碎片（ef 条件全部进入模型输入）
│        │
│        ▼
│    pre_retrieved_chunks = _DocWrapper(doc) for each document   exp1_agentic_rag L529-534
│        │
│        ▼
│    pipeline.run(question, pre_retrieved_evidence=pre_retrieved_chunks)   pipeline.L1213
│        │
│        ├─[2] analyzer.analyze(question)      → TaskAnalysis（task 恒 GENERAL，≈0 LLM）
│        │        pipeline.L1235
│        ├─[3] planner.plan(task_analysis)      → ExecutionGraph（LLM 出图，失败回退模板）
│        │        pipeline.L1242 / planner.L83-108
│        ├─[4] executor.execute(graph, pre_retrieved_evidence)   pipeline.L104
│        │        └ 注入 context.evidence_cache["__external_retrieval__"]（跳过 retrieve 节点）
│        │        └ GraphExecutor 走图：按节点类型 dispatch
│        │            ├─ organize 节点 → EvidenceOrganizer 去重/分组/get_context
│        │            ├─ decide 节点   → LLM 判 sufficient/insufficient（pipeline._evaluate_condition L853）
│        │            ├─ reason 节点   → PromptBuilder.build → format_prompt("general") → LLM（L582-614）
│        │            └─ verify 节点   → LLM 判 pass/fail（L915-968）
│        ▼
│    _produce_final_answer(question, context, graph)   ← 从完整证据再生最终答案（1 次 LLM）pipeline.L971
│        └ 输出 answer
│
└─ score_ef = scorer.rule_based_score(q, ref, ans_ef)    _e2e_agentic_ef_35_pairs L129
      metrics/industrybench_scorer.rule_based_score L612：纯规则 0-3（零 LLM）
      └ compute_coverage(ref, cand) L465：字符 Jaccard + 关键词2-5字覆盖 + 实体(标准号/数值)覆盖加权
        coverage≥0.60→3  ≥0.35→2  ≥0.12→1  else 0

mean_score_ef = mean(score_ef over 35)   → _e2e_agentic_ef_35_pairs.acc()
```

**逐节点说明（谁在消耗 token / 谁在定分）**

| 节点 | 是否 LLM | 幂等性 | 对分数的实际影响 |
|---|---|---|---|
| 外部检索 + 回捞 | 否 | 确定 | 决定**证据池**（ef 多注入原始碎片，token +135%） |
| analyzer | ≈否（task 中和为 GENERAL） | 确定 | 微弱：仅影响下游模板选择 |
| planner 建图 | 是 | 否（随机） | 决定节点数/分支，间接影响答案 |
| decide（LLM） | 是 | 否 | 若判 insufficient 触发二次链路，改变证据组合 |
| organize | 否 | 确定 | 证据去重/分组，决定喂给 reason 的拼接顺序 |
| reason（LLM） | 是 | 否 | 生成中间推理，但**最终答案不一定用它** |
| verify（LLM） | 是 | 否 | 只做通过/继续判定，不改答案文字 |
| **_produce_final_answer（LLM）** | 是 | 否 | **真正决定 answer 文字** → 直接决定 0-3 分 |
| rule_based_score | 否 | 确定 | 把 answer 和 ref 做字符/关键词/实体覆盖 → 评分 |

**关键结论（读完代码的 3 个事实）**
1. **ef 路线的"多一次检索"不增加分数上限**：`pre_retrieved_evidence` 注入后所有 `retrieve` 节点被跳过（pipeline.L133-139），
   回捞只是**扩大证据池**，最终仍是同一个 `_produce_final_answer` 从证据池生成一次答案、同一个 scorer 打分。
2. **分数的最终决定者是 `_produce_final_answer` 那次 LLM 调用的输出**，而非 decide/verify 分支。碎片越多，
   该次 LLM 越可能被噪声块干扰 → 与报告「ef 用 +135% token 换 +0.086 分、且 2 题降分」完全一致。
3. **评分是确定无 LLM 的**（`RuleBasedScorer`）：只看 answer 与 ref 的覆盖度，不看证据多寡——所以把高覆盖碎片
   拿掉、却只靠堆碎片，对分数没有帮助；这正是「证据聚焦」优于「无条件回捞」的根因。

---

## 端到端量化归因总表（每一段「值多少钱」）

| 环节 | 量化结论 | 证据强度 |
|---|---|---|
| 任务/题型划分（format） | 信息量≈0；读标注 `_format` > LLM 判定（62.5%）≈随机 | 2049 题面信号全零 + A/B |
| 策略规划图分支 | 模板选择，简单/高级两档，复杂度杠杆而非精度杠杆 | 代码路径实测 |
| 检索 recall | Hit@1≈0.46~0.55，**主要瓶颈** | 端到端语义指标 |
| 检索×format 正交 | format 不改变召回，capability 决定召回 | format×capability 矩阵 |
| 组织/合并 | 可保住完整 chunk，但可能 prompt 膨胀 | 证据拼接逻辑实测 |
| format 定制答题 | 内容分无增益（Δrule≈-0.12）、结构分 +25pt（选择题） | A/B 同源检索 |

---

## 一句话结论

在这套工业 agentic 系统里，从「任务分析→规划→检索→组织→推理→评分」的完整链路中，
**可量化、可重复的实质增益主要来自检索召回（capability 驱动）与答题结构；任务/题型（format）划分是低内容价值、中结构价值的前端环节，且因题面无可判别信号，其判定必须复用 CSV 标注（`_format`）而非 LLM 反推。**
任务维度（`task`）已在当前实现中被中和化为恒 `GENERAL`（旧 73%/48% 为历史关键词分类评测，不再复现），
而检索仍限定内容上限，故整体改善应优先投在**检索召回**这一主瓶颈上。

---

## 复现命令

```powershell
# 阶段1 中和化现状（task 恒 general / format 直读标注）；旧关键词分类评测见 json 回溯
python LINS-Industrial\_verify_neutralize_bypass.py
python LINS-Industrial\_eval_task_analyzer.py

# 阶段2 策略规划器专项量化（LLM 建图 vs fallback、verify/二次检索占比）
python LINS-Industrial\_probe_planner_stats.py --n 8 --scorer rule

# 阶段4/7/8 端到端 agentic 实验（需要 DEEPSEEK_KEY）
python LINS-Industrial\experiments\exp1_agentic_rag.py

# 阶段7 format A/B 答题定制量化（默认读 CSV _format 真值）
python LINS-Industrial\_ab_format_first.py --samples 16

# 阶段1-7 依赖的既有探测（format×capability、format 信号缺失）
python LINS-Industrial\_probe_recal_format_x_cap.py
python LINS-Industrial\_probe_format_signal_missing.py
```

> 相关文档：`docs/agentic_framework_and_runtime.md`（含 §11-12 原始端到端链）、
> `docs/task_classification_stage_analysis.md`（任务分析专项）、
> `docs/analyzer_rule_classification_audit.md`（task 维度中和化审计）、
> `docs/analyzer_neutralize_bypass_quant.md`（中和化旁路量化）、
> `docs/prompt_builder_integration_report.md`（PromptBuilder 接入）。
