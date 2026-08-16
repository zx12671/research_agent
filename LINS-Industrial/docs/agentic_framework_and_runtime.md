# Agentic 框架逻辑与运行流程

本文只描述 `LINS-Industrial/agentic` 包，不展开知识库、检索器和评估目录。`agentic` 的职责是：把一个问题转换成任务类型，再转换成可执行的工作流图，最后在图上调度检索、证据组织、推理、判断、合并和验证节点。

## 1. 总体架构

```text
Question
   ↓
TaskAnalyzer.analyze()
   ↓ TaskAnalysis
StrategyPlanner.plan()
   ↓ ExecutionGraph (DAG)
GraphExecutor.execute()
   ├─ retrieve   检索
   ├─ organize   证据组织
   ├─ decide     条件判断
   ├─ merge      多路证据合并
   ├─ reason     生成中间推理/答案
   ├─ verify     答案质量验证
   └─ end        结束
   ↓
_produce_final_answer()
   ↓
PipelineResult / dict
```

包的主要文件分工：

| 文件 | 角色 |
|---|---|
| `task_types.py` | 定义任务、图节点、执行图、上下文结果的数据结构 |
| `analyzer.py` | 问题分类，生成 `TaskAnalysis` |
| `planner.py` | 根据任务分析生成 `ExecutionGraph` |
| `pipeline.py` | 图执行器和总入口，负责状态、分支和最终输出 |
| `organizer.py` | 对证据去重、分组并生成带引用上下文 |
| `solver.py` | 独立的任务推理器，提供结构化推理工作流 |
| `prompt_builder.py` | 按任务类型生成提示增强指令 |
| `prompts.py` | 标准 RAG prompt 模板和格式化函数 |

## 2. 核心数据模型

### 2.1 `TaskType` 与 `TaskAnalysis`

`TaskType` 支持：`comparison`、`diagnosis`、`calculation`、`selection`、`procedure`、`standard_interpretation`、`explanation`、`general`。

`TaskAnalysis` 是分析阶段的输出，包含：

- `task`：任务类型。
- `confidence`：分类置信度。
- `reasoning`：分类说明。
- `reasoning_complexity`：低/中/高。
- `requires_multi_source`、`requires_formula`、`requires_step_reasoning`：下游策略提示。
- `preferred_source`、`expected_evidence`：证据偏好。

### 2.2 `GraphNode` 与 `ExecutionGraph`

`GraphNode` 的关键字段是 `id`、`type`、`action`、`params`、`next`、`branches`、`fallback`。其中：

- `next` 表示普通顺序边。
- `branches` 表示 `decide` / `verify` 的条件边。
- `fallback` 表示节点异常时的备用节点。

`ExecutionGraph` 保存节点字典、入口节点、任务类型和完整的 `task_analysis`。它还提供拓扑排序、按类型查找、执行指令提取和可视化能力。

### 2.3 `ExecutionContext`

执行期间所有中间状态都放在 `ExecutionContext`：

```text
question
evidence_cache[node_id]       每个检索节点的证据
last_retrieval_results        最近一次检索结果
organized_evidence[node_id]   每个组织节点的结果
last_organized                最近一次组织结果
reasoning_results[node_id]    每个推理节点的结果
last_reasoning_result         最近一次推理结果
decisions[node_id]            decide 结果
verdicts[node_id]             verify 结果
accumulated_context           合并后的完整上下文
execution_log                 节点执行日志
```

## 3. 阶段一：任务分析

入口是 `AdaptiveAgenticPipeline.run(question)`，第一步调用 `TaskAnalyzer.analyze(question)`。

当前代码是轻量、确定性的关键词分类器：不调用 LLM，也不改写原问题。命中关键词后返回对应任务类型，未命中返回 `general`；其他语义字段使用保守默认值（例如 `requires_multi_source=False`）。构造函数保留 `llm_client` 参数只是为了兼容旧接口，实际不会调用。

这意味着当前 `agentic` 的“自适应”主要来自后续执行图模板和节点参数，而不是由 LLM 重新理解问题。

## 4. 阶段二：策略规划

`StrategyPlanner.plan(task_analysis)` 有两级策略：

1. 首先尝试调用 LLM，让它返回 JSON 图。
2. JSON 解析或结构校验失败时，使用 `_fallback_graph()` 的任务模板。

LLM 图必须至少包含一个 `retrieve` 和一个 `end`，节点引用必须存在；`decide` 必须有分支，`verify` 必须有 `pass` / `fail` 分支。

### 4.1 简单图

适用于 `calculation`、`procedure`、`explanation`、`general`：

```text
retrieve_1 → organize_1 → reason_1 → end
```

### 4.2 高级图

适用于 `comparison`、`selection`、`diagnosis`、`standard_interpretation`：

```text
retrieve_1 → organize_1 → decide_1
                              ├─ sufficient → reason_1 → verify_1
                              │                         ├─ pass → end
                              │                         └─ fail → retrieve_2
                              └─ insufficient → retrieve_2 → merge_1 → reason_1
```

高级模板会强制启用较大的 `retrieve_k`、多查询和 fusion；`retrieve_2` 使用 `targeted=True`、`evidence_guided=True`，从原始证据中抽取标准号和数值参数构造补充查询。

## 5. 阶段三：图执行

`GraphExecutor.execute()` 的执行机制：

1. 创建 `ExecutionContext`。
2. 可选地把 `pre_retrieved_evidence` 放入 `evidence_cache`。
3. 从 `entry_points` 建立队列。
4. 取出节点，按类型分发给 `_handle_*`。
5. 节点成功后通过 `_get_next_nodes()` 决定后继。
6. 节点异常时记录日志，并将 `fallback` 节点加入队列。
7. 执行结束后统一调用 `_produce_final_answer()`。

队列采用“反向插入”实现近似深度优先；`visited` 防止同一节点重复执行，`max_loop_iterations` 用于限制异常循环。

## 6. 各类节点的实际行为

### `retrieve`

读取 `retrieve_k`、`multi_query`、`use_ked`、`use_fusion`、`min_score` 等参数，调用注入的 retriever。多查询时调用 `multi_query_retrieve()`，补充检索时先执行 `_build_evidence_guided_query()`，再把结果标准化成证据列表并写入 `evidence_cache`。

### `organize`

从当前证据取得文档，构造 `ExecutionDirective(module="organization")`，调用 `EvidenceOrganizer.execute()`。结果保存到 `last_organized` 和 `organized_evidence`。

`EvidenceOrganizer` 的实际处理是：

- 按 chunk id / 内容相似度去重。
- 按 comparison、symptom、candidate、formula、chronological、topic 分组。
- 保留去重后的所有文档和原始检索顺序。
- 通过 `get_context()` 输出带 citation 编号的上下文。

当前版本的主路径不会进行二次相关性过滤，分组主要影响组织方式和展示顺序。

### `decide`

调用 `_evaluate_condition()`，把问题和最多约 1500 字符的证据预览发给 LLM，要求只返回 `sufficient` / `insufficient` / `yes` / `no`。调用失败时默认返回 `sufficient`，即继续回答而不是补充检索。

### `merge`

收集 `evidence_cache` 中所有检索结果，优先保留 organizer 生成的完整原始 chunk 文本，再附加所有原始证据和推理结果。推理摘要只作为参考，不替代原始证据，避免关键标准号或数值在摘要中丢失。

### `reason`

代码注释和 `TaskSolver` 设计意图是“执行结构化推理工作流”，但当前 `GraphExecutor._handle_reason()` 实际执行的是：

```text
TaskAnalysis + ExecutionGraph + evidence
    → PromptBuilder.build()
    → prompts.format_prompt(task_key="general")
    → 直接调用 LLM
    → last_reasoning_result
```

也就是说，当前运行时没有调用注入的 `solver.solve()`；`TaskSolver` 仍作为独立组件存在并有完整的任务 prompt，但不是 `GraphExecutor` 的实际 reason 节点实现。这是理解当前代码时最重要的“设计与实现差异”。

### `verify`

先做本地硬检查：答案为空或少于 10 个字符、证据数为 0，直接失败；否则调用 LLM 判断 `pass` / `fail`。调用异常时，当证据数至少为 2 默认为通过。

### `end`

设置 `execution_complete=True` 并写入日志。真正的最终答案仍由图遍历结束后的 `_produce_final_answer()` 统一生成。

## 7. 最终答案生成

`_produce_final_answer()` 不直接复用推理节点的文本作为最终答案，而是重新组织完整证据并再次调用 LLM：

1. 优先使用 `OrganizedEvidence.get_context()`。
2. 如果已有推理结果，将其放入 `REASONER_SUMMARY_REFERENCE`，仅作为辅助参考。
3. 用 `format_prompt(task_key="general")` 生成最终 prompt，并附加 `PromptBuilder` 指令。
4. 调用 OpenAI-compatible `chat.completions` 接口。
5. 返回答案、证据全文、证据数量、分组数、prompt 长度、执行图和节点日志。

因此复杂任务通常至少有两次生成相关 LLM 调用：一次 `reason`，一次最终答案生成；高级图还可能增加 `decide` 和 `verify` 调用。

## 8. 适配与消融

`AdaptiveAgenticPipeline.run_with_ablation()` 先生成完整图，再按配置改写：

- `adaptive_retrieval=False`：所有检索节点恢复为 `k=10`、单查询、KED 开启。
- `adaptive_organize=False`：统一按 topic 组织。
- `adaptive_reasoning=False`：统一使用 general 推理参数。
- `conditional_branching=False`：移除 decide / verify，并重连前驱到后继。
- `multi_hop_retrieval=False`：在无条件分支时移除 merge 节点。

这套机制用于比较不同 Agent 组件对结果的影响，但它修改的是执行图参数和边，不是替换整个 pipeline。

## 9. 当前运行时需要特别注意的事实

### 9.1 `pre_retrieved_evidence` 的“跳过检索”语义

`GraphExecutor.execute()` 的注释说明传入外部证据后应跳过所有 `retrieve` 节点，并把证据直接给后续节点。但当前 `execute()` 只把外部证据写入上下文，`_execute_node()` 仍会按图调度 `retrieve` handler；代码中没有显式的 `if external: skip retrieve` 分支。

因此，若调用方传入外部证据，应以执行日志和 `evidence_cache` 为准确认实际是否真的跳过了内部检索，不能只依据 `used_external_retrieval` 字段判断。

### 9.2 `visited` 会限制验证重试

图设计允许 `verify(fail) → retrieve_2`，但同一节点只执行一次；如果图形成更长的回环，`visited` 会阻止已执行节点再次进入。因此当前实现更接近“一次补充检索/一次验证分支”，不是无限验证循环。

### 9.3 证据可能被多次拼接

`merge` 会同时加入 organizer 上下文和 `evidence_cache` 原始 chunk；这是为了召回安全，但可能让 prompt 变长、同一内容出现多次。分析上下文长度时应同时看 `evidence_length` 和 `prompt_length`。

## 10. 一次完整调用的时序

```text
AdaptiveAgenticPipeline.run(question)
  1. analyzer.analyze(question)
  2. planner.plan(task_analysis)
  3. executor.execute(graph, question)
     3.1 retrieve_1
     3.2 organize_1
     3.3 [高级图] decide_1
         ├─ sufficient → reason_1
         └─ insufficient → retrieve_2 → merge_1 → reason_1
     3.4 [高级图] verify_1
         ├─ pass → end
         └─ fail → retrieve_2（受 visited 限制）
  4. _produce_final_answer(question, context, graph)
  5. 补充 task_analysis / execution_graph / execution_nodes 等元数据
```

 一句话概括：**agentic 是一个以 `ExecutionGraph` 为核心的任务自适应调度器；Analyzer 决定任务，Planner 决定图，Executor 维护状态并调度节点，Organizer 负责证据结构化，当前 Reason 节点直接调用 LLM，最终阶段再基于完整证据生成一次答案。**

## 11. 前端任务/题型分析（format）的量化验证

阶段一 `TaskAnalyzer` 与阶段七的 `PromptBuilder` 都依赖"任务/题型类型"。为判断这个前端环节是否是真正的价值来源，做了严格的 A/B 量化实验（脚本 `LINS-Industrial/_ab_format_first.py`，数据集标注字段 `_format`，16 条按四类题型分层，检索 k=10 完全同源）：

**实验设计（控制变量）**：管线 A（基线）用统一答题提示直接回答；管线 B 先让 LLM 判定题型（format），再用 format 定制提示回答。两条管线使用**同一批** `hybrid_retrieve` 文档与相同顺序构造 evidence，唯一差异是"是否按 format 定制答题 prompt"。

**结果 1 — format 判定精度低（前端可靠性不足）**：

- LLM 题型判定总体准确率仅 **62.5%（10/16）**，加权 F1 **56.3%**。
- 混淆矩阵显示误差集中在"填空题被误读"：4 条填空题判对 0 条（2 条→选择题、2 条→问答题），还有 1 条计算题被误判为填空题。
- 这类题目题干多为长描述、无 `___` 显式占位符，LLM 难以仅凭语义稳定区分题型。

**结果 2 — format 定制对内容分几乎无增益、结构分有正收益（后端收益）**：

| 指标 | A（统一提示） | B（format 定制） | Δ |
|---|---|---|---|
| rule_score (0-3) | 2.688 | 2.562 | **-0.125** |
| 关键词覆盖 cov% | 65.6 | 62.4 | **-3.2pt** |
| 实体覆盖 ent_cov% | 25.0 | 25.0 | +0.0 |
| 结构命中 struct% | 50.0 | 75.0 | **+25.0pt** |
| 冗余引导语 verb% | 0.0 | 0.0 | +0.0 |

按题型分层进一步说明：选择题 struct 0%→100%（format 提示促使输出 A/B 选项结构，唯一明显正向收益）；但填空题因 format 被误判成其他类、定制方向错误，rule 3.00→2.75（负向伤害）。

**量化结论**：

1. **format/任务题型判定是一个"低内容价值、中结构价值"的前端环节**：对答案内容分（rule / 关键词 / 实体覆盖）几乎无增益——在检索同源前提下 Δrule≈0，实体覆盖持平；唯一实质正收益是答题**结构**（选择题输出 A/B 选项），这更多是"表述风格"而非"回答内容"的变化。

2. **判定准确率低的主因是"题源信号缺失"，不是 LLM 分类能力差**。对全量 CSV（追问答题 1163 / 填空 601 / 选择 252 / 计算 33，共 2049 条）在题面文本的显式题型信号做统计：

   | 题型 | 疑问句结尾 | 含填空符(`___`/空括号) | 含选项字母 | 平均长度 |
   |---|---|---|---|---|
   | 问答题 | 99.8% | 0.0% | 0.0% | 51 |
   | 填空题 | 100.0% | 0.0% | 0.0% | 51 |
   | 选择题 | 100.0% | 0.0% | 0.0% | 52 |
   | 计算题 | 100.0% | 0.0% | 0.0% | 61 |

   即：**四类题型在题面中几乎无显式题型符号**（填空符 0%、选项 A-D 0%、且几乎都以疑问句收尾）。唯一的弱信号是"计算词/数字"（选择 0.4% / 填空 12.5% / 计算 63.6%），但它更接近能力（capability）而非题型（format），且覆盖率远不足以精确分类。因此**真人也无法仅凭 question 稳定区分"选择"与"填空"**——`_format` 是一份与题面文本解耦的标注元数据。这解释了为什么 A/B 实验中模型把 4 条填空题 0 判对且倾向输出"A/B 选项"——不是模型不会分类，而是输入里本没有那个信号。

3. **参考答法形态与 format 标签也存在解耦**：部分标注为填空题/问答题的条目，其参考答案本身就是"从集合中选一个名词/短语"（例如填空条目的标准答法可写成 `B. 故障信号触点`）。这进一步说明 format 是标签层概念，与题目作答形态并不严格对应。

4. **综合归因与建议**：与检索环节解耦同样成立——此前"format×capability 检索矩阵"已证明 format 对检索召回基本无关（主轴是 capability）。因此量化上 **format/任务题型分析不构成 agentic 在该工业数据集上的有效增益杠杆**；而其判定准确率<65% 属于**"题源无可判别信号"**造成的上限，而非模型不足。若要在系统中保留 format 定制，应**直接复用数据集已标注的 `_format` 字段（或句法/heuristic）作为题型来源**，不要从 question 文本反推，更不要依赖 LLM 判定——LLM 在此输入的合理预期就是贴近随机/偏向多数类。

> 数据与完整逐条记录：`results/ab_format_first/ab_format_first_<ts>.md`、`ab_rows_<ts>.json`、`detect_rows_<ts>.json`；题源信号缺失全量统计：`results/probe_format_signal_missing/format_signal_missing.json`。

## 12. 端到端量化链路梳理：从任务分析到结果输出

下面把"从 CSV 载入 → 任务/题型分析 → 检索 → 证据组织 → 推理 → 最终答案 → 评分 → 指标输出"整条链路逐一实测，并标注每一阶段的量化度量、数据接口与已知归因。所有环节都已在仓库实际运行确认（实验脚本 `experiments/exp1_agentic_rag.py`，驱动脚本 `_ab_format_first.py`）。

**阶段 0 — 数据载入与标注**
- 数据源：`INDUSTRYBENCH_CSV`（全量 2049 条，追问答题 1163 / 填空 601 / 选择 252 / 计算 33）。
- 关键列：`question`（题面）、`answer`（参考）、`_format`（题型中文标注，与题面解耦的元数据）、`capability`（能力）、`difficulty`（难度）。
- 量化：题型分布极不均衡（问答题占 57%），`load_question_dataset` 可配额抽样做按题型分层实验。
- 注意点：`_format` 是标注真值，不参与 agentic 关键字任务分类（见阶段 1）。

**阶段 1 — 任务/题型分析（TaskAnalyzer）**
- 入口：`AgenticRAGEngine.answer(question, retrieval_k)` → `pipeline.run(question=...)` → `analyzer.analyze(question)`。
- 实际行为（代码实测）：**轻量确定性关键词分类器，不调用 LLM**。命中关键词返回 task，未命中返回 `general`；`requires_multi_source` 等语义字段用保守默认值。
- 量化（格式定制实验 `_ab_format_first.py`，format_source=llm）：
  - 直接把"题型"交给 LLM 判定 → 准确率 **62.5%（10/16）**、加权 F1 **56.3%**；填空题 4 判 0，误差集中在填空↔选择/问答混淆。
  - 对 2049 条题面统计显式题型信号全部≈0：填空符 0%、选项字母 0%、四类都 99.8%+ 以疑问句收尾 → **题面本身无可判别信号，真人/LLM 都无法从 question 反推出 `_format`**。
  - 结论：把 `_format` 标注直接当作题型来源（`format_source=annotated`，读 CSV 真值）是唯一可靠做法，已在 `_ab_format_first.py` 落地。

**阶段 2 — 策略规划（StrategyPlanner）**
- `planner.plan(task_analysis)`：LLM 出图失败时回退任务模板；简单图 `retrieve→organize→reason→end`，高级图 `decide/verify` 分支 + 二次检索 `retrieve_2(targeted, evidence_guided)`。
- 量化：高级图（comparison/selection/diagnosis/standard_interpretation）至少 2 次生成类 LLM 调用（reason + 最终答案），还可能加 decide/verify；简单图 1 次 reason + 1 次最终答案。
- 注意点：图由关键词 task 触发，task 又由关键词触发 → 规划阶段的"自适应"本质是模板选择，不是新语义理解。

**阶段 3 — 图执行（GraphExecutor）**
- 按 `ExecutionGraph` DAG 调度 retrieve/organize/decide/merge/reason/verify/end 节点，`ExecutionContext` 维护 `evidence_cache` / `organized_evidence` / `reasoning_results` 等状态。
- 关键实测差异：`reason` 节点直接走 `PromptBuilder.build() → format_prompt("general") → LLM`，**没有调用注入的 `TaskSolver.solve()`**（solver 是独立未接入组件）；`verify` 结合本地硬检查(空/过短/0证据) + LLM pass/fail 判断。

**阶段 4 — 检索（Retriever）**
- `retrieve` 节点读 `retrieve_k/multi_query/use_ked/use_fusion/min_score`，多查询走 `multi_query_retrieve`，高级图二次检索用 `_build_evidence_guided_query()`。
- 量化（端到端语义检索指标，35 样本修复版 `exp1_agentic_rag_fix` / 20 样本 `20260801_223628`）：Semantic Hit@1 约 **0.46~0.55**、Hit@3≈Hit@1、Hit@10 0.51~0.65、Semantic MRR 0.47~0.57、NDCG 0.77~0.90。→ 检索是召回瓶颈之一：Top-1 命中率不到一半，"答案对不对"高度依赖证据是否落在参考内容里。
- 正交性结论（此前 format×capability 矩阵实测）：召回主轴是 **capability**，与 format 基本无关。

**阶段 5 — 证据组织（EvidenceOrganizer）**
- 按 chunk id/相似度去重，按 comparison/symptom/candidate/formula/chronological/topic 分组，保留全部去重后文档与原始顺序，`get_context()` 输出带 citation 的上下文。
- 量化：`organize` 直接决定 `evidence_cache` 是否保留完整 chunk；`merge` 会同时并入 organizer 上下文与原始 chunk → prompt 可能膨胀（看 `evidence_length` 与 `prompt_length`）。

**阶段 6 — 评分与指标输出（Scorer）**
- 回答质量：`RuleBasedScorer.rule_based_score(0-3)` / `compute_coverage` / `compute_entity_coverage`；可选 LLM judge + 安全违规 SV 检查。
- 检索质量：`RetrievalEvaluator` 产出 recall@k / precision@k / hit@k / MRR / NDCG；另加语义 `semantic_hit@k / semantic_mrr / semantic_ndcg`（词/粒度 jaccard ≥阈值）。
- 端到端总体量化（多批实测交叉验证）：

  | 运行 | 样本 | avg rule/3 | Hit@1 | Hit@3 | NDCG | 耗时/样本 |
  |---|---|---|---|---|---|---|
  | exp1_agentic_rag_fix | 35 | 1.71 | 0.457 | 0.457 | 0.769 | 9.1s |
  | exp1_agentic_rag_20260801 | 20 | 1.55 | 0.550 | 0.550 | 0.895 | 18.9s |

  - 平均分稳定在 **1.55~1.71 / 3.0**，SV 违规 0；按难度 easy>medium>hard，按能力"标准规范与术语/安全合规与风险控制"偏弱（0.50~1.38）。

**端到端量化归因总表（让"每段值多少钱"透明）**

| 环节 | 量化结论 | 证据强度 |
|---|---|---|
| 任务/题型划分（format） | 信息量≈0；读注释 `_format` > LLM判定（62.5%）≈ 随机 | 2049 题面信号全零 + A/B 实验 |
| 策略规划的图分支 | 模板选择，简单/高级两档，主要是复杂度而非精度杠杆 | 代码路径实测 |
| 检索 recall | Hit@1≈0.46~0.55，是主要瓶颈 | 端到端语义指标 |
| 检索×format 正交 | format 不改变召回，capability 才决定 | format×capability 矩阵 |
| 组织/合并 | 可保住完整 chunk，但可能致 prompt 膨胀 | 证据拼接逻辑实测 |
| format 定制答题 | 内容分无增益(Δrule≈-0.12)、结构分 +25pt(选择题) | A/B 同源检索 |

**一句话端到端结论**：在这套工业 agentic 系统里，从"任务分析 → 规划 → 检索 → 组织 → 推理 → 评分"的完整链路中，**可量化、可重复的实质增益主要来自检索召回（capability 驱动）与答题结构；任务/题型（format）划分是低内容价值、中结构价值的前端环节，且因题面无信号其判定必须复用 CSV 标注而非 LLM 反推。**

（顶层架构另一份文档：`docs/lins_industry_system_architecture.md`；端到端实验报告源：`results/experiments/exp1_agentic_rag_*/report.md`、`ab_format_first/ab_format_first_*.md`。）


