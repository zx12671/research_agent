# SkillLINS：基于技能编排与验证门控 Agentic Loop 的高效医学问答框架

> **文档状态：Research Proposal / Draft**
>
> 本文档提出一项基于 LINS 的后续研究：将原有角色式模块重构为类型化 Skills，通过单一持久 Controller、显式 Evidence-State、Function Call 和外部验证门控构建 Agentic Loop，并在等模型、等工具和等预算条件下与 Multi-Agent 系统进行比较。

## 1. 研究背景与论文定位

### 1.1 背景

[LINS](https://www.nature.com/articles/s41467-025-64142-2) 通过 MAIRAG、KED、Link-Eval 以及 PRA、SKA、QDA、PCA 等模块，提高了医学问答的回答质量、证据可信度和可追溯性。其主要流程已经具备检索、证据评估、问题拆解、生成和冲突检查等能力，但这些能力主要以角色模块和自然语言判断串联，停止条件、状态迁移及校验过程仍不完全显式。

与此同时，Self-RAG、CRITIC 等工作表明，模型可以通过检索、反馈和自我修订改善答案；近期 Agent 研究则进一步关注多 Agent 系统的成本、协调失败和公平评测问题。相关工作包括：

- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [CRITIC](https://arxiv.org/abs/2305.11738)
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502)
- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)

本研究不把“更多 Agent”作为默认设计，而是区分三个概念：

1. **能力**：检索、拆解、验证、生成等任务能力；
2. **Skill**：具有严格输入输出接口、可独立测试的能力单元；
3. **Agent 拓扑**：这些能力由一个持续 Controller 调用，还是由多个具有独立上下文的 Agent 协调完成。

### 1.2 核心研究问题

> 在共享状态、顺序依赖和证据约束明显的任务中，单一持久 Controller 配合结构化 Skills、显式 Evidence-State 和验证门控停止机制，能否比角色分离的 Multi-Agent 系统取得更好的质量—成本—可靠性 Pareto 前沿？

研究不主张单 Agent 普遍优于 Multi-Agent，而是识别两种架构各自的适用边界：

- 对共享状态强、顺序依赖强的任务，Agentic Loop 可能减少上下文复制、协调开销和状态不一致；
- 对可并行分解、需要上下文隔离或真正异构专家能力的任务，Multi-Agent 仍可能具有优势。

### 1.3 暂定论文标题与方法名称

英文标题：

> **From Roles to Skills: Verifier-Gated Agentic Loops for Cost-Efficient Evidence-Grounded Medical Question Answering**

方法名称：

> **SkillLINS**

### 1.4 预期贡献

1. **Role-to-Skill Compilation**

   将 LINS 中 PRA、SKA、QDA、PCA 等角色模块编译为无持久人格、可复用、具有严格类型约束的 Skills，从而分离“能力设计”和“Agent 拓扑”。

2. **Explicit Evidence-State Loop**

   引入统一任务状态，显式记录子问题、已接受和已拒绝证据、答案主张、引用关系、验证错误、调用历史和剩余预算。

3. **Verifier-Gated Termination**

   模型只能提出 `finish` 请求，最终停止由系统级外部验证门控决定，减少过早停止、无证据结论和虚假引用。

4. **Quality–Cost Pareto Evaluation**

   在模型、检索器、语料、提示信息和预算一致的条件下，对比单循环、Multi-Agent 和原始 LINS，隔离系统拓扑本身的贡献。

5. **Applicability Boundary**

   分析 Agentic Loop 在何种任务结构下优于 Multi-Agent，以及 Multi-Agent 在何种条件下仍然占优。

## 2. 研究问题与假设

### RQ1：角色是否必须由独立 Agent 实现？

比较原始 LINS、基于相同 Skills 的 Multi-Agent 系统和单 Controller Skill 系统，判断角色分离是否产生独立收益。

**H1：** 在相同模型、工具和预算下，SkillLINS 的答案质量不低于原始 LINS，同时减少模型调用、上下文重复和 Token 消耗。

### RQ2：显式状态是否优于自然语言上下文中的隐式状态？

**H2：** Evidence-State 能减少重复检索、遗忘已有证据、前后决策不一致和无效工具调用。

### RQ3：验证门控能否让模型正确停止？

**H3：** 与模型通过自然语言自行判断停止相比，Verifier-Gated Termination 将显著降低过早停止率、无证据主张率和无效引用率。

### RQ4：Agentic Loop 与 Multi-Agent 的优势边界在哪里？

**H4：**

- 对共享状态强、顺序依赖强的任务，单 Controller 更具成本和稳定性优势；
- 对可并行分解、需要上下文隔离或异构能力的任务，Multi-Agent 的优势更明显。

### RQ5：优势来自技能化、循环控制还是验证器？

通过消融实验分别移除显式状态、停止门控、语义证据验证、引用验证、问题拆解、预算控制和失败后的修订循环，以定位真实性能来源。

## 3. SkillLINS 方法设计

### 3.1 总体流程

```text
用户问题
   ↓
Controller 读取 LoopState
   ↓
选择一个 Skill 并生成结构化 Function Call
   ↓
执行 Skill，更新证据和任务状态
   ↓
Controller 决定继续检索、拆解、生成、修订或请求 finish
   ↓
外部 Verifier Gate
   ├── 验证通过 → 返回答案
   ├── 验证失败 → 返回结构化错误，继续循环
   └── 预算耗尽或证据不足 → abstain
```

Controller 不维护 PRA、SKA、QDA 等独立人格，只维护一个任务状态。每个 Skill 完成有限、可测试的操作。

### 3.2 公共状态接口

统一的 `LoopState` 至少包含：

```text
question
subgoals[]
evidence_ledger[]
rejected_evidence[]
draft_answer
claim_ledger[]
validation_reports[]
iteration
tool_call_count
retrieval_count
token_budget
remaining_budget
status
termination_reason
```

字段语义如下：

- `evidence_ledger`：保存证据 ID、来源、检索查询、文本、相关性和支持的子目标；
- `rejected_evidence`：保存被判定为无关、冲突或质量不足的证据及原因；
- `claim_ledger`：保存答案中的事实主张及其引用证据；
- `validation_reports`：保存验证失败原因，不允许通过自然语言覆盖旧错误；
- `status`：只能取 `running`、`finished`、`abstained`、`budget_exhausted`；
- `termination_reason`：记录通过验证、证据不足、预算耗尽或达到硬性循环上限等停止原因。

所有运行过程输出统一的 `RunTrace` JSONL，支持回放、失败分析和成本统计。核心公共类型包括：

- `LoopState`
- `SkillCall`
- `SkillResult`
- `ValidationReport`
- `RunTrace`

### 3.3 Skills 设计

| Skill | 主要功能 | 核心输出 |
|---|---|---|
| `decompose_question` | 将问题拆为可验证子目标 | 子目标及依赖关系 |
| `retrieve_evidence` | 对指定子目标检索证据 | 带唯一 ID 的证据列表 |
| `assess_evidence` | 判断证据相关性与充分性 | `sufficient`、`partial` 或 `irrelevant` |
| `assess_self_knowledge` | 提供检索方向或关键词 | 不得作为最终证据 |
| `draft_answer` | 使用已接受证据生成答案 | 答案和 claim–citation 映射 |
| `verify_claims` | 检查每项事实主张是否被支持 | 支持、矛盾和缺失证据列表 |
| `verify_citations` | 检查引用存在性和对应关系 | 引用准确率与覆盖率 |
| `revise_answer` | 根据验证错误修订答案 | 新答案及修订记录 |
| `finish` | 提交停止申请 | 候选最终答案 |
| `abstain` | 明确报告证据不足 | 原因和缺失信息 |

`assess_self_knowledge` 只能帮助拆解问题、生成检索词或发现知识缺口，不能让医学事实绕过证据验证。

### 3.4 Function Call 约束

所有 Skills 使用严格 JSON Schema：

- 禁止未声明字段，并设置 `additionalProperties: false`；
- 枚举值必须显式定义；
- 证据只能通过合法 evidence ID 引用；
- Controller 不能直接修改验证结果；
- Function 参数解析失败时返回结构化错误；
- Skill 执行结果必须写回 `LoopState`；
- 相同参数的重复调用优先从缓存返回，并计入重复调用指标。

实现时使用 DeepSeek 当前支持的严格 Tool Call 接口，并固定完整模型 ID、API 版本、实验日期和参数。接口设计参考 [DeepSeek Tool Calls 文档](https://api-docs.deepseek.com/guides/tool_calls/)。

### 3.5 停止机制

Controller 可以提出 `finish`，但系统仅在以下条件全部满足时接受：

1. 已生成非空答案；
2. 所有必要子目标均已关闭；
3. 每个医学事实主张至少具有一条有效证据；
4. 不存在未解决的证据冲突；
5. 引用 ID 全部合法；
6. 语义验证器判定证据足以支持答案；
7. 没有待处理的验证错误。

验证失败时返回：

```text
failure_type
failed_claims[]
missing_evidence[]
contradictions[]
invalid_citations[]
recommended_next_actions[]
```

Controller 根据错误选择重新检索、重新拆解或修订答案。默认停止边界为：

- 最多 8 次 Agentic Loop；
- 最多 3 次检索调用；
- 连续两次检索没有新增有效证据时停止检索；
- 预算耗尽或证据仍不充分时必须 `abstain`；
- 不允许在门控失败后强制完成。

### 3.6 验证器

采用组合式外部门控。

#### 确定性验证

检查 JSON Schema、引用 ID、字段完整性、预算、子目标状态、重复调用和未处理错误。

#### 隔离上下文语义验证

使用独立 DeepSeek 验证调用，只输入问题、证据和答案，不输入 Controller 的推理历史，判断：

- entailment；
- contradiction；
- evidence sufficiency；
- claim–citation alignment。

#### 离线评测验证

正式实验不只依赖同一模型作为裁判。数据集标签、确定性引用指标和开源 NLI 或 Link-Eval 类指标作为主要依据，LLM Judge 仅作为补充分析。

## 4. 实验设计

### 4.1 数据集

#### 医学主实验

- **PubMedQA**：采用官方专家标注划分；
- **MedQA-USMLE**：采用官方测试集；
- **MedQA Mainland China**：用于跨语言和医学知识鲁棒性验证。

#### 跨领域验证

- **MuSiQue**：用于测试多跳、证据组合和问题拆解能力。

MuSiQue 的作用不是证明 SkillLINS 在所有任务上优于 Multi-Agent，而是检验优势是否与共享状态和顺序推理有关。

### 4.2 检索语料

核心实验禁止使用实时搜索，以保证可复现性：

- PubMedQA 使用冻结的 PubMed 摘要语料；
- MedQA 使用固定版本的公共医学教材和 PubMed 快照；
- MuSiQue 使用其公开上下文或固定支持文档集合；
- 使用 BGE-M3 和 FAISS 构建统一索引；
- 保存语料版本、文档哈希、索引配置和检索缓存。

所有方法共享同一语料、嵌入模型、索引和 `top-k`。主实验使用 `top-k=5`，在分层子集上测试 `top-k∈{3,10}`。

### 4.3 对比方法

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | Direct DeepSeek | 无检索基础线 |
| B1 | Vanilla RAG | 标准单次检索生成 |
| B2 | Original LINS MAIRAG | 原始论文方法 |
| B3 | Fixed Skill Pipeline | 无动态循环的技能流水线 |
| B4 | ReAct-style Single Agent | 有工具但无显式状态和停止门控 |
| B5 | Matched Multi-Agent Skills | 与 SkillLINS 使用相同 Skills 的 Multi-Agent |
| Ours | SkillLINS | 显式状态、技能循环和验证门控 |

B5 必须与 SkillLINS 使用相同模型、Skill 描述、检索器、证据和预算，只改变上下文是否分散到多个角色 Agent。这样才能将差异归因于系统拓扑，而非能力或工具差异。

### 4.4 公平预算

设置三档总生成预算：

- 4K Token；
- 8K Token；
- 16K Token。

预算统计覆盖所有模型调用，包括 Controller、角色 Agent、验证器和修订调用。所有对比方法具有相同的：

- 总 Token 上限；
- 最大工具调用数；
- 最大检索次数；
- 语料和检索器；
- 模型版本；
- 温度和输出限制。

主实验使用 `deepseek-v4-flash`；在三个数据集的分层 300 题子集上使用 `deepseek-v4-pro` 复验模型容量敏感性。默认温度为 0。正式运行前必须再次核对并记录服务商当时实际提供的模型 ID。

### 4.5 评价指标

#### 任务质量

- PubMedQA、MedQA：Accuracy；
- MuSiQue：Exact Match、Token F1；
- 拒答场景：Selective Accuracy、Risk–Coverage Curve。

#### 证据质量

- Evidence Recall@k；
- Citation Precision、Recall、F1；
- Supported Claim Rate；
- Unsupported Claim Rate；
- Contradiction Rate；
- Evidence Sufficiency。

#### 循环控制质量

- Finish Precision：被系统接受的完成请求中真正正确的比例；
- Premature Stop Rate；
- Invalid Finish Rejection Rate；
- Recovery-after-Rejection Rate；
- Budget Exhaustion Rate；
- 重复或无效 Skill Call 比例；
- 平均循环步数；
- 正确拒答率。

#### 效率

- 输入、输出和推理 Token；
- LLM 调用次数；
- 检索次数；
- 端到端延迟；
- 单题美元成本；
- 不同预算下的质量—成本 Pareto 前沿。

不使用单一“效果/成本”比值替代原始数据，同时报告质量、成本和 Pareto 支配关系。

### 4.6 消融实验

在每个数据集的分层样本上依次测试：

- 无显式 `LoopState`；
- 无 Verifier Gate，由模型自行停止；
- 无语义证据验证；
- 无引用验证；
- 无问题拆解；
- 无预算感知；
- 无验证失败后的修订；
- 将 Skills 替换回独立角色 Agent；
- 允许内部知识直接通过停止门控。

最后一项预计会提高覆盖率但降低证据可靠性，用于展示医学场景中门控约束的必要性。

### 4.7 边界实验

根据任务结构为样本增加三个属性：

- 顺序依赖程度；
- 子问题并行度；
- 跨子问题共享证据程度。

比较不同结构区间内 SkillLINS 和 Multi-Agent 的性能差异，验证：

- 高共享状态、高顺序依赖是否更适合单 Controller；
- 高并行度、低共享状态是否缩小差距或使 Multi-Agent 占优。

这部分将使论文从“提出一个新框架”提升为“研究 Agent 架构选择规律”。

## 5. 统计分析与可信性

- 完整测试集在温度 0 下运行一次；
- 每个数据集选取分层 300 题进行三次独立重复实验；
- Accuracy 差异使用 McNemar 检验；
- 成本、步数和引用指标使用配对 Wilcoxon 检验；
- 使用 10,000 次配对 Bootstrap 计算 95% 置信区间；
- 多重比较采用 Holm 校正；
- 同时报告效应量，不只报告显著性；
- Pareto 分析报告每种方法在三档预算下的支配次数和前沿面积；
- LLM Judge 对方法名称、运行轨迹和成本信息保持盲测。

## 6. 实施阶段与时间安排

### 第 1–2 周：安全与基线修复

- 移除代码中的明文 API Key 并立即轮换；
- 改为环境变量和不入库配置；
- 扫描当前分支和 Git 历史中的敏感信息；
- 修复现有 DeepSeek 测试脚本语法问题；
- 补充缺失数据下载、校验和和目录检查；
- 固化 Original LINS 的可复现实验结果。

不在本研究分支中合并当前分叉的 `master`，以 `main` 上的 LINS 实现作为方法基线。

### 第 3–5 周：Skill 化与统一接口

- 将 PRA、SKA、QDA、PCA 包装为类型化 Skills；
- 实现 `LoopState`、`SkillCall`、`SkillResult`、`ValidationReport` 和 `RunTrace`；
- 建立严格 Function Call Schema；
- 增加工具缓存、预算计算和轨迹回放。

### 第 6–8 周：Agentic Loop 与停止门控

- 实现单 Controller 循环；
- 实现确定性验证器和隔离语义验证器；
- 实现 finish rejection、修订和 abstain；
- 在约 30 个开发样本上完成端到端调试。

### 第 9–11 周：公平基线

- 接入 Direct、Vanilla RAG 和 Original LINS；
- 实现 Fixed Skill Pipeline；
- 实现无门控 ReAct 基线；
- 实现同 Skills、同预算的 Matched Multi-Agent。

### 第 12–15 周：正式实验

- 每个数据集先运行 100 题试验；
- 只使用开发集修订 Prompt 和门控规则；
- 冻结配置后运行 DeepSeek Flash 全量实验；
- 运行三档 Token 预算；
- 完成 DeepSeek Pro 分层子集复验。

### 第 16–18 周：消融和边界分析

- 完成组件消融；
- 分析停止错误、重复调用、证据冲突和拒答行为；
- 按任务并行度、共享状态和顺序依赖分组；
- 构建质量—成本 Pareto 图和失败分类体系。

### 第 19–22 周：论文与开源材料

- 撰写方法、实验和安全限制；
- 整理 Prompt、Schema、配置和环境锁定文件；
- 发布去除密钥后的代码；
- 发布可公开的运行轨迹、评测脚本和统计脚本；
- 准备匿名仓库和复现说明。

## 7. 测试与验收标准

### 7.1 工程测试

- 每个 Skill 的 Schema 合法性和异常输入测试；
- 非法 evidence ID 必须被拒绝；
- 门控失败后不得直接完成；
- 超预算后不得继续调用模型；
- 相同输入的缓存结果必须一致；
- `RunTrace` 可以完整回放一次决策过程；
- 固定随机种子和缓存条件下结果可复现；
- 仓库及实验日志不得包含密钥。

### 7.2 研究成功标准

满足以下任一质量—成本条件：

1. 在至少两个数据集上 Pareto 支配 Original LINS 和 Matched Multi-Agent，且第三个数据集无显著退化；或
2. 准确率差异控制在 1 个百分点以内，同时平均 Token 成本降低至少 20%；或
3. 在相同成本下取得统计显著的质量提升。

同时必须满足：

- 无效完成或无证据完成相对减少至少 30%；
- 引用正确性显著提高；
- 主要结论具有 95% 置信区间；
- 优势能够通过消融定位到显式状态、门控或技能编排，而不是 Prompt 差异；
- 完整报告 Multi-Agent 胜出的任务区间。

如果没有达到优势标准，则将论文重定位为：

> 单 Agent Agentic Loop 与 Multi-Agent 在等预算、等能力条件下的系统性边界研究。

不得在结果不支持时声称“单 Agent 普遍优于 Multi-Agent”。

## 8. 论文结构

1. **Introduction**：Multi-Agent 的角色冗余、状态复制、成本和停止不可靠问题；
2. **Related Work**：LINS、Self-RAG、CRITIC、ReAct、Multi-Agent 失败分析和成本评测；
3. **Role-to-Skill Formalization**：角色、Skill 和系统拓扑的形式化区分；
4. **SkillLINS**：Evidence-State、Function Calls、Agentic Loop 和验证门控；
5. **Experimental Protocol**：等能力、等工具、等预算的比较；
6. **Results**：质量、证据、成本和停止可靠性；
7. **Ablation and Boundary Analysis**：贡献来源及适用边界；
8. **Safety and Limitations**：医学风险、同模型验证偏差、数据污染和拒答；
9. **Conclusion**：从“增加 Agent”转向“显式控制状态、工具和停止条件”。

## 9. 复现与开源产物

最终匿名仓库至少包含：

- 完整 Skills JSON Schema；
- Controller 与 Verifier 配置；
- 冻结语料和索引 manifest；
- 数据下载脚本与 checksum；
- 所有基线的统一运行入口；
- Prompt 版本和实验配置；
- 去除敏感信息的 RunTrace；
- 成本、延迟和 Token 统计脚本；
- 统计检验和制图脚本；
- 环境锁定文件和最小复现说明。

## 10. 明确假设与默认决定

- 方法采用 training-free、Prompt-controlled 的第一版，不进行强化学习或 Controller 微调；
- 医学问答是主验证领域，MuSiQue 负责跨领域边界验证；
- 使用单一持久 Controller，不创建长期存在的专家角色 Agent；
- 验证器由确定性规则和隔离上下文的 DeepSeek 语义验证组成；
- `finish` 只是停止申请，系统门控拥有最终决定权；
- 原始 LINS 保持不变作为基线，新方法通过包装现有能力实现；
- 核心实验使用冻结语料，不使用实时搜索；
- 论文主张限定为质量—成本—可靠性的条件性优势，不宣称单 Agent 对 Multi-Agent 的普遍替代。
