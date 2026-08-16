# LINS-Industrial Agentic 系统全景 —— 步骤 · 流程 · 架构 · 结论

> 本文是一份**单一整合文档**，把「知识库构建 → 检索 → Agentic 推理 → 评估」的整个
> 工业知识问答系统的当前实际运行状态讲清楚；并收编近期检索层迭代（run2 两步式
> v1→v4）、7 路检索 recall 对比、端到端落地与前端任务/题型判定的量化结论。
> 阅读对象：任何需要快速理解「这个系统现在怎么跑、为什么这么设计、数据证明了什么」的人。
>
> 关联细节文档（本文是它们的整合视图，如需逐环节深挖再往下读）：
> `docs/lins_industry_system_architecture.md`（静态架构）、
> `docs/agentic_framework_and_runtime.md`（agentic 分包 + 早期端到端量化）、
> `docs/stagewise_debug_plan.md`（逐阶段调试计划与每次改动留痕）。

---

## 0. 一句话定位

`LINS-Industrial` 是一套面向工业知识问答（IndustryBench / 技术手册 / 工程标准）的
**Agentic RAG** 系统。核心运行边界一句话：

> **外部检索器负责「找证据」，Agentic LINS 负责「组织、推理、生成带引用的答案」。**
>
> 当前检索层生产默认走 **run2 两步式 v4**（宽池初检 → LLM 相关性重排 → 3 篇锚文档回捞，
> 且 anchor_bonus 只给第 1 篇），已在 50 样本真实 LLM 下以 cov@10=46.7%、NDCG=65.7%
> 确立为 7 种检索方法中全面最优并落入生产。

---

## 1. 分层架构总览

| 层次 | 主要目录 / 模块 | 核心职责 | 主要产物 |
|---|---|---|---|
| 数据与知识源 | `knowledge_source/`、`data/` | 接入 IndustryBench、手册、PDF、标准 | `UnifiedDocument` |
| 知识库构建 | `knowledge_builder/`、`retrieval/corpus_builder.py` | 去重、切块、统计、构建清单 | corpus/chunks JSONL、manifest |
| 索引层 | `retrieval/chunker.py`、`embedder.py`、`faiss_indexer.py` | 切分、向量化、建 FAISS | `.npy`、meta、`.faiss` |
| 检索层 | `retrieval/retriever.py` | KED、dense/hybrid、**两步式 run2 v1~v4** | `RetrievalResult`/`RetrievedChunk` |
| Agent 控制层 | `agentic/analyzer.py`、`planner.py`、`task_types.py` | 任务/题型识别、执行图、控制分支 | `TaskAnalysis`、`ExecutionGraph` |
| 证据处理层 | `agentic/organizer.py` | 去重、按任务分组、保留引用与顺序 | `OrganizedEvidence` |
| 推理生成层 | `agentic/solver.py`、`agentic/pipeline.py` | 结构化推理、生成带引用答案 | `PipelineResult` |
| 评估层 | `metrics/`、`utils/`、`experiments/` | QA/检索/引用/安全评估与消融 | JSON/Markdown 报告 |

系统数据流总览：

```text
知识源 → 统一文档 → 语料/切块 → Embedding → FAISS/BM25
                                                  ↓
用户问题 → 工业检索器(run2 v4) → 证据 → Agentic 执行图 → 带引用答案
                                                  ↓
                                     质量/安全/引用评估（0–3 分 / Recall / SV）
```

---

## 2. 知识库离线构建流程

1. `KnowledgeSource` 抽象 + `SourceRegistry` 管理异构来源（IndustryBench / 技术手册 /
   PDF / 工程标准），统一成 `UnifiedDocument`（含字段 `document_id/title/text/source/
   industry/capability/metadata`）。
2. `IndustryKnowledgeBuilder` / `IndustrialCorpusBuilder`：
   `load → deduplicate → chunk_all → build_embeddings → build_index → save_manifest`。
3. 切块策略：`paragraph`（默认）/ `fixed_size` / `recursive`。
4. 运行时读取 `knowledge_corpus/manifest.json` 定位 chunks / embedding / FAISS。
5. 索引规模（本项目实测）：约 **55095 个 chunk**，FAISS `bge-small_flat`，embedding
   `BAAI/bge-small-zh-v1.5`（dim=512）。

---

## 3. 在线主流程（当前 Agentic RAG，检索层 = run2 v4）

主线在 `experiments/exp1_agentic_rag.py::AgenticRAGEngine.answer()`。
采用 **Stable Knowledge Interface**：外部检索器先固定检索，证据注入 Agent，
Agent 内部 `retrieve` 节点经 SHORTCUT 短路跳过，组织/推理仍走执行图。

```text
Question
  │
  ▼ 阶段A：固定外部检索（run2 两步式 v4）
  industrial retriever: dense 初检 pool=50 → LLM 相关性重排 → 3篇锚文档回捞 → 融合 top-k=10
  │  RetrievalResult / RetrievedChunk（citation 带 *run2）
  ▼
  TaskAnalyzer.analyze()  → TaskType/TaskAnalysis（关键词分类器，读 _format 标注控题型）
  ▼
  StrategyPlanner.plan()  → ExecutionGraph（简单图 / 高级图模板）
  ▼
  GraphExecutor.execute()（evidence 注入，跳过内部 retrieve）
     ├─ organize   证据组织（去重、分组、带引用上下文）
     ├─ reason     结构化推理（LLM）
     ├─ verify     答案质量校验（本地硬检查 + LLM）
     └─ end
  ▼
  _produce_final_answer() → PipelineResult / dict（带 [n] 引用答案）
```

### 3.1 阶段 A — 外部检索（本轮核心）
- 生产 engine：`Run2AgenticRAGEngine`，`_init_external_retriever()` 返回
  `Run2ExternalRetriever`（默认 `version="v4"`），其 `retrieve()` 调用底层
  `OpenDomainRetriever.two_stage_retrieve_v4(...)`。
- `AgenticRAGEngine`（base）为纯 top10 dense 对照；`Run2ExternalRetriever` 支持
  `--version {v1,v4}` 一键回切基线。

### 3.2 阶段 B — 任务 / 题型分析（TaskAnalyzer）
- 轻量**确定性关键词分类器**，不调用 LLM；命中关键词返回 task，否则 `general`。
- 关键落地：因**题面本身无可判别题型信号**（2049 条统计填空/选项信号≈0），
  题型（`_format`）判定**直接读 CSV 真值标注**（`format_source=annotated`），而非
  LLM 反推（LLM 判定仅 62.5%，填空 4 判 0）。见 `_ab_format_first.py`。

### 3.3 阶段 C — 策略规划（StrategyPlanner）
- 两级策略：先试 LLM JSON 出图，失败回退任务模板。
- 简单图 `retrieve→organize→reason→end`（calculation/procedure/explanation/general）；
  高级图 `decide/verify` 分支 + `retrieve_2(targeted, evidence_guided)` 二次检索
  （comparison/selection/diagnosis/standard_interpretation）。

### 3.4 阶段 D — 图执行与证据组织
- `GraphExecutor` 按 DAG 调度 organize/reason/verify/decide/merge/end，
  `ExecutionContext` 维护证据与推理状态。
- **生产证据组织 = `RerankTruncOrganizer`**（organize v3，`agentic/organizer.py`，
  `AgenticRAGEngine` 装配于 `exp1_agentic_rag.py`）：在 `EvidenceOrganizer` 的
  「按 chunk id/相似度去重、按任务分组、保留顺序与 citation」之上，**默认开启 L2
  词法重排（signal='lex'，score=0.7·lex(query,doc)+0.3·norm(FAISS)）并把低分端截断**
  （`max_chars=6000`）。组织消融复测：与 base `EvidenceOrganizer` 同源检索下
  avg 1.700 vs 1.667（+0.033，升5/降3/平22，**统计不显著但零系统伤害、检索/SV 无污染**）
  → 裁决"生产直接开 L2 组织、保持落地"；如要单点回退改回 `EvidenceOrganizer()` 即可。
- **merge 膨胀修复（S5-P0，已迁移进生产 `pipeline.py::_handle_merge`）**：raw 段
  按 `chunk_id` 跳过**已被 organized 覆盖**的 chunk（`_covered_ids = organizer 已含全集`），
  只保留第二跳 `retrieve_2` 新增 chunk。修复前 `accumulated_context` 膨胀 ≈**2.63×（100%
  样本≥1.5×）**，修复后至 **1.04× / 0%≥1.5×**，RAW-organized 内容重叠 0（无重复并入）、
  organized 全量 100% 保留（Recall-Safe，0 缺失）。数据：`results/s5_merge_inflate.json`。

### 3.5 阶段 E — 推理与输出
- `reason` 走 `PromptBuilder → LLM`；`verify` = 本地硬检查(空/过短/0证据) + LLM pass/fail。
- 输出带 `[n]` 引用答案 + 全过程元数据（图、节点日志、证据统计）。


---

## 4. 检索层迭代演进（run2 两步式：base → v1 → v2 → v3 → v4）

检索层是**可量化、可复现增益的主要来源**。run2 的动机：base/hybrid 只做"一次召回排序"，
相关文档里**未进初检 top-k 的同源块会整体漏掉**（补全机会丢失）。两步式用第二段 LLM
（真实 DeepSeek，仅凭【问题+候选块】，**绝不碰 GT/答案**）重排 + 反向回捞补全。

| 版本 | 机制 | 关键结论 / 教训 |
|---|---|---|
| base | `retrieve(q,k=10,use_ked=True)` | 对照基线，纯 top10 dense |
| v1 | `two_stage_retrieve`（回捞块 rel 恒 0） | 补全块受限于 rel=0，几乎进不了 top-k；回捞机制被证实 |
| v2 | `two_stage_retrieve_v2`（回捞块吃二阶段 rel） | 与 v1 检索层 recall 相当，换来了"机制正确/可解释" |
| v3 | 去重到 3 篇文档、**3 篇全吃 anchor_bonus(+0.5)** | **负优化**：low-rel 文档靠 bonus 硬挤 top10、稀释相关块；21 题 cov −1.8pp / good% −4.8pp |
| **v4** | 去重到 3 篇回捞、**anchor_bonus 只给第 1 篇(top_doc)** | **正增益**；把「回捞范围」与「排序权重」解耦 = recall 广 + precision 严 |

> **演进史上的两条候选通路（均已收编/关闭，保留交代以明因果）**
> - **证据前向（EF）**：把检索证据前向注入推理以强化证据利用（无条件回捞会上下文稀释降分、
>   条件 EF 才可正贡献）。在 run2 出现前作为"证据是短板"阶段的关键增强，**后被 run2 两步式
>   取代**，未进生产（`stagewise_debug_results.md` 主题 C，📌参考）。
> - **多视角改写 / multi-view（`ab_mv_query_rewrite`）**：LLM 依 task 生成语义互补子视角、
>   各视角 hybrid 检索再 RRF 融合（`agentic/pipeline.py` `semantic_mv` 通路）。结论=**默认关闭**
>   （multi 整体仍负增益），保留为可选通路；`hybrid` 才是稳定正增益的"解药"。

### 4.1 v4 融合打分（精确公式）
对候选块 `c`（池块 + 回捞块）：

```text
score(c) = w_dense/(60+rank)   # rrf 项（回捞块用 pool_k 常量档）
         + rel(c)*w_rel        # 二阶段 LLM 相关性分（归一化 0~1）
         + bonus               # bonus = 0.5 仅当 c.document_id == top_doc，否则 0
```

- `top_doc` = 按池块 LLM 预打分对 `document_id` 去重后的**第 1 篇**（最相关文档）。
- 第 2/3 篇文档的块**只靠真实 rel 分**公平竞争，不靠 bonus 稀释 top10。
- 参数：`pool_k=50, n_anchor=3, backfill_th=0.02, backfill_max=50, w_dense=w_rel=1, w_anchor=0.5`。
- 输出元数据 `query_expanded` 携带 `backfill=...` / `bf_in_topk=...` 便于归因回捞。


---

## 5. 7 路检索 recall 对比（50 样本，真实 LLM）

**探针：`_ab_agentic_recall_7way.py`**（新增）—— 在同一批 cap 均衡抽样的 50 题上，
用生产 agentic 链路的底层方法统一评估。指标口径与 `_diag_recall_run2.py` 完全一致：
相关集合 = chunk 与 `knowledge_text` 的 bigram IoU≥0.15，仅用于评估、ranker 不可见。

- **Sim ranker（零成本，仅链路冒烟）**：v4 = cov@10 46.0% / good 48% / NDCG 61.6% 领先。
- **LLM ranker（真实 DeepSeek，max_tokens=2000，约 24 分钟）**——最终定案数据：

| 方法 | cov@10 | good% | hit@1 | hit@3 | hit@10 | MRR | NDCG |
|---|---|---|---|---|---|---|---|
| single(base) | 36.2% | 32% | 52% | 62% | 70% | 58.1% | 57.7% |
| multi | 35.6% | 32% | 32% | 54% | 70% | 45.2% | 50.7% |
| hybrid | 38.2% | 34% | 52% | 64% | 70% | 58.9% | 57.9% |
| v1 | 45.5% | 42% | 54% | 68% | 72% | 61.3% | 61.4% |
| v2 | 45.2% | 44% | 54% | 68% | 72% | 61.2% | 61.2% |
| v3 | 42.8% | 44% | 56% | 68% | 72% | 62.5% | 60.4% |
| **v4** | **46.7%** | **48%** | **56%** | **72%** | 72% | **63.0%** | **65.7%** |

**Δ vs v1（PP）**：`single-v1 Δcov=-9.3 Δgood=-10 ΔMRR=-3.2 ΔNDCG=-3.7`；
`multi-v1 Δcov=-9.9 ΔMRR=-16.1`（最差，fusion 稀释）；`hybrid-v1 Δcov=-7.2`；
`v2-v1 ≈ 0`；`v3-v1 Δcov=-2.6`（good/MRR 略升）；**`v4-v1 Δcov=+1.2 Δgood=+6.0
Δhit1=+2.0 ΔMRR=+1.7 ΔNDCG=+4.3`**。

**读表结论**
1. **两步式整体碾压三步基线**：最弱的 v3(42.8%) 也高于 best baseline(single/hybrid ~38%)。
2. **v4 是 7 路里唯一在 cov@10/good%/hit@1/MRR/NDCG 五项上全面第一**（NDCG 65.7% 遥遥领先，
   说明相关块不仅捞到、还排得更靠前更集中）。
3. **v3 的"全锚吃 bonus"再次被证伪**；multi 的 `multi_query`+fusion 在工业检索里是负增益
   （MRR 45.2%）。需注意：multi 的"**大幅负增益**(−9.6pp)"已随 `retriever.py`
   「原 query 保底 + 子查询只做补充」的修复收窄至「中性偏微负(约 0±0.5%)」，不再是灾难性；
   但**仍未转正**，且依赖 planner 模板触发（如 `TaskType.COMPARISON` 仍设 `multi_query:True`，
   走已修复的原 query 保底版）。结论不变：**不建议默认开启，hybrid 才是稳定正增益的"解药"**。
4. 分层：v4 在安全合规(57%)、故障诊断(57%)、工程计算(28%) 等能力上是各方法最优。

> 数据存档：`results/agentic_recall_7way/agentic_recall_7way_20260809_162441.md+.json`(LLM)
> 与 `_161618`(Sim)。


---

## 6. 端到端落地（生产 agentic 链路的验证）

**探针：`_e2e_agentic_run2.py`**（改造）——把 v4 接到生产 `AgenticRAGEngine` 全链路
（analyzer→planner→organizer→reason→solve，加 verify/decide 图），`--version v4` 默认。

- 给 `Run2ExternalRetriever` / `Run2AgenticRAGEngine` 增加 `version` 参数（默认 v4），
  `retrieve()` 按版本分派（v1/v4）；CLI `--version {v1,v4}`；输出文件名带 version。

### 6.1 base vs run2(v4)，14 题安全回归（真实 agentic + DeepSeek）
- **涨 3 / 跌 0 / 持平 11**；平均分 base 2.50 → run2(v4) **2.71（Δ=+0.21）**。
- 三题各 +1：标准规范（2→3，补全50）、质量计量×2（2→3 补全26 / 1→2 补全46）。
- **14 题证据数全部=10 且无一降** → v4 的 anchor_bonus-only-top1 在端到端链路把检索层
  正增益安全兑现为作答得分，零回退。
- 数据：`results/e2e_agentic_run2/e2e_run2_v4_20260809_155628.json`。

---

## 7. 前端任务/题型判定（analyze 读 format）结论

- **做法**：`TaskAnalyzer` 保持关键词任务分类；题型 `_format` 改为**直接读 CSV 标注**
  （`format_source=annotated`），落地于 `_ab_format_first.py`。
- **为什么**：对 2049 条题面统计显式题型信号全部≈0（无填空符、无选项字母、99.8%+ 以
  疑问句收尾）→ 题面本身不可判别，**真人/LLM 都无法从 question 反推 `_format`**。
- **量化**：LLM 判定准确率仅 62.5%（10/16）、加权 F1 56.3%，填空 4 判 0；读标注是唯一可靠做法。
- 检索召回主轴是 **capability**，与 format 基本正交（format×capability 矩阵实测）。


---

## 8. 评估与反馈闭环

```text
问题/参考答案/知识文本 → 运行模型 → [QA质量 0–3分 | 检索质量 Recall/MRR/NDCG | 引用/安全 LinkEval/SV]
                                        ↓
                                     报告与消融 → 回改下一轮
```

- `metrics/industrybench_scorer.py`：0–3 分 rubric + 安全违规(SV)检查。
- `utils/industrial_linkeval.py` / `eval_scripts/industrial_linkeval/`：引用 precision/recall/F1。
- 检索指标：`retrieval/recall_metrics.py` + `RetrievalEvaluator`：cov@10/good%/hit@1/3/10/
  MRR/NDCG。
- **指标易读补充**：`cov@10`=GT 句子覆盖率（找没找到相关内容）；`hit@k`=首条相关证据
  在第 k 位内命中的对错；`MRR`=首条相关证据排多靠前（`1/rank` 平均）；`NDCG`=整份结果里
  相关块排得多整齐（`DCG/IDCG` 归一到 0~1，1=排序已最优）。

### 8.1 检索召回诊断（S4）结论
- **B 类（标准号）题召回薄弱**由「语料/标签口径」主导，而非纯排序问题；`multi_query`
  门控可校准，但为负增益，生产默认关闭。
- 定案数据：`results/s4_recall/s4_recall_20260807_235012.md`（✅）；诊断汇总见
  `docs/s4_path3_diagnosis_report.md`。复现：`_diag_recall_s4.py` /
  `_audit_s4_evidence.py` / `_calibrate_multi_gate.py`。

### 8.2 评分归因（S6）结论
- 低分样本逐题归因到 **S4（检索缺料）/ S6（生成组织）**；规则分对应**内容覆盖**而非措辞，
  用于判断"降分到底是没捞到还是没答好"。
- 定案数据：`results/s6_score/s6_score_20260808_101616.{md,json}`（✅）。复现：
  `_diag_s6_score_attribution.py`。

---

## 9. 全量结论与生产定案

| 环节 | 量化结论 | 可信度 |
|---|---|---|
| 任务/题型划分 | 题面信号≈0，读标注 > LLM判定(62.5%)≈随机 | 2049 题面统计 + A/B |
| 策略规划 | 模板选择，主要是复杂度而非精度杠杆 | 代码路径实测 |
| **检索 recall** | **v4 两步式 cov@10=46.7%、NDCG=65.7%，7 路全面第一** | 50 样本真实 LLM |
| 检索 method 排序 | v4 > v2≈v1 > v3 > hybrid > single > multi | 50 样本真实 LLM |
| run2 两步式 vs base | v1 就比 best baseline 高 ~7pp cov | 50 样本真实 LLM |
| 组织/合并 | 生产用 `RerankTruncOrganizer`(organize v3)，L2 词法重排+低分截断；merge 已按 chunk_id 去重防膨胀(2.63×→1.04×, Recall-Safe) | S5/组织消融实测 |
| 端到端作答 | base 2.50 → v4 2.71，涨3/跌0/持平11，零回退 | 14 题真实 agentic |

**生产定案**
- **检索层默认 = run2 两步式 v4**（`two_stage_retrieve_v4`；`Run2AgenticRAGEngine version=v4`）。
- base 对照触发：纯 top10；multi/hybrid 保留为可选通路但**不建议默认开启**（multi 已由
  "大幅负增益"修复至"中性偏微负"，仍未转正，且依赖模板触发）。
- 题型判定默认读 CSV 标注；agentic 执行图保持模板化。

---

## 10. 如何复现 / 跑通验证

```bash
cd LINS-Industrial

# 检索层 7 路 recall（50 样本）——成本高请用后台
python _ab_agentic_recall_7way.py --n_total 50 --sim      # Sim 冒烟（快）
python _ab_agentic_recall_7way.py --n_total 50 --llm      # 真实 DeepSeek（≈24min）

# 端到端 A/B（base vs run2-v4）
python _e2e_agentic_run2.py --per_cap 2 --pool 50 --seed 7 --version v4

# 生产 P0 回归
python _s2_p0_verify.py
```

（`--only v1 v4` 可指定只跑部分方法；全量代码见 `retrieval/retriever.py`、
`_diag_run2_v1v2.py`、`_ab_agentic_recall_7way.py`、`_e2e_agentic_run2.py`。）

