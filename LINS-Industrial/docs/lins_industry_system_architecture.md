# lins_industry 系统框架与核心流程

> 本文基于 `LINS-Industrial` 当前代码整理，重点描述可运行的工业知识问答 / Agentic RAG 主线，并标注与旧版 LINS 调用的边界。

## 1. 系统定位

`LINS-Industrial` 是在 LINS 多智能体检索增强框架上面向工业知识问答场景的扩展层，主要承担四类职责：

1. 将 IndustryBench、技术手册、工程标准等异构资料统一成工业知识文档。
2. 将文档构造成可检索的 chunk、向量和 FAISS 索引。
3. 针对工业问题执行 KED 查询扩展、向量检索、多查询融合和证据组织。
4. 通过任务分析、执行图、结构化推理和引用生成完成回答，并用 IndustryBench / LinkEval 进行评估。

系统可以概括为：

```text
知识源 → 统一文档 → 语料/切块 → Embedding → FAISS/BM25
                                               ↓
用户问题 → 工业检索器 → 证据 → Agentic 推理管线 → 带引用答案
                                               ↓
                                      质量/安全/引用评估
```

## 2. 分层架构

| 层次 | 主要目录/模块 | 核心职责 | 主要产物 |
|---|---|---|---|
| 数据与知识源层 | `knowledge_source/`、`data/` | 接入 IndustryBench、技术手册、PDF、工程标准等 | `UnifiedDocument` |
| 知识库构建层 | `knowledge_builder/`、`retrieval/corpus_builder.py` | 去重、切块、统计、生成构建清单 | corpus JSONL、chunks JSONL、manifest |
| 索引层 | `retrieval/chunker.py`、`embedder.py`、`faiss_indexer.py` | 文本切分、向量化、FAISS 建索引 | `.npy`、meta JSONL、`.faiss` |
| 检索层 | `retrieval/retriever.py` | KED、Embedding、FAISS、可选 BM25、多查询融合 | `RetrievalResult` / `RetrievedChunk` |
| Agent 控制层 | `agentic/analyzer.py`、`planner.py`、`task_types.py` | 识别任务、生成执行图、控制分支 | `TaskAnalysis`、`ExecutionGraph` |
| 证据处理层 | `agentic/organizer.py` | 去重、按任务分组、保留引用和检索顺序 | `OrganizedEvidence` |
| 推理生成层 | `agentic/solver.py`、`agentic/pipeline.py` | 按任务执行结构化推理、生成带引用答案 | `PipelineResult` |
| LINS 兼容层 | 同级 `LINS-main/`、`model.model_LINS.LINS` | 复用原始 MAIRAG、多智能体和旧评估接口 | `MAIRAG` 返回值 |
| 评估层 | `metrics/`、`utils/`、`eval_scripts/`、`experiments/` | QA、检索、引用、安全和消融评估 | JSON/Markdown 报告 |

## 3. 知识库离线构建流程

### 3.1 知识源接入与统一

`knowledge_source/` 通过 `KnowledgeSource` 抽象和 `SourceRegistry` 管理不同来源。每种来源最终转换成 `UnifiedDocument`：

```text
document_id / title / text / source / industry / capability / metadata
```

典型来源包括：

- `IndustryBenchSource`：从 IndustryBench CSV 读取问题关联知识。
- `TechManualQASource`：技术手册问答资料。
- `PDFManualSource`：PDF 手册。
- `EngineeringStandardsSource`：工程标准。

`SourceRegistry.load_all()` 负责批量加载，`search_all()`、`filter_by_industry()`、`filter_by_capability()` 提供统一查询。

### 3.2 语料构建

`IndustryKnowledgeBuilder` / `IndustrialCorpusBuilder` 的标准链路为：

1. `load_industrybench()` 或 `add_*()`：加载 CSV、TXT、JSONL 等资料。
2. `deduplicate()`：按文档内容 hash / `document_id` 去重，并合并非空元数据。
3. `chunk_all()`：将长文档切成检索单元。
4. `build_embeddings()`：使用 BGE 等模型生成向量。
5. `build_index()`：构建 FAISS 索引。
6. `save_manifest()`：保存文档数、chunk 数、来源、模型、索引路径等元数据。

切块支持三种策略：

- `paragraph`：按段落切分，默认策略。
- `fixed_size`：固定字符长度并保留 overlap。
- `recursive`：按段落、句子递归切分。

运行时主要读取 `knowledge_corpus/manifest.json`，并通过 manifest 找到 chunks、embedding 和 FAISS 文件。

## 4. 在线问答主流程（当前 Agentic RAG 主线）

实际主线位于 `experiments/exp1_agentic_rag.py` 的 `AgenticRAGEngine.answer()`。为保证检索可复现，当前实现采用“Stable Knowledge Interface”：先由外部工业检索器固定检索，再把证据注入 Agent，Agent 内部的 `retrieve` 节点被跳过。

### 阶段 0：请求进入

入口通常是实验脚本中的 `AgenticRAGEngine.answer(question)`，评估脚本再对返回答案进行打分。输入至少包含问题；选择题还包含 options，评估样本包含参考答案、行业、能力和难度。

### 阶段 1：固定外部检索

`IndustrialRetriever` 封装 `retrieval.retriever.OpenDomainRetriever`：

```text
question
  → KED 关键词提取/扩展
  → query embedding
  → FAISS dense search（可选 BM25/hybrid、多查询融合）
  → top-k RetrievedChunk
  → RetrievalResult
```

关键行为：

- 默认 `k=10`，可由 `retrieval_k` 覆盖。
- `use_ked=True` 时执行查询扩展。
- 支持 `multi_query_retrieve()`、fusion/union、按文档或内容聚合。
- 检索结果带 `chunk_id`、`document_id`、`score`、`rank`、`industry`、`capability`、`citation` 等元数据。
- 若 embedding 非法，检索器会尝试原问题回退；最终失败会在 `RetrievalResult.retrieval_failed*` 留痕。

### 阶段 2：任务分析

`TaskAnalyzer.analyze(question)` 输出 `TaskAnalysis`，当前轻量实现主要依据规则识别任务类型；设计上也支持 LLM 生成丰富语义字段。任务类型包括：

`comparison`、`diagnosis`、`calculation`、`selection`、`procedure`、`standard_interpretation`、`explanation`、`general`。

同时给出复杂度、是否需要多来源、是否需要公式/分步推理、偏好知识源和期望证据类型。

### 阶段 3：策略规划

`StrategyPlanner.plan(task_analysis)` 输出 `ExecutionGraph`。优先尝试 LLM 生成并校验图结构，失败时使用按任务类型定义的 fallback 模板。

图节点类型：

- `retrieve`：检索证据。
- `organize`：组织证据。
- `decide`：判断证据是否充分。
- `merge`：合并多路证据。
- `reason`：调用 LLM 执行推理。
- `verify`：校验答案质量。
- `end`：结束。

典型图：

```text
retrieve_1 → organize_1 → decide_1 ── sufficient ─→ reason_1 → verify_1 → end
                                  └─ insufficient → retrieve_2 → merge_1 ─┘
```

对于简单问题通常是 `retrieve → organize → reason → end`；比较、诊断、选择等复杂问题可启用多跳检索、条件分支和验证。

### 阶段 4：执行图与证据组织

`GraphExecutor` 在 `agentic/pipeline.py` 中按图遍历节点，维护 `ExecutionContext`，其中保存：

- 每次检索结果 `evidence_cache`。
- 最近一次检索结果 `last_retrieval_results`。
- 最近一次组织结果 `last_organized`。
- 节点执行日志、分支结果和中间答案。

在当前主线中，外部检索结果通过 `pre_retrieved_evidence` 注入，因此所有内部 `retrieve` 节点直接跳过；组织和推理仍按执行图运行。若直接调用 `AdaptiveAgenticPipeline.run()` 且不传外部证据，则会执行图中的内部检索节点。

`EvidenceOrganizer.execute()` 当前是轻量处理：

1. 去重，避免同一 chunk / 高相似内容重复。
2. 按 `comparison`、`symptom`、`candidate`、`formula`、`chronological`、`topic` 等策略分组。
3. 保留 FAISS 原始排序、所有去重后的文档和 citation，不做二次相关性过滤。
4. 生成 `OrganizedEvidence.get_context()`，把证据格式化成带 `[1]`、`[2]` 等引用编号的上下文。

### 阶段 5：任务感知推理与答案生成

`TaskSolver.solve()` 接收问题、组织后的证据和 `ExecutionDirective`：

```text
任务类型/推理类型
  → 任务专用 system prompt
  → 明确的 workflow steps
  → 证据上下文 + citation 约束
  → DeepSeek/OpenAI-compatible LLM
  → answer
```

推理类型对应不同工作方式，例如比较、诊断、计算、选择、流程、标准解释和一般说明。默认要求每个关键结论引用证据 chunk，并在末尾执行完整性、准确性和引用检查。

### 阶段 6：输出封装

`AdaptiveAgenticPipeline.run()` 最终返回字典 / `PipelineResult`，主要包含：

- `answer`：最终回答。
- `task_analysis`：任务识别结果。
- `execution_graph`：实际执行图。
- `execution_nodes`：节点执行日志。
- `node_count`、`second_retrieval_triggers`：运行统计。
- `used_external_retrieval`：是否使用 Stable Knowledge Interface。
- 组织后的 group/document 统计和引用信息。

## 5. 两条运行路径的边界

### 当前推荐路径：Agentic RAG v2

```text
IndustrialRetriever（固定检索）
  → TaskAnalyzer
  → StrategyPlanner
  → GraphExecutor（注入外部 evidence，跳过 retrieve 节点）
  → EvidenceOrganizer
  → TaskSolver
  → 带引用答案
```

它对应 `experiments/exp1_agentic_rag.py` 当前重设计版本，检索和 Agent 推理解耦，便于公平比较和稳定复现实验。

### 兼容/旧路径：LINS MAIRAG

部分 `eval_scripts/industry_eval/*`、`experiments/exp1_qa.py`、`exp3_citation.py`、`exp4_ablation.py` 直接实例化 `LINS-main/model/model_LINS.py` 的 `LINS`，调用 `MAIRAG()` 或 `chat()`。这条路径由 LINS-main 自带的多智能体检索/推理逻辑驱动，可能包含自身检索、PRM 重排和引用生成，不等同于 Agentic v2 的执行图。

因此分析实验结果时，应明确记录：

- 是 `AgenticRAGEngine` 还是 `LINS.MAIRAG`。
- 是否使用外部固定检索。
- 是否启用 adaptive retrieval / organize / reasoning。
- 是否走 closed-book、普通 RAG 或 agentic RAG。

## 6. 评估与反馈闭环

评估层不参与在线回答，但用于验证各核心阶段：

```text
问题/参考答案/知识文本
          ↓
      运行模型
          ↓
 ┌────────┼─────────┐
 QA质量   检索质量   引用/安全
 0–3分    Recall@k   Precision/Recall/F1
 SV调整   MRR/NDCG  LinkEval
          ↓
       报告与消融
```

- `metrics/industrybench_scorer.py`：IndustryBench 0–3 分 rubric，安全违规（SV）检查，并输出 adjusted score。
- `eval_scripts/industrial_linkeval/`：按 QA、填空、选择题和检索格式分流评估。
- `utils/industrial_linkeval.py`：抽取回答中的 `[n]` 引用，计算 citation precision/recall/F1，并做陈述一致性、流畅度检查。
- `experiments/exp1_qa.py`：端到端问答。
- `exp2_retrieval.py`：检索 Recall@k、MRR、NDCG 等。
- `exp3_citation.py`：引用质量。
- `exp4_ablation.py`：比较各自适应组件开关对效果的影响。

## 7. 关键对象与数据流映射

| 对象 | 产生模块 | 消费模块 | 含义 |
|---|---|---|---|
| `UnifiedDocument` | `knowledge_source` | builder/corpus | 统一的原始知识文档 |
| `IndustrialDocument` | `retrieval/corpus_builder.py` | chunker | 语料构建阶段的文档记录 |
| `DocumentChunk` / `KnowledgeChunk` | chunker/builder | embedder/index/retriever | 可检索文本块 |
| `RetrievedChunk` | `OpenDomainRetriever` | organizer/solver/LINS | 带分数和引用的检索证据 |
| `RetrievalResult` | retriever | AgenticRAGEngine/评估器 | 一次检索的完整结果 |
| `TaskAnalysis` | analyzer | planner/pipeline | 问题任务语义 |
| `ExecutionGraph` | planner | GraphExecutor | 可执行工作流 DAG |
| `OrganizedEvidence` | organizer | solver/pipeline | 去重、分组、带引用上下文 |
| `PipelineResult` | pipeline | 实验/评估 | 最终回答和全过程元数据 |

## 8. 一句话总结

`lins_industry` 的核心不是“检索后直接让 LLM 作答”，而是把工业知识库构建、可复现检索、任务感知的执行图、证据组织、结构化推理和引用/安全评估串成一条可观测流水线；当前最重要的运行边界是：**外部检索负责找证据，Agentic LINS 负责组织、推理和生成答案**。

