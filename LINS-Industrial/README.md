# 🏭 LINS-Industrial: 工业场景知识问答评估框架

> **基于 LINS (Multi-Agent Retrieval-Augmented Framework) 的工业垂直领域扩展**

LINS-Industrial 将 [LINS 多智能体检索增强框架](https://github.com/zx12671/research_agent) 从医学领域扩展到**工业领域**，专注于评估 LLM 在工业知识问答场景下的表现，并严格对标 **IndustryBench** 论文（[arXiv:2506.19875](https://arxiv.org/abs/2506.19875)）的评估协议。

---

## 📋 项目概况

| 项目 | 说明 |
|------|------|
| **框架** | LINS (MAIRAG + KED + Link-Eval) |
| **评估基准** | IndustryBench（1060条，10行业 × 7能力 × 3难度） |
| **主 LLM** | DeepSeek-chat |
| **检索器** | BGE / text-embedding-3-large |
| **知识库** | industry_kb（自定义工业知识库） |
| **评分协议** | 0-3 分制 + 安全违规 (SV) 惩罚调整 |
| **实验管线** | QA / 检索 / 引用 / 消融 四组标准化实验 |

---

## 📁 目录结构

```
LINS-Industrial/
│
├── run_industrial_eval.py          # 🏁 工业评估主入口脚本
├── test_industrial.py              # ✅ 配置快速验证脚本
│
├── config/
│   └── industrial_config.yaml      # ⚙️ 工业场景配置文件
│
├── data/                           # 📦 原始数据文件
│   ├── industrybench/              # 📊 IndustryBench 标准数据集 (1060条)
│   │   └── huggingface_dataset.csv
│   ├── industry_kb/                # 📚 工业知识库文本
│   │   └── industry_kb.txt
│   └── factorywave/                # 🔧 FactoryWave 工业数据
│
├── knowledge_source/               # 🆕 知识源抽象层
│   ├── __init__.py
│   ├── unified_document.py         #    UnifiedDocument: 统一文档格式
│   ├── base_source.py              #    KnowledgeSource (ABC) 抽象基类
│   ├── industrybench_source.py     #    IndustryBench 知识源 (CSV → UnifiedDocument)
│   ├── tech_manual_qa_source.py    #    技术手册问答知识源
│   ├── pdf_manual_source.py        #    PDF手册知识源
│   ├── engineering_standards_source.py # 工程标准知识源
│   └── source_registry.py          #    知识源注册中心
│
├── knowledge_builder/              # 🆕 知识库构建管线
│   ├── __init__.py
│   └── builder.py                  #    IndustryKnowledgeBuilder: 一键构建管线
│                                   #    load → deduplicate → chunk_all → 
│                                   #    build_embeddings → build_index → save_manifest
│
├── retrieval/                      # 🆕 开放域检索模块
│   ├── __init__.py
│   ├── corpus_builder.py           #    语料库构建 (IndustryBench/TXT/JSONL → JSONL)
│   ├── chunker.py                  #    文档切分 (paragraph/fixed_size/recursive)
│   ├── embedder.py                 #    向量嵌入 (BGE/text2vec/m3e)
│   ├── faiss_indexer.py            #    FAISS 索引 (flat/ivf/hnsw)
│   └── retriever.py                #    OpenDomainRetriever + KED + RAGPipeline
│                                   #    Question → KED → Embedding → FAISS → Top-k
│
├── knowledge_corpus/               # 🗄️ 知识库运行时目录 (自动生成)
│   ├── __init__.py
│   ├── manifest.json               #    构建清单
│   ├── sources/                    #    原始语料 (JSONL)
│   ├── chunks/                     #    文本块 (JSONL)
│   ├── embeddings/                 #    向量嵌入 (npy + meta)
│   └── index/                      #    FAISS 索引 (.faiss)
│
├── metrics/
│   └── industrybench_scorer.py     # 📊 核心评分模块（对标论文协议）
│       ├── ScoringRubric           #    0-3 评分标准定义
│       ├── SafetyViolationChecker  #    安全违规检查
│       ├── JudgeLLM                #    Judge 模型封装 (DeepSeek API)
│       ├── IndustryBenchScorer     #    完整评分 Pipeline
│       └── RuleBasedScorer         #    快速规则评分器（零成本）
│
├── utils/
│   └── industrial_linkeval.py      # 🔗 Link-Eval 工业适配版
│       └── IndustrialLinkEval      #    引用评估器 (precision/recall/F1)
│
├── experiments/                    # 🆕 标准化实验管线
│   ├── __init__.py                 #    统一导出
│   ├── config.py                   #    实验全局配置 (路径/模型/API Key)
│   ├── exp1_qa.py                  #    QA 性能评估 (quick/rag/closed_book 三模式)
│   ├── exp2_retrieval.py           #    检索性能评估 (Recall@k/MRR/NDCG)
│   ├── exp3_citation.py            #    引用准确性评估 (Precision/Recall/F1)
│   └── exp4_ablation.py            #    消融实验 (baseline→+PRA→+SKA→+QDA→+PCA→full)
│
├── eval_scripts/
│   ├── industry_eval/              # 📝 论文级评估脚本
│   │   ├── eval_industrybench.py       # 论文级评估主脚本 (quick/rag/closed_book)
│   │   └── eval_industrybench_rag.py   # RAG 模式独立评估
│   │
│   ├── rag_eval/                   # 🆕 全新 RAG 评估管线
│   │   ├── __init__.py
│   │   └── eval_rag.py             #    端到端 RAG 评测 (retriever → prompt → LLM → scorer)
│   │
│   └── industrial_linkeval/        # 🆕 工业 LinkEval 详细评估引擎
│       ├── __init__.py
│       ├── format_detector.py      #    自动检测问题格式 (QA/Fill-in-Blank/MC)
│       ├── qa_evaluator.py         #    QA 评估器 (语义相似度/LLM Judge)
│       ├── blank_evaluator.py      #    填空题评估器 (精确/归一化/同义词匹配)
│       ├── mc_evaluator.py         #    选择题评估器 (准确率)
│       ├── retrieval_evaluator.py  #    检索评估器 (Recall@k/Precision@k/MRR/NDCG)
│       └── linkeval_core.py        #    核心引擎 + 统一 EvalReport
│
├── results/                        # 📈 评估结果
│   ├── industrybench_eval_quick_*.json          # 开卷模式
│   ├── industrybench_eval_closed_book_*.json    # 闭卷模式
│   ├── industrybench_eval_rag_*.json            # RAG 模式
│   └── experiments/                              # 实验管线输出 (JSON + Markdown)
│
├── show_baselines.py               # 📊 展示所有基线结果对比
├── show_summary.py                 # 📊 显示最新评估结果摘要
│
├── _build_kb.py                    # 🔨 知识库构建第一阶段
├── _build_industry_kb.py           # 🔨 知识库构建第二阶段 (DeepSeek 生成)
├── _gen_industry_embedding.py      # 🔨 生成嵌入向量
├── _check_embedding.py             # 🔨 检查嵌入文件
│
├── _debug_view.py                  # 🔍 评估结果快速查看
├── _debug_view2.py                 # 🔍 数据字段结构查看
├── _debug_view3.py                 # 🔍 查找所有键名
│
└── LINS-main/                      # 📎 LINS 核心模块链接
    └── model/
        └── retriever/
```

---

## ⚠️ 架构澄清：Industrial LINS 的正确两阶段设计

> **注意：以下是在代码实现基础上的**理想架构 **说明。当前代码部分区域（如 `eval_rag.py`）仍处于过渡状态，将检索和生成混在了一起（直接用 LLM 而非 LINS），需按此架构重构。**

### 正确架构

```
┌─────────────────────────────────────────────────────────────────┐
│                第一阶段: IndustrialRetriever (独立检索)           │
│                                                                  │
│  Question → KED(关键词提取) → Embedding → FAISS Search          │
│  → Multi-Query Fusion → Evidence Aggregation                    │
│                                                                  │
│  输出: Evidence (检索到的 top-k 知识块 + 元数据)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        Evidence
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                第二阶段: LINS (MAIRAG, retrieval=False)           │
│                                                                  │
│  输入: Question + Evidence (作为 context 传入)                    │
│  行为: 只做 Multi-Agent Reasoning + Citation + Answer Generation │
│  不调用自己的检索器，完全依赖 External Evidence                   │
│                                                                  │
│  输出: Answer with Citations                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                评估管线 (metrics/ + eval_scripts/)               │
│  IndustryBenchScorer (0-3 评分 + SV 调整)                       │
│  IndustrialLinkEval (FormatDetector → EvalReport)               │
└─────────────────────────────────────────────────────────────────┘
```

### 关键区别

| 组件 | ❌ 错误用法 | ✅ 正确用法 |
|------|-----------|-----------|
| **IndustrialRetriever** | 检索后直接丢给普通 LLM → 生成答案 | 检索后输出 **Evidence**，传给 LINS |
| **LINS (MAIRAG)** | MAIRAG(retrieval=True) — 用自己的检索器再检索一次 | MAIRAG(retrieval=False, context=evidence) — 只用传入的 evidence |
| **Pipeline** | 检索 → LLM(DeepSeek) → 答案 | 检索 → LINS(推理+引用+生成) → 带引用的答案 |

### 为什么这样设计？

1. **分离关注点**：IndustrialRetriever 负责"找到相关知识"，LINS 负责"推理并引用知识"
2. **避免重复检索**：两个检索器各自检索会造成证据不一致
3. **完整的 LINS 能力**：MAIRAG 的多智能体推理、Citation 生成、Link-Eval 评估依然可用
4. **模块化**：可以独立替换检索器或 LINS，互不影响


---

## 🧩 核心组件详解

### 1️⃣ 知识源抽象层 (`knowledge_source/`)
统一管理多种异构工业知识源的接入：
- **`base_source.py`**: `KnowledgeSource` 抽象基类，定义 `discover()` / `load()` / `load_all()` / `search()` / `stats()` 接口
- **`unified_document.py`**: `UnifiedDocument` 统一文档格式（`document_id`, `title`, `text`, `source`, `industry`, `capability`, `metadata`）
- **`industrybench_source.py`**: 从 IndustryBench CSV 适配为 UnifiedDocument
- **`source_registry.py`**: 注册中心，管理所有可用的知识源

### 2️⃣ 知识库构建管线 (`knowledge_builder/`)
`IndustryKnowledgeBuilder` 一键自动化管线：
1. `load()` — 从多个 KnowledgeSource 加载文档
2. `deduplicate()` — 去重
3. `chunk_all()` — 切分文本块
4. `build_embeddings()` — 生成向量嵌入
5. `build_index()` — 构建 FAISS 索引
6. `save_manifest()` — 保存构建清单

### 3️⃣ 开放域检索模块 (`retrieval/`)
- **`chunker.py`**: 支持 `paragraph`（段落）、`fixed_size`（固定长度+overlap）、`recursive`（递归优先段落→句子）三种切分策略
- **`embedder.py`**: 支持 BGE-small(384d)、BGE-base(768d)、text2vec(768d)、m3e(768d) 多种嵌入模型
- **`faiss_indexer.py`**: 支持 flat(精确)、ivf(近似)、hnsw(近似) 三种索引类型
- **`retriever.py`**: `OpenDomainRetriever` 完整管线（KED关键词提取 → 嵌入 → 检索 → 多查询融合 → 证据聚合）+ `RAGPipeline` 端到端生成

### 4️⃣ 标准化实验管线 (`experiments/`)
四组标准化实验，继承统一配置，输出 JSON + Markdown 报告：

| 实验 | 文件 | 评估内容 | 核心指标 |
|------|------|---------|---------|
| **实验一** | `exp1_qa.py` | QA 性能评估 | 原始分(0-3) + SV 调整 + 分层统计 |
| **实验二** | `exp2_retrieval.py` | 检索性能评估 | Recall@k / Precision@k / MRR / NDCG |
| **实验三** | `exp3_citation.py` | 引用准确性评估 | Citation Precision / Recall / F1 |
| **实验四** | `exp4_ablation.py` | 消融实验 | 平均分 + SV 率 + 耗时 + 分布 |

### 5️⃣ 评分协议 (`metrics/industrybench_scorer.py`)
严格对标 IndustryBench 论文 §4.2 节的评估协议：

**原始评分 (0-3):**
- **Score 3 (Correct):** 与标准答案高度一致，包含所有关键信息
- **Score 2 (Acceptable):** 方向正确但有关键缺失
- **Score 1 (Partial):** 部分相关但不一致
- **Score 0 (Incorrect):** 不相关或错误

**安全违规 (SV) 调整:**
- 检查是否违反源文本中的安全约束
- 违反则 `adjusted_score = 0`
- 否则 `adjusted_score = raw_score`
- 最终得分: `Final(SV) = Mean(adjusted_scores)`

### 6️⃣ 工业 LinkEval 评估引擎 (`eval_scripts/industrial_linkeval/`)
自动检测问题格式并路由到对应评估器：
- `format_detector.py`: 自动识别 QA / Fill-in-Blank / Multiple Choice
- `qa_evaluator.py`: 语义相似度 + LLM Judge + 证据一致性
- `blank_evaluator.py`: 精确匹配 / 归一化匹配 / 同义词匹配
- `mc_evaluator.py`: 选择题准确率
- `retrieval_evaluator.py`: Recall@k / Precision@k / MRR / NDCG
- `linkeval_core.py`: 核心引擎，生成统一的 `EvalReport`（JSON + Markdown + Console）

---

## 📊 评估结果汇总

| 模式 | Raw Mean | Adj Mean | SV Rate |
|------|----------|----------|---------|
| quick (开卷) | — | — | — |
| rag (检索) | — | — | — |
| closed_book (闭卷) | — | — | — |

> 结果文件位于 `results/` 目录，使用 `show_baselines.py` 或 `show_summary.py` 查看。

---

## 🔧 快速使用

```bash
# 1. 设置 API Key（或在 experiments/config.py 中已内置 fallback）
export DEEPSEEK_API_KEY='sk-your-key'

# 2. 测试配置
cd LINS-Industrial
python test_industrial.py

# 3. 运行论文级评估（10 个样本，规则评分）
python eval_scripts/industry_eval/eval_industrybench.py --mode quick --num 10
python eval_scripts/industry_eval/eval_industrybench.py --mode rag --num 10
python eval_scripts/industry_eval/eval_industrybench.py --mode closed_book --num 10

# 4. RAG 独立评估（手动检索 industry_kb + DeepSeek）
python eval_scripts/industry_eval/eval_industrybench_rag.py

# 5. 端到端 RAG 评测（新管线）
python eval_scripts/rag_eval/eval_rag.py --llm deepseek --k 5

# 6. 实验管线
python experiments/exp1_qa.py --mode quick --num_samples 10      # QA 性能评估
python experiments/exp2_retrieval.py --num_samples 50 --topk 10  # 检索性能评估
python experiments/exp3_citation.py --num_samples 30              # 引用准确性评估
python experiments/exp4_ablation.py --num_samples 20              # 消融实验

# 7. 构建工业知识库（一键全流程）
python -c "from knowledge_builder.builder import build_knowledge_base; build_knowledge_base()"

# 8. 查看结果
python show_summary.py
python show_baselines.py
```

---

## 🏗️ 知识源架构：新旧两套体系

项目有两套并行的知识处理体系：

### 旧体系（原始 LINS 风格）
- `_build_kb.py` → `_build_industry_kb.py` → `_gen_industry_embedding.py`
- 直接在 `data/industry_kb/` 操作，通过 `industry_kb_embedding.json` 文件检索
- 用于 `eval_industrybench_rag.py`

### 新体系（2024 重构）
- **`knowledge_source/`**: 统一的 KnowledgeSource 抽象 + UnifiedDocument 输出
- **`knowledge_builder/`**: IndustryKnowledgeBuilder 一键管线
- **`retrieval/`**: 开放域检索（KED + Embedding + FAISS + RAGPipeline）
- 输出到 **`knowledge_corpus/`** 下统一管理
- 用于 `eval_rag.py` 和所有实验脚本

---

## 🔗 与 LINS-main 的关系

```
LINS/                       # 根项目
├── README.md               # LINS 主项目介绍（医学领域，多智能体RAG）
├── LINS-main/              # LINS 核心框架代码
│   ├── model/model_LINS.py # 核心 LINS 模型 (MAIRAG / KED / AEBMP / MOEQA)
│   ├── model/database.py   # 数据库接口 (PubMed / Bing / 本地知识库)
│   └── metric/             # 原始 Link-Eval 评估指标
│
├── LINS-Industrial/        # ← 本项目（工业扩展）
│   └── ...                 # 上述所有文件
│
└── metric/                 # 根级别评估指标
```

- LINS-Industrial **复用** LINS-main 的 `model_LINS.py` 中的 `LINS` 类（用于 LLM 调用）
- LINS-Industrial **新增** 了工业专用的知识源抽象层、检索管线、知识库构建器
- LINS-Industrial **新增** 了完整的评估体系（IndustryBenchScorer + IndustrialLinkEval）
- LINS-Industrial **新增** 了标准化实验管线（exp1~exp4）

---

## 📄 许可

本项目基于 [Apache-2.0 License](../LICENSE)。
