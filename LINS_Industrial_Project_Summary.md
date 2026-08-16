# 🏭 IndustrialLINS Framework — 完整项目总结

> **论文级框架描述：基于 LINS 多智能体检索增强框架的工业领域扩展，专注工业知识问答评估**

---

## 一、Framework Overview

**IndustrialLINS** 将 LINS (Multi-Agent Retrieval-Augmented Framework) 从医学领域扩展到工业领域，严格对标 **IndustryBench** 论文（[arXiv:2506.19875](https://arxiv.org/abs/2506.19875)）的评估协议。

### Architecture: 四层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    IndustrialLINS Framework                        │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ① Knowledge Layer  (知识层)                              │    │
│  │  ├── KnowledgeSource (抽象知识源: CSV/TXT/PDF/JSONL...)    │    │
│  │  ├── UnifiedDocument 统一文档格式                          │    │
│  │  ├── IndustryKnowledgeBuilder 知识库构建管线              │    │
│  │  └── Knowledge Corpus (chunks + embeddings + FAISS index) │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ② Retrieval Layer  (检索层)                              │    │
│  │  ├── KED (Keyword Extraction & Decomposition)             │    │
│  │  ├── Embedding Search (BGE / text2vec / m3e → FAISS)     │    │
│  │  ├── Multi-Query Fusion (RRF / Union)                    │    │
│  │  ├── Evidence Aggregation                                │    │
│  │  └── Context Construction                                 │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ③ Reasoning Layer  (推理层)                              │    │
│  │  ├── MAIRAG (Multi-Agent IRAG)                           │    │
│  │  ├── LLM Generator (DeepSeek-chat)                       │    │
│  │  └── Answer Generation with Citation                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ④ Evaluation Layer  (评估层)                             │    │
│  │  ├── Three Eval Modes: quick / rag / closed_book          │    │
│  │  ├── IndustryBenchScorer (0-3 + SV)                      │    │
│  │  ├── IndustrialLinkEval (Precision/Recall/F1)            │    │
│  │  ├── RetrievalEvaluator (Recall@k / MRR / NDCG)          │    │
│  │  ├── FormatDetector → QA/Blank/MC Evaluator              │    │
│  │  └── Experiment Pipeline (exp1~exp4)                     │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 关键参数

| 项目 | 说明 |
|------|------|
| **评估基准** | IndustryBench（1060条，10行业 × 7能力 × 3难度） |
| **主 LLM** | DeepSeek-chat |
| **嵌入模型** | BGE-small-zh-v1.5 (384d) / text2vec / m3e |
| **向量索引** | FAISS (flat / ivf / hnsw) |
| **知识库** | industry_kb（自定义工业知识库） |
| **评分协议** | 0-3 分制 + 安全违规 (SV) 惩罚调整 |

---

## 二、四层架构详解

### ① Knowledge Layer（知识层）

**目标：** 统一管理多种异构工业知识源，构建可检索的知识库。

```
knowledge_source/           ← 知识源抽象层
├── base_source.py          KnowledgeSource (ABC): discover() / load() / search() / stats()
├── unified_document.py     UnifiedDocument: {doc_id, title, text, source, industry, capability, metadata}
├── industrybench_source.py IndustryBenchSource: CSV → UnifiedDocument（10行业×7能力×3难度）
├── tech_manual_qa_source.py 技术手册问答知识源
├── pdf_manual_source.py    PDF手册知识源
├── engineering_standards_source.py 工程标准知识源
└── source_registry.py      知识源注册中心

knowledge_builder/          ← 知识库构建管线
└── builder.py              IndustryKnowledgeBuilder:
                            load → deduplicate → chunk_all → build_embeddings → build_index → save_manifest

knowledge_corpus/           ← 知识库运行时（自动生成）
├── manifest.json           构建清单
├── sources/                原始语料 (JSONL)
├── chunks/                 文本块 (JSONL)
├── embeddings/             向量嵌入 (npy)
└── index/                  FAISS 索引 (.faiss)
```

**Pipeline:**
```
load(doc_sources) → deduplicate() → chunk(strategy=paragraph|fixed|recursive)
→ embed(model=bge|text2vec|m3e) → build_index(type=flat|ivf|hnsw) → manifest.json
```

---

### ② Retrieval Layer（检索层）

**目标：** 从大规模知识库中高效检索与问题最相关的文本块。

```
retrieval/
├── chunker.py       TextChunker: paragraph / fixed_size(size+overlap) / recursive(paragraph→sentence)
├── embedder.py      EmbeddingGenerator: BGE(384d) / text2vec(768d) / m3e(768d)
├── faiss_indexer.py FaissIndexer: flat(exact) / ivf(approximate) / hnsw(hierarchical)
├── corpus_builder.py 语料库构建器（从多种源提取 → JSONL）
└── retriever.py     ★核心文件★ OpenDomainRetriever + KED + RAGPipeline
```

**Retrieval Pipeline:**
```
Question
   ↓
KED (Keyword Extraction & Decomposition)
   ↓  TECH_PATTERNS: CNC, PLC, 过载保护, 6RA70, IGBT...
   ↓  extract_keywords() → decompose_queries()
Embedding Search (BGE / text2vec / m3e)
   ↓  ← FAISS Index
Top-k Chunks (可配置 k)
   ↓  (可选)
Multi-Query Fusion (RRF / Union)
   ↓  (可选)
Evidence Aggregation (dedup + rerank)
   ↓
Context Construction → LLM Generator
```

---

### ③ Reasoning Layer（推理层）

**目标：** 基于检索到的证据，生成准确且可引用的工业领域回答。

**核心组件：**
- **LINS-MAIRAG**：复用 LINS-main 的多智能体检索增强生成模型
- **三种评估模式**：
  - `quick` (Oracle 开卷)：直接注入 ground-truth knowledge_text，**上限参考**
  - `rag` (检索增强)：Question → KED → Retriever → Knowledge Corpus → Context → Answer，**目标评估**
  - `closed_book` (闭卷)：无外部知识，仅依赖模型内部知识，**下界参考**

**Answer with Citation**: 生成答案时自动标注引用编号 [1], [2]...，源自检索到的文档片段。

---

### ④ Evaluation Layer（评估层）

**目标：** 严格对标 IndustryBench 论文协议，提供全面的评估体系。

```
metrics/
└── industrybench_scorer.py     ★核心评分模块★
    ├── ScoringRubric           0-3 评分标准定义
    ├── SafetyViolationChecker  安全违规检查（拒绝回答/回避模式检测）
    ├── JudgeLLM                Judge 模型封装 (DeepSeek API)
    ├── IndustryBenchScorer     完整评分 Pipeline
    └── RuleBasedScorer         快速规则评分器（零成本，关键词匹配）

eval_scripts/
├── industry_eval/              ← 论文级评估脚本
│   ├── eval_industrybench.py       主评估脚本 (quick / rag / closed_book)
│   └── eval_industrybench_rag.py   RAG 独立评估
├── rag_eval/                   ← 全新 RAG 评估管线
│   └── eval_rag.py                 端到端 RAG 评测 (retriever → prompt → LLM → scorer)
└── industrial_linkeval/        ← 工业 LinkEval 评估引擎
    ├── format_detector.py      自动检测问题格式 (QA / Fill-in-Blank / Multiple Choice)
    ├── qa_evaluator.py         QA 评估器（语义相似度 / LLM Judge）
    ├── blank_evaluator.py      填空题评估器（精确/归一化/同义词匹配）
    ├── mc_evaluator.py         选择题评估器（准确率）
    ├── retrieval_evaluator.py  检索评估器 (Recall@k / Precision@k / MRR / NDCG)
    └── linkeval_core.py        核心引擎 + 统一 EvalReport

experiments/                    ← 标准化实验管线
├── config.py                   实验全局配置
├── exp1_qa.py                  实验一: QA 性能评估（quick / rag / closed_book 三模式）
├── exp2_retrieval.py           实验二: 检索性能评估（Recall@k / MRR / NDCG）
├── exp3_citation.py            实验三: 引用准确性评估（Precision / Recall / F1）
└── exp4_ablation.py            实验四: 消融实验（baseline→+PRA→+SKA→+QDA→+PCA→full）
```

### 补充：什么是 Retrieval Top-k？

**Top-k** 是检索层的一个核心超参数：**回答一个问题时，从知识库中检索多少个最相关的文本块（chunk）作为 LLM 的上下文。**

```
知识库 (10,000 chunks)
        ↓  检索器计算每个 chunk 与问题的语义相似度
        ↓  按相似度降序排列
        ↓ 
Top-3: [chunk#42(0.92), chunk#101(0.87), chunk#007(0.81)]  →  拼成上下文给 LLM
```

| k 值 | 效果 | 代价 |
|------|------|------|
| **k 太小**（如 k=1） | 可能漏关键证据，回答不准确 | 上下文短，速度快 |
| **k 适中**（如 k=5） | 能找到足够证据，效果最佳 | 上下文合理 |
| **k 太大**（如 k=20） | 噪声增多，LLM 可能混淆 | 上下文长，速度慢，API 费用高 |

**各实验使用的 top-k：**
- `exp1_qa.py`：k=5（QA 性能评估）
- `exp2_retrieval.py`：评估 k=1,3,5,10（看 Recall@k 随 k 增长的曲线）
- `exp3_citation.py`：k=5（引用评估）
- `exp4_ablation.py`：k=5（消融实验）

---

#### 评分协议 (IndustryBench §4.2)

**原始评分 0-3:**
| 分数 | 等级 | 说明 |
|------|------|------|
| **3** | Correct | 与标准答案高度一致，包含所有关键信息 |
| **2** | Acceptable | 方向正确但有关键缺失 |
| **1** | Partial | 部分相关但不一致 |
| **0** | Incorrect | 不相关或错误 |

**安全违规 (SV) 调整:**
- 违反源文本安全约束 → `adjusted_score = 0`
- 否则 `adjusted_score = raw_score`
- 最终: `Final(SV) = Mean(adjusted_scores)`

**RuleBasedScorer（轻量版）：** 参考答案与预测答案的关键词匹配率评分
| 匹配率 | 得分 |
|--------|------|
| ≥ 80% | 3 |
| ≥ 50% | 2 |
| ≥ 20% | 1 |
| < 20% | 0 |

---

## 三、完整目录结构（代码级映射）

```
LINS-Industrial/
│
├── run_industrial_eval.py          # 🏁 主入口：初始化 LINS 工业版 → MAIRAG → 结果
├── test_industrial.py              # ✅ 配置快速验证
├── config/industrial_config.yaml   # ⚙️ 工业场景配置文件
│
├── data/                           # 📦 原始数据
│   ├── industrybench/huggingface_dataset.csv  # 基准 (1060条)
│   ├── industry_kb/industry_kb.txt            # 自定义工业知识库
│   └── factorywave/                           # FactoryWave 数据
│
├── knowledge_source/               # ← Knowledge Layer
├── knowledge_builder/              # ← Knowledge Layer
├── knowledge_corpus/               # ← Knowledge Layer (运行时)
├── retrieval/                      # ← Retrieval Layer
├── metrics/                        # ← Evaluation Layer (评分)
├── eval_scripts/                   # ← Evaluation Layer (评估管线)
├── experiments/                    # ← Evaluation Layer (实验管线)
├── utils/industrial_linkeval.py    # ← Evaluation Layer (引用评估)
│
├── results/                        # 📈 评估结果输出
├── show_baselines.py               # 📊 基线对比
├── show_summary.py                 # 📊 最新结果摘要
│
├── _build_kb.py / _build_industry_kb.py / _gen_industry_embedding.py  # 🔨 旧体系构建工具
└── LINS-main/                      # 📎 LINS 核心模块链接
```

---

## 四、与父项目 LINS-main 的关系

```
LINS/                       # 根项目 (Multi-Agent RAG Framework)
├── README.md               # 医学领域介绍
├── LINS-main/              # 核心框架
│   ├── model/model_LINS.py # ★ LINS 类 (MAIRAG / KED / AEBMP / MOEQA)
│   ├── model/database.py   # 数据库接口 (PubMed / Bing)
│   └── metric/             # 原始 Link-Eval
│
├── LINS-Industrial/        # ← IndustrialLINS Framework (本框架)
│   ├── Knowledge Layer     #   knowledge_source/ + knowledge_builder/
│   ├── Retrieval Layer     #   retrieval/
│   ├── Reasoning Layer     #   (复用 LINS-main model_LINS.py)
│   └── Evaluation Layer    #   metrics/ + eval_scripts/ + experiments/
│
└── metric/                 # 根级别评估指标
```

| 维度 | LINS-main (医学) | IndustrialLINS (工业) |
|------|-----------------|---------------------|
| 知识源 | PubMed / Bing | IndustryBench / TXT / PDF / JSONL |
| 检索 | Bing + PubMed | BGE + FAISS + 开放域检索 |
| 评估 | Link-Eval (医学) | IndustryBenchScorer (0-3+SV) + IndustrialLinkEval |
| 实验 | 医学 QA 评测 | 四种标准化实验 (QA/检索/引用/消融) |
| 架构 | 单体 | 四层分层架构 (Knowledge/Retrieval/Reasoning/Evaluation) |

**复用：** LINS-main 的 `model_LINS.py`（LLM 调用、MAIRAG 多智能体推理）

**新增：**
- 工业专用的知识源抽象层（knowledge_source/）
- 开放域检索管线（retrieval/：BGE + FAISS + KED + RAGPipeline）
- 知识库构建器（knowledge_builder/：一键构建管线）
- 完整评估体系（metrics/ + eval_scripts/ + experiments/）
- 四种标准化实验（QA / 检索 / 引用 / 消融）
- 新知识源（IndustryBench, PDF手册, 技术手册QA, 工程标准）

---

## 五、代码快速使用

```bash
# 1. 设置 API Key
export DEEPSEEK_API_KEY='sk-your-key'

# 2. 测试配置
cd LINS-Industrial
python test_industrial.py

# 3. 构建工业知识库（一键全流程）
python -c "from knowledge_builder.builder import build_knowledge_base; build_knowledge_base()"

# 4. 运行快速评估（10个样本，规则评分）
python eval_scripts/industry_eval/eval_industrybench.py --mode quick --num 10

# 5. RAG 评估（手动检索 industry_kb + DeepSeek）
python eval_scripts/industry_eval/eval_industrybench_rag.py

# 6. 端到端 RAG 评测（新管线）
python eval_scripts/rag_eval/eval_rag.py --llm deepseek --k 5

# 7. 实验管线
python experiments/exp1_qa.py --mode quick --num_samples 10      # QA 性能评估
python experiments/exp2_retrieval.py --num_samples 50 --topk 10  # 检索性能评估
python experiments/exp3_citation.py --num_samples 30              # 引用准确性评估
python experiments/exp4_ablation.py --num_samples 20              # 消融实验

# 8. 查看结果
python show_summary.py
python show_baselines.py
```

---

## 六、主要依赖

- **LLM**: deepseek-chat API（主模型）、OpenAI API、Ollama（本地部署）
- **检索**: sentence-transformers (BGE)、FAISS、numpy
- **嵌入模型**: BAAI/bge-small-zh-v1.5（默认384维）
- **LangChain**: langchain_deepseek / langchain_openai / langchain_ollama
