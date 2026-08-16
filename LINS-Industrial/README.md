# LINS-Industrial：工业领域多智能体检索增强框架

> 基于 **LINS (Multi-Agent Retrieval-Augmented Framework)** 核心框架的工业垂直领域扩展，用于评估和提升 LLM 在**工业知识问答**场景下的表现，严格对标 **IndustryBench**（arXiv:2506.19875）评估协议。

本项目将 LINS 从医学领域拓展到工业领域（机械制造、工业安全、工艺规范、故障诊断等），构建了多智能体自适应管线（Adaptive Agentic Pipeline），并配套完整的分阶段调试（S1~S6）、消融（A/B）与端到端（E2E）评测脚本。

---

## 项目特性

| 特性 | 说明 |
|------|------|
| 自适应多智能体管线 | TaskAnalyzer -> StrategyPlanner -> GraphExecutor 结构化图执行 |
| 多策略检索 | BGE-M3 稠密检索 + BM25（jieba）+ 关键词查询重写 |
| 多源知识库 | IndustryBench 数据集 + knowledge_corpus（chunk / 嵌入 / FAISS 索引）|
| 完整评测体系 | IndustryBenchScorer、IndustrialLinkEval、检索 Recall 指标 |
| 分阶段可量化调试 | _diag系列 / _ab消融 / _probe探针 / _e2e端到端脚本 |
| DeepSeek 主模型 | 基于 deepseek-chat，兼容 OpenAI 等 LLM |
| 工业场景 | IndustryBench 工业知识问答（标准规范、工艺、故障诊断等）|

---

## 项目概况

| 项目 | 说明 |
|------|------|
| 框架 | LINS (MAIRAG + KED + Link-Eval) |
| 评估基准 | IndustryBench（10 行业 x 7 能力 x 3 难度）|
| 主 LLM | DeepSeek-chat |
| 检索器 | BGE（BGE-M3）/ text-embedding-3-large |
| 知识库 | industry_kb + knowledge_corpus（BGE-M3 向量 + FAISS）|
| 评分协议 | 0~3 分制 + 安全违规 (SV) 惩罚调整 |
| 实验管线 | QA / 检索 / 引用 / 消融 四组标准化实验（exp1~exp4）|

---

## 快速开始

### 1. 环境准备

```bash
conda create -n LINS python=3.11 -y
conda activate LINS
pip install -r requirements.txt
```

依赖说明：本仓库依赖同级的 LINS-main 核心框架（run_industrial_eval.py 会自动加载 ../LINS-main）。请确保仓库结构完整。

### 2. 设置 DeepSeek API Key

本项目通过环境变量读取 API Key（代码不硬编码密钥）：

```bash
# PowerShell
$env:DEEPSEEK_API_KEY="sk-你的key"
# Linux / macOS
export DEEPSEEK_API_KEY=sk-你的key
```

安全提示：请勿将 API Key 硬编码进文件，代码一律从 os.environ.get('DEEPSEEK_API_KEY') 读取。

### 3. 准备 BGE-M3 嵌入模型（使用 BGE 检索器时）

BGE 检索器需要 BGE-M3 模型。若本地不存在 model/retriever/bge/bge-m3/，运行时将从 HuggingFace（BAAI/bge-m3）自动下载。因模型文件较大（约 2GB），未纳入 Git 跟踪。

---


---

## 目录结构

```
LINS-Industrial/
├── run_industrial_eval.py          # 工业评估主入口
├── test_industrial.py              # 配置快速验证
├── test_agentic_rag.py             # 智能体 RAG 管线测试
├── show_baselines.py / show_summary.py   # 结果展示
├── agentic/                        # 自适应多智能体管线
│   ├── analyzer.py                 #   任务分析与格式归一化
│   ├── planner.py                  #   策略规划（StrategyPlanner）
│   ├── organizer.py                #   证据定向组织（RerankTrunc）
│   ├── solver.py                   #   任务求解
│   ├── pipeline.py                 #   图执行引擎（GraphExecutor）
│   ├── prompt_builder.py / prompts.py
│   └── task_types.py               #   任务类型 / 执行图与指令
├── retrieval/                      # 开放域检索模块
│   ├── corpus_builder.py           #   语料库构建
│   ├── chunker.py                  #   文档切分
│   ├── embedder.py                 #   向量嵌入
│   ├── faiss_indexer.py            #   FAISS 索引
│   ├── retriever.py                #   OpenDomainRetriever
│   └── recall_metrics.py           #   检索指标
├── knowledge_source/               # 知识源抽象层
├── knowledge_builder/              # 知识库构建管线
├── knowledge_corpus/               # 运行时语料/嵌入/索引（自动生成）
├── metrics/                        # industrybench_scorer 等
├── utils/                          # industrial_linkeval 工具
├── eval_scripts/                   # 评估脚本
├── experiments/                    # 实验管线（config / exp1~exp4）
├── config/                         # 配置（industrial_config.yaml）
├── data/                           # IndustryBench 数据 + industry_kb
├── docs/                           # 设计文档 / 分析报告
├── results/                        # 实验结果输出
└── requirements.txt                # 依赖清单
```

顶级实验/调试脚本（_ 开头）：
- _diag系列：系统性阶段诊断（S2/S3/S4/S5/S6）
- _ab系列：A/B 消融实验（BM25、查询重写、证据组织等）
- _probe系列：单项探针验证（召回、格式化、证据损失等）
- _e2e系列：端到端评测

---

## 自适应多智能体管线（Agentic Pipeline）

```
用户问题
   │
   ▼
TaskAnalyzer   → 任务分类 + 格式归一化 (infer_format_heuristic)
   │
   ▼
StrategyPlanner → 规划执行图 (ExecutionGraph / 图节点)
   │
   ▼
GraphExecutor  → 按图调度
   │  ├─ 检索（BGE 稠密 + BM25 + 查询重写）
   │  ├─ EvidenceOrganizer（证据定向 / RerankTrunc）
   │  └─ TaskSolver（推理求解）
   │
   ▼
最终回答（带引用）
```

---

## 评估评分协议

- 分值：0~3（0=不正确 / 1=部分相关 / 2=可接受 / 3=完全正确）
- 安全违规 (SV) 调整：违反源文本安全约束则 adjusted=0，否则 adjusted=raw
- 最终得分：Final(SV) = Mean(adjusted_scores)
- 检索指标：Hit@k、Precision@k、MRR、NDCG（retrieval/recall_metrics.py）
- 引用评估：IndustrialLinkEval（precision / recall / F1）

评分器：metrics/industrybench_scorer.py（完整评分 Pipeline + 零成本 RuleBasedScorer）
引用评估：eval_scripts/industrial_linkeval/（QA / Fill-in-Blank / MCQ 自动检测路由）

---

## 知识源架构：两套并行体系

- 旧体系（原始 LINS 风格）：_build_kb.py → _build_industry_kb.py → 直接在 data/industry_kb/ 操作，用于 eval_industrybench_rag.py
- 新体系（重构）：knowledge_source/（知识源抽象）→ knowledge_builder/（一键管线）→ retrieval/（开放域检索）→ 输出到 knowledge_corpus/，用于 eval_rag.py 及所有实验脚本

---

## 与 LINS-main 的关系

本项目复用 LINS-main 的 model_LINS.LINS 类（LLM 调用）、retriever_model（BGE 检索）。

- 新增：工业知识源抽象、开放域检索管线、知识库构建器、完整评估体系（IndustryBenchScorer + IndustrialLinkEval）、标准化实验管线（exp1~exp4）、自适应智能体管线

---

## 依赖

完整依赖见 [requirements.txt](./requirements.txt)，核心包括：openai、httpx、numpy、torch、transformers、FlagEmbedding、jieba、pandas、tqdm，以及 LINS-main 框架所需依赖（biopython、datasets、feedparser、playwright、modelscope、scikit-learn 等）。

---

## 重要说明

- 大模型文件：BGE-M3 等模型二进制文件（>100MB）未纳入 Git 跟踪，需本地提供或自动下载
- API Key 安全：所有密钥通过环境变量 DEEPSEEK_API_KEY 注入，切勿硬编码
- 依赖 LINS-main：运行前请确保仓库包含 ../LINS-main 完整框架

---

## 许可

本项目基于 [Apache-2.0 License](../LICENSE)。

## 运行入口脚本

```bash
# 1. 工业场景快速评估入口
python run_industrial_eval.py

# 2. 配置快速验证
python test_industrial.py

# 3. 独立评估（规则评分）
python eval_scripts/industry_eval/eval_industrybench.py --mode quick --num 10
python eval_scripts/industry_eval/eval_industrybench.py --mode rag --num 10
python eval_scripts/industry_eval/eval_industrybench.py --mode closed_book --num 10

# 4. RAG 独立评估
python eval_scripts/industry_eval/eval_industrybench_rag.py

# 5. 端到端 RAG 评测（新检索管线）
python eval_scripts/rag_eval/eval_rag.py --llm deepseek --k 5

# 6. 标准化实验管线
python experiments/exp1_qa.py --mode quick --num_samples 10
python experiments/exp2_retrieval.py --num_samples 50 --topk 10
python experiments/exp3_citation.py --num_samples 30
python experiments/exp4_ablation.py --num_samples 20

# 7. 构建工业知识库
python -c "from knowledge_builder.builder import build_knowledge_base; build_knowledge_base()"

# 8. 查看结果
python show_summary.py
python show_baselines.py
```