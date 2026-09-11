# LINS 项目复现工作整理报告

> 基于 zx12671/research_agent master 分支的 LINS 原版复现工作
> 报告范围：LINS-main 及其相关复现、DeepSeek 适配与医学评测工作
> 说明：LINS-Industrial、IndustryBench、IHER 等工业方向内容不纳入本报告

---

## 一、项目概况

本报告对当前仓库 `zx12671/research_agent` master 分支中的 LINS 原版复现工作进行整理，并对附件《LINS 项目复现工作阶段性报告》中的结论进行**实证核对与数据补充**。

原版 LINS 来自 [WangSheng21s/LINS](https://github.com/WangSheng21s/LINS)，论文为 *LINS: A Multi-Agent Retrieval-Augmented Framework for Enhancing the Quality and Credibility of LLMs' Medical Responses*（Nature Communications 2025）。其核心是 **MAIRAG**（多智能体迭代检索增强生成）、**KED**（关键词提取退化检索）与 **Link-Eval**（基于引用的自动化评估）。

当前工作的本质是：**在保留 LINS 算法本体的前提下，把运行后端从 OpenAI 生态迁移到 DeepSeek + 本地 BGE-M3，并重建一套自包含的医学评测驱动**。工作阶段已由早期的"代码恢复与运行问题排查"进入"可运行基线形成—正式实验复现—结果核验"阶段。

---

## 二、范围与边界

| 模块/目录 | 定位 | 本报告处理 |
|---|---|---|
| `LINS-main/` | LINS 原版复现与 DeepSeek 适配核心目录 | 纳入 |
| `LINS-main/model/` | LINS 核心模型、Retriever、Database | 纳入 |
| `LINS-main/{evaluate,metric,add_dataset}/` | 评测与数据集相关模块 | 纳入 |
| `test_deepseek_lins_full.py`、`LINS-main/test_deepseek.py`、`lins_test_common.py` | 完整测试与医学评测脚本 | 纳入 |
| `eval_results_lins_full/` | LINS（含 RAG）评测输出 | 纳入 |
| `eval_results_llm_baseline/` | 纯 LLM 零样本对照组输出 | 纳入 |
| `LINS/` | 原版 LINS 参考仓库（含 `add_dataset/oncokb`） | 仅作对照基线 |
| `LINS-Industrial/` | 工业方向二次研究项目 | **排除** |
| `LINS_industry_img/` 等 | 工业方向配套资料 | **排除** |
| IHER / IndustryBench 相关实验 | 工业检索研究内容 | **排除** |

---

## 三、总体进展

代码层面基本恢复，运行层面基本打通，核心算法模块已具备可运行实现，自动化评测框架已经建立；但严格意义上的论文级实验复现仍需完成正式规模运行、参数冻结、重复实验与结果核验。

| 工作阶段 | 状态 | 说明 |
|---|---|---|
| 原版 LINS 代码恢复 | ✅ 已完成 | 核心工程结构与主要模块已恢复 |
| 运行环境与依赖修复 | ✅ 基本完成 | 已解决主要依赖与兼容性问题，并上 CI |
| DeepSeek API 适配 | ✅ 已完成 | 支持 DeepSeek 作为主要运行模型 |
| 核心模块恢复 | ✅ 已完成 | MAIRAG、KED、HERD 等主要模块可运行 |
| 测试体系建设 | ✅ 基本完成 | 已建立快速测试与完整评测两类脚本 |
| 医学数据集评测 | 🟡 进行中 | 流程已建立，但存在链路缺陷，见第七章 |
| 论文级复现核验 | 🟡 进行中 | 需与原论文设置和结果逐项对齐 |
| Baseline 冻结 | 🔴 待完成 | 尚未固定配置并做重复性验证 |

---

## 四、已完成的主要工作

### 4.1 LINS 原版工程恢复

`LINS-main/` 已形成较完整的工程结构，已超出"简单下载原始代码"的阶段，属于面向当前运行环境的工程恢复。

```
LINS-main/
├── add_dataset/                 # 自定义数据集添加（add_database.py / pdf2embedding.py）
├── evaluate/                    # 评测脚本与数据集
│   ├── eval_medqa_star_ALL.py
│   ├── arguments.py
│   └── evaluate_data/
│       ├── pubmedqa/data/       # ori_pqal.json (1000) + test_ground_truth.json (500)
│       ├── medqa_us/data/       # medqa_us_test.json
│       ├── medqa_mainland/data/ # medqa_mainland_test.json
│       └── medqa_taiwan/data/   # medqa_taiwan_test.json
├── metric/                      # evaluator.py / scorer.py
├── model/
│   ├── extracting/  fetching/  filtering/  retriever/  searching/
│   ├── model_LINS.py            # 核心 LINS 类
│   ├── database.py              # LINS_Database（Pubmed/Bing/HERD 等）
│   ├── retriever_model.py       # LINS_Retriever（BGE / text-embedding-3-*）
│   ├── chat_llms.py             # LLM 调用封装
│   └── prompts.py  utils.py
├── Link_Eval.py                 # 原版 LinkEval
├── Link_Eval_DeepSeek.py        # DeepSeek 版 LinkEval（新增）
├── lins_test_common.py          # 公共测试工具（新增）
└── test_deepseek.py / test_lins.py
```

### 4.2 环境与依赖问题修复

复现过程中针对原项目运行环境与当前开发环境的差异进行了多项修复：

- **缺失源码恢复**：补全 `model/` 下的 `extracting/`、`fetching/`、`filtering/`、`searching/` 等子模块，修正 `add_dataset/pdf2embedding.py` 的缺陷。
- **类名修正**：`database.py` 中类名为 `Pubmed` 而非 `DataBase`；`retriever_model.py` 中为 `Text_embedding_3_large` 而非 `Retriever`（对应 commit `d343a1c`、`167e232`）。
- **依赖兼容性固定**：`pyarrow==14.0.1` 与 `datasets==2.12.0` 的兼容问题（commit `36f8a0c`）。
- **CI 建设**：新增 `.github/workflows/test_imports.yml`，校验核心 import（`LINS`、`Text_embedding_3_large`、`Pubmed`、`Evaluator`、`Scorer`、`utils`），降低后续实验因环境问题失败的风险。
- **移除外部重依赖**：原 `Link_Eval.py` 依赖 `transformers` 的 `T5-11B`（NLI）与 `UniEval`（流畅度），并需外网 `med_linker_search`；CI 因此失败（commit `c23151d`）。

### 4.3 DeepSeek API 适配

已在运行环境与模型接口层面完成适配，原则上**不改变 LINS 的核心算法流程**。经逐文件比对（行数差）：

| 文件 | 原版行数 | LINS-main 行数 | 增量 | 判断 |
|---|---|---|---|---|
| `model/model_LINS.py` | 683 | 769 | +86 | 有改动（key 路由） |
| `model/chat_llms.py` | 113 | 258 | +145 | 有改动（新增后端） |
| `model/retriever_model.py` | 200 | 207 | +7 | 微调 |
| `model/database.py` | 424 | 435 | +11 | 微调 |
| `model/prompts.py` | 356 | 356 | **0** | **完全未动** |
| `Link_Eval.py` | 147 | 147 | **0** | **完全未动** |

**关键结论：Prompt 模板与 LinkEval 原文件零改动，证明"算法是原版的，只换了 LLM 后端与评测驱动"。**

具体改动点：

1. **`chat_llms.py` 新增 `DeepSeek` 类**，并在 `chatllms.__init__` 增加分支：

   ```python
   if 'deepseek' in self.model_name:
       self.model = DeepSeek(llm_keys=llm_keys, model_name=self.model_name)
   elif 'qwen' in self.model_name and ('api'|'turbo'|'plus'|'max'):
       self.model = QianWen(llm_key=llm_keys, model_name=self.model_name)
   ```

2. **`model_LINS.py` 新增 `get_llm_key()` 路由函数**，按模型名自动选择 key（`DeepSeek_keys` / `Gemini_keys` / `QianWen_keys` / `LLM_keys`）。
3. **`LINS.__init__` 新增参数** `QianWen_keys`、`DeepSeek_keys`，并将 `retriever` / `database` 初始化前置，使 key 异常时检索器仍可先行建立。

> ⚠️ 需注意：当前 `test_deepseek.py`、`test_deepseek_lins_full.py`、`Link_Eval_DeepSeek.py` 中仍存在**明文硬编码的 API Key**。git 历史显示曾专门做过"移除硬编码、改用环境变量"的提交（`1ec5eb1`），但随后被 revert（`6f7c08c`）。这属于必须整改的安全项，详见 7.4。


### 4.4 LINS 核心流程恢复

`model_LINS.py` 已恢复核心入口与主要工作流程，保留多智能体检索、证据筛选、答案生成与一致性检查等关键机制。

**MAIRAG 核心流程：**

```
问题输入
  → Passage Retrieval（多源检索）
  → Passage Relevance Analysis（PRA / PRM 段落相关性评估）
  → Gold Evidence 筛选
  → 答案生成
  → Passage Coherence Analysis（PCA）
  → 冲突检测
  → 必要时重新生成答案
```

已恢复的主要方法（`LINS-main/model/model_LINS.py`）：

| 方法 | 行号 | 作用 |
|---|---|---|
| `chat` | 58 | 多轮对话 |
| `chat_for_evaluation` | 76 | 评测专用对话入口 |
| `PRM` | 116 | 段落相关性评估（Passage Relevance Module） |
| `GRM` | 约 130 | 指南相关性评估（Guideline Relevance Module） |
| `keyword_extraction` | 180 | 关键词抽取 |
| `KED_search` | 232 | KED 检索 |
| `get_passages` | 275 | 检索 + 向量过滤 |
| `MAIRAG` | 380 | 多智能体迭代检索增强生成 |
| `HERD_search` | 439 | 多级/多源检索 |
| `AEBMP` | 509 | 循证医学实践 |
| `MOEQA` | 651 | 医嘱解释 |
| `MAIRAG_options` | 665 | 选项式 MAIRAG（选择题评测用） |

### 4.5 KED 与多源医学检索恢复

KED 已恢复为可独立测试的检索模块。其基本逻辑为：**先用原始问题检索，结果不足时进行关键词抽取，并通过关键词组合及逐步退化策略继续检索**。

从 `KED_search()` 实现可见的退化策略：

1. 用原始问题检索；
2. 无结果 → 调用 `keyword_extraction()` 抽取关键词；
3. 关键词过短（`len <= 3`）→ 跳过降级，避免无效检索；
4. 关键词用 `AND` 连接，无结果则逐个**移除末尾关键词**再检索；
5. 仍无结果 → 返回 `None`；含 rate limit 处理与重试。

同时 HERD 相关多源检索逻辑得到保留，可按配置从 **Guidelines / PubMed / Bing** 获取证据（`HERD_search()`，行 439），并交给后续答案生成流程。检索结果与问题会用 `self.retriever` 做向量相似度过滤，取最相关段落。

### 4.6 上层医学应用模块恢复

除 MAIRAG 外，代码还保留：

- **AEBMP**（循证医学实践）— `LINS.AEBMP()`
- **MOEQA**（医嘱解释 / Medical Order QA）— `LINS.MOEQA()`
- **Medical Text Explanation**（医学文本解释）
- **PICO**（临床问题结构化）
- **SKA**（自知识分析）
- **QDA**（问题分解）
- **PCA**（段落一致性分析）

因此当前复现并非仅恢复一个基础 RAG，而是基本保留了 LINS 面向医学问答场景的主要应用接口。

### 4.7 自动化测试与评测框架建立

已形成**两类**脚本：

**① 快速验证 —— `LINS-main/test_deepseek.py`（397 行）**

覆盖 2.1–2.14 共 14 个子测试：基础多轮对话、MAIRAG 完整功能、PRM 段落相关性、SKA+QDA 降级路径、多智能体流程、KED 关键词退化检索、QDA 问题分解降级、HERD 多级检索、AEBMP、PubmedQA 前向评估、LinkEval 引用评估等。另含 `test_reasoner` / `test_reasoner_with_retriever` 两个 reasoner 场景。

**② 完整评测 —— `test_deepseek_lins_full.py`（623 行）**

组织三数据集评测 + 准确率统计 + LinkEval 指标计算，并保存为结构化文件。

**③ 公共工具 —— `LINS-main/lins_test_common.py`（338 行）**

被上述两类脚本共用，包含：

| 函数 | 作用 |
|---|---|
| `load_pubmedqa_samples` | 加载 PubMedQA 样本 |
| `test_database_config` | 测试数据库配置 |
| `test_ked_retriever` | 测试 KED 检索 |
| `print_metric_summary` | 打印指标汇总 |
| `should_force_maybe` | **后处理规则（本分支新增，原版没有）** |
| `load_linkeval_data` | 加载 LinkEval 数据 |
| `evaluate_single_linkeval_sample` | 单样本 LinkEval 评估 |


---

## 五、与原版 LINS 评测入口的差异（关键工程决策）

原版评测入口为 `LINS/evaluate/eval_pubmedqa_star_ALL.py`（PubMedQA）与 `eval_medqa_star_ALL.py`（MedQA），在当前环境下无法直接运行：

| 原版评测脚本的依赖 | 当前环境的障碍 |
|---|---|
| `arguments.py` 的 `get_medlinker_args()` | 需命令行传参；默认 `--task_list medqa_ch medqa_us medqa_tw`，无 pubmedqa |
| 默认 `gpt-4o-mini` + `OPEN_API_KEY` | 无法访问 OpenAI |
| `--retriever_ckpt_path` 本地 checkpoint | 路径与 `LINS-main` 实际布局不符 |
| 检索结果需**预先落盘** `search_results/...jsonl` | 必须两阶段流程，需先离线跑检索 |
| `Link_Eval.py` 依赖 `med_linker_search` | 外部包装不上，CI 失败 |
| `--device cuda:1` | 单卡/无卡环境不适用 |

因此 `test_deepseek_lins_full.py` 的**本质是一个自包含的单文件评测驱动（driver）**：把"检索 → 生成 → 后处理 → 评估 → 落盘"整条链在一次运行内串起来，不依赖 CLI 参数、离线检索产物与外部包。

| 维度 | 原版 `eval_pubmedqa_star_ALL.py` | `test_deepseek_lins_full.py` |
|---|---|---|
| 入口形式 | `argparse` + `get_medlinker_args()` | 无参数，配置写在文件头部常量区 |
| LLM | `gpt-4o-mini`，`OPEN_API_KEY` | `deepseek-chat`（主 + 辅助） |
| 检索器 | `text-embedding-3-large`（OpenAI API）或预生成文件 | 本地 `BGE` / `bge-m3`，完全离线 |
| 数据集 | 仅 PubMedQA | PubMedQA + MedQA-US + MedQA-Mainland |
| 覆盖方法 | 遍历 `method_list=['chat','Original_RAG','MAIRAG',...]` | 仅 MAIRAG |
| 答案解析 | `answer_map={A:yes,B:no,C:maybe}`，扫描 A/B/C | PubMedQA 同；MedQA 直接用 `options_response` |
| 后处理 | 无 | **新增** `should_force_maybe()` 四规则 |
| 并发 | `ThreadPoolExecutor` + `num_batch` 批处理 | 串行 for 循环（可读性好，速度慢） |
| 评估 | 仅 `acc` | 准确率 + LinkEval 五指标 |
| 断点续跑 | 有 `num_begin/num_end` | 无（失败需整批重跑） |
| 输出 | 单个 `results.jsonl` | `*_details.jsonl` + `*_results.json` + `linkeval_results.json` + 汇总 |

---

## 六、当前评测体系与实际结果

### 6.1 评测数据集

| 数据集 | 规模 | 当前使用 |
|---|---|---|
| PubMedQA (`ori_pqal.json`) | **1000** 条（ground truth 500） | 脚本配置 20 条 |
| MedQA-US (`medqa_us_test.json`) | **1273** 条 | 脚本配置 15 条 |
| MedQA-Mainland (`medqa_mainland_test.json`) | **3426** 条 | 脚本配置 15 条 |
| MedQA-Taiwan | 已就位 | 未启用 |

### 6.2 实际运行结果（来自结果文件，非 README 示例）

**① LINS 完整框架结果（`eval_results_lins_full/`，2026-05-25）**

MedQA 各 15 条：

| 数据集 | 正确/总数 | 准确率 |
|---|---|---|
| MedQA-US | 13/15 | **86.7%** |
| MedQA-Mainland | 13/15 | **86.7%** |

逐题明细（MedQA-US，含 KED 检索条数）：

```
test_0  GT=C ✓ ked=3     test_8  GT=C ✓ ked=12
test_1  GT=E ✓ ked=20    test_9  GT=A ✓ ked=12
test_2  GT=C ✓ ked=14    test_10 GT=E ✓ ked=0
test_3  GT=D ✗(选B) ked=1
test_4  GT=B ✓ ked=7     test_11 GT=D ✓ ked=1
test_5  GT=E ✗(选C) ked=1
test_6  GT=D ✓ ked=1     test_12 GT=B ✓ ked=1
test_7  GT=C ✓ ked=20    test_13 GT=E ✓ ked=4
                         test_14 GT=D ✓ ked=1
```

MedQA-Mainland 15 条中错 2 题（`test_0` 选 D、`test_10` 选 E）。合计 30 题错 4 题。

**② 纯 LLM 零样本基线（`eval_results_llm_baseline/`，无检索、无 prompt 工程）**

各 5 条：MedMCQA 0.40、PubMedQA 0.00、MedQA-US 0.40、MedQA-Mainland 0.40。

**③ 早期探索结果（`eval_results/`，各 5 条）**

| 数据集 | LINS(deepseek) | 纯 LLM 基线 | 差值 |
|---|---|---|---|
| MedMCQA | 0.60 (3/5) | 0.40 (2/5) | +0.20 |
| PubMedQA | 0.20 (1/5) | 0.00 (0/5) | +0.20 |
| MedQA-US | 0.60 (3/5) | 0.40 (2/5) | +0.20 |
| MedQA-Mainland | 0.60 (3/5) | 0.40 (2/5) | +0.20 |

**④ 全量 no-RAG 直答对照（`medqa_direct_summary.json`，2026-06-04）**

| 数据集 | 正确/总数 | 准确率 |
|---|---|---|
| MedQA-US | 897/1273 | **70.5%** |
| MedQA-Mainland | 3035/3426 | **88.6%** |

### 6.3 LinkEval 结果（`eval_results_lins_full/linkeval_results.json`，5 条 PubMedQA）

| 指标 | 数值 |
|---|---|
| Citation Precision (CP) | **0.881** |
| Citation Recall (CR) | **0.813** |
| F1 Score | **0.834** |
| Statement Correctness (SC) | **0.200** ⚠️ |
| Statement Fluency (SF) | **0.950** |

### 6.4 评价指标说明

DeepSeek 版 LinkEval（`Link_Eval_DeepSeek.py`）对齐论文核心指标，并用 DeepSeek 替代两个重型依赖：

| 指标 | 主要用途 | 本分支实现方式 |
|---|---|---|
| CSP | 引用集合层面的精确性 | 论文对齐 |
| CP | 引用准确性 | `compute_citation_metrics` |
| CR | 引用覆盖情况 | `compute_precision_and_recall` |
| F1 | 综合 CP 与 CR | 调和平均 |
| SC | 陈述正确性 | 语句间无矛盾 + 答案正确 |
| SF | 陈述流畅性 | `DeepSeekFluency`（替代 UniEval） |
| NLI 蕴涵判断 | SC/CSP 的底层判据 | `DeepSeekNLI`（替代 T5-11B） |

---

## 七、当前阶段需要特别说明的问题

### 7.1 "代码复现完成" ≠ "论文实验复现完成"

已基本完成代码与运行环境层面的恢复，但严格的论文级复现还需确认原论文中的模型、Retriever、数据集、Prompt、Top-k、模块开关等实验参数是否与当前版本一致，并通过正式运行获得可追溯结果。**当前更准确的结论是"已形成可运行的 LINS 复现基线"，而非"全部实验已完全复现"。**

### 7.2 README 示例结果不能直接作为正式实验结果

README 与测试代码中的示例输出仅用于说明运行格式，正式汇报应以实际运行生成的 JSON/JSONL 为准。本报告第六章的数据**全部来自结果文件实测统计**，未引用 README 示例数字。

### 7.3 工程适配与算法改进需要明确区分

DeepSeek 适配、依赖修复、接口兼容与测试工具重构属于 **Engineering Adaptation**，不应被描述为对 LINS 算法本身的改进。后续若在 LINS 基线之上开展新算法研究，应明确划分 **"LINS Reproduction Baseline"** 与 **"New Method"**。

### 7.4 需整改的具体缺陷（本次核对新发现）

以下为对代码与结果文件逐项核对后发现的**实际问题**，属于进入正式实验前必须处理的前置项：

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| 1 | **API Key 明文硬编码** | `test_deepseek.py:16`、`test_deepseek_lins_full.py:15`、`Link_Eval_DeepSeek.py:43` 均写死 `sk-94ccad...` | 安全事故；且与"改用环境变量"的提交方向相悖 |
| 2 | **PubMedQA 链路未跑通** | `eval_results_lins_full/pubmedqa_details.jsonl` 仅 **5 条**（脚本设定 20 条），`predicted_answer` 全空，第 5 条 `is_correct` 为 `null`（GT=unknown） | PubMedQA 结果不可用 |
| 3 | **SC 指标异常** | 引用质量 F1=0.834，但 SC 仅 0.2，两者矛盾 | 疑为 SC 的 prompt 或解析逻辑对 DeepSeek 输出不兼容 |
| 4 | **LINS+RAG 全量对照缺失** | LINS+RAG 仅 15 条；no-RAG 已全量（1273/3426） | 两者规模不可比，结论不成立 |
| 5 | **MedQA-Mainland 未见提升** | LINS+RAG 15 条 86.7% vs no-RAG 全量 88.6% | ⚠️ 危险信号：需扩量验证 RAG 是否真的有增益 |
| 6 | **结果文件未入库** | `.gitignore` 第 20–22 行忽略所有 `eval_results*/` | 成果无法追溯，不利复现核验 |
| 7 | **工作区不干净** | `LINS-Industrial/` 整目录显示为已删除未提交；另有 `task_progress.md`、`LINS_industry_img/` 等 | 干扰 baseline 边界，与"排除工业方向"的定位不符 |
| 8 | **`LINS/` 与 `LINS-main/` 代码重复** | 两份 `metric/`、`evaluate/`、`model/searching/` 等 | 易混淆，需明确哪份是唯一复现基线 |
| 9 | **无重复运行/稳定性验证** | 结果均为单次运行 | 无法确认非随机波动 |
| 10 | **评测脚本串行执行** | 完整评测为 for 循环 | 全量跑批（1273+3426 条）耗时不可接受 |

### 7.5 关于结论的审慎表述

附件报告已正确指出"已形成可运行的 LINS 复现基线"。在此基础上，本报告补充强调：**当前尚不足以得出"RAG 提升了回答质量"的结论**，原因是：(a) 样本量过小（15/20 条）；(b) PubMedQA 链路失效；(c) 唯一可比的 MedQA-Mainland 全量数据点上，RAG 版本反而不及 no-RAG。这三点必须通过 7.4 中 #2、#4、#5 的整改来消除。


---

## 八、下一阶段工作计划

在附件计划基础上，结合 7.4 的实测缺陷，调整如下：

| 优先级 | 任务 | 目标 | 产出 |
|---|---|---|---|
| **P0** | 移除硬编码 API Key | 统一改为环境变量 + 轮换密钥 | 安全整改记录 |
| **P0** | 修复 PubMedQA 答案抽取 | 从 5 条扩至 20+ 条有效样本 | PubMedQA 逐样本结果 |
| **P0** | 冻结实验配置 | 统一 LLM、Retriever、Top-k、Prompt、模块开关 | Baseline 配置表 |
| **P0** | LINS+RAG 全量运行 | MedQA-US(1273) + MedQA-Mainland(3426) | 逐样本结果文件 |
| **P0** | 完成 LinkEval 统计 | 统一计算各指标，并排查 SC 异常 | 指标汇总表 |
| **P1** | 重复运行/稳定性验证 | 确认非单次随机波动 | 均值、方差或置信区间 |
| **P1** | 原论文结果对齐 | 逐项比较实验设置与结果 | Reproduction 对照表 |
| **P1** | 整理干净 Baseline | 将 `LINS-main` 作为独立基线，清理工作区 | 独立实验目录/说明 |
| **P2** | 结果入库 | 关键 JSON 汇总提交版本库 | 可追溯结果集 |
| **P2** | 形成最终汇报材料 | 总结复现结果与差异原因 | LINS Reproduction Report |

---

## 九、建议的最终实验组织方式

为避免 master 分支中的工业方向代码影响 LINS baseline，后续实验建议**以 `LINS-main` 为唯一复现代码范围**。实验记录按"代码版本—环境—模型—数据集—配置—运行结果—统计结果"的顺序保存。

推荐的实验结构：

```
LINS Baseline
├── Original LINS 配置（如可获得）
├── DeepSeek-LINS 配置
├── PubMedQA          (1000 条，正式规模)
├── MedQA-US          (1273 条，正式规模)
├── MedQA-Mainland    (3426 条，正式规模)
├── LLM-Baseline 对照 (no-RAG, 同规模)
└── LinkEval          (CP / CR / F1 / SC / SF)
```

后续其他研究方法应在该 baseline 固定后独立加入，避免将工程适配与算法改进混在同一实验结果中。

---

## 十、阶段性结论

截至目前，LINS 原版复现工作已完成从代码恢复到可运行系统的主要建设工作。`LINS-main` 的核心模型、Retriever、Database、KED、MAIRAG、HERD、多源医学检索以及 LinkEval 等主要模块均已具备较完整实现；DeepSeek API 适配与主要环境兼容问题也已处理，并已上 CI 保障可复现性。

**当前项目已具备作为后续实验基线的基本条件。下一阶段核心任务应从"继续修代码"转向"严格做实验"**：固定配置、实际运行、保存原始结果、重复验证、与原论文设置逐项对照，并最终形成一套可追溯、可复核的 LINS reproduction baseline。

同时需强调三点：

1. 本报告有意**排除** LINS-Industrial、IndustryBench、IHER 及工业场景相关内容，上述内容属后续独立研究方向，不应与当前 LINS 原版复现工作的进度和成果混合统计。
2. 当前已积累的实验数字**不足以支撑"RAG 提升回答质量"的结论**（样本量小 / PubMedQA 失效 / MedQA-Mainland 全量上 RAG 不及 no-RAG），需按第八章 P0 项整改后再下判断。
3. 所有指标数字均取自实际结果文件，README 与测试脚本中的示例输出不作为实验依据。

---

## 附：当前工作状态速览

| 类别 | 状态 |
|---|---|
| 代码恢复 | 🟢 基本完成 |
| 环境适配 | 🟢 基本完成 |
| DeepSeek 适配 | 🟢 完成（但 Key 硬编码待整改） |
| 核心算法模块 | 🟢 基本恢复 |
| 测试框架 | 🟢 已建立 |
| CI 可复现性 | 🟢 已上 GitHub Actions |
| MedQA 小样本评测 | 🟢 已跑（各 15 条，86.7%） |
| 全量 no-RAG 对照 | 🟢 已跑（1273 / 3426 条） |
| LinkEval 引用指标 | 🟡 已跑（5 条），SC 指标异常待排查 |
| PubMedQA 评测 | 🔴 未跑通（仅 5 条且答案抽取失效） |
| LINS+RAG 全量实验 | 🔴 待完成 |
| 重复运行/稳定性验证 | 🔴 待完成 |
| 论文级复现核验 | 🟡 进行中 |
| 最终 Baseline 冻结 | 🔴 待完成 |
| LINS-Industrial / IHER | ⚪ 本报告排除 |

---

*报告生成依据：`LINS-main/` 源码与行数比对、`eval_results*/` 结果文件实测统计、git 提交历史、`.github/workflows/test_imports.yml`、以及附件《LINS 项目复现工作阶段性报告》。*
