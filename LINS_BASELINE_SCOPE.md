# LINS Reproduction Baseline — 范围说明

本文件定义 LINS 原版复现工作的**实验边界**。所有 LINS 复现相关的实验、结果统计与汇报，均以本文件划定的范围为唯一口径。

---

## 1. 唯一复现基线：`LINS-main/`

`LINS-main/` 是 LINS 原版复现与 DeepSeek 适配的**唯一核心目录**，所有正式实验只在此范围内进行。

```
LINS-main/
├── model/                    # 核心：LINS / Retriever / Database / LLM 封装
│   ├── model_LINS.py         # 核心 LINS 类（MAIRAG / KED / AEBMP / MOEQA / HERD）
│   ├── chat_llms.py          # LLM 调用封装（DeepSeek / GPT / Gemini / Qwen）
│   ├── retriever_model.py    # 检索器（本地 BGE-M3 / OpenAI embedding）
│   ├── database.py           # 数据库（pubmed / bing / guidelines / omim / oncokb）
│   ├── prompts.py            # Prompt 模板（与原版一致，未改动）
│   ├── utils.py
│   └── extracting/ fetching/ filtering/ retriever/ searching/
├── evaluate/                 # 评测脚本与数据集
│   ├── evaluate_data/        # pubmedqa / medqa_us / medqa_mainland / medqa_taiwan
│   └── eval_medqa_star_ALL.py, arguments.py
├── metric/                   # evaluator.py / scorer.py
├── add_dataset/              # 自定义数据集添加
├── Link_Eval.py              # 原版 LinkEval（未改动）
├── Link_Eval_DeepSeek.py     # DeepSeek 版 LinkEval
├── lins_test_common.py       # 公共测试工具
├── test_deepseek.py          # 快速验证脚本
└── test_lins.py              # 原版调用示例
```

配套的评测驱动位于仓库根目录：

| 路径 | 作用 |
|---|---|
| `test_deepseek_lins_full.py` | 完整评测驱动（三数据集 + LinkEval） |
| `evaluate_data/` | 补充数据集（MedQA-US 全量等） |
| `model/retriever/bge/bge-m3/` | 本地 BGE-M3 向量模型 |
| `eval_results/` | 早期小样本结果（5 条） |
| `eval_results_lins_full/` | LINS（含 RAG）评测输出 |
| `eval_results_llm_baseline/` | 纯 LLM 零样本对照组输出 |
| `LINS_复现工作整理报告.md` | 阶段工作整理报告 |

---

## 2. 明确排除的范围

以下内容**不计入** LINS 原版复现工作，已从版本库中移除（提交 `cc36088`）：

| 排除项 | 说明 |
|---|---|
| `LINS-Industrial/` | 工业方向二次研究项目（376 个文件） |
| `LINS_industry_img/` | 工业方向配套图片材料 |
| `LINS_Industrial_Project_Summary.md` | 工业方向项目总结 |
| `task_progress.md` | 工业方向进度记录 |
| IndustryBench / IHER 相关实验 | 工业检索研究内容 |

这些内容属于**后续独立研究方向**，应在 LINS baseline 冻结之后单独开展，并在独立目录中记录，不与 LINS 复现结果混合统计。

---

## 3. 边界维护规则

1. **新增 LINS 复现内容** → 放入 `LINS-main/` 或根目录对应的评测输出目录。
2. **新增工业方向内容** → 不得放入本仓库的 baseline 范围；`.gitignore` 已对工业方向路径设置排除规则。
3. **区分工程适配与算法改进**：
   - DeepSeek 适配、依赖修复、接口兼容、测试工具重构 = **Engineering Adaptation**
   - 在 LINS 之上提出的新方法 = **New Method**，必须独立标注，不得与复现结果混报。
4. **实验记录顺序**：代码版本 → 环境 → 模型 → 数据集 → 配置 → 运行结果 → 统计结果。

---

## 4. 实验记录结构（推荐）

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

---

## 5. 基线配置冻结状态

| 配置项 | 当前值 | 状态 |
|---|---|---|
| 主 LLM | `deepseek-chat` | 已定 |
| 辅助 LLM | `deepseek-chat` | 已定 |
| Retriever | `BGE`（本地 `bge-m3`） | 已定 |
| Database | `pubmed` | 已定 |
| 评测方法 | `MAIRAG` | 已定 |
| Top-k | 待统一 | 🔴 未冻结 |
| Prompt | 沿用原版（未改动） | 已定 |
| 模块开关 | 待统一 | 🔴 未冻结 |
| 随机性 | 结果均为单次运行 | 🔴 待重复验证 |

> ⚠️ 基线尚未完全冻结。在 Top-k / 模块开关统一并完成重复运行验证之前，实验结果不具备正式可比性。

---

*相关文档：`LINS_复现工作整理报告.md`（阶段工作整理）、`README.md`（项目总览）。*
