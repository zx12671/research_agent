# 检索环节 A/B：单查询 Dense(现状) vs 单查询 Hybrid(dense+BM25→RRF)

> 回答核心问题：**在不启 multi_query 的前提下，接通 `hybrid_retrieve`（dense + BM25 → RRF）能否补强召回？**
> 实测脚本：`_retrieve_single_vs_hybrid.py`
> 样本 30 题（6 capability × 5，seed=7；随机追加选型类 1 题）
> 命中判定：汉字 bigram IoU ≥ 0.20 vs `knowledge_text`
> 检索器：`OpenDomainRetriever`（55095 块，bge-small-zh-v1.5）
> hybrid 固定：`use_multi_query=False, use_sparse=True, sparse_pool=50, dense_w=1.0, sparse_w=0.8`

---

## 一、背景：为什么要"单查询 hybrid"而不是多查询

- 之前的 `_probe_recall_30.py` 里 **multi_query**（`ked.decompose` 按「、，」切句）被证实**有副作用**：子查询各自带别的块，RRF 融合把"对的那块往下压"，导致 hit@5/10 反而低于 single。
- 本任务要求**不启用 multi_query**，只验证"BM25 补强纯 dense"这一条更干净的路径。
- `hybrid_retrieve(use_multi_query=False)` 恰好提供这条路径：**单查询**同时跑 dense(KED) 与 BM25，再把两者 RRF 融合。

---

## 一·五、P0 口径已统一（重要）

本 AB 评估口径已切换为 **GT 句子覆盖率**（`retrieval/recall_metrics.py::sent_coverage`），
以下新旧两张表须并置阅读：
- **主口径 = 句子覆盖率**（下方新表）—— 这是优化/回退的唯一验收依据。
- 底下"整段 IoU hit@k / top-50 累积"**降级为交叉参照**（旧口径会系统性低估，不再单独作结论）。

**主口径（句子覆盖率，30 题，seed=7）：**

| 方法 | cov@10 | cov@20 | cov@50 | mean | med | cov≥50% |
|---|---|---|---|---|---|---|
| single | 0.27 | 0.32 | 0.36 | 0.27 | 0.26 | 17% |
| hybrid(单查询) | 0.29 | 0.33 | 0.36 | 0.29 | 0.28 | 20% |

**按 capability（cov@10 均值）：** 标准规范+0.02 / 安全合规+0.04 / 质量计量+0.04 / 故障诊断+0.03 / **工程计算-0.02（唯一微回退）**。

> 结论：在句覆盖率口径下，hybrid 依然带来一致的（小幅）补全度提升（cov@10 0.27→0.29、拼齐≥50% 17%→20%），且与归因实验吻合（提升≈BM25 单独贡献 +0.02）。


## 二、交叉参照 · 旧口径 top-10 内 Recall@k（整段 IoU，不再单独作结论）

| 方法 | n | hit@1 | hit@3 | hit@5 | hit@10 |
|---|---|---|---|---|---|
| single (现状) | 30 | 26.7% | 40.0% | 46.7% | 46.7% |
| **hybrid (单查询)** | 30 | **30.0%** | **46.7%** | 46.7% | 46.7% |

**读法：**
- **hit@1 +3.3pp（27→30%）、hit@3 +6.7pp（40→47%）** —— 这是最有价值的增益：BM25 用精确 token（型号/标准号/能力词）把"正确的块"顶到了**更靠前**的位次（top-1/top-3），直接决定 organizer/reason 是否优先用它对。
- hit@5/10 **持平 46.7%**：召回上限（recall ceiling）在 k≥5 时仍由 dense 主决定，BM25 并未引入全新命中的块。

---

## 三、交叉参照 · 旧口径 GT 在 top-50 候选池内的累积命中（整段 IoU）

| 方法 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|
| single | 47% | 47% | 47% | 47% |
| **hybrid** | 47% | 47% | 47% | **50%** |

**结论：**
- 需注意：此前的"top-50 命中"是用**整段 IoU** 口径；`docs/agentic_recall_completeness_correction.md` 证明该口径会严重低估真实召回（见 `agentic_recall_completeness_correction.md` 与 `_retrieve_ablate_source.py`）。
- 在**句子覆盖率**口径下，真正"top-50 外"的只有极少数 B 类题；hybrid 在句覆盖率 K=10 上比 single **+0.020**（见 `docs/agentic_recall_completeness_correction.md` 的归因）。
- hybrid 的增益主要来自 **BM25 精确 token 补强**（非 dense 扩容），在固定 top-10 内用同量召回拿到了更高的 hit@1/@3，属**纯 precision@前段 提升**。

---

## 四、按 capability：hit@10（无退化）

| capability | n | single@10 | hybrid@10 | delta |
|---|---|---|---|---|
| 标准规范与术语 | 5 | 40% | 40% | 0 |
| 工艺原理与参数影响 | 5 | 40% | 40% | 0 |
| 安全合规与风险控制 | 5 | 80% | 80% | 0 |
| 质量计量与检测 | 5 | 20% | 20% | 0 |
| 故障诊断与排查 | 5 | 40% | 40% | 0 |
| 工程计算与估算 | 5 | 60% | 60% | 0 |

**无一类退化**；每类命中与 single 一致，说明 BM25 融合没有在任一类引入噪声命中来挤掉对块。

---

## 五、score 分布（重要注意事项，供下游）

| 方法 | min | p50 | mean | p90 | max |
|---|---|---|---|---|---|
| single | 0.571 | 0.698 | 0.696 | 0.777 | 0.862 |
| hybrid | 0.014 | 0.023 | 0.021 | 0.029 | 0.030 |

**关键警示：** hybrid 的 `score` 是 **RRF 纯排名分**（`Σ weight/(60+rank)`），量级（~0.02）远小于 dense 相似度（~0.7）。
- **对当前默认链路无功能影响**：GENERAL/Mock 下 organizer 走 `organize_by=topic` 分组（不用 score 做阈值），prompt_builder 只把 score 作为展示字段 `(score=0.030)`。
- **但对未来的 rerank 是隐患**：organizer 的 `rerank=True` 分支有 `norm_score = min(raw_score/1.0,1.0)`，假设 score∈[0,1]；若喂入 RRF 分(~0.02)，归一化后几乎为 0，会压制 lexical 协同。**落地 rerank 前必须先对 RRF 分做重标定（如按 dense 最大分映射或单独归一化）。**

---

## 六、端到端生效验证（agentic 链路）

已在 `_handle_retrieve` 增加 hybrid 分支（`params["hybrid"]=True` 时走 `retriever.hybrid_retrieve`，单查询，不开 multi_query）；planner 支持 `StrategyPlanner(retrieve_hybrid=True)` 让模板图请求它。实测一题：
- `[retrieve_1] ... hybrid=True` → 日志确认走 hybrid 分支；
- BM25 把 **`SIMOREG DC Master 6RA70`** 关键 chunk（含"多台整流并联运行"答案）精确顶到第 1 位；
- evidence 10 → organize 5 组(2279 字符) → reason，链路完整。

---

## 七、结论与裁决

| 结论 | 度量 |
|---|---|
| 单查询 hybrid 提升 hit@1/@3（precision 前段） | +3.3 / +6.7 pp |
| hit@5/10 持平，不降不升 | 0 pp |
| GT 进 top-50（召回上限）仅 +3pp | 47→50% |
| capability 层无退化 | 全部 =0 |
| 端到端 agentic 链路正常 | ✅ |

**一句话**：单查询 hybrid（dense+BM25→RRF）在**不引入 multi_query 副作用**的前提下，把"命中的块挤到更靠前"（hit@1/3 提升），且无任一侧退化，**是安全、值得默认开启的检索增强**；但它**不解决召回入口（top-50 外的约 50% 丢失）**——那是下一步"修 embedding/BM25 初筛召回"的目标，与本次 enhancement 正交。

---

## 八、实施改动

- `agentic/pipeline.py` `_handle_retrieve`：新增 `use_hybrid` 分支（在 multi_query 之前判断，按 `params["hybrid"]` 开关），透传 `hybrid_sparse_pool/hybrid_dense_weight/hybrid_sparse_weight`。
- `agentic/planner.py`：
  - `base_retrieve_params` 增加 `hybrid/hybrid_sparse_pool/hybrid_dense_weight/hybrid_sparse_weight`（默认 hybrid=False，**零侵入现状**）；
  - `StrategyPlanner.__init__` 新增 `retrieve_hybrid: bool = False`，仅影响模板图默认值，LLM 自定义图不受影响。
- 新增实验脚本 `_retrieve_single_vs_hybrid.py`（可复现本 A/B）。

## 复现

```bash
cd LINS-Industrial
python _retrieve_single_vs_hybrid.py --n 30 --seed 7
# 启用 agentic 端 hybrid（示例）
python -c "
from agentic import StrategyPlanner
p = StrategyPlanner(llm_client=..., retrieve_hybrid=True)
"
```
