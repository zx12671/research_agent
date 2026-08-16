# 结论：task/分类维度在下沉(证据先验)场景同样无增益——A/B 实测

## 背景与问题

上一轮已通过 16 样本 A/B（task→Solver 专门 prompt，avg_delta=0.0）证明：
**TaskAnalyzer 输出的 task 分类对最终 QA 结果无贡献**（覆盖受限 + 作用无增益）。

本实验转换"施加场"，回答剩余可能：把 task/format 分类沉降为**证据层先验信号**
（仅 rescore/重排 Evidence，绝不下发 Solver），量化对 ground-truth 召回的得失。

## 实验设置（`_ab_evidence_prior_rerank.py`）

- 样本：2040 全量按 `(capability, _format)` 交叉分层抽样，121 题（25 单元）。
- 统一真实检索：`OpenDomainRetriever.retrieve(q, k=10, use_ked=True)`（KED+dense+fallback）。
- 命中判定：AiOU≥0.15 vs `knowledge_text`。
- 先验构造：format(题型)+task 关键词 → 期望 capability 集合（口径与 analyzer KEYWORD_MAP 对齐）。
- 三档排序对比：
  - `base`：原检索序
  - `small`：命中期望 capability 的 chunk `score+=0.15` 后重排
  - `hard`：期望 capability 分区硬插到非期望之前

## 结果（n=121；hit@1/3/5/10 为命中占比，improved/unchanged/worsened 为命中位次位移）

| 方法 | hit@1 | hit@3 | hit@5 | hit@10 | improved | unchanged | worsened |
|---|---|---|---|---|---|---|---|
| base   | 50.4% | 57.9% | 61.2% | **64.5%** | — | — | — |
| small  | 38.0% | 46.3% | 56.2% | 64.5% | 1 | 97 | **23** |
| hard   | 34.7% | 43.8% | 55.4% | 64.5% | 1 | 93 | **27** |

先验前提检验：**真值 capability ∈ 期望集合 仅 35.8%**（39/109）。

## 解读（三条硬证据）

1. **前提不成立**：format/task→capability 的映射只有 1/3 命中真值类别，
   在 64% 的题上是错误信号 → 重排 = 人为扰动。
2. **只伤顺序、不动集合**：`hit@10` 三档完全持平（64.5%）说明先验不改变"找回哪些块"，
   只破坏 top-k 窗口内顺序 → 命中被压到 5 名之后（top-5 截断场景实打实丢分）。
3. **worsened ≫ improved**（23~27 : 1）：位移分布压倒性劣化，与先验方向一致预期相反。

## 结论（与既有 A/B 合并的闭环）

- **task/分类维度：施加于 Solver prompt → avg_delta=0（无增益）**
- **task/分类维度：施加为 Evidence 先验重排 → hit@1/3/5 显著下降（有害）**
- 杠杆不在"分类维度信号化"，其价值上限被两个事实锁死：
  KED+dense 已把 ground-truth 排在窗口内；而人工先验对真值类别的覆盖仅 35.8%。

**处置建议**：
- TaskAnalyzer 的 task 分类**不作为 Solver / 检索的先验信号**，维持纯观测/报告用途。
- 后续若提升召回，方向应转向**证据本身**（Recall 上限、KED query 扩展、chunk 粒度/质量），
  而非给现有 top-k 窗口叠加入工排序先验。

## 复现

```bash
cd LINS-Industrial
python _ab_evidence_prior_rerank.py
```
