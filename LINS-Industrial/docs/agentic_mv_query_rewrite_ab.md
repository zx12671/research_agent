# A/B 结论：语义级多视角 Query 改写（P1 补全度主杠杆）—— 证据池句覆盖 + 完整 AdaptiveAgenticPipeline

> 结论一句话：**多视角改写 + 融合在完整 agentic 链路证据池上，句覆盖最多 +0.012（mv_hybrid）、
> 且 hit@1 翻倍（12%→38%），但未转化为答案质量提升（--real 下 base 2.25 → mv 2.00，回退 0.25）。
> 多视角改写不是补全度的主杠杆，按纪律不默认开；真正的杠杆仍是证据前向（TOP-1 chunk 再造第二轮）。** ⚠️证伪性结论

---

## 一、目标与方法

验证「语义级多视角 query 改写」（由 LLM 依 TaskAnalysis 的 task/expected_evidence 生成 2~4 个
语义互补子视角）在**完整 `AdaptiveAgenticPipeline`** 上对证据池句覆盖率与最终答案质量的真实增益。

三臂（完整 pipeline 全链路，仅 external 检索不同，注入方式与 `_e2e_agentic_ef.py` 同构）：

| 臂 | 检索方式 |
|----|----------|
| `base` | 完整 pipeline + single dense + KED（生产现状） |
| `mv_dense` | 多视角改写 → 各视角 dense → RRF 融合 |
| `mv_hybrid` | 多视角改写 → 各视角 hybrid(dense + BM25jieba, sparse_w=0.2) → RRF 融合 |

- LLM 双模式：`--mock`（MockLLM + 确定性视角，零成本，测证据池句覆盖传导）；
  `--real`（真实 DeepSeek + 真实语义视角，测最终答案质量）。
- 主口径 = `retrieval/recall_metrics.sent_coverage`（证据池 = `answer()` 返回的 `retrieval_result.documents`，
  即喂给 agent 的唯一外部证据源）；交叉参照 = 整段 IoU hit@k；--real 附加 `RuleBasedScorer`。
- 样本：`_ab_mv_query_rewrite.py` 沿用 `_ab_bm25_jieba.py` 口径（6 capability × 5 = 30 题, seed=7）。

## 二、结果

### 2.1 主口径：证据池句覆盖率 cov@10（mean/median/≥50%）

**mock 30 题（确定性视角，测传导）**

| 臂 | mean | median | ≥50% |
|----|------|--------|------|
| base | 0.273 | 0.255 | 17% |
| mv_dense | 0.276 | 0.260 | 20% |
| mv_hybrid | 0.276 | 0.260 | 17% |

**real 8 题（真实语义视角，测答案质量）**

| 臂 | cov@10 mean | hit@1 | RuleBasedScorer mean |
|----|-------------|-------|----------------------|
| base | 0.335 | 12% | **2.250** |
| mv_dense | 0.334 (−0.001) | 25% | 2.000 (−0.250) |
| mv_hybrid | 0.347 (**+0.012**) | **38%** | 2.000 (−0.250) |

（real 4 题子集曾现 mv_dense 答案质量 1.5→2.0 假升；扩到 8 题后反转，base 反超，以 8 题为准。）

### 2.2 capability 分层（mock 30 题，mv_hybrid cov@10）

- `标准规范与术语`：0.231 → 0.259（+0.029，最大增益——标准号维度匹配强）
- `安全合规与风险控制`：0.438 → 0.466（+0.027）
- `故障诊断与排查`：0.377 → 0.392（+0.015）
- `质量计量与检测`：0.110 → 0.057（**−0.053，明显回退**）
- `工程计算/工艺原理`：基本不变

## 三、解读：为什么多视角不是主杠杆

1. **证据池句覆盖增益微弱**：mv_hybrid 完整链路 +0.012，与检索层 `_ab_bm25_jieba.py` 的 hybrid cov@10
   +0.011 同量级——**检索层增益在传导到完整 pipeline 证据池时没有放大，被 organizer/证据组织吸收**。
2. **mv_dense 中性（−0.001）**：纯 dense 多视角基本不增益，说明多视角分解本身收益有限；
   增益主要来自 hybrid 的 sparse 通道（BM25jieba 精确匹配），而非"多视角语义分解"。
3. **答案质量回退（−0.25）**：句覆盖小幅↑ 但最终答案 ↓——**多视角+融合把低相关块并入证据池，
   "上下文稀释"吃掉多出来的句子覆盖**；`质量计量与检测`的 −0.053 是这种带偏的集中体现。
4. **唯一稳定亮点 = hit@1 翻倍**：mv_hybrid 把最相关块顶到首位（12%→38%）——但这是"排序优化"，
   未转化为"拼齐答案"（≥50% 题占比持平或略降）与答案质量。

## 四、结论与建议

### 判定（遵循"无正增益则不默认开"纪律）
- **多视角改写不默认开**：不做 pipeline 注入、不改 `base_retrieve_params`、生产保持现状。
- mv_hybrid 的 hit@1 优势有独立价值，但被答案质量回退抵消；不构成开通理由。

### 增量下沉（若后续要继续此方向）
- 优先验证「证据前向」：TOP-1 chunk 再造第二轮查询（planner `retrieve_2`/`evidence_guided` 骨架已有，
  docs/agentic_recall_entrance_diagnosis.md 已定位它是查找真 hard-miss 的保险，比多视角分解更贴杠杆）。
- 若要保留 hit@1 优势，可考虑"仅在多视角证据池命中但答案质量受控时启用 sparse"，做门控而非全量融合。

### 交付物
- 新增：`agentic/query_rewriter.py`（独立改写器，含 mock/真实双模式）、`_ab_mv_query_rewrite.py`（三臂 A/B）
- 结果：`results/ab_mv_query_rewrite_mock.json`（30 题）、`results/ab_mv_query_rewrite_real.json`（8 题）
- 生产代码零改动（pipeline.py / planner.py / retriever.py 均未动，`hybrid=False` 现状保持）
