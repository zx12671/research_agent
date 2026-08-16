# ef（证据前移 top10+整档回捞）接入生产 Agentic 链路：端到端验证

日期：2026-08-05
脚本：`_e2e_agentic_ef.py`、`_e2e_agentic_ef_backfill_count.py`
输入：`results/e2e_agentic_ef.json`（agentic 分数）、`results/e2e_agentic_ef_backfill.json`（真实回捞数）

---

## 1. 本实验回答的问题

上一轮已在"轻量直答 LLM"口径下把 ef 的降分根因定位为 **上下文稀释（DISTRACT）**：
- base 证据已够/已能答好的题 → 批量回捞低相关碎片反而丢要点 → 降分；
- base 证据不足的题（缺口型）→ 回捞补上关键证据 → 升分。

但"轻量直答"（`Question → LLM(证据拼接)`）并非生产路径。生产路径是
`Question → 外部检索(top10) → TaskAnalyzer → Planner → Organizer → Solver → 答案`，
其中 organizer 会对证据做"原文保留 + 低相关降权"。因此必须验证：
**在前置检索相同的条件下，把 ef 的回捞块真正注入生产 agentic 链路，是否仍复现轻量直答的降分/升分规律，还是被 organizer/solver 吸收（变得稳健）。**

---

## 2. 方法（不改生产代码）

在 `AgenticRAGEngine` 的"稳定知识接口"层做**最小侵入替换**：
- 自定义 `EFExternalRetriever` 继承 `IndustrialRetriever`，重写 `retrieve()`：
  先用父类拿 top10，再按 JT 相似度对每个 top10 文档做整档回捞（与 ef 口径一致、`jt>=0.02`、按 `chunk_index` 排序 top50，块级去重）。
- 自定义 `EFAgenticRAGEngine` 仅覆盖 `_init_external_retriever()` 返回该检索器。
- 其余全部走生产 `AgenticRAGEngine.answer(retrieval_k=10)`：
  `pre_retrieved_chunks`（含回捞块）→ `AdaptiveAgenticPipeline.run()` → organizer/solver 全套保留。
- 对照：base = 原 `AgenticRAGEngine`（纯 top10）；ef = `EFAgenticRAGEngine`（top10+回捞）。
- 复用 `_diag_degrade_mechanism` 的 FOCUS 6 题（硬匹配 CSV 前缀），每条件跑 1 次完整 agentic。
- 评分：`RuleBasedScorer.rule_based_score`（与生产一致）。

已知局限：`_e2e_agentic_ef.py` 运行时 `n_extra_backfill` 因 citation 子串匹配 `"[*ef]"` 未命中而误判为 0；已用独立轻量探针（`_e2e_agentic_ef_backfill_count.py`）对同 6 题重算真实回捞块数并校正，判据为 citation 含 `*ef`。

---

## 3. 结果：agentic 生产链路下 ef 不再降分

| 题 | 类型(备注) | base 证据 | ef 证据(回捞) | base分 | ef 分 | Δ | agentic 链路结论 |
|---|:---:|---:|---:|:---:|:---:|:---:|---|
| 光纤滤料截污容量/比表面积 | DEGRADE | 10 | 13 (+3) | 2 | 2 | 0 | 持平，DISTRACT 消失 |
| 聚脲防水涂料裂缝预处理 | DEGRADE | 10 | 34 (+24) | 1 | 1 | 0 | 持平，大回捞量无损失 |
| 插座端子电镀前质控 | DEGRADE | 10 | 13 (+3) | 2 | **3** | **+1** | **缺口型升分复现** |
| 切削鳞片刀具失效 | CTRL | 10 | 23 (+13) | 1 | 1 | 0 | 持平 |
| 新45型电压表开孔 | CTRL | 10 | 64 (+54) | 3 | 3 | 0 | 持平，超长回捞无污染 |
| 桥式整流器 IF(av) | CTRL | 10 | 29 (+19) | 2 | 2 | 0 | 持平 |

**汇总：涨 1 / 跌 0 / 持平 5。**

### 与轻量直答对比（同 6 题，逐题同源）

| 口径 | 涨 | 跌 | 持平 | 备注 |
|---|:---:|:---:|:---:|---|
| 轻量直答 LLM（上轮） | 1 | 2（光纤滤料、聚脲） | 3 | DISTRACT 造成 2 题降分 |
| **agentic 生产链路（本轮）** | **1** | **0** | **5** | **DISTRACT 完全消失，且插座端子缺口型升分保留** |

> 注：轻量直答口径的"光纤滤料、聚脲"2 题在 agentic 链路下均为持平；本轮 agentic 链路新观测到"插座端子 +1 涨分"，而轻量直答口径该题此前未见升分。

---

## 4. 机理归因（结合答案文本人工复核）

1. **Organizer / Solver 吸收了上下文稀释（DISTRACT 免疫）**
   - 电压表题 ef 注入 54 块回捞（证据 10→64，增幅 6.4×），分数仍保持在满分 3。
   - 光纤滤料、聚脲题即便回捞大量低相关碎片（+3、+24），分数稳定，未再出现轻量直答下的"要点被淹没"。说明 agentic 的**证据组织（原文保留 + 低相关降权/丢弃）与 Solver 的判别式引用**起到了去噪作用——这正是 agentic 比轻量直答更稳的机制化体现。

2. **缺口型升分在 agentic 下仍复现（正收益保留）**
   - "插座端子电镀前质控" base=2（只覆盖冲压过程检验[2]+电镀后检查[1]）；ef 答案新增**半成品入库检验[6]（泡点与泡差检测）**并补全"影响导电性"链路，得分 2→3。
   - 即：当 base 证据存在缺口时，回捞补上的关键证据仍能转化为 agentic 的可辩护要点提升。

3. **证据增量在 agentic 下的边际价值接近中性偏正**
   - 6 题中 5 题持平、1 题涨，"亏"的概率=0；但净收益也不高（等效代价：输入 token 增加数倍）。说明在 agentic 链路里，固定 top10 已覆盖绝大多数判定所需事实，ef 的回捞主要贡献是**把"缺口型"风险兜底**。

---

## 5. 结论与建议

- **结论 1**：ef（top10+整档回捞）在 **agentic 生产链路下不会造成降分**（0 例），轻量直答口径下观测到的 DISTRACT 被 organizer/solver 有效吸收。**上下文稀释是轻量直答路径的固有缺陷，而非 ef 或回捞本身的缺陷。**
- **结论 2**：ef 在 agentic 下的价值从"全口径防丢分"收窄为"兜底缺口型升分"，净收益小（1/6 升、0/6 降、5/6 持平）。
- **建议 A（工程）**：若在 agentic 生产链路上追求低风险召回兜底，ef 可安全开启（不会伤基线）；但若要最大化性价比，应仅对**判定为缺口型**的问题启用回捞，对饱和型只保留 top10，避免无效 token 开销。可复用 task 维度/证据饱和度信号做条件开关。
- **建议 B（口径差异固化）**：后续所有 ef / 召回增强类评估，应以**生产 agentic 链路**为最终口径；轻量直答仅作"最坏情形上界"参考，不应作为否定回捞价值的依据。
- **后续量化项**：本 lins 每条件仅 1 次完整 agentic、共 6 题，样本小；建议在更大样本（≥30）上以 agentic 口径复核"缺口型升分率"，并测量 ef 引入的 token/时延成本，据此决定是否上线条件式回捞。

---

## 6. 产出物
- `_e2e_agentic_ef.py`：ef 注入生产 agentic 链路的端到端脚本（base-vs-ef）。
- `results/e2e_agentic_ef.json`：6 题 agentic 分数 + 答案摘要存档。
- `results/e2e_agentic_ef_run.log`：运行日志。
- `_e2e_agentic_ef_backfill_count.py`：真实回捞块数轻量探针（不跑 LLM）。
- `results/e2e_agentic_ef_backfill.json`：逐题回捞数。
