# S4→S5 组织消融 · 真实 exp1_agentic 30 题全量对照（裁决复测修正版）

> 结论一句话：**上一版 +0.167 的结论被证伪。修正注入点后的同期干净复测，
> organize_v3（RerankTruncOrganizer）avg 1.700 vs base（EvidenceOrganizer）avg 1.667，
> 仅 **+0.033（升5/降3/平22）**，落在 base 自身 ±0.033 抖动区间内，**统计不显著**。

## ⚠️ 方法论关键修正（为什么 +0.167 是假的）
- **上轮注入点无效**：`AdaptiveAgenticPipeline` **没有 `self.organizer`**（组件挂在内部
  `GraphExecutor.executor.organizer`）。上轮 `_exp1_agentic_rag_opt.py` 只写了
  `eng._pipeline.organizer = new_org`（死属性）+ `eng._organizer = new_org`，**真实执行
  路径 `executor.organizer` 实际仍是 `EvidenceOrganizer`**。
- 所以上轮 organize_v3 根本没切换组织器，+0.167 是 **LLM 采样随机噪声**（id2 案件印证：
  base 本轮 HDMI 题得 3 分、上轮得 1 分，纯覆盖率阈值边界抖动）。
- **本轮已修正**：`eng._pipeline.executor.organizer = new_org`，并实证 executor.organizer=
  RerankTruncOrganizer（见复测日志 `_recheck_log.txt`）。

## 复测口径（同期干净对照，剥离随机）
| 组 | organizer | avg | semantic_hit@1 | SV |
|---|---|---|---|---|
| **base** | EvidenceOrganizer | **1.6667** | 0.4667 | 0 |
| **organize_v3** | RerankTruncOrganizer | **1.7000** | 0.4667 | 0 |
| Δ | — | **+0.033** | 0（检索未动） | 0 |

- 样本：同题库前 30、同顺序、同 `run_agentic_rag_experiment(agentic_rag)` 脚本（base）与
  `run_optimized_experiment(organize_v3)`，同 EnhancedScorer rule 口径，dense 检索不变。
- 存放：`results/experiments/experiments/recheck_base/`、`recheck_ov3/`（逐题 JSON）。

## 逐题 diff（recheck: base → ov3）
- 升 5：id9、id21、id24、id25、id27（各 +1）
- 降 3：id2、id7、id18（各 -1，其中 id7 上轮还是"升"，进一步证明逐题抖动随机）
- 平 22
- **净 +0.033**，且升降题与上轮（±6/…/±1）**完全不同** → 逐题结果不可复现，属采样噪声。

## id2 降分机理（第一步排查定性）
- **排除** "词法重排把题干复述噪声顶前 / 截断误伤真证据"：base 与 ov3 的
  `matched_count` 均为 3、semantic 检索指标 4 位小数全同 → 组织器对 id2 证据输入近似等价。
- **根因**：题目问"标准HDMI 的物理尺寸/结构优势"，**证据池本就不含该知识点**（base 也
  没答出 14mm/4.5mm）。RuleBasedScorer 阈值 `coverage≥0.12→1分`；BASE 答案含建设性选型
  推断（螺丝锁定/法兰固定/应力释放）coverage 略过 0.12，OV3 更防御式"证据不足"跌破阈值。
  属 **0.12 覆盖率阈值边界的措辞随机**，非系统性组织伤害（本轮 ov3 id2 得 3 分、base id2 得 1 分，
  完全反向上轮 base 1 / 上上轮 ov3 0 的组合）。

## 判据结论（修正后）
- **S4 词法重排+截断在 30 题上不产生统计显著增益**（+0.033，base 固有抖动 ±0.033）。
- **也不构成系统性伤害**（净正值、平 22 题、SV 0、检索/Semantic 全同）。
- 上轮"组织侧 ROI 高于检索侧、可回收 ~0.17"的说法**不再成立**，应撤回。

## 生产处置
- 生产 `experiments/exp1_agentic_rag.py` 装配点已落地 `RerankTruncOrganizer()`
  （`agentic/organizer.py` 新增生产级类）。**因无系统性伤害、零成本**，保留，但**不宣称增益**；
  回退单点即可改回 `EvidenceOrganizer()`。
- ⚠ 若未来要再评估组织侧，必须：① 用 `executor.organizer` 正确注入；② 更大样本或 ≥3 轮
  取均值，避免落入 base 固有抖动。

## 建议
1. **组织侧 ROI 证伪** → 不再优先投入 S4 词法重排/截断调参。
2. 生产保留 RerankTruncOrganizer 作为**中性零成本**选项，但不对业务宣称提升。
3. 若要继续，改看 S3 检索侧（hybrid/mv，1.67 与 base 1.667 同水平，同为中性）或换更高 ROI 方向。
