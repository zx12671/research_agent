# LINS-Industrial 有效测试数据汇总索引
> 配套文档：`docs/stagewise_debug_plan.md`（逐阶段调试计划）、
> `docs/agentic_system_whitepaper.md`（系统全景白皮书）。
>
> 这份索引把整个优化过程中**有效且重要**的测试数据文件分主题整理，避免"报告和数据
> 文件太多、找不到关键结论"。每一条给出：**结论一句话 + 数据文件 + 复现脚本 + 状态**。
> 状态图例：✅定案（生产依据）/ 📌参考（中间佐证）/ 🗑已废弃（负优化，留作反面）。

---

## 主题 A. 检索 recall 与 run2 两步式（最核心）

### A1. run2 两步式 vs 现状三方法（single/multi/hybrid）—— 起点
- **结论**：run2 两步式（LLM 重排 + 反向回捞）cov@10 显著高于 best baseline（single/hybrid）
  ~35%→41%，机制＝**补全 > 排序**。
- **数据**：`results/run2_vs_s4/run2_vs_s4_20260808_220121.{md,json}`（✅定案起点）。
- **复现**：`python _diag_recall_run2.py --per_cap 4 --pool 60 --sim`。

### A2. run2 变体迭代 v1→v2→v3→v4（21 题，检索层）
- **结论**：v1≈v2；**v3（3 篇全吃 bonus）是负优化**（cov −1.8pp/good −4.8pp，低相关文档靠
  bonus 硬挤 top10）；**v4（bonus 只给 top1）回正**（Δcov +2.2pp、ΔNDCG +2.6pp）。
- **数据**：`results/run2_v1v2/run2_v1v2_20260809_133116.{md,json}`（✅定案）。
- **复现**：`python _diag_run2_v1v2.py --per_cap 3 --pool 50 --llm`。

### A3. 7 路检索 recall，**50 样本真实 LLM（最终定案）**
- **结论**：**v4 是 7 路唯一五项（cov/good/hit1/MRR/NDCG）全面第一**——
  cov@10 46.7% / good 48% / hit@1 56% / hit@3 72% / MRR 63.0% / NDCG 65.7%。
  两步式整体碾压三步基线；multi 最差（MRR 45.2%，fusion 稀释）；v3 依旧不如 v4。
- **数据**：`results/agentic_recall_7way/agentic_recall_7way_20260809_162441.{md,json}`（✅定案）；
  `agentic_recall_7way_20260809_161618.{md,json}` 📌=Sim 冒烟对照。
- **复现**：`python _ab_agentic_recall_7way.py --n_total 50 --llm`（≈24min）/ `--sim`（快）。

### A4. 生产端到端落地（base vs run2-v4，14 题）
- **结论**：v4 端到端 **涨3/跌0/持平11**，平均分 base 2.50→v4 2.71（Δ+0.21）；证据数全=10
  零回退。生产 `Run2AgenticRAGEngine` 默认 v4。
- **数据**：`results/e2e_agentic_run2/e2e_run2_v4_20260809_155628.json`（✅定案）。
- **复现**：`python _e2e_agentic_run2.py --per_cap 2 --pool 50 --seed 7 --version v4`。

---

## 主题 B. 前端任务/题型判定（analyze 读 format）

### B1. 题面题型信号缺失
- **结论**：2049 条题面显式题型信号全部≈0（无填空符/选项字母/99.8%+疑问句收尾）
  → 真人/LLM 都无法从 question 反推 `_format`。
- **数据**：`results/probe_format_signal_missing/format_signal_missing.json`（✅定案）。
- **复现**：`_probe_format_signal_missing.py`。

### B2. format 误读 / LLM 判定
- **结论**：LLM 判题型仅 62.5%（10/16）、加权 F1 56.3%、填空 4 判 0；**读 CSV 标注是唯一可靠**。
- **数据**：`results/probe_format_misread/format_misread_results.json`、
  `results/frontend_judgment/frontend_judgment_20260808_000813.{md,json}`（✅定案版本，其余时间戳📌）。
- **复现**：`_ab_format_first.py`（已落地 `format_source=annotated`）。

---

## 主题 C. 证据前向（EF）与证据利用

### C1. 证据利用/退化机制
- **结论**：调低/关闭前向证据会显著降分；EF 的收益靠"把检索证据前向喂给推理"，这是
  agentic 端到端能力的关键（与 run2 的"证据是短板"结论互相印证）。
- **数据**：`results/diag_evidence_utilization.json`、`diag_evidence_utilization_35.json`（📌）、
  `results/diag_degrade_mechanism.json`（📌）。
- **复现**：`_diag_evidence_utilization.py`、`_diag_degrade_mechanism.py`。

### C2. EF 端到端 + 条件 EF（35 题）
- **结论**：无条件下 EF 全量回捞可能引入噪声；条件 EF（门控）在 35 题对均分有正贡献，
  但增益有限且需门控信号。作为 run2 出现前的证据增强阶段，最终被 run2 两步式取代。
- **数据**：`results/ab_ef_pairs_35.json`（📌）、`results/e2e_agentic_ef.json`（📌）。
- **复现**：`_e2e_agentic_ef.py`、`_ab_conditional_ef_35`、`_e2e_agentic_ef_35_pairs.py`。

---

## 主题 D. 证据组织与合并（S5）

### D1. merge 上下文膨胀
- **结论**：修复前 `accumulated_context` 膨胀≈2.63×（100% 样本≥1.5×）；修复后至 1.04×/
  0%≥1.5×，RAW-organized 重叠 0，organized 全量 100% 保留（Recall-Safe）。
- **数据**：`results/s5_merge_inflate.json`（✅）vs `s5_merge_inflate_BEFORE.json`（📌修复前基线）。
- **复现**：`_diag_merge_inflate.py`。

### D2. organize 变体对比
- **结论**：organize v3（分组+去重+词法置顶/截断）vs base 组织是同源检索下的对照，
  主要价值在回收"证据在但失焦"的传导缺口；结论汇总 `ab_organize_v3_vs_base_30.md`。
- **数据**：`results/experiments/ab_organize_v3_vs_base_30.md`（📌，其内部 recheck_base/
  recheck_ov3 逐题 JSON 已随 `experiments/` 归入 `results/_archive/experiments/`）；
  `results/_archive/experiments/ab_organize_util_10.json`（📌，20 样本组织消融）。


---

## 主题 E. 检索召回诊断（S4）

- **结论**：B 类（标准号）题召回薄弱由"语料/标签口径"主导而非纯排序；multi_query 门控
  可校准；诊断结论汇总自 `docs/s4_path3_diagnosis_report.md`、`docs/agentic_recall_*_diagnosis.md`。
- **定案数据**：`results/s4_recall/s4_recall_20260807_235012.md`（✅最终，其余 `s4_recall_*`
  时间戳📌）；关键探测 `s4_evidence_audit.json`、`multi_gate_calib.json`、
  `bn_std_spec_focus.json`（📌）。
- **复现**：`_diag_recall_s4.py`、`_audit_s4_evidence.py`、`_calibrate_multi_gate.py`。

---

## 主题 F. 评分归因（S6）

- **结论**：低分样本逐题归因到 S4(检索缺料)/S6(生成组织)；规则分对应内容覆盖而非措辞。
- **数据**：`results/s6_score/s6_score_20260808_101616.{md,json}`（✅）。
- **复现**：`_diag_s6_score_attribution.py`。

---

## 主题 G. 其他 / 归档说明

### G1. 仍可能用到的中间数据（📌 参考/已并入定案）
- `results/task_analyzer_classification_eval.json`（analyzer 分类评估）、
  `results/ab_bm25_jieba.json` / `ab_bm25_jieba_grid.json`（BM25 权重网格）、
  `results/ab_mv_query_rewrite_*.json`（multi-view 改写，结论=默认关闭）、
  `results/planner_stats/*`（planner 成本审计）、`results/exp X 结论见白皮书 §9`。

### G2. 历史一次性运行产物（已归档到 `results/_archive/`，非删除）
- **`results/experiments/` 现只保留 4 项有效内容**：
  `exp1_agentic_rag_20260801_223628`、`exp1_agentic_rag_fix`（被框架文档引用的端到端
  定案运行）、`organizer_ab_fixed`（`experiments/ab_fix_stats.py` 生成的修复统计）、
  `ab_organize_v3_vs_base_30.md`（D2 结论文档）。
- **其余 34 个历史时间戳运行目录已移入 `results/_archive/experiments/` 完整保留可回溯**
  （`exp1_qa_*`、早期 `exp1_agentic_rag_2026xxxx`、`organizer_ab`、`recheck_*`、
  `regress_*`、`exp1_promptfix_30`、`exp1_qa_closedbook_baseline_01`、以及
  `experiments/experiments/` 嵌套重复中的 `bm25j30`/`opt_*`/`recheck_*`）。
- **归纳**：日常查阅只看 `results/experiments/` 这 4 项 + 本文档各主题的 ✅定案文件；
  历史原始逐题数据全部在 `_archive` 下，需要时再翻。

