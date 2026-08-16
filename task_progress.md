# Agentic 项目系统分析任务清单

## 后续排查总原则（分而治之 · 分阶段可量化）
- 把系统拆成 S1~S6 六个独立阶段，每阶段设可量化指标与处置阈值，见
  **`LINS-Industrial/docs/stagewise_debug_plan.md`**。
- 只对指标最落后的阶段动手（当前判定 **S4 检索 Hit@1 0.5 → 目标 0.7** 为首要优先），
  修完重测该阶段 + 端到端再进下一阶段；**不做盲目多路试错**。

## 任务S4-1：检索基线诊断（局部完成 · 判定 FAIL）
- [x] 新增 `LINS-Industrial/_diag_recall_s4.py`（官方 `RetrievalEvaluator` + capability/industry 分层 + single/multi/hybrid 对比 + k-sweep + IOU 命中判定）。
- [x] 实测（56 样本，7 cap×8，IOU=0.15）：`results/s4_recall/s4_recall_20260807_215812.md`
  - single hit@1=**50.0%**、MRR=55.8%、NDCG=55.5%；**multi(26.8%)/hybrid(48.2%) 不优于 single**（增强负增益）；
  - 能力差异 max-min=**37.5%>20%** → **未达判定标准（FAIL）**；
  - 语料缺口：工程计算 chunk 1%、故障诊断 0%（无料可检）；标准规范 chunk 63% 却 hit@1 仅 50%（超饱和匹配差）；
  - k-sweep(hybrid)：@10=62%→@50=71% → GT 本就在 top10，缺口在检索本身而非截断。
- [ ] 下一步：a) 补工程计算/故障诊断语料缺口；b) 定位 KED multi 为何降分（先保 single 基线）；c) 标准规范排序/措辞失配。


## 任务1：analyze 直接读 format 落地 + 前端任务判定 ✅
- [x] `analyze(question, format=...)` 已具备直接读 format（normalize_format → TaskAnalysis.format），`_verify_format_analyzer.py` 验证 format 落位正确。
- [x] **生产链路落地**：`agentic/pipeline.py` 的 `run()` / `run_with_ablation()` 新增可选 `format=""` 参数并透传给 `analyzer.analyze(question, format=format)`。
- [x] **引擎落地**：`experiments/exp1_agentic_rag.py` 的 `AgenticRAGEngine.answer()` 新增可选 `format=""` 参数并透传至 `_pipeline.run/run_with_ablation`。
- [x] 兼容性：不传 format（空串）保持原 heuristic 行为；已用集成检查验证 run/run_with_ablation/answer 三层透传 + 空串兼容（ALL PASS）。
- [x] 前端任务判定量化：`_probe_frontend_judgment.py` 新增「落地后走查」维度（analyze 直读 `_format` 真值），与 heuristic/LLM 对比 format 判定准确率（LLM 判 vs ground truth）。新结果：`LINS-Industrial/results/frontend_judgment/frontend_judgment_20260807_213906.md|json` —— 落地后 format acc=100%，生产 heuristic≈29%（落地前），LLM 判定本身 acc<80% → 前端 format 应优先靠 CSV 真值。
- 相关文件：`_ab_format_first.py`、`_probe_format_misread.py`、`agentic/analyzer.py`、`agentic/pipeline.py`、`experiments/exp1_agentic_rag.py`。

## 任务S6：评分/输出阶段检查（评分误伤清洗 + 低分能力域归因）✅
- [x] 新增 `LINS-Industrial/_diag_s6_score_attribution.py`（零 LLM、确定性、可留痕），产物 `results/s6_score/s6_score_20260808_101616.md|json`。
- [x] **动作1 评分误伤清洗**：rule_score 与 cov 单调对应（真实 6 题 score3→cov0.71 / score2→0.46~0.55 / score1→0.07~0.09）；「实体正确/转述不同」题（纤维滤料 ent_cov=1.00、score2）判 Acceptable(2) **不误伤**；对 35 题中 6 个低分样本构造「正确重述答案」**6/6 重述后均判≥2**（0 误伤候选）→ 规则分容忍换措辞、只反映内容覆盖。
- [x] **动作2 低分归因**：故障诊断2/质量计量2/安全合规1/标准规范1 共 6 个低分(≤1)样本，用完整 question 本地 top10 检索，`sent_coverage(ref, evidence)` 全部 **=0.00**（连简短答案句都没拼齐）→ **全部归因召回侧缺料（S4 治理域）**，S6(证据足但产出差) 0 题。
- [x] **S6 判定『通过』**：低分能力域主因归 S4（证据前向/multi-query/query 改写已在 S4 治理）；S6 评分口径与生成表述无系统性误伤。


## 阶段一：代码结构梳理
- [ ] 阅读 README 及核心架构文档
- [ ] 梳理 agentic/ 目录各模块（task_types, planner, solver, organizer, prompt_builder, prompts, pipeline）
- [ ] 梳理 retrieval/ 及 metrics/ 支撑模块

## 阶段二：执行流程测试
- [ ] 测试 pipeline 完整流程（任务分析 -> 检索 -> 求解 -> 输出）
- [ ] 测试各 agentic 组件单独功能
- [ ] 追踪从输入问题到最终输出的数据流转

## 阶段三：量化分析
- [ ] 运行实验（exp1_agentic_rag 等）
- [ ] 对比各组件对最终评分的影响
- [ ] 量化执行流程各阶段的耗时/调用次数/效果

## 阶段四：汇总报告
- [ ] 输出完整的功能/流程/量化分析报告
