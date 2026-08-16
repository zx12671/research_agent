# Agentic 后续排查与优化方案（分而治之 · 分阶段可量化）

> **核心原则**：把难题拆解为**相互独立、可单独量化**的阶段，对每一阶段设定明确的
> 准确率/得分指标与处置阈值。排查只针对指标最落后的阶段动手（"哪个阶段拖后腿就查哪个"），
> 修完一个阶段再重测整体——**不做盲目多路试错**。每阶段改动都必须能复现"改动前/后"对比。

---

## 0. 分阶段总览（指标口径与当前基线）

Agentic 系统在生产中是一条六阶段链路。**端到端分低 ≠ 某一处坏**，必须逐段归因：

| 阶段 | 组件 / 入口 | 可量化指标 | 当前基线（实测来源） |
|---|---|---|---|
| **S1** 任务/题型分析 | `TaskAnalyzer.analyze` | format 判定 acc；task 判定 acc/F1 | format 落地后 **100%**（读 `_format` 真值，`frontend_judgment`）；task **恒 general**（中性化，口径不适用） |
| **S2** 策略规划 | `StrategyPlanner.plan` | 图谱类型、LLM 调用次数、超时回退率 | 高级图 2~4 次生成调用；存在 LLM 出图失败回退模板 |
| **S3** 图执行 | `GraphExecutor` | reason/decide/verify 节点实际命中、solver 接入率 | reason 走 `PromptBuilder` 而非 `TaskSolver.solve()`（solver 未接入） |
| **S4** 检索 | `Retriever` / `hybrid_retrieve` | semantic Hit@1/3、MRR、NDCG、recall@k | Hit@1 **0.46~0.55**；Hit@10 0.51~0.65；NDCG 0.77~0.90 |
| **S5** 证据组织 | `EvidenceOrganizer` | 去重保留率、上下文完整度、prompt 膨胀度 | 组织保留完整 chunk；`merge` 可能膨胀 prompt |
| **S6** 评分/输出 | `RuleBasedScorer` | rule_score(0-3)、cov、ent_cov、struct | avg **1.55~1.71/3.0**；SV=0；"标准规范与术语/安全合规"偏弱 0.50~1.38 |

**分而治之的查询路径**：整体 avg rule 低 → 依次检查 S4（证据是否召回）→ S5（证据是否进 prompt）
→ S6（评分是否对"答案对但表述不同"误伤）；S2/S3 只有在 S4/S6 都达标后再看（否则是"包装问题"）。

---

## S1 · 任务/题型分析 ——  目标：format 判定稳定可达

- **指标**：format acc / 加权F1；task acc/F1（LLM 判 vs ground truth）。
- **当前基线**：format 落地后=100%（读 `_format` 真值归一化）；LLM 判 format acc<80%（不可靠）；
  task 因 analyzer 中性化恒 general（该维度已决定不参与行为决策）。
- **排查动作**：
  1. 无需再优化 format（真值直读是最优解）。若某场景拿不到 `_format`，才回退。
  2. 确认生产链路`pipeline.run / AgenticRAGEngine.answer` **确实把 `_format` 传进 analyze**（任务1已落地，用 `frontend_judgment` 复查）。
- **判定"通过"**：`_probe_frontend_judgment.py` 跑出 `落地后 acc=100%`。
- **现成工具**：`_probe_frontend_judgment.py`、`_verify_format_analyzer.py`、`_ab_format_first.py`。

## S2 · 策略规划 ——  目标：图谱选择与调用成本可控

- **指标**：图类型分布（简单/高级）、每样本 LLM 生成调用次数、LLM 出图失败/回退模板率、plan 耗时。
- **当前基线**：高级图（comparison/selection/diagnosis/standard_interpretation）≥2 次生成调用 + 可能 decide/verify；简单图 1+1 次。
- **排查动作**：统计 2049 题的图类型分布与超时回退率；定位"回退模板却仍耗时"的极端样本。
- **判定"通过"**：回退率≈0；生成调用次数与图类型一致（无冗余二次检索）。
- **现成工具**：`_system_flow_quant.py`、`_perf_fullwalk.py`（耗时/调用统计）、`results/planner_stats/`。

> **—— S2 量化留痕（20260808 · 检验完成 · 判定：✅ 达标，P0/P1 均已落地）——**
> **阶段结论**：真实 LLM 下**图分布 100% 高级、回退率 0**、LLM 调用固定 **4/题（planner+reason+verify+final）**，
> retrieve 成本与图类型**解耦**（非 LLM 调用，走 FAISS/BM25，按需触发）。唯一真冗余项 = 主检索在
> 注入 `pre_retrieved_evidence` 后仍重复执行 —— 已由 **P0 短路**修复。S2 不再列为阻塞项。
>
> **① S2 审计（60 次真实执行 · `_recheck_err.txt` · `_s2_scan.py`）**
> - 图类型：**100% 高级（60/60 含 `verify_1`）、0 纯简单线性图、0 fallback、0 decide**；5 种模式皆 LLM 动态生成。
>   `retrieve_2` 规划 87%、真实触发仅 40%（紧跟 verify fail，**按需非冗余**）。
> - LLM 调用：恒 **4/题** = `planner(1)+reason(1)+verify(1)+final(1)`。
> - **关键缺陷**：`retrieve_1` 执行 60/60——即便已注入 `pre_retrieved_evidence`（日志仅"文案式 skip"，实际重跑检索）→ 冗余主检索。
>
> **② P0 修复（已实施并单测 4/4 通过 · `pipeline.py` L367-408 SHORTCUT 块）**
> - `GraphExecutor._handle_retrieve` 新增短路段：主检索（`retrieve`/`retrieve_1`，`not is_followup`）遇
>   `__external_retrieval__` 即短路、直接消费注入证据 → **消除 60/60 冗余重检索**；follow-up `retrieve_2` 保留
>   （verify-fail 召回）；证据按 `retrieve_k` 截断维持 k 语义。验证：`_s2_p0_verify.py`（隔离单测，零模型依赖）。
>
> **③ P1 数据补全（50 样本真实 DeepSeek LLM，20260808 · `_s2_p1_real50.py` → `results/s2_p1_real50.json`）**
> - 图分布：**advanced=50/50（ratio 1.0）、simple=0、error=0** —— 与 60 样本审计结论一致。
> - **fallback 率 = 0.0**（真实 LLM 出图零回退模板）。
> - 耗时：**plan 平均 4027ms/题、run 平均 14110ms/题**；图平均 **4.96 节点/题**（advanced 5 节点为主）。
> - LLM 调用：**固定 4/题**（`planner=50`、`reason=100`、`verify=50`；reason 2 次、verify 1 次、planner 1 次）。
> - **P0 真实链路复验**：每题日志均出现 `[retrieve_1] SHORTCUT: using pre-retrieved evidence ... skipped internal retrieval`，
>   `evidence keys` 保留 `__external_retrieval__` —— 主检索短路真实生效，无回退/错误。
> - **口径修正**：S2 指标从"LLM 调用次数"修订为"**总成本 = LLM 调用(恒 4±0) + 检索次数(retrieve_1 短路后=1，另按需 retrieve_2)**"。
>
> **处置**：S2 **通过**。2049 题全量无需再跑真实 LLM（60+50 样本图表结构/回退/耗时均已收敛且一致）。
> 建议后续条目：① follow-up `retrieve_2` 的 evidence 聚焦/限量（先前 件③ 决策待定）；② 输出格式对"标准规范/安全合规"题型的评分偏弱可留作 S6 观察。留痕文本格式与 S3/S4 对齐。

## S3 · 图执行 ——  目标：每个节点确实执行预期逻辑

- **指标**：reason/decide/verify 节点命中率、evidence→prompt 传递完整率、`TaskSolver` 接入率。
- **当前基线**：reason 实际走 `PromptBuilder.build() → format_prompt("general")`，**未调用注入的 `TaskSolver.solve()`**（solver 独立未接入）。
- **排查动作**：
  1. 决定 `TaskSolver` 是否应接入 reason 节点（否则 solver 是死代码，耗费维护）。
  2. ~~检查 `merge` 是否把 organizer 上下文 + 原始 chunk 都并入导致 prompt 膨胀~~ **（✅ 20260808 已修复落地：`_handle_merge` raw 段按 chunk_id 跳过已被 organized 覆盖的 chunk，膨胀 2.63×→1.04×、重复率 1.0→0.0、0 缺失；数据见 S5/`results/s5_merge_inflate.json`+`_BEFORE.json`）**。
- **判定"通过"**：存在明确方案（接入或移除 solver）；prompt 长度与 evidence 长度比例受控（<阈值，见 S5）。**merge 项已达标**。
- **现成工具**：`_system_flow_quant.py`、`experiments/exp1_agentic_rag.py`（图执行日志）。

> **—— S3 量化留痕（20260808 · 检验完成 · 判定：达标，选"不接入 TaskSolver"——）**
> - 检验脚本：`_diag_s3_executor.py`（真实 faiss + MockLLM 走完整 `AdaptiveAgenticPipeline.run`，注入"任何方法被调即抛异常"的 `ThrowingSolver` 刺探；12 题样本，可复现）→ `results/_diag_s3_executor.json`。
> - **节点实际命中率/题 = retrieve:1.0 / organize:1.0 / reason:1.0 / end:1.0**（简单图；decide/verify/merge 仅在 advanced 图触发，属预期）；**LLM 调用点 reason=2 次/题（reason 节点真实执行，走标准 prompt+PromptBuilder）**。
> - **TaskSolver 接入率 = 0.0**：`ThrowingSolver` 全程未被触碰 → 证明 `_handle_reason`/全部处理器从不调用 `TaskSolver.solve()`；且 `pipeline.py` 中 `self.solver` 仅出现在 `__init__` 赋值（L104），无任何引用点 —— solver 是被注入但从不通电的死代码。
> - **裁决（满足"存在明确方案=接入或移除"）**：**选择"不接入"**。理由：① 现有 reason 标准 prompt+PromptBuilder 已过 S6 验收（评分公平、低分主因归 S4 召回，端到端瓶颈不在 reason prompt）；② `TaskSolver._solve_general` 系统提示比标准 prompt 更简陋（无输出格式约束），`_solve_with_workflow` 依赖 `directive.params.workflow_steps` 而模板图 reason 仅给 `["synthesize"]` —— 接入反而有换掉已验证 prompt 的降级风险。**处置**：`TaskSolver` 保留为"可选、未来多步推理再做"的模块，不强制注入、不接入 reason；在 `solver.py` 顶部标"未接入"说明（防误用/误维护）。merge 项已于 20260808 修复达标。
> - **S3 判定 = 通过（明确方案已确认；merge 达标；prompt 受控）**，不再列为阻塞项。


> **—— run2 两步式检索（EF 改进版）与 baseline 三方法对比留痕（20260808 · 数据已存档）——**
> - **动机/方法（构造可复现）**：run1=当前 agentic 三检索（single/multi/hybrid）；run2=两步法：① 现有 dense 初步检索取宽池 pool_k=50；② 真实 DeepSeek（`deepseek-chat`）**仅凭【问题+候选块】**逐块给 0-1 适配分（prompt 绝不注入 answers/knowledge_text/GT，规避"直接读答案"），取高相关锚块的 `document_id` **反向回捞同源文档其余块**（`chinese_jt(query,块)>=0.02` 过滤防噪，复用 EF 口径），再融合排序 top-k。实现：`retriever.py::OpenDomainRetriever.two_stage_retrieve`（新增）+ 可注入 `RelevanceRanker`（`LLMRelevanceRanker` 真实打分 / `SimRelevanceRanker` 零成本兜底）+ 探针 `_diag_recall_run2.py`。
> - **实测（21 题，7 capability×3，seed=7，pool=50，主口径 cov@10）**：**run2_llm=41.6%/good38%** > hybrid 35.7%/29% > single 33.3%/24% > multi 32.5%/24%；较 best baseline(single/hybrid)**+5.9pp**；交叉参照 hit@1=43%、hit@3=52%、MRR=48.4%（各行最高/并列最高）。数据见 `results/run2_vs_s4/run2_vs_s4_20260808_220121.md+.json`。
> - **机制归因（补全>排序）**：零成本对照 run2_sim=39.1%（仍高于全部 run1）→ **文档级反向补全本身即带来增益，LLM 打分再叠加**（+2.5pp）。按 capability 最大收益在历史短板：工程计算 cov@10 27%→44%（+17pp）、故障诊断 35→46%、选型与替代 24→39%；无任何能力回退低于 baseline。
> - **成本**：21 题级 LLM 打分，平均每题候选 50 块、累计 prompt chars≈159k；每题约 +一个 LLM 调用点（远小于生产 agentic 每题固定 4 次 LLM），可接受。
> - **（续·20260809）run2 v1 缺陷实证 + 方案B v2 小样本对比**：
>   - **缺陷确认（set 级必然性）**：`two_stage_retrieve`（v1）里 `ranked=ranker.score(query,pool_chunks)` 只给【50 池块】打分；而回捞块 `backfilled` 被第二步B 定义为"不在池内"（`if cid in pool_ids: continue`）。于是 `backfilled ∩ ranked.keys() = ∅`（集合论必然），`ranked.get(cid,0.0)` 对每个回捞块恒 0 → 回捞块融合分固定 `1.0/110+0+0.5=0.5091`，永远挤不进 top10（5 题实证 100% 落榜）。
>   - **方案B（v2）**：新增 `two_stage_retrieve_v2`，调整顺序为「捞池→B先回捞→把【池块+回捞块】合并为一组候选→统一 `ranker.score` 给全体候选（含回捞块）真实 rel→融合」，回捞块得以带 rel 参与竞争。代价：ranker 打分 50+N 块，成本略升。
>   - **小样本实证（pool=50, n_anchor=3, k=10）**：
>     - **Sim 零成本 ranker**：回捞块 v1 进 top10 累计=0；**v2 累计=11**（id=184 进6、1077 进4、32 进1；1369 仍 0——rel 全非零(0.37~0.58)但 top10 门槛被池内锚块顶到 1.1585，属"回捞块 rel 不足以越过门槛"的合法挤压，可调 w_anchor/w_rel）。v2 下回捞块 rel 全覆盖且非零（0.35~0.92），证明打分覆盖缺陷已修复。
>     - **真实 LLM ranker**：暴露第二个瓶颈——默认 `LLMRelevanceRanker(max_tokens=900)` 对 56~64 块（50池+回捞）的长 prompt 输出易被截断，回捞块在 prompt 末尾丢分（两次同参运行 rel 覆盖不稳）。**将 `max_tokens` 提到 2000 后实测 cand=60 全覆盖（回捞 10/10 都有分）**。生产侧推进方案B 需同时放大 `max_tokens` 或对候选**分批打分**。
>   - **结论**：方案B（v2）在机制上确能修复"回捞块 rel=0"缺陷并让回捞块进 top-k；在真实 LLM 下须配合放大 `max_tokens`（或分批）才稳定生效。探针可复现：`_diag_run2_v2_ab.py`；v2 实现 `retriever.py::two_stage_retrieve_v2`。
>   - **（续·20260809）v1 vs v2 检索层 recall 指标（与 S4 同口径，21 题=7cap×3, seed=7, pool=50, LLM ranker max_tokens=2000，探针 `_diag_run2_v1v2.py`）**：
>     - **结论先行：v2 修复机制缺陷但 recall 指标无增益**。v1=v2 在全部主口径上几乎完全持平，MRR/NDCG 微降。明细（k=10）：
>       - cov@10：v1=41.6% / v2=41.6%（**Δ+0.0pp**）
>       - good%：38% / 38%（+0.0pp）
>       - hit@1：43% / 43%（+0.0pp）；hit@3：52% / 52%（+0.0pp）
>       - MRR：48.5% / 48.4%（**-0.1pp**）；NDCG：50.2% / 49.9%（**-0.2pp**）
>     - **机制解释**：v2 让 15 个回捞块真正进入 top-10（v1=0），但**这些"池外同源边角块"本身的 rel 中等（0.2~0.6）、进榜位置靠后（多位于 6~10 名），且极少命中 GT 答案句子** → 只抬升 top10 尾部覆盖，对 句子级主口径 cov@10/good% 与前置命中 hit@1/@3 零贡献；反而因它们挤占中低位、把原本排更前的池块微降，致 MRR/NDCG 微回落。
>     - **落地判断**：v1 与 v2 在检索层 recall 上**等价**（v2 不 worse，也不更好）。v2 的增量仅"回捞块进 top10 的机制正确性"，无 recall 指标兑现。故**方案 B 不作为 run2 检索层的落地收益点**；若生产要采用，价值在"机制正确/可解释性"，而非 score 提升。数据存档 `results/run2_v1v2/`。
>   - **（续·20260809）锚文档去重迭代：v3 负优化 → v4 正增益（21 题 LLM，pool=50，探针 `_diag_run2_v1v2.py`）**：
>     - **v3（去重到 3 篇、3 篇全部吃 anchor_bonus）反而更差**：Δcov −1.9pp / Δgood −4.8pp / ΔNDCG +0.0pp。原因：3 篇文档全体块都 +0.5 bonus，低相关文档靠 bonus 硬挤 top10、稀释真正相关块，句子级主口径大跌 → 上一版"s广度=3篇"方向被证伪。
>     - **v4（去重到 3 篇回捞，但 anchor_bonus 只给第 1 篇/rel 最高那篇）正增益**：Δcov **+2.2pp**、ΔMRR +0.4pp、ΔNDCG **+2.6pp**、Δgood/hit1/hit3 +0.0pp；回捞进 top10=13（比 v3 的 20 更克制，因第 2/3 篇块只靠 rel 竞争）。分层抬升：工艺原理 52→59%、质量计量 67→72%、故障诊断 46→48%，无能力回退。
     - **机制**：把「回捞范围」与「排序加权」解耦——范围广（3 篇补全找漏网），权重严（仅 top1 吃 bonus、其余靠真实 rel 择优录取）= **recall 阶段广、precision 阶段严**（hierarchical）。v4 是 run2 检索层目前**唯一在 cov@10 上实质优于 v1/v2 的变体**。实现 `retriever.py::two_stage_retrieve_v4`；数据 `results/run2_v1v2/run2_v1v2_20260809_133116.md+.json`。
>   - **（续·20260809·生产落地）v4 接入生产 agentic 端到端 → 14 题 base vs run2(v4)（`_e2e_agentic_run2.py --version v4`）**：
>     - 给 `Run2ExternalRetriever`/`Run2AgenticRAGEngine` 增加 `version` 参数（默认 `"v4"`；`v1`=原 two_stage_retrieve），`retrieve()` 按 version 分派。CLI 加 `--version {v1,v4}`，输出文件名带 version。
>     - **端到端 A/B（per_cap=2, seed=7，真实 agentic 编排 + DeepSeek）**：**涨 3 / 跌 0 / 持平 11**；平均分 base 2.50 → run2(v4) **2.71，Δ=+0.21**。三题 +1：标准规范（2→3，补全50）、质量计量×2（2→3 补全26 / 1→2 补全46）。**14 题全部证据数=10 且无一降** → v4 的 anchor_bonus-only-top1 在端到端链路把检索阶段的正增益安全兑现为作答得分，零回退。生产端默认即 v4。数据 `results/e2e_agentic_run2/e2e_run2_v4_20260809_155628.json`。
>   - **（续·20260809·agentic 检索 7 路 recall，50 样本，探针 `_ab_agentic_recall_7way.py`）**：在生产 agentic 链路的底层检索方法上定量对比 **single/multi/hybrid/v1/v2/v3/v4**，指标 cov@10+good%+hit@1/3/10+MRR+NDCG，seed=7 抽 50 题（cap 均衡）。
>     - **Sim ranker（零成本，仅验证链路）**：**v4 全面领先** cov@10=**46.0%**/good%=**48%**/MRR=59.5/NDCG=**61.6**；v2=44.5%；v3=42.8%；v1=42.6%；hybrid=38.3%；multi=35.6%；single=36.2%。vs v1：Δcov=+3.5pp Δgood=+8.0pp ΔNDCG=+5.5pp；single/multi/hybrid 均落后 v1。注：Sim 下 multi 的 fusion 结果被嵌入打分失真（hit@1 32% 异常低），仅作冒烟。
>     - **LLM ranker（真实 DeepSeek，max_tokens=2000，50 样本 ≈24min）**：**v4 是 7 路里唯一在 cov@10/good%/MRR/NDCG/hit@1 上全面第一** —— cov@10=**46.7%**/good%=**48%**/hit@1=**56%**/hit@3=**72%**/MRR=**63.0%**/NDCG=**65.7%**；v1=45.5/42/54/68/61.3/61.4；v2=45.2/44/54/68/61.2/61.2；v3=42.8/44/56/68/62.5/60.4；hybrid=38.2/34/52/64/58.9/57.9；single=36.2/32/52/62/58.1/57.7；multi=35.6/32/32/54/45.2/50.7。vs v1：Δcov=+1.2pp Δgood=+6.0pp Δhit1=+2.0pp ΔMRR=+1.7pp ΔNDCG=+4.3pp；v1/v2 持平、v3 cov 回落(−2.6pp)、single/multi/hybrid 全部落后。**两步式整体碾压三步基线**（v1 就比 best baseline 高 ~7pp cov）；multi 最差（fusion 稀释）。v4 生产落地定案成立。数据 `results/agentic_recall_7way/agentic_recall_7way_20260809_162441.md+.json`。


## S4 · 检索 ——  当前**首要排查点**（Top-1 命中率不到一半）

- **指标**：semantic Hit@1/3、MRR、NDCG、recall@k；分 capability 分层召回。
- **当前基线**（更新至 20260807·22:16 实测，56 样本 / 7 capability×8 · 多口径并列）：
  - 主口径（句子覆盖率 cov@10）：**single=33.7%** / good%(拼齐率)=28.6%；`multi(24.1%)` / `hybrid(30.0%)` **均再低于 single** → multi 负增益在口径修正后更明确；**（此 multi/hybrid 24.1%/30.0% 为 20260807 改造前值；路径2 落地后 multi/hybrid 分别回 33.4%/36.0%，负增益被消除——见代码 retriever.py L584-588/L824-826）**；
  - 交叉参照（整段 IoU hit@1, k=10）：**single=50.0%**、MRR=55.8%、NDCG=55.6%；multi(26.8%) / hybrid(48.2%) 均不优于 single；**（multi/hybrid 为改造前值）**；
  - 能力差异（hit@1）**max-min=37.5% > 20%**；工程计算 good%=0%、质量计量 12%（补全度重灾区）；chunk 占比最高的是标准规范 63%、最低工程计算 1.0%/故障诊断 0.4%（**非零**，经 `_audit_s4_evidence.py` 交叉验证 top10 内仍可捞到相关块）；标准规范 chunk 63% 但 hit@1 仅 50%（超饱和却匹配差）；
  - k-sweep(hybrid)：@10=62% → @20=64% → @50=71% → **GT 大多本就在 top10，缺口在排序/补全而非截断**；**（k-sweep 为 20260807 改造前值；路径2 落地后重跑为 66/68/68/71）**。
- **排查动作**：
  1. 分层统计：`_diag_recall_s4.py --per_cap 8` 已跑（口径已升级为 sent_coverage 主口径 + 整段 IoU 交叉参照），见 `results/s4_recall/s4_recall_20260807_221619.md`。逐题明细 `s4_rows_20260807_221619.json`。**（注：原 rows 文件被误删且未入 git，已按 seed=7 用改造后代码重跑重建；single 主口径逐位一致，multi/hybrid/k-sweep 为改造后值并已在文件内 `_restore_note` 如实标注差异，详见 `s4_recall_20260807_235012.md` 对照。）**
  2. 对照 `_probe_format_misread.py`：format 身份可检索，瓶颈是问题措辞与标注语义失配（确认成立，不并入检索主轴）。
  3. **多查询/混合增强实测为负增益** → 下一步定位 KED 分解质量 / 停用 multi 分支。**（20260808 路径2 已落地：原查询保底+提权后 multi/hybrid 负增益消除，见 retriever.py；本条为改造前判断留痕）**
- **判定（当前 FAIL）**：semantic Hit@1 ≥ 0.7 且各 capability 差异 <20%。
> **—— S4 量化留痕 & 归因（可溯源层）——**

### 1. 每个关键量化的【数据来源】→【方法来源】对照表（证据链）

| 关键量化 | 数值（本阶段） | 数据来源 | 方法/口径来源 |
|---|---|---|---|
| single hit@1=50% / MRR / NDCG | 50.0% / 55.8% / 55.6% | `_diag_recall_s4.py --per_cap 8` → `results/s4_recall/s4_recall_20260807_221619.md`【A】 | 官方 `eval_scripts/industrial_linkeval/retrieval_evaluator.py::RetrievalEvaluator`('standard'); 相关集合=整段 IoU≥0.15 的 chunk (交叉口径) |
| single 句子覆盖率 cov@10=33.7% / good%=28.6% | 33.7% / 28.6% | 同上 | `retrieval/recall_metrics.py::sent_coverage`(**P0 主验收口径**, GOOD_COV=0.5)；定理: 对长 GT 免于整段 IoU 低估，见 `docs/agentic_recall_completeness_correction.md` |
| multi / hybrid 均 ≤ single（负增益，改造前值） | cov@10: 24.1% / 30.0% vs single 33.7%；hit@1: 26.8% / 48.2% vs 50.0% | `_diag_recall_s4.py --per_cap 8`(改造前 20260807) → `s4_recall_20260807_221619.md`【A】；路径2 落地后重跑 multi/hybrid 回 33.4%/36.0% | `OpenDomainRetriever.multi_query_retrieve`(fusion) / `hybrid_retrieve`(dense+BM25→RRF)；归因逻辑见 `docs/agentic_recall_entrance_diagnosis.md`§二 |
| 能力间 hit@1 max-min=37.5% > 20% | 工程计算 38% / 工艺原理 75% | 同上【B】 | 同【B】capability 分层聚合；差异阈值=20%(项目通过标准) |
| 工程计算/故障诊断 chunk 占比 | 1.0% / 0.4%（非零，528 / 213 块）；top10 内 avg 相关块=1.00 / 2.12 | `_audit_s4_evidence.py` → `results/s4_recall/s4_evidence_audit.json`(20260808) | 语料 capability 标签 + `s4_rows_20260807_221619.json` 入池交叉验证；**修正初版"≈0%/无米下锅"表述** |
| k-sweep @10=62%→@50=71%（改造前值） | @20=64%, @50=71% | `_diag_recall_s4.py --per_cap 8`(改造前 20260807)→【D】(hybrid)；路径2 后重跑 66/68/68/71 | 候选池放大累积命中；判定: 池外增量小→排序/补全问题而非截断 |
| BM25 通道贡献 +0.020(cov@10) | +0.020、BM25 纯增益但非主杠杆 | `docs/agentic_recall_entrance_diagnosis.md`§二表 | `_retrieve_ablate_source.py` 控制变量(去 BM25 / 去扩容)；结论: BM25 补"精确 recall"非"语义补全" |
| Format 失配非检索主轴 | format 身份已可检索 | `docs/agentic_recall_entrance_diagnosis.md` + `_probe_format_misread.py` | 见【E】；format 仅作二级过滤字段 |

### 2. 口径自审（透明性）—— 为什么"50%"与"33.7%"都可入账

- **整段 IoU(hit@1=50%)会系统性低估**真实召回（GT 是数百字长文被 chunk 切碎，single chunk 只盖开头几十字），曾误导出"50% GT 在 top-50 外"，实际 top-10 覆盖≥1 句的题占 87%（见 `agentic_recall_completeness_correction.md`）。
- 因而 **S4 验收以句子覆盖率 cov@10 为唯一主口径**（与仓库 recall_metrics P0 规范一致），整段 IoU 降为**交叉参照**。两种口径下**结论方向一致 = FAIL**：
  - 主口径 fail：cov@10=33.7%(平均只拼齐 GT 答案 1/3)、good%=28.6%(仅 ~3 成题拼得齐)，远低于"拼齐/召回"目标；
  - 交叉口径 fail：hit@1=50% < 70% 且能力差 37.5% > 20%。
  - 因此"不达标"判定**稳健于口径**，不存在口径误判。

### 3. 为什么数据不达标 —— 归因分析（按证据强度）

1. **主因=补全度(Completeness)而非召回入口(Recall@Top)**
   - 证据：cov@10 平均仅 33.7%（拼不齐 GT 答案），而 top-10 内"能沾到边"的题高达 87%；k 从 10→50 覆盖率仅+~9pp，且 @50 累积命中也仅 71% → **瓶颈是 query 编码抓不全散落在多个 chunk 的答案碎片，不是"没进池子"，更不是"库缺知识"**（库缺已证伪，见 correction 文档）。
2. **能力差异(37.5%) = chunk 分布不均衡 × 检索补全双因素**（`_audit_s4_evidence.py` 交叉验证修正）
   - 修正初版"无米下锅"表述：工程计算/故障诊断**不是零块**（528/213 块），且 top10 内**仍能捞到相关块**（avg=1.00/2.12）；
   - 工程计算 good%=0% 实为 **"数值/计算型 GT 句子覆盖率天然低"**（多步骤计算答案，单块只盖开头）+ 相关块绝对量少（528 块/1%）；故障诊断 cov10=42.8% 其实不算差，good%=38% 受 0.5 阈值卡点掩盖；
   - 标准规范 chunk 63% 但 hit@1 50% → **超饱和却匹配差（排序/措辞失配）**，属另一类问题。
   → **语料量级不均是硬约束（影响补全上限），排序/措辞失配是软约束（影响命中）**，修检索能救"超饱和但命中低"与"有料未顶上"，救不了"数值型 GT 覆盖率天然低"。
3. **multi 负增益的代码级根因 = RRF 加权依赖"子查询次序"而非"相关性/质量"**（`retriever.py` 已核）
   - `_fusion_merge`(retriever.py:886-892)：`weight = len(results) - rank_weight` 给**第 0 个元素最大权重**，且 `score += weight/(60+chunk.rank)` **只用子查询内部 rank、无相关性门控**；hybrid_retrieve 的 `_rrf_merge_lists`(859-878) 原本按 rank 等权融合。
   - 后果：KED 分解出的空泛/低价值子查询（如"列出风险"）也带回一堆低相关块参与融合，把原查询、高相关的单一子结果**稀释出 top-n**——这是 multi/hybrid cov@10 低于 single 的确定性机制。
   - **处置（已落地 20260808，见 §6）**：改为【原 query 保底融合】——把原始完整查询作为权重最高成员参与 RRF（multi 放首位借 `_fusion_merge` 次序权；hybrid 显式给 1.2× 权重）。质量门控候选被数据证伪（top1 分布重叠），故走保底而非门控。
4. **BM25 已实测=+0.020 纯加分小额**，非召回上限主杠杆；中文 tokenizer 落地前不作优化重点（`agentic_bm25_jieba_ab.md`）。

### 4. 结论
- **判定 = FAIL**（主口径 cov、交叉口径 hit@1 双双未达标）；修复优先级与三条路径见下方 §5。
- 每步优化后**沿用本表口径回归**（sent_coverage 主口径 + 整段 IoU 交叉参照 + k-sweep），凡出现回退即判假信号。

### 5. 下一步（按分而治之，经 `_audit_s4_evidence.py` 证据修正后重新排序）
  1. ✅ **【已落地 20260808·路径2】保 single 基线 + 修 multi 负增益**：见下方 §6 留痕。multi/hybrid 负增益已消除（multi cov@10 24%→33%，hybrid 30%→36%），single 基线不变。
  2. 🔍 **【已分析未修改 20260808·路径3→已判"非检索算法问题"】标准规范 B 类池外失配——见 `docs/s4_path3_diagnosis_report.md`(含 §6 证伪)**：
     - 早期发现：标准规范 49% 题 GT 连 top-50 都不进(B类)、纯排序(A类)仅 4%；
     - **P0「标准号精确检索」被双重否决**：① 属答案侧标识泄漏(审稿风险)；② 更根本——含标准号 chunk 对 GT 句覆盖仅 2.2%，**根本无料可捞**，P0 无效；
     - **A/B(BM25 标准号加权)亦收回**：标准号只命中"标准标题块"，不指向承载 GT 的文本；
     - **根本病灶=数据集/评测构造**：标准规范 GT 是库外撰写知识，与语料字面/句面几乎不重叠 → 低召回主要非检索器缺陷，而是 **GT 来源与语料不同构**。
     - **处置**：不再投入标准号检索旁路/加权；若要站住脚，方向是**语义级命中判据 + 如实披露此类结构性低估**（属评测设计，超检索算法范畴）。
  3. **补语料量级（工程计算/故障诊断 528/213 块）**：非"零块"但仍偏少；经 `_audit_s4_evidence.py` 证实相关块能进 top10，诊断集中在覆盖率而非不可检索。

### 6. 【路径2 落地留痕 20260808】修 multi 负增益：改了什么 / 怎么改 / 前后量化对比
- **目标**：消除 multi(24%)/hybrid(30%) 相对 single(33.7%) 的负增益，且**不污染 single 基线**。
- **方法选择（数据驱动，先证伪→再定案）**：
  1. 候选①〔子查询质量门控〕**被数据证伪**：`results/s4_recall/multi_gate_calib.json` 显示有效/无效子查询的 top1 余弦相似度分布高度重叠（0.58~0.85），阈值不可靠地分离，拍阈值会误杀/放行双向失控 → 弃用。
  2. 候选②〔原 query 保底融合〕**离线模拟验证通过**：`_sim_multi_groundfix.py` → `multi_groundfix_sim.json`：给原 query 最高权重（2:1）融入 RRF，multi 的 cov@10 从 24.4%→33.8%（≥single 33.7%），hit@1 27%→45%，good% 23%→30%。
- **改了什么（最小侵入，仅 `retrieval/retriever.py` 两处）**：
  1. `multi_query_retrieve`：Step2 把【原始完整 query】的 `dense_search` 结果置于 `all_results[0]`（`_fusion_merge` 的 `weight=len-fold-rank_weight` 自动给首位最高权重）→ 原查询高相关结果不再被低价值子查询稀释。
  2. `hybrid_retrieve`：dense 侧【始终保底原始完整 query 并给 1.2× 权重】，子查询等权 1.0，且去重（跳过与原 query 相同的子查询）→ 与离线模拟的"原 query 高权重"口径一致。
- **验证口径**：`_diag_recall_s4.py --per_cap 8`（同 56 样本，sent_coverage 主口径 + 整段 IoU 交叉参照）。
- **前后量化对照**（pre=`s4_recall_20260807_221619.md` / post=`s4_recall_20260807_224857.md`）：

| 方法 | cov@10 前→后 | good% 前→后 | hit@1 前→后 | NDCG 前→后 |
|---|---|---|---|---|
| single（基线） | 33.7%→**33.7%** | 29→29 | 50.0→**50.0** | 55.6→**55.5**（✅不变） |
| multi | 24.1%→**33.4%**(+9.3) | 23→**29** | 26.8→**32.1**(+5.3) | 35.2→**48.4**(+13.2) |
| hybrid | 30.0%→**36.0%**(+6.0) | 27→**32** | 48.2→**50.0**(+1.8) | 51.2→**56.3**(+5.1) |
| k-sweep(hybrid) | @5 61→**66** / @10 62→**68** | — | — | — |

- **诚实结论**：
  1. multi 负增益**已消除**（cov@10 24.1→33.4，≈single；NDCG +13.2pp），hybrid 额外受益至**反超 single**（cov@10 36.0、NDCG 56.3）。
  2. **single 基线完全未受影响**（两项 main 指标前后逐位相等），符合"先保 single"铁律，无回归。
  3. 但 **整体仍 FAIL**：single 的 cov@10 与 hit@1 未变，能力差 37.5% 未变 → 本步只修复"多查询负收益"，**未触及 single 自身的召回上限**（那是路径3/补全度问题）。
  4. **可回退**：若后续某 AB 在未复测 multi/hybrid 时依赖旧行为，还原两处改动即可（改动点已带注释 + 本留痕），或直接切 single 作为默认入口不受影响。
- **通用性复测（20260808·`_verify_multi_generalization.py`, 3 seed×42题, 真实落地 retriever 验证）**：
  | seed | single cov | multi cov | hybrid cov | multi≥single 题占比 | hybrid≥single 题占比 |
  |---|---|---|---|---|---|
  | 7 | 30.5% | 30.5% | **32.5%** | 79%* | **95%** |
  | 11 | 33.3% | 31.9% | **34.9%** | 83%* | **93%** |
  | 42 | 28.2% | 28.4% | **30.0%** | 79%* | **90%** |
  **逐题正/负/零分布（126题累计）**：
  - multi Δcov(multi−single)：**正23 / 负20 / 零83**，非零均值≈0 → **multi=中性持平**：其价值是"消除系统性被稀释的负增益"，非逐题更优。≥single 题占比 79~83% 中大量为"平局(==)"。
  - hybrid Δcov(hybrid−single)：**正37 / 负9 / 零80**，非零均值 **+0.05**，正负比 4:1 → **hybrid=真实通用正向**（广泛分布于非零样本，非局部夸大；平局未灌水）。
  **诚实定性**：multi 保底 = "不再更差的中性入口"；hybrid 保底 = "稳定略优于 single 的入口"。**生产上建议用 hybrid 或 single，multi 不作为提升 recall 的手段**。24.4→33.8% 为离线模拟(权重2:1)上限估计；落地真实值以本表 seed 汇总与 `224857.md` 为准，方向一致但更保守。
- **产物**：`multi_gate_calib.json`（证伪①）、`multi_groundfix_sim.json`（验证②）、`s4_recall_20260807_224857.md`（落地后）+ `_calibrate_multi_gate.py`/`_sim_multi_groundfix.py`（复现脚本）。
- **现成工具**：`_diag_recall_s4.py`（官方 RetrievalEvaluator + capability/industry 分层 + 方法对比 + k-sweep，口径已升级为 sent_coverage 主口径）、`_audit_s4_evidence.py`（路径证据审计，`results/s4_recall/s4_evidence_audit.json`）、`eval_scripts/industrial_linkeval/retrieval_evaluator.py`、`retrieval/recall_metrics.py`、`_diag_recall_*.py`、`_probe_format_misread.py`。

## S5 · 证据组织 ——  目标：证据完整且 prompt 不膨胀

- **指标**：去重后保留率、`get_context()` 是否含全部命中 chunk、prompt/evidence 长度比。
- **当前基线**：`organize` 保留全部去重后 chunk 与原始顺序；`merge` 可能同时并入 organizer + 原始 chunk → prompt 膨胀。
- **排查动作**：抽样统计 evidence_length 与最终 prompt_length 分布；定位超长样本（高 recall 但被截断丢分）。
- **判定"通过"**：缺失 chunk 0；超长样本有明确截断策略且不丢核心命中。**（20260808 实测精化：organize/get_context 保真通过即满足本判据；但 merge 分支 `_handle_merge` 重复并入使实际 prompt 膨胀 2.63×，列为独立待修缺陷，S5 判定降级为"有条件通过"——详见下）**
- **现成工具**：`_system_flow_quant.py`（evidence/prompt 长度）、`_diag_evidence_utilization.py`。

> **—— S5 量化留痕（20260808·判**有条件通过·merge 膨胀为正式发现**）——**
> - 口经：生产对标 `RerankTruncOrganizer`（truncate=True,max_chars=6000）/ 对照 `EvidenceOrganizer`（纯保真），N=120 全能力均匀（`_diag_evidence_s5.py`）→ `results/evidence_s5_audit.json`；**并新增真实节点处理器复核 `_diag_merge_inflate.py` → `results/s5_merge_inflate.json`**（回应"是否脱离真实 agentic 系统"的质疑）。
> - **完整性**：get_context 含全部去重后 chunk = **100%**，缺失 chunk = **0**；去重保留率 prod/pure = **99.4% / 99.4%**。
> - **直接路径（organize→sufficient→reason，`_handle_reason`）**：evidence 均值/p95 = 2054/2621 字符，prompt 均值/p95 = 3789/4376，**evidence/prompt=53.8%**，超长(>6000/8000)样本=**0**——**该路径无膨胀**（`organized.get_context()` 与原始 evidence **if/elif 二选一，非叠加**）。
> - **⚠️ merge 分支膨胀（正式发现）**：图走到 `decide=insufficient→retrieve_2→merge_1` 时走 **`_handle_merge`，非 `_handle_reason`**。真实处理器复核：merge 把 `last_organized.get_context()` + `evidence_cache` 全部 raw chunks **拼接**，**同 chunk 被输出两遍** → `accumulated_context / organize_only ≈ **2.63×**（max 2.79）、100% 样本 ≥1.5×、内容重复率 **1.0**`（N=40）。**真实基准日志（60 次图执行）merge 触发 ≈ 28%（17 次）**——非偶发。此前 "_diag_evidence_s5 只测 direct-reason 路径故误报 merge 无忧"，已修正。
> - **结论**：S5 **证据组织语义（organize/get_context 保真、0 缺失）通过**；但 **merge 节点重复并入构成真实 prompt 膨胀** → 整体降级为**有条件通过**，缺陷列入待修（建议：`_handle_merge` 去重——合并时跳过已被 organized 覆盖的 chunk）。详见 `docs/s5_evidence_organization_audit.md`。


## S6 · 评分/输出 ——  目标：评分"公平反映答案内容"

- **指标**：rule_score(0-3)、关键词/实体覆盖、结构命中；按难度/能力分层；LLM judge 参照。
- **当前基线**：avg **1.55~1.71/3.0**、SV=0；"标准规范与术语/安全合规与风险控制"偏弱（0.50~1.38）；difficulty easy>medium>hard。
- **排查动作**：
  1. 先**清洗评分误伤**：答案内容对但表述不同是否被 rule 判低（用 cov/ent_cov 交叉看）。
  2. 对低分能力域抽 10 题回看：是召回问题（S4）还是生成表述问题（S6）。
- **判定"通过"**：低分能力域已归因到 S4 或 S6，且已在该阶段治理。
- **✅ 已检查（20260808）** `_diag_s6_score_attribution.py` → `results/s6_score/s6_score_20260808_101616.md|json`，双口径归因：
  - **动作1 评分误伤清洗（通过）**：rule_score 与 cov 单调对应（真实 6 题 score3→cov0.71 / score2→0.46~0.55 / score1→0.07~0.09）；"实体正确/转述不同"（纤维滤料 ent_cov=1.00、score2）**不误伤**；对 6 个低分样本构造"正确重述答案"**6/6 重述后均判≥2**（0 个误伤候选）→ 规则分容忍换措辞、只反映内容覆盖。
  - **动作2 低分归因（通过）**：35 题真实端到端中 6 个低分(≤1)样本（故障诊断2/质量计量2/安全合规1/标准规范1）**全部**用完整 question 本地 top10 检索，证据句覆盖 `sent_coverage(ref, evidence)` **=0.00**（连简短答案句都没拼齐）→ **归因召回侧缺料（S4 治理域）**，S6(证据足产出差) **0 题**。
  - **验收：S6 判定『通过』** → 低分能力域主因已归 S4（证据前向/multi-query/query 改写已在 S4 治理），S6 评分口径与生成表述无系统性误伤。

---

## 执行顺序建议（贪心：先修指标最差且影响最大的）

1. **先 S4 检索召回**（Hit@1 0.5 → 目标 0.7）—— 这是端到端分数的主要上界。
2. 再 **S6 评分口径**（排除"对了但没对措辞"的误伤），避免误判为检索问题。
3. 再 **S3/S5**（solver 死代码 / prompt 膨胀）清理架构噪声。
4. **S2** 控调用成本；**S1** 保持当前落地即可，仅每轮回归复查。

> 每阶段改动后：重跑该阶段指标 **+** 端到端 `exp1_agentic_rag_fix`（avg rule & Hit@k），
> 记录"改动前/后"差值，达标才进入下一阶段。**不做一次性多阶段混改**。
