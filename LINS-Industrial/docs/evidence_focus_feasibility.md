# 件③ 证据聚焦（带依据直答 + 硬护栏）· 可行性分析

> 目的：在执行件③前，先论证"A 类收益最大 / 直击 12 题利用失败"这个前提是否成立，
> 以决定该做轻档、重档，还是暂不做。本文不落地任何代码，仅给裁定与依据。

## 一、现状基底（已核实）

生产 reasoning 层已有两层"基于证据"提示，但都是**软约束**：

- `SYSTEM_PROMPT`（`agentic/prompts.py` L32-41）：已写 `Cover EVERY key fact supported by
  the evidence`、`repeat the value EXACTLY`、`Prefer a partially-supported answer over
  "insufficient evidence"`、`NEVER refuse to answer when at least one retrieved chunk is
  relevant`。
- `_general_prompt`（L629-638）：`Base your answer on the evidence above`、
  `extract EVERY available fact`、`clearly distinguish supported vs uncertainty`、
  `Only refuse with "insufficient evidence" when there is NOTHING relevant`。
- `_MINIMAL_EXTRA`（`agentic/prompt_builder.py` L76-80）：`基于提供的证据给出你最好的答案`。

→ **件③轻档的目标约束，与已在生产生效的提示高度重合，区别只在"更严格（要求标注证据缺失）"。**
这不是从零加约束，而是把软指令"变硬"。重档（先精选再抽取式作答）则是新增组织+改写链路。

## 二、实证校验（N=35，`results/probe_evidence_forward_e2e_35.json` 重算）

### 断言 1：“证据到位但模型失焦/漏抽 = 12 题” —— 属实，但**不是 A 类专属**
用 `base<3 且 ef≤base`（排除 base=3 分地板）近似真利用瓶颈，35 题里确有 **12 题**：
- 按 cls 分层：**B 类 9 / A 类 3**（≈3:1）。
- 升分组（ef>base）6 题同样 **B 5 / A 1**。
→ "12 题利用失败"这个数字被当成 A 类瓶颈来源是**误导**；它是 A/B 合计，B 类是 A 类的 3 倍。

### 断言 2：“对 A 类收益最大，直击 A 类失焦” —— **不成立**
A 类总共 6 题基线分布 = 3/2/1 / 1/2/3：
- 2 题已 3 分满分（节水灌溉水池、两线制压力）→ **地板，任何改动都到顶**；
- 3 题卡 2/2/1 分（GB/T 2423、钼蓝滴定、测量离散性）→ 与 B 类同款"可涨未涨"瓶颈；
- 仅 1 题（纺织材料 1→2）是证据前向**真正让 A 类升分**。
→ 件③若只瞄准 A 类，受用面仅 3 题，其中 2 题是标准数值抽取型（GB/T2423、钼蓝）——这 2 题
  恰是"cover EVERY key fact + 逐值重复"能对症的；但即便全救回，对 A 类总分贡献也就
  +1~+2（3 题里还有 2 满分的既得利益不能动）。**"对 A 类收益最大"缺乏数据支持。**

### 断言 3：“证据到位但利用不足”这个机制的**存在性是成立的**
虽非 A 类专属，但 12 题可涨未涨 + `docs/agentic_evidence_utilization.md` 已将其列为
"下一步实验 #2 证据聚焦/摘要、#3 带证据直答门禁"——**件③轻档=报告建议 #3，重档≈建议 #2**。
机制方向与既有结论一致，只是原报告的优化说辞是**全局**的（跨 A/B），不是 A 类专属。

## 三、重档与件①结论的冲突（关键）
件①结构探针已实测：生产 top-10 证据总量仅 ~1.5-1.8k 字符，`max_chars=6000` 下
`drop%=0`、`ctx_base==ctx_rt`——**RerankTruncOrganizer 的"砍噪声"在真实链路上空转**。
重档"先用 RerankTruncOrganizer 把 top-10 精选成最相关片段"在 top-10 尺度下**无噪声块可砍**；
且本数据集多数是 `Question → 一句话 → Answer` 的直答题，**抽取式作答改写会破坏轻量答法**，
对大量已满分的"地板题"和简单直答题是净风险、非收益。

## 四、裁定与建议

| 档 | 建议 | 理由 |
|---|---|---|
| **轻档**（reason 层加一条硬约束，默认关） | **建议执行** | 成本≈0（仅 `_handle_reason` 注入一条约束，默认 None 不改变行为）；对症 12 题可涨未涨的"失焦/自由作答"；与 `SYSTEM_PROMPT`"别拒绝作答"不冲突（是"标注无依据点"非"拒绝作答"）；可干净 A/B。 |
| **重档**（RerankTrunc 精选 + 抽取式作答） | **暂不执行** | 与件①结论冲突（top-10 无噪声可砍）；抽取式改写破坏直答题与地板题，净风险；成本高收益不确定。 |

**口径修正**：件③应定性为**全局**的"证据聚焦 + 硬护栏"手段，不要以"A 类收益最大"作宣传——
数据不支持（A 类真瓶颈 3 题且含 2 满分地板；B 类是 3 倍）。若执行，A/B 目标应是全量均值，
预期对 12 题可涨未涨中的 B 类多数 + A 类 GB/T2423/钼蓝 2 题起效；A 类的上限仍受
件②（按需触发）与检索侧（④组 cov<0.5 23 题）约束。

*数据来源：`results/probe_evidence_forward_e2e_35.json`（N=35）重算；`docs/agentic_evidence_utilization.md`。*
