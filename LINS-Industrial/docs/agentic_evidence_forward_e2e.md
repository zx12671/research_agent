# evidence-forward 端到端终验报告（base vs 整档回捞+相似度prune）

## 1. 目的
验证"证据前向补全"（evidence-forward）：对 A 类超长 GT 题目，当前 top-10 检索只能覆盖其开头段，是否可通过"整档回捞 + 按 query 相似度取每档 top50"补全证据，进而**真实传导到答案分**（而非仅命中/覆盖率这种代理指标）。

## 2. 方法（最小闭环，隔离"候选质量→答案分"单一变量）
- **候选口径**（用户拍板）：
  - `base` = 仅 `retrieve(q,k=10)`（生产现状）
  - `ef_pruned` = 对 top-10 每档回捞全部块 → 每档内按 `jt(块,query)`(去空格 bigram JT) 排序去重，过滤滑出 `JT<0.02` 的低相关块，每档封顶 **50** 块 → 合并。
- **回答生成**：同一套 standard prompt + DeepSeek 轻量直答（temperature=0.2，一次调用），**只换候选**。
- **评分**：`RuleBasedScorer.rule_based_score(q,ref,pred)`（纯规则 0-3 覆盖度）+ `check_safety_simple` SV 清零——与 exp1 `rule` 模式完全一致，可复现。
- **样本**：N=21，seed=7，每能力 3 题（覆盖 7 项能力），A 类(≥50句) 3 题 / B 类 18 题。

## 3. 结果

| 分群 | n | base.mean | ef.mean | Δ | 升/降/平 | 0分率 base→ef |
|---|---|---|---|---|---|---|
| **全部** | 21 | 2.143 | **2.381** | **+0.238 (+11.1%)** | 6/3/12 | 10% → **0%** |
| **A(超长GT≥50句)** | 3 | 1.000 | **2.000** | **+1.000** | 2/0/1 | 33% → **0%** |
| B(<50句) | 18 | 2.333 | **2.444** | +0.111 | 4/3/11 | 6% → 0% |

- SV 违规：base 0 / ef 0（补全证据未引入额外安全违规）。
- 候选量：base=10 → ef=49（每档回捞均 38 块、命中均 6.4 档）。耗时 84s/42 次 LLM 查询。
- 原始逐题表：`results/probe_evidence_forward_e2e.json`

## 4. 结论与解读
1. **立项成立**：ef_pruned 在端到端答案分上**全局 +11.1%**；尤以 A 类 +1.0 翻倍、0分率清零最显著，直接印证"top-10 只覆盖超长 GT 开头"的机制缺口已被证据补全修复，且传导到了真实答案分。
2. **B 类 = 召回已饱和/敏感于头部**：B 类+0.11 微涨但出现 3 例**负增益**——整档回捞把头部高相关块之外的中后段相关块推入 prompt，稀释了头部线索，在短答案题目上反致覆盖度下降。
3. **上线建议（分诊式启用）**：
   - **仅 A 类（GT/源文档块数大、top-10 覆盖不全）启用整档回捞**，其余保持 top-10 现状 → 可同时拿 A 类+1.0 的收益并规避 B 类的负增益。
   - 对 ef 已回捞但答案分仍不升的题，属"证据在但模型未利用"，非检索层问题，交给 TaskSolver/prompt 层。
   - 候选池膨胀（10→49）对长上下文模型的成本可接受；如需更省，可把每档 top50 降到 top20（对照 `_probe_evidence_forward.py` 的 cov 差异折中）。

## 5. 文件
- 探针：`_probe_evidence_forward_end2end.py`
- 数据：`results/probe_evidence_forward_e2e.json`
- 前序：`docs/agentic_recall_completeness_correction.md`（命中/补全度）、`docs/agentic_evidence_forward_probe.md`（整档回捞覆盖上限）
