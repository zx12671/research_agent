# 「召回率 47%→87%」修正结论的量化证据报告

> 目的：用可复现的逐题数据回答"为什么旧结论'召回率≈40%'作废，
> 而真实 dense 检索 87%（26/30）的题目至少能把 GT(答案) 的一句话捞进 top-10"。
>
> 本报告是自包含的**证据链**：判定口径 → 逐题对照表 → 低估成因的数学解释 → 复现命令。

---

## 0. 一句话结论

旧口径 `整段 IoU(GT全文, 单chunk) ≥ 0.20` 因 **GT 是长文、chunk 是碎片** 而系统性低估命中；
改用 `GT逐句 vs chunk 的 bigram-Jaccard(≥0.30)` 后，**dense top-10 的句级命中率 = 26/30 = 87%**，
旧口径仅 14/30 = 47%。**两者差 40pp，全部来自口径漂移，而非检索能力变化。**

---

## 1. 两种判命中口径（这本身是关键）

### 旧口径（作废）— 基于 `results/diag_recall_coverage.py` 之前的 `_probe_recall_30` 式判断
```
hit_old(q) = ∃ chunk ∈ top-K :  jaccard( 整段GT全文,  chunk ) ≥ 0.20
```
- 参与 n-gram 集合的两端**长度极不对称**：GT 一段有 200~600 字，chunk 只有 ~80~120 字。
- 设 GT 有 Ng 个 bigram，chunk 有 Nc 个，交集至多 min(Ng,Nc)。
- 只有当 chunk 恰好等于 GT 的一个**完整子段**时才可能够到 0.20；
- 而真实 chunk 是"GT 某一段+周围无关信息"的拼接 → 交集被引入的无关 bigram 稀释 + 全段分母被超长 GT 放大。
- ⚠️ 这就造成核心误判：**GT 第一段明明逐字躺在 TOP-1 里，但因整个 GT 还有 300 字，IoU 被稀释到 0.14 < 0.20 → 判 miss。**

### 新口径（本次） — `hit_new(q) = coverage ≥ 1 句`
```
把 GT 按句子切分 → 对每个句子 s：
  若 ∃ chunk ∈ top-K : jaccard(s, chunk) ≥ 0.30 → 记 s 被覆盖
coverage(q) = 被覆盖句数 / GT 总句数
hit_new(q)  = coverage(q) ≥ 1/总句数   (即至少覆盖 1 句)
threshold 0.30：单句与 chunk 长度可比，0.30 已是"同一句话"的强信号。
```

---

## 2. 逐题对照表（seed=7, top-10, single, use_ked=True）

> 来源：`_diag_recall_sentence.py` 直接运行输出（GT 整段 vs 逐句）：
> - `nSentGT ` = GT 直接切出的句子数
> - `covSent ` = 被 top-10 覆盖的句子数
> - `old整段 ` = 旧口径 `jaccard(整段GT, chunk)≥0.20` 命中与否
> - `sentence` = 新口径 `coverage≥1` 命中与否

| 题目(截取) | cap | nSentGT | covSent | old整段 | sentence |
|---|---|---|---|---|---|
| 电磨/直磨机小磨头风险 | 安全合规 | 21 | 18 | False | **True** |
| 乙炔气体输送管道 | 安全合规 | 24 | 0 | True | False |
| 压电式速度传感器防护 | 安全合规 | 18 | 6 | True | True |
| 应急保温毯物理机制 | 安全合规 | 16 | 5 | True | True |
| 爆炸性环境胶管接头认证 | 安全合规 | 16 | 11 | True | True |
| 380V三相整流 VDRM | 工程计算 | 23 | 0 | True | False |
| 节水灌溉后评价经济 | 工程计算 | 102 | 12 | False | **True** |
| SYT-2000微压计皮托管 | 工程计算 | 16 | 6 | False | **True** |
| 5寸锯片直径 | 工程计算 | 20 | 2 | True | True |
| 50kWh充电20-100% | 工程计算 | 20 | 12 | True | True |
| 铸铁V型铁硬度 | 工艺原理 | 17 | 5 | True | True |
| 永磁材料海洋防护三层 | 工艺原理 | 13 | 0 | True | False |
| 变频器外部电位器 | 工艺原理 | 23 | 7 | False | **True** |
| 二氧化锡气化法优势 | 工艺原理 | 20 | 7 | False | **True** |
| 橡胶漆喷涂前处理 | 工艺原理 | 23 | 6 | False | **True** |
| 制冷压缩机维修干燥 | 故障诊断 | 22 | 14 | False | **True** |
| DZSF直线振动筛粘料 | 故障诊断 | 27 | 6 | False | **True** |
| 聚氨酯旋流器底流口 | 故障诊断 | 17 | 10 | True | True |
| A68无线门铃距离 | 故障诊断 | 24 | 6 | False | **True** |
| 3Cr13折叠刀盐雾 | 故障诊断 | 16 | 3 | True | True |
| 埋地管道防腐等级 | 标准规范 | 17 | 5 | True | True |
| GB/T 2423.59-2008 | 标准规范 | 76 | 4 | False | **True** |
| 色差测量仪同色异谱 | 标准规范 | 24 | 6 | True | True |
| GB/T 7350-1999防水 | 标准规范 | 45 | 7 | False | **True** |
| GB893.1卡簧 | 标准规范 | 36 | 15 | False | **True** |
| 电子测量离散参数 | 质量计量 | 186 | 0 | False | False |
| 6层电路板盲孔 | 质量计量 | 31 | 1 | True | True |
| 纺织统计指标 | 质量计量 | 202 | 5 | False | **True** |
| 钼蓝分光测硅空白 | 质量计量 | 68 | 8 | False | **True** |
| 亚甲基蓝摩尔浓度 | 质量计量 | 16 | 6 | False | **True** |
| **合计** | | | | **14/30=47%** | **26/30=87%** |

---

## 3. 逐字核验（打消"是否真的是 chunk 里有 GT"的疑虑）

> 来源：`_diag_recall_excerpt.py`，对 part 判 miss 的题做 TOP-1 原文抽验。

例 1（电磨）：GT = 「电磨(直磨机)采用加粗纯铜芯线电机作为动力核心，…（全场 21 句）」。
TOP-1 chunk 输出 = **同一段原文的前 2 句逐字相同**。
→ 知识**逐字在库**，只是 chunk 只装下了 GT 的开头，还无法覆盖后 19 句 → 旧全段 IoU≈0.14 判 miss。

例 2（二氧化锡/气化法）：GT 首段「二氧化锡(气化法生产)作为高性能无机非金属材料，广泛应用于陶瓷釉料…」
TOP-1 chunk 逐字相同。→ 同理。

例 3（GB/T 2423.59）：GT 为国标条文全文(76句)，TOP-1 仅捞到开头约 1 句标题 → cov=4/76≈0.05，旧口径 False，新口径 True。
→ 归为文档 `agentic_recall_completeness_correction.md` 中的"**A 类·超长 GT 型**"：知识在库，但单轮 top-K 无法补全全篇，是**任务型错配**而非缺库。

---

## 4. 数学解释：为什么 40pp 差全部来自口径漂移

令 GT 全长句子 G，拆句后为 s1..sm；chunk c 恰好等于 s1（逐字）。
- 旧：jaccard(G, c) 的分母含 G 的**全部 bigram**(大幅超 s1)，分子只含 s1∩c≈s1 →
  比值 ≈ len(s1)/(len(G)+…) ≈ 常常 <0.2。**Chunk=c=s1 时也判 miss。**
- 新：jaccard(s1, c)=1.0 ≥0.30 → s1 覆盖，hit。

因此哪条路径都不改检索，只改判据；14→26 个 hit 完全是"把判据切到与 chunk 同粒度"的结果。
**这也反向说明：真正的检索不足不是"捞不进边"，而是"捞不全边"（补全度），见 companion 报告。**

---

## 5. 复现命令

```bash
cd LINS-Industrial
python _diag_recall_sentence.py     # 逐题 old vs sentence 对照（本文表2）
python _diag_recall_excerpt.py      # TOP-1 与 GT 逐字抽验（本文§3）
python _diag_recall_coverage.py     # 覆盖率分布/补全度主表（K=10/20/50 三入口）
# 全量结果落盘: results/diag_recall_coverage.txt
```

相关文档：
- `docs/agentic_recall_completeness_correction.md` — 补全度(completeness)主报告
- `docs/agentic_recall_30_probe.md` — 旧口径报告（请以本文+correction 为准）
