# 分析：bge-small 是否限制 agentic 检索效果？——结论"够用为主，瓶颈非模型"

> 触发：综合评估 embedding 模型 BAAI/bge-small-zh-v1.5 是否为检索瓶颈，还是已够用。
> 方法：**已有 A/B 归因文档 + 两条 VDRM 决定性实证**（`_demo_vdrm_probe.py` / `_demo_vdrm_verify.py`）+ 模型官方定位。
> 结论：**当前主瓶颈不在模型大小，而在分块粒度/query 表达；bge-small 够用，VDRM 这类"hard-miss"实为长 chunk 稀释伪命题，改写 query 即 rank1。**

---

## 一、先看项目自己的归因（30 题，句覆盖率口径，`agentic_recall_entrance_diagnosis.md`）

| 类 | n | 真因 | 是否是"模型小" |
|---|---|---|---|
| **C 常见弱覆盖** | 23 | 答案碎片分散在库内、query 一条向量抓不全 → 补全度低 | 否（query/分块） |
| **A 超长 GT** | 5 | GT 60~202 句，单轮 top-K 天然拼不齐 | 否（任务型） |
| **B 真hard-miss** | 2 | 语义鸿沟；深挖后**仅 1 题(VDRM)**是真信号失灵 | 唯一疑似模型 |

**决定性读数**：
- `dense 放大 top-20→50，句覆盖率完全不涨（0.27=0.27=0.27）` → 瓶颈不是"池子小"，是"query 编码抓不全/排序"。
- BM25(jieba) 复活后 `cov@10 +0.020`、hybrid top-10 多盖 GT 8/30 题 → BM25 是补精确 recall 的配角。
- multi_query(切逗号) **负增益 −0.029**，hybrid 覆盖 0 增益 → **query 分解不是杠杆**。

---

## 二、VDRM 决定性实证（推翻"hard-miss 需换模型"）

**题**："380V线电压三相整流电路中，应选断态重复峰值电压(VDRM)不低于多少伏的普通晶闸管？" → 答 1200V

**库内真相关联 chunk `80775e037f73`**（关键句，埋在约 200 字 chunk 中段）：
> "…选型时需重点关注…断态重复峰值电压（VDRM）…例如，工作于 **380V线电压的三相整流电路中**…应选择 **VDRM不低于1200V**的晶闸管…"

**（当前索引/裸 query 实测）原文 vs 改写 query 的 rank**：

| query | dense rank | bm25 rank |
|---|---|---|
| 原完整长句（use_ked=False） | **13** | **1** |
| 改写1：`三相整流电路 选 VDRM 不低于1200V 的普通晶闸管` | **1** | **1** |
| 改写2：术语直陈短查询 | 15 | 1 |
| 改写3：标准选型问句 | 26 | 6 |

**结论**：
1. **答案 chunk 在 bge-small 下早就被检索到**（dense top13、bm25 top1），只是**没进 top10**（dense 13 略超 10）。
2. **改写 query 后 dense → rank1** → **杠杆 100% 在 query 表达 / 分块粒度，不在模型**。
3. 文档里"dense rank=562"是**旧口径 / use_ked=True（KED 长 query 拉低）**下的读数——**恰好佐证 KED 把 query 拉长反而降 dense 精度**，也是"流程/配置而非模型"的又一证据。

**机制解释（为什么 dense rank=13 而文档说 562）**：答案埋在长 chunk 中段，bge-small 是**单向量、偏短句匹配**的 bi-encoder，query 向量与"含大量旁支参数的长 chunk 向量"相似度被稀释；但**同主题邻居在 top 附近 → 是排序精度问题，不是召回缺失**。

---

## 三、换更大模型能救多少？（模型官方定位）

| 模型 | 参数 | 定位 | 对 VDRM 该类“长 chunk 稀释”是否对症 |
|---|---|---|---|
| **bge-small-zh-v1.5**（现用） | 24M | 短句/语义检索，小档最强之一 | — |
| bge-large-zh-v1.5 | ~326M | 单向量 bi-encoder，T2Retrieval 分数级提升 | **仍单向量、偏短本，对长 chunk 稀释改善有限** |
| **bge-m3** | ~568M | **多语言 + 多粒度(dense+sparse+colbert多向量) + 8192长文本**，官方明确主打 LongDocRetrieval | **真正对症**（长文本 + 稀疏自蒸馏 + 多向量），但需重建索引、成本高，且有更便宜的流程优化等价方案 |

> 关键：**bge-large 不是 VDRM 类的解药（同是单向量短本模型）；bge-m3 才是，但被"分块细粒度 + query 改写"以 0 成本等价覆盖。**

---

## 四、综合判断

### 1. bge-small **不是当前检索效果的主瓶颈**（够用为主）
- 23/30 是补全度问题（query/分块/排序），5/30 是超长 GT 任务型，深挖后真·检索信号失灵仅 1 题。
- 那 1 题(VDRM) **实测答案已在 top13 + bm25 top1，改写 request 即 rank1** → 更接近"流程/口径"而非模型能力。
- 现有 C-MTEB 小档第一档的性能，对短句语义检索够用。

### 2. bge-small 的**真实、机制性弱项**（换模型才真正受益的地方）
- **单向量 + 偏短句**：对"答案埋在 512 字符长 chunk 中段"的匹配会被稀释 → 这是 bge-m3(长文本+多向量) 或**更细的分块**能救的部分。
- **KED 把 query 拉长会降 dense 精度**（VDRM 实测 rank13→562 的机制即此）→ 这是配置问题，改不改模型都会影响。

### 3. 最优路径（按 ROI，模型几乎排在最后）
| 优先级 | 动作 | 依据 | 是否动模型 |
|---|---|---|---|
| P0 | **分块粒度**（长 chunk 再切到"一个参数事实=一块"） | VDRM 答案埋在 200 字 chunk 中段，切细后 dense 命中即提前 | 否 |
| P0 | **query 改写**（语义级、去冗余聚焦判别键，替代 KED 拉长） | 实测 VDRM 改写→dense rank1 | 否 |
| P1 | 保留 BM25(jieba) 作精确门控/兜底 | bm25 对型号/标准号 rank1，VDRM 已被其 top1 捞到 | 否 |
| P1 | A 类超长 GT 做题级检索预算/证据前向 | 5 题任务型 | 否 |
| P3 | **再评估换 bge-m3**（或重排 reranker） | 仅当上述完成后仍剩"长文档语义"缺口 | 是（成本高） |

> 一句话：**先用 0 成本把"分块细粒度 + query 改写"做掉，bge-small 大概率够用；换 bge-m3 只在确认"长 chunk 语义稀释"是剩余主缺口时才投，且需重建 55k 向量索引。**

---

## 复现
```bash
cd LINS-Industrial
python _demo_vdrm_probe.py        # 原文 vs 改写 query 的 dense/bm25 rank（VDRM 案例）
python _demo_vdrm_verify.py       # 答案 chunk 库内定位 + 长 chunk 中段句子展示
# 已有证据：docs/agentic_recall_entrance_diagnosis.md / agentic_bm25_jieba_ab.md / agentic_query_vision_ab_result.md
```
