# 检索环节 A/B：jieba 中文分词重做 BM25 → sparse_weight 权重甜点

> 回答核心问题：**按 `docs/agentic_recall_completeness_correction.md` 的建议，给 BM25 接入 jieba 中文分词后，能否让"假性失效"的 BM25 通道复活，并为 hybrid 融合找到一个"只补不扰"的权重甜点？**
>
> 实测脚本：`_ab_bm25_jieba.py`（三入口收敛）、`_ab_bm25_jieba_grid.py`（sparse_weight 网格）
>
> 样本 30 题（6 capability × 5，seed=7），检索器 `OpenDomainRetriever`（55095 块，bge-small-zh-v1.5），主口径 = GT 句子覆盖率 `retrieval/recall_metrics.py::sent_coverage`，交叉参照 = 整段 IoU hit@k。
>
> 结论一句话：**jieba 让 BM25 从"死通道"(覆盖率恒 0) 复活为"真实通道"(93% 题能捞到相关块、cov@10 +0.011)，但仍非召回上限主杠杆；`sparse_weight` 最优解 = 0.2（增益最高 + 扰动最低），三处默认已由 0.8 改为 0.2。**

---

## 一、背景：BM25 为什么"假性失效"

此前 `docs/agentic_recall_completeness_correction.md` 的量化证据链：

| 入口 | K=10 | K=20 | K=50 | 结论 |
|---|---|---|---|---|
| dense+bm25 | 0.27 | 0.32 | 0.36 | 混合=纯 dense，无增益 |
| dense_only | 0.27 | 0.32 | 0.36 | 同左 |
| bm25_only | **0.00** | **0.00** | **0.00** | **BM25 彻底失效** |

根因（两层，本次核实）：
1. **分词层**：旧 `_tokenize` 中文用"整段 + 2/3字字符滑窗 n-gram"——无词边界，词项与 query 错位，且 n-gram 在 `BM25Okapi` 的 IDF 下被高频字符组合稀释。
2. **环境层（本次新发现）**：当前 conda 环境 `rank_bm25` 未安装 → `BM25_AVAILABLE=False` → `hybrid_retrieve` 里 `if use_sparse and BM25_AVAILABLE` 分支**根本不执行**，`sparse_pool_ranked=[]`，RRF 只融合 dense。即当时 hybrid 的结果在机械层面就等于纯 dense。

---

## 二、改造：`retrieval/retriever.py::_tokenize`

- 中文改为 **jieba 精确模式切词**（替代字符 n-gram），新增辅助方法 `_get_jieba_tokens()`。
- **保留字母数字正则**（型号/标准号/数值参数如 `gb/t 26467`、`SIMOREG-6RA70`、`380v`）——工业语料的关键精确命中信号，jieba 词项覆盖不到。
- 去重保留（同一文本内重复词项不放大词频）。
- dense 主路径 `retrieve()` 完全不受影响；BM25 索引每次新进程按新分词重建，无残留缓存。

---

## 三、三入口 × K 收敛表（jieba 后，30 题 seed=7）

**主口径（GT 句子覆盖率）：**

| 入口 | K=10 | K=20 | K=50 | cov≥50%(K10) |
|---|---|---|---|---|
| single（纯 dense，现状） | 0.273 | 0.323 | 0.356 | 17% |
| hybrid（dense+BM25-jieba→RRF, w=0.8） | 0.281 | 0.328 | 0.358 | 17% |
| bm25_only（纯 BM25-jieba） | 0.183 | 0.217 | 0.262 | 7% |

**复活/增益信号：**

| 信号 | 旧实现 | 新实现(jieba) |
|---|---|---|
| bm25_only@K=50 覆盖≥1 句 GT 的题 | 0/30 (0%) | **28/30 (93%)** |
| bm25_only cov@10/50 | 0.000 / 0.000 | 0.183 / 0.262 |
| hybrid cov@10 vs single | +0.000 | **+0.008**（K20 +0.005, K50 +0.002）|
| hybrid top-10 多盖 GT 句的题 | 0/30 | 8/30 |

**交叉参照（整段 IoU hit@k，top-10）：**

| 入口 | hit@1 | hit@3 | hit@5 | hit@10 |
|---|---|---|---|---|
| single | 27% | 40% | 43% | 47% |
| hybrid | 30% | 40% | 47% | 47% |
| bm25_only | 30% | 37% | 37% | 40% |

### 读法
- **Q1：BM25 是否复活？→ 是。** bm25_only 从"一个都捞不到"变成 93% 的题能捞到相关块（K=50 覆盖率 0.26），hybrid 有持续正增益（全 K 档非负）。
- **Q2：hybrid 是否实质提升？→ 有限，仍是配角。** cov@10 仅 +0.008；bm25_only 的 h@10(40%) 仍低于 dense(47%)——**recall 上限(completeness)仍主要由 dense 决定**。

### 隐患（第 3 处：扰动前段）
- **质量计量与检测类 cov@10 退化**：hybrid 0.110 → 0.070（−0.040），本轮唯一负向 capability。
- bm25_only 精准度仍低于 dense → BM25 冲入前段过头时会挤占 dense 优质块。

→ 由此启动 sparse_weight 网格，找"只补不扰"的权重。

---

## 四、sparse_weight 网格（30 题 seed=7）→ 权重甜点

脚本 `_ab_bm25_jieba_grid.py`，对 `sparse_weight ∈ {0.8, 0.6, 0.4, 0.2}` 分别跑 hybrid，对照 single 基线，dense/BM25 与权重无关只算一份。

| 入口 | cov@10 | cov@20 | cov@50 | ΔCov10 | h@10 | 顶新块 | 质量计量Δ |
|---|---|---|---|---|---|---|---|
| single(dense) | 0.273 | 0.323 | 0.356 | — | 14 | — | base(0.110) |
| hybrid **w=0.8**（旧默认） | 0.281 | 0.328 | 0.358 | +0.008 | 14 | 8 | −0.040 |
| hybrid w=0.6 | 0.280 | 0.323 | 0.358 | +0.007 | 14 | 7 | −0.040 |
| hybrid w=0.4 | 0.280 | 0.323 | 0.356 | +0.007 | 14 | 8 | −0.040 |
| hybrid **w=0.2**（新默认） | **0.284** | 0.323 | 0.356 | **+0.011** | 14 | 8 | **−0.015** |

**按 capability cov@10**：安全合规 +0.027、故障诊断 +0.015、标准规范 +0.039（均正，各档保持）；工程计算/工艺原理持平；质量计量是唯一退化侧。

### 结论：`sparse_weight = 0.2` 是"只补不扰"最优解
| 判据 | w=0.8 | w=0.2 |
|---|---|---|
| 补：cov@10 增益 | +0.008 | **+0.011（更高）** |
| 扰：质量计量退化 | −0.040 | **−0.015（收窄 62%）** |
| 扰：h@10 下降 | 0（持平） | 0（持平） |
| 补：顶新块数 | 8 | 8（保持） |
| cov@20/50 | +0.005/+0.002 | 持平/+0.000（不退化） |

关键机制：降权**不牺牲"补"的能力**（新块数不变、cov@10 反升），因为 BM25 冲入前段的低质块被压低后不再挤占 dense 优质位；同时**显著缓解"扰"**（质量计量退化收窄）。不存在"零退化"完美点——质量计量在 dense 侧本就最弱(0.110 打底)，任何 sparse 融合都会轻微扰动它，但 w=0.2 是全局最优。

---

## 五、已落地改动（sparse_weight 0.8 → 0.2）

为保证"启用 hybrid 时用的是 A/B 验证过的最优点"，三处默认值统一改为 0.2：

| 文件 | 位置 | 改动 |
|---|---|---|
| `retrieval/retriever.py` | `hybrid_retrieve` 签名 | `sparse_weight: float = 0.8 → 0.2` |
| `agentic/pipeline.py` | `_handle_retrieve` params.get | `"hybrid_sparse_weight", 0.8 → 0.2` |
| `agentic/planner.py` | `base_retrieve_params` | `"hybrid_sparse_weight": 0.8 → 0.2` |

> 说明：planner 的 `base_retrieve_params` 是模板图 params 注入源，会覆盖 pipeline 的兜底默认，故三处必须一致改才闭环。
>
> **对生产零影响**：`hybrid` 参数默认仍为 `False`（`pipeline.py::_handle_retrieve` / `planner` 的 `_default_retrieve_hybrid`），当前生产仍是纯 dense `retrieve()`；本次改动只影响"未来启用 hybrid 时"的默认权重。

---

## 六、结论与决策

1. **jieba 接入有效**：BM25 从"死通道"(覆盖率 0) 复活为"真实通道"(93% 题捞到相关块)，但 **recall 上限仍主要由 dense 决定**，BM25 是温和增益的配角，不改变"补全度主杠杆在 query/证据前向"这一既定判断。
2. **sparse_weight=0.2**：实现"只补不扰"——cov@10 增益最高(+0.011)且扰动最低(质量计量退化收窄到 −0.015)。
3. **生产默认保持 `hybrid=False`**（纯 dense），本改动仅预设了启用 hybrid 时的正确默认；是否在后续把 hybrid 用作 hard-miss 兜底（dense 零覆盖时启）可按需再评估。

## 复现
```bash
cd LINS-Industrial
python _ab_bm25_jieba.py --n 30 --seed 7          # 三入口收敛 + 复活/增益信号
python _ab_bm25_jieba_grid.py --n 30 --seed 7 --weights 0.8 0.6 0.4 0.2
# 结果: results/ab_bm25_jieba.json / results/ab_bm25_jieba_grid.json
# 依赖: jieba 0.42.1 + rank_bm25 0.2.2（缺失时 BM25 通道自动降级为空）
```

