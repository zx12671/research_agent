# evidence-forward 可行性快检（第3项立项把关）

> 结论：**可行且零成本，形式比原设想更优** —— 无需第二次 LLM query，可用 `document_id` 精确回捞该文档全部块。
> 快检方式：纯静态读 `industrybench_chunks.jsonl`，不检索、不加载模型，秒级出结果。

## 快检脚本
`_probe_evidence_forward_feasibility.py`（只依赖 `knowledge_corpus/chunks/industrybench_chunks.jsonl` + CSV，30 样本、seed=7）

## 1. 索引分块结构事实（chunk 元数据 schema）

每一条 chunk 自带完整分块元数据：

| 字段 | 示例 | 说明 |
|---|---|---|
| `chunk_id` | `3f3d53d2895b` | 块唯一 ID |
| `document_id` | `a49135506044d598` | **所属文档 ID（精确回捞键）** |
| `chunk_index` | `0` | **该文档内第几块（0 起）** |
| `total_chunks` | `6` | **该文档切分后的总块数** |
| `source` | IndustryBench | 来源 |
| `capability` / `industry` | 选型与替代 / 冶金… | 分类标签 |
| `content` | … | 块文本 |

**关键发现**：既有 `document_id`，又有 `chunk_index` 和 `total_chunks`。
→ 这意味着 evidence-forward **不一定需要"第二段 query → 二次检索"**；可以直接用 `document_id` 在 `self.chunks` / JSONL 中**精确回捞该文档的全部块**（包括 top-10 未出现在候选池的中后段），并按 `chunk_index` 还原文档内部顺序。

## 2. Q1：同一 document_id 是否有多个 chunk —— 强成立

- 语料共 **55,095** 块，**2,048** 个独立 `document_id`；
- 块数 > 1 的 doc 达 **2,028 个（99%）**；
- 单 doc 块数最大达 **523**（<sup>1</sup>）。

→ 同源"顺藤摸瓜"有海量可捞内容，非孤块单例。

## 3. Q3：A 类超长 GT 的源文档块数 —— 靶区完全命中

30 样本（seed=7）中，用"GT 首句 vs chunk 头 bigram overlap"定位每题的源文档，得到：

- **A 类（超长 GT ≥ 50 句）共 5 题，全部可顺藤**（源 doc 块数 235 / 148 / 88 / 71 / 52），且首段 JT 高达 0.62–0.93 → 首段能**准确定位到正确文档**；
- **全部 30 题中 30/30 源 doc 块数 ≥ 2** → 方案覆盖面远超 A 类靶区。

| 典型 A 类题 | GT 句数 | 首段 JT | 源 doc 块数 |
|---|---|---|---|
| 纺织测量离散程度 | 202 | 0.87 | 235 |
| 电子测量离散特性 | 186 | 0.62 | 148 |
| GB/T 2423.59—2008 | 76 | 0.68 | 88 |
| 节水灌溉后评价 | 102 | 0.84 | 71 |
| 钼蓝分光光度测硅 | 68 | 0.93 | 52 |

## 4. 实现形态（比原设想更优：免第二次 LLM query）

**第0级：doc-回捞（推荐，确定性、零 LLM）**
1. 第一轮 `retrieve(q, k=10)` 拿 top-10 chunk，提取 `(document_id, chunk_index, total_chunks)`；
2. 用 `document_id` 从 `self.chunks` / JSONL **直接回捞该 doc 全部块**；
3. 与 top-10 合并，按 `chunk_index` 保证文档内部顺序，可按"doc 命中即优先带整档"或位置加权进候选；
4. 仅检索/候选侧改动，不动 planner/solver。

（若第0级不足，落回"top-1 块实体 → 二次 query → 二次检索"的第1级；但快检表明第0级已覆盖 A 类全部。）

## 5. 成本与风险

| 维度 | 结论 |
|---|---|
| 计算 | 纯 dict 回捞，无向量检索、无 LLM，秒级 |
| 改动面 | 检索候选合并逻辑，可回滚、不影响现有 top-10 命中 |
| 风险 | ① 整档回捞可能带进噪声块（可用 JT/score 过滤或限制每 doc 最多取 N 块）；② 与"仅取 top-10"口径不一致，需明确验收对比基准 |

## 6. 建议立项（待确认）

下一步写 `_probe_evidence_forward.py`，30 样本、seed=7、句级覆盖率口径（与 A/B 一致），对比：
`base(retrieve top-10)` vs `base+doc回捞`；
验收：cov≥50% 达标题数（当前 5/30）是否翻倍、平均覆盖率是否抬升。

---
<sup>1</sup> 快检脚本第 Q1 节输出：块数分布前 15 = [523, 417, 410, 402, 395, 389, 385, 369, 338, 329, 309, 297, 296, 288, 286]。
