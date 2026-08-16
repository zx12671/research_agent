# S5 证据组织量化检查报告 —— 判定【有条件通过（merge 膨胀为正式发现）】
> 状态：检查完成 · **未改任何生产代码**（仅诚实补测，修正此前误报）
> 生成：20260808 · 对标 `docs/stagewise_debug_plan.md` S5 检查项
> 脚本：`_diag_evidence_s5.py`（N=120，全能力均匀抽样）+ `_diag_merge_inflate.py`（N=40，**真实节点处理器**复核）
> 结果：`results/evidence_s5_audit.json` + `results/s5_merge_inflate.json`
> 口径：生产对标 `RerankTruncOrganizer`（落地组织器, truncate=True, max_chars=6000）/ 对照 `EvidenceOrganizer`（纯保真）
> 触发修正：针对"脚本是否脱离真实 agentic 系统"质疑，新增直连 GraphExecutor 真实节点 `_handle_merge` 的复核，发现并修正此前"merge 无忧"误报。

---

## 一、检查目标与判据
S5 目标：**证据完整且 prompt 不膨胀**。
- 判据A 完整性：`get_context()` 是否含全部【生产 organizer 去重后】chunk → 缺失应 = 0。
- 判据B 去重保留率：organize 后 chunk 数 / 检索命中数（生产截断不得造成 chunk 丢失）。
- 判据C 长度与比：evidence 长度、full_prompt 长度、evidence/prompt 比、超长样本(> 6000/8000 字符)。
- 判据D merge/重复膨胀：reason 是否把 organizer + 原始 chunk 同时并入 prompt；证据串内是否出现重复内容块。

## 二、量化结果（N=120，覆盖选型/工艺/标准/质量/诊断/安全/工程计算）
| 判据 | 指标 | 结果 | 判定 |
|---|---|---|---|
| A 完整性 | get_context 含全部去重后 chunk | **100%** | ✅ |
| A 完整性 | total_missing_chunks / missing_case | **0 / 0** | ✅ 无缺失 |
| B 保留率 | prod_dedup_retention / pure | **99.4% / 99.4%** | ✅ 截断不丢块 |
| C 长度 | ctx 均值 / p95 | 2054 / 2621 字符 | ✅ 轻量 |
| C 长度 | prompt 均值 / p95 | 3789 / 4376 字符 | ✅ 远低于窗口 |
| C 比 | evidence / prompt | **53.8%** | ✅ 非膨胀 |
| C 超长 | ctx >6000 / >8000 | **0 / 0** | ✅ 无截断丢分样本 |
| 截断 | cases_losing_gt_core（被截 block 含 GT≥0.30） | **0** | ✅ |
| D merge | 直接路径(`_handle_reason`)证据串内重复>1 | **0** | ✅ 直接路径无重复并入 |
| D merge | **merge 分支(`_handle_merge`)膨胀 accumulated/organize_only** | **2.63×（max 2.79）** | ⚠️ 同 chunk 被输出两遍 |
| D merge | merge 分支样本 ≥1.5× 占比 / 内容重复率 | **100% / 1.0** | ⚠️ 正式发现 |

## 三、关键归因（诚实澄清 + 触发修正）
1. **直接路径无膨胀（原先结论，仍成立）**：`pipeline.py::_handle_reason` 中
   `evidence_str = organized.get_context() if organized else self._evidence_to_string(evidence)`
   为 **if/elif 二选一，非叠加** → 在 organize→sufficient→reason 路径下不会同时并入 organizer 产物 + 原始 chunk，120 题证据串内同内容指纹重复 = 0。
2. **⚠️ merge 分支存在真实重复并入（触发修正后的正式发现）**：当图走到
   `decide=insufficient → retrieve_2 → merge_1` 时，走的是 **`_handle_merge`（pipeline.py:707）而非 `_handle_reason`**。该处理器拼接：
   ① `last_organized.get_context()`（全量 chunk verbatim）＋② `evidence_cache` 全部 raw chunks（同一 chunk 内容再输出一次）＋③ reasoner summaries。
   → **同 chunk 被输出两遍**。真实节点处理器复核（N=40）：
   `accumulated_context / organize_only ≈ **2.63×**（max 2.79）、100% 样本 ≥1.5×、内容重复率 **1.0**`。
   **真实基准日志（60 次图执行）merge 触发 ≈ 28%（17 次）** —— 非偶发情形，约三成问题受影响。
3. **为何此前误报"merge 无忧"**：`_diag_evidence_s5.py` 只调 `organizer.execute + format_prompt(evidence=ctx)`，**绕过了 GraphExecutor 的 merge 节点**，仅覆盖 direct-reason 路径 → 漏测 `_handle_merge` 的二次并入。新增 `_diag_merge_inflate.py`（直连真实 `_handle_organize`/`_handle_merge`）后修正。
4. **为何生产截断不触发**：`RerankTruncOrganizer` 默认 max_chars=6000，而 120 题 ctx 均长 2054、p95 2621，普遍远低于阈值 → 截断分支实测不触发（0 超长样本）。"高 recall 但被截断丢分"在本语料量级下不成立。
5. **早前 3 例"缺失"为计数口径假阳性**：初版按 `(score=\d+\.\d+)` 计数，部分 chunk 正文含字面 "(score=…)" 造成误计；改为"content 前缀句归一后必须在 ctx 中逐字出现 + n_doc==organized_count"双条件后，真实缺失 = 0。

## 四、结论
- **S5 证据组织语义通过**：organize/get_context 保真完整（0 缺失、100% 保真、去重保留 99.4%），直接路径不膨胀（evidence/prompt 53.8%、无超长、无重复并入）。group 仅作展示视图，Recall-Safe 契约成立。
- **✅ merge 分支已修复（20260808 落地生产代码）**：原缺陷——`_handle_merge` 把 organizer 全量 ctx + evidence_cache 全部 raw chunks 同时并入 → 同 chunk 输出两遍（膨胀 2.63×、重复率 1.0、触发率≈28%）。已在 `agentic/pipeline.py::_handle_merge` 落地修复：**raw 阶段按 chunk_id 跳过已被 organized 覆盖的 chunk，仅保留第二跳新增 chunk**（`retrieve_2` 中不在 first-hop organized 的 chunk）。属独立小改，不影响 direct-reason 路径。
- **修复效果（`_diag_merge_inflate.py`，N=40 真实节点处理器 A/B）**：
  | 指标 | 修复前 | 修复后 |
  |---|---|---|
  | accumulated/organize_only 平均 | **2.63×** | **1.04×** |
  | 最大 | 2.79× | 1.08× |
  | 样本 ≥1.5× 占比 | **100%** | **0%** |
  | RAW 与 ORGANIZED 内容重叠 | ~1.0 | **0.0** |
  | organized 全量在 accumulated | — | **true（100% 保留，Recall-Safe 0 缺失）** |
  - 基线存档 `results/s5_merge_inflate_BEFORE.json`（2.63×），修复后结果 `results/s5_merge_inflate.json`（1.04×）。
- 整段判定：**merge 修复后整体通过**（直接路径不膨胀 + merge 路径去重后 1.04×、0 缺失、无重复并入）。
- 后续建议（非本次问题）：对偶发超长标准类 chunk 可加"单块长度上限"的软监控，但当前无样本触发。

## 复现
```bash
python _diag_evidence_s5.py     # N=120 直接路径完整性与长度
python _diag_merge_inflate.py   # N=40 真实节点处理器 merge 分支膨胀复核（修复后 1.04×）
```
