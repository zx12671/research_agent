# 件① 提纯（证据精选）· 结构探针报告

- 探针：`_probe_organize_ab.py`（零-LLM，纯组织侧结构度量）
- 明细：`results/organize_ab_structure_35.json`（35 题，A=6 / B=29）
- 口径：同一 top-10 检索结果（`OpenDomainRetriever.retrieve(q,k=10)`），
  `EvidenceOrganizer`（无提纯） vs `RerankTruncOrganizer`（提纯档=生产默认）对比。

## 一、背景与决策点
生产已装配 `RerankTruncOrganizer`（`experiments/exp1_agentic_rag.py` L630-631），
且 `agentic/pipeline.py` 的 `GraphExecutor._handle_organize` 通过 `self.organizer.execute(...)`
不传参、吃其子类默认 `rerank=True/truncate=True/signal=lex/max_chars=6000`，＝提纯档在生产真实执行路径上 **打开**。
但此前从未按 A/B 类分层量化它对**证据输入**到底改了什么。

## 二、落地改动（默认行为不变）
- `agentic/pipeline.py::_handle_organize` 新增 `node.params["organize_panel"]` 显式开关：
  - 缺省/None → 保持现状（走 organizer 自身默认）；
  - `{"rerank":False,"truncate":False}` → 显式关闭提纯档（= EvidenceOrganizer 基线，供 A/B）；
  - `{"rerank":True,...,"max_chars":N}` → 显式开/调参提纯档。
- 纯增量：不改检索、不改 reason prompt、不改默认行为，可单点回退。

## 三、实测结论（35 题，离线 HF_HUB_OFFLINE=1）

| cls | n | in | base_n | rt_n | drop% | ctx_base | ctx_rt | lexB@5 | lexR@5 | reorder |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 6 | 10 | 9.8 | 9.8 | 0% | 1438 | 1438 | 0.13 | 0.13 | 2.3 |
| B | 29 | 10 | 9.9 | 9.9 | 0% | 1797 | 1797 | 0.09 | 0.10 | 3.1 |
| ALL | 35 | 10 | 9.9 | 9.9 | 0% | 1735 | 1735 | 0.10 | 0.11 | 3.0 |

### 解读
1. **reorder≈3.0**（B 3.1 / A 2.3）：L2 词法顶置在生产真实路径**生效**，平均每题 top-5
   约 3 块被换位——重排是提纯档当前唯一真实贡献。
2. **drop%=0.0、ctx_base==ctx_rt**：`max_chars=6000` 下，当前 top-10 证据总量仅 ~1.5-1.8k 字符，
   **远低于截断阈值 → 截断完全空转**。提纯档"砍噪声"能力在实际链路上从未启用。
3. **对 A 类弱**：A 类 `lexB@5=lexR@5=0.13` 持平（B 类微升 0.09→0.10）；A 类 reorder 也最低(2.3)。
   "词法顶置"对 A 类(GT 超长、证据点散布)帮助有限。
4. 单独把 `max_chars` 调小并不能救：实测 `--max_chars 2000` 与 `6000` 结果完全一致
   （真实 top-10 总长 <2000，两个阈值都截不掉）。截断要对 A 类起作用，需**放大证据池**
   （多 hop / 更大召回）使其超阈，或配 `max_chunks` 强制按量截。

### 判定
- 件①"接入"已完成（原为隐性默认，现显式可开关 + 可 A/B 分层量化）。
- **结构性收益有限的重排为主、砍噪声空转**；对 A 类是中性偏弱。
- 这说明"提纯"不是 A 类(GT 超长)的主解药——A 类的缺口更在**件②按需触发+件③证据聚焦**
  （证据到位后如何抽取利用），而非把已有 top-10 再修剪。

## 四、回退
- 探针用 `EvidenceOrganizer` 即无提纯；生产改回 `EvidenceOrganizer()`（`exp1_agentic_rag.py` L631）即可。
- `organize_panel` 开关本身不改默认，风险为零。
