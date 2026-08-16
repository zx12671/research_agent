# 核查：P1 语义级多视角改写(SemanticQueryRewriter)接入生产的真实端到端结果

> 任务：把 P1 语义级多视角改写从实验脚本接入生产 agentic 路径，并在真实 agentic 路径复跑。
> 结论：**完成了生产接入（默认关、零侵入现状、链路跑通），但真实端到端 12 题验证显示：
>  语义多视角对已较强的 base(dense+hybrid) 无平均增益（cov −0.38pp、hit@1/3/5 完全一致、
>  hard-miss 0 捞回）；唯一可测的正向信号是"相关块 rank 只前移不后退(3 题前移/0 后退)"。
>  语义鸿沟 hard-miss 并未因此路径被实质解决** —— 与前几轮文档判定一致，不应默认开。

---

## 一、生产接入落地（agentic/planner + pipeline，默认关，零侵入现状）

### 1. `agentic/pipeline.py`
- `GraphExecutor.__init__` 新增可选参数 `semantic_rewriter=None`（存 `self.semantic_rewriter`）；默认 None → 生产行为不变。
- `_handle_retrieve` 新增 P1 生产分支 **`semantic_mv`**（`params.get("semantic_mv", False)`，默认关，优先级最高）：
  - 条件：`semantic_mv=True` 且 executor 注入了 rewriter，才进；
  - `views = semantic_rewriter.rewrite(question, graph.task_analysis)`（含原 query 首保底）；
  - 每个视角 `hybrid_retrieve(v, k, use_ked, use_sparse=True, sparse_pool=50, dense_weight=1.0, sparse_weight=0.2)`；
  - `results = retriever._fusion_merge(per_view, k)` → 与其它分支同类型（含 `.chunks`）落到证据消费。
  - 异常/未注入 rewriter → 静默降级标准检索，生产零回归。
- `AdaptiveAgenticPipeline.__init__` 透传 `semantic_rewriter` 给 executor。

### 2. `agentic/planner.py`
- `StrategyPlanner.__init__` 新增 `retrieve_semantic_mv=False`（与 `retrieve_hybrid` 同构，默认关）；
- `base_retrieve_params` 增加 `semantic_mv/mv_max_views/mv_sparse_weight`（随 `_default_retrieve_semantic_mv`）；
- 高级任务模板 `retrieve_1`：开启时 `semantic_mv=True` 并**关闭机械 multi_query/use_fusion**（避免语义视角×切逗号叠加稀释）；`retrieve_2` 补检索同理 keep semantic_mv。

### 3. 驱动脚本 `_e2e_semantic_mv_prod.py`
真实 agentic 路径组装（TaskAnalyzer→StrategyPlanner 模板图→按 `_handle_retrieve` 分派出证据池），
base / sem_mv 两臂同题库对比；`--real` 走真实 DeepSeek（analyzer/planner + LLM 语义视角），
`--mock` 走确定性视角链路回归。

## 二、真实端到端验证（--real，seed=7，12 题，真实 DeepSeek 语义视角）

| 指标 | base(现状) | sem_mv(语义多视角) |
|---|---|---|
| avg_cov@10 | 39.64% | 39.26%（delta −0.38pp） |
| hit@1/3/5 | 3/4/4 | 3/4/4（完全一致） |
| 相关块 rank 前移/后退 | — | **前移 3 / 后退 0** |
| base top-10 未命中(IoU≥0.20) | 7 题 | sem_mv 捞回 **0** 题 |

链路已确认真实走通：`[rewriter] mock=False`（真实 LLM 语义视角），sem_mv 臂 `params.semantic_mv=True,
multi_query=False`，各语义视角确实注入新 chunk（如题1 语义视角新增 5~9 个新 chunk_id）。

## 三、为什么"语义鸿沟 hard-miss 未被解决"（诚实分析）

1. **评测口径的"伪 hard-miss"**：7 个 base `hit=None` 中有相当一部分，其**相关块已在 base top-10 内**，
   只是整段 bigram IoU<0.20 硬阈值没判中（如题3 除冰液：base top-10 第7 已有"按 AS5901 进行喷水防冰
   试验"，融合后该块 rank 前提到第4/5）→ 检索层本就没 miss，是口径误判，语义改写救不了"假 miss"。
2. **真正的检索 miss（术语改写同义 VDRM 类）**：LLM 语义视角调换措辞后，dense+BM25 仍搜不到 GT——
   单纯"多视角改写 + 融合"无法把 top-10 外的东西拉进来。需要更强机制（证据引导二次检索、精确 token
   命中、或扩大候选池），不在本路径能力内。
3. **`_fusion_merge` 的 RRF + 原 query 首保底**使 top-10 与 base 高度重合，语义视角的独有贡献
   （新 chunk）多被截断在 10 之外——这解释了"有新增 chunk 却 cov 持平"。

## 四、结论与后续建议

- ✅ **接入已落地且真实链路验证过**：`semantic_mv` 生产分支经真实 Agentic pipeline 触发，rewriter
  真实 LLM 语义视角生效，各视角 hybrid 检索注入新证据，原 query 保底融入 RRF。
- ⚠️ **不建议默认开**：12 题真实对比无平均增益、hit 无变化、hard-miss 0 捞回；唯一正向信号是
  "相关块 rank 只前移不后退(3/0)"，这是微弱且不转化为答案质量提升的信号。与
  `docs/agentic_mv_query_rewrite_ab.md` L72-73 既有判定一致。
- 🔧 **要真正治 hard-miss，候选方向**（非本路径范围，留待后续）：
  a) 提高每视角候选池（topk_per_view 20→50）+ 增大 `retrieve_k`，给 RRF 更多语义新 chunk 进 top-N 的空间；
  b) 引入证据引导二次检索（`_build_evidence_guided_query`，已有）+ 稀疏精确命中的强结合；
  c) 对 base hit=None 的样本做 **IoU 阈值可解释的诊断**（区分"真 miss"与"口径误判"），避免把后者计进 hard-miss。

## 复现
```bash
python _e2e_semantic_mv_prod.py --max-q 12 --seed 7 --real   # 真实语义视角（需 DEEPSEEK_KEY）
python _e2e_semantic_mv_prod.py --max-q 12 --seed 7          # mock 确定性视角链路回归
```
产物：`results/_e2e_semantic_mv_prod_real.json`、`results/_e2e_semantic_mv_prod_mock.json`
