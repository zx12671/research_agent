# 方案：P1 语义级多视角 Query 改写 + 句覆盖口径下完整 AdaptiveAgenticPipeline A/B

> 状态：**已执行 → 证伪性结论（不默认开）**；完整结果见 `docs/agentic_mv_query_rewrite_ab.md`
> 关联主线：`docs/agentic_bm25_jieba_ab.md`、`docs/agentic_recall_entrance_diagnosis.md`、
>          `docs/agentic_query_vision_ab_result.md`、`docs/agentic_retrieve_hybrid_ab.md`
> 复用先例：`_e2e_agentic_ef.py`（完整 agentic 链路注入 A/B）、`_system_flow_quant.py`（MockLLM/真实 LLM 双模式）、
>           `_ab_bm25_jieba.py`（句覆盖率主口径 + K 收敛表）
> 对应脚本：`_ab_mv_query_rewrite.py`、模块 `agentic/query_rewriter.py`

---

## 0. 一句话目标

把 P1「补全度主杠杆 = 语义级多视角 query 改写」落在**完整 `AdaptiveAgenticPipeline`** 上，
并以 **GT 句子覆盖率 `sent_coverage`** 为主验收口径做三臂 A/B，验证改写对检索证据池与最终答案的
真实增益（或证伪——沿用既有纪律：无增益则不默认开）。

---

## 1. 背景与依据（为什么做多视角、为什么口径是句覆盖）

1. **上一轮已定位主杠杆方向**：`docs/agentic_query_vision_ab_result.md` 明确——
   「若继续走分解方向 → 必须换成**语义级子视角**（复用 planner task/expected_evidence 生成 2~4 个
   独立视角 query，而非切逗号）；当前正则子串稀释语义是根因」。P1 正是接这个结论顺延做。

2. **既有 multi_query 是"机械子句"**：`multi_query_retrieve`（retriever.py:558）依赖 `ked.decompose()`
   按标点/子句切分原查询，不是"语义视角"——对工业长问（标准号 + 参数 + 场景同句）覆盖不足。

3. **第二跳是确定性抽取（无 LLM）**：`_build_evidence_guided_query`（pipeline.py:239）从已命中 chunk
   抽取标准号/数值对，够精确但只覆盖"判别键"，不覆盖"语义视角缺失"型补全。

4. **口径已统一为句覆盖**：`retrieval/recall_metrics.py::sent_coverage()` 是唯一主验收口径
   （P0 已落地）。检索层 A/B（`_ab_bm25_jieba.py`）已用，但**从未在完整 agentic pipeline 的证据池上
   用过句覆盖**——本轮补上这一段空白（检索层 cov@10 本身 +0.008 的增益，要看是否传导到最终证据池）。

5. **纪律约束**：生产 `hybrid=False` 默认纯 dense，零影响。因此多视角改写器**新建独立模块，
   不进生产默认分派**，仅 A/B 脚本注入验证；有正向结论再考虑沉淀为可选参数。

---

## 2. 语义级多视角改写器（新模块 `agentic/query_rewriter.py`）

### 2.1 定义

```python
class SemanticQueryRewriter:
    def __init__(self, llm_client, model_name="deepseek-chat",
                 num_views=3, views_per_call=4, temperature=0.3):
        ...
    def rewrite(self, question: str, task_analysis=None) -> List[str]:
        # 返回 1+ 个语义子视角 query 列表（至少含原问题保底）
```

### 2.2 输入信号（复用 planner 已有语义信号，呼应文档建议）

- `question`（原文）
- `task_analysis.task`（8 类推理任务：comparison/diagnosis/...）
- `task_analysis.expected_evidence`（comparison/symptom/specification/formula/...）
- `task_analysis.original_question`

### 2.3 Prompt 设计（LLM 生成 2~4 个语义视角）

```
你是工业知识库检索的查询改写器。给定一个工业问答与它的任务类型/期望证据类型，
生成 2~4 个"语义互补"的子视角检索 query。要求：
- 每个视角聚焦一个独立语义面（例如：标准号维度 / 数值参数维度 / 对象原理维度 / 工程实操维度）
- 保留型号/标准号/数值原文，不拆分
- 视角之间尽量不重叠，合起来覆盖原问题全部语义
- 输出 JSON 数组（仅 query 字符串列表）
```

**视角数自适应**：comparison→≥3（A 和 B 分述 + 对比准则）；procedure→步骤维度；
calculation→公式+参数维度；general→对象+场景维度。保底 `[question]`。

### 2.4 容错

- LLM 失败/超时/非 JSON → 降级 `[question]`（与纯 base 等价，不引入噪声）
- 每视角去重、截断 ≤120 字
- 视角数 θ：num_views 个 → 逐个 dense/hybrid 检索 → `_fusion_merge`/`_union_merge` 融合 top-k
  （复用 retriever 现有融合，见 3.2）

---

## 3. A/B 注入方式（不改生产默认分派）

### 3.1 驱动入口：`AgenticRAGEngine.answer()`（exp1_agentic_rag.py:404）

`AgenticRAGEngine` 内部即构造并使用 `AdaptiveAgenticPipeline`（exp1_agentic_rag.py:628），
因此"完整 `AdaptiveAgenticPipeline` A/B"与 `_e2e_agentic_ef.py` 同构：
`base_engine = AgenticRAGEngine(...)` 原样；增强引擎用子类/包装覆写检索分派注入改写器。

### 3.2 注入点：`pipeline.GraphExecutor` 的检索分派（pipeline.py:311 `_handle_retrieve`）

- 现状分派：`hybrid` 分支 / `multi_query_retrieve` 分支 / 单 `retrieve()`。
- 注入方案：新建 `MVGraphExecutor(subclass of GraphExecutor)`，覆写 `_handle_retrieve`：
  - 调用 `SemanticQueryRewriter.rewrite(q, task_analysis)` 得 views
  - 每个 view 走 `retriever.hybrid_retrieve(view, k, use_sparse)`（dense 臂即 `retrieve()`）
  - 用 `retriever._fusion_merge`/`_union_merge` 并池 top-k
  - 其余 organize/reason/verify/final 全走生产原逻辑
- A/B 脚本装配：`MVAgenticRAGEngine(AgenticRAGEngine)` 用 `MVGraphExecutor` 替换默认 executor。

> 说明：不改 `pipeline.py` 生产分派；通过子类覆写完成注入，**零侵入生产默认路径**。
> 若 A/B 正反馈，后续再把改写器作为 `SemanticQueryRewriter` 可选组件注入 `GraphExecutor.__init__`。

### 3.3 LLM 双模式（复用 `_system_flow_quant.py` 的 MockLLM/CountingLLM 骨架）

- `--mock`（默认）：改写器用 MockLLM 返回可控视角 JSON；零成本、确定性，只验证链路/覆盖率卡点不崩。
- `--real`：改写器用真实 DeepSeek；测真实端到端准确率 `RuleBasedScorer.rule_based_score`。

---

## 4. 三臂 A/B 设计

| 臂 | 检索方式 | 说明 |
|----|----------|------|
| `base` | 完整 pipeline 原样（单查询 dense + KED） | 生产现状基线 |
| `mv_dense` | 多视角改写 → 各视角 dense → 融合 | P1 主验证（不含 sparse）|
| `mv_hybrid` | 多视角改写 → 各视角 hybrid(0.2) → 融合 | P1+hybrid 联合（对照 sparse 增益）|

- 样本：沿用 `_ab_bm25_jieba.py` 的 `sample_rows` 口径（6 capability × 5 = 30 题，seed=7）；
  为控成本 `--real` 默认子集（如 8~10 题），`--mock` 全 30 题跑通。
- 数据行：`results/ab_mv_query_rewrite.json / _mock / _real / ...`

---

## 5. 评估指标（句覆盖口径，多口径并列）

主口径（唯一验收依据）：
- **证据池句覆盖率** `sent_coverage` @ K=10/20/50
  - 证据池定义：pipeline 最终喂给答案生成器的 chunk 集（`retrieval_result` / evidence_cache）
  - 与检索层 `_ab_bm25_jieba.py` 呼应，形成"检索层→证据池"传导对照
- **cov@10 ≥50% 题占比**（"拼齐答案"节流点）

交叉参照：
- 整段 IoU `hit@k`（`doc_hit_position`，不作为结论）
- 按 capability 分层 cov@10
- `--real`：`RuleBasedScorer.rule_based_score`（最终答案正确率）
- 耗时对比（多视角改写 + 多路检索的代价）

---

## 6. 交付物清单

**新增**
- `agentic/query_rewriter.py` —— 语义级多视角改写器（含 JSON 解析容错、降级保底）
- `_ab_mv_query_rewrite.py` —— 三臂 A/B（mock/real 双模式，句覆盖口径 + scorer）
- `results/ab_mv_query_rewrite_*.json` —— 结果
- `docs/agentic_mv_query_rewrite_ab.md` —— 方案 + 结果沉淀

**（视 A/B 结论）修改**
- 若正反馈：pipeline.py 增加 `SemanticQueryRewriter` 可选注入 + `base_retrieve_params` 侧开关（仍默认关）

**不动**
- 生产默认分派（`hybrid=False` 纯 dense 现状）、`_tokenize`/BM25 既有成果

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 多视角改写合并后语义稀释 → 不如 base | 用 `_union_merge`（并集）保持基 recall；分数做加权；失败即"不默认开" |
| --real 成本高/耗时长 | --real 限子集；mock 全量先验证链路 |
| 改写器 LLM 输出非法 | 严格 JSON 解析 + 保底 `[question]`，非法即退化 base 等价 |
| 融合机制不理想 | 先测 union（保 recall），再测 fusion；两实现都在 retriever 内现成 |

---

## 8. 执行顺序（确认后）

1. 建 `agentic/query_rewriter.py`（含 mock 视角生成）
2. 建 `_ab_mv_query_rewrite.py`（三臂 + mock 模式跑 30 题，验证无崩溃、看 cov 卡点）
3. `_ab_mv_query_rewrite.py --real --max-q 8` 跑真实子集，取 `rule_based_score`
4. 汇总结果 → 写 `docs/agentic_mv_query_rewrite_ab.md`
5. 正反馈 → 上报可选开通开关；零/负反馈 → 记录证伪结论，不默认开
