# StrategyPlanner 专项量化探针报告 (v2)

- 样本数: 8 | scorer: rule

## 1. LLM高级图 vs fallback 形状
- advanced(含 verify/decide/多检索或 meta 声明): 8 (1.0)
- fallback 形状(4节点线性无高级): 0 (0.0)
- 规划图 meta: branching_true=1.0, multi_hop_true=1.0, verify_true=1.0
## 2. 节点数
- 规划图平均节点数: 6.875
- 执行日志平均节点数: 5.5
## 3. 二次检索 (planned)
- 规划图 retrieve 节点>1 占比: 1.0
- 平均 retrieve 节点数: 2
- 说明: result 无 second_retrieval_triggers 字段；此处按规划图 retrieve 节点数>1 统计 planned 二次检索；执行层是否真正触发以 execution log / 运行日志为准（实测 retrieve_2 会真实二次检索，未像 retrieve_1 那样被 pre_evidence 短路）。
## 4. verify 节点
- 平均 verify 节点数: 1.0
## 5. 按规划图分组的 Accuracy/SV 交叉
- advanced 组: count=8, avg_adj=1.25
- fallback 组: count=0, avg_adj=None
## 6. 总览
- overall avg_adjusted: 1.25
- SV 违规数: 0

> 注：advanced/fallback 按 execution_graph 真实结构（node_summary/node_types/has_*）判定，见模块 docstring。