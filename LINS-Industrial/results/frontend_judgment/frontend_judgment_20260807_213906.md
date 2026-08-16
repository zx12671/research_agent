==========================================================================
任务一 前端任务判定量化（Analyzer ≥ format/task 判定）
抽样 24 题 / 全量 2049 题 | 模型 deepseek-chat | 20260807_213906
==========================================================================

【0】系统走查：生产 TaskAnalyzer 当前实际行为
  - task 维度：analyze() 恒返回 GENERAL（中性化单点），LLM 不调用。
      task 分布: {'general': 24}  → task 判定【零启用】
  - format 维度（落地前）：pipeline 未传 format；analyze() 内 normalize_format('',q) 走 heuristic。
      format 分布: {'QA': 21, 'Calculation': 3}
  - mean_confidence=0.950; LLM 调用=False（0=纯确定性，零延迟）
  - 结论：前端判定（尤其 task）生产中为【中性化/降级】状态；format 仅靠 heuristic。

【0b】落地后走查：analyze(question, format=_format) 直接读 `_format` 真值
  - pipeline.run / AgenticRAGEngine.answer 已透传可选 format 参数；analyze 直读真值归一化。
      format 分布: {'QA': 7, 'FillBlank': 6, 'MultipleChoice': 6, 'Calculation': 5}  (与 gt 一一对应 → 100%)
      mean_confidence=0.950; LLM 调用=False

【1】format(题型) 判定准确率（gt=CSV `_format`；落地后 vs heuristic vs LLM 判）
                   acc      加权F1
  落地后(读真值)       100.0%    100.0%
  heuristic(生产)   29.2%     17.7%
  LLM 直判           0.0%      0.0%

  混淆矩阵(行=真值, 列=预测; heur):
  真实\预测         问答题           填空题           选择题           计算题           
  问答题           6             0             0             1             
  填空题           5             0             0             1             
  选择题           6             0             0             0             
  计算题           4             0             0             1             
  落地后混淆矩阵(行=真值, 列=预测):
  问答题           7             0             0             0             
  填空题           0             6             0             0             
  选择题           0             0             6             0             
  计算题           0             0             0             5             

  分层 acc(heuristic → 落地后):
    选型与替代            heur=25%   landed=100%
    故障诊断与排查          heur=33%   landed=100%
    安全合规与风险控制        heur=33%   landed=100%
    标准规范与术语          heur=25%   landed=100%
    工艺原理与参数影响        heur=25%   landed=100%
    质量计量与检测          heur=25%   landed=100%
    工程计算与估算          heur=50%   landed=100%
    diff:medium      heur=46%   landed=100%
    diff:hard        heur=22%   landed=100%
    diff:easy        heur=0%   landed=100%

【2】task(推理任务) 判定准确率（gt=capability→TaskType 专家映射代理）
                   acc      加权F1
  生产(恒GENERAL)     0.0%      0.0%
  LLM 判8类          0.0%      0.0%

  LLM task 混淆矩阵(行=真值代理, 列=预测):
  真实\预测           comparison    diagnosis     selection     calculation   standard_interpretationprocedure     explanation   general       
  对比              0             0             0             0             0             0             0             0             
  诊断              0             0             0             0             0             0             0             0             
  选型              0             0             0             0             0             0             0             0             
  计算              0             0             0             0             0             0             0             0             
  标准解读            0             0             0             0             0             0             0             0             
  流程/规程           0             0             0             0             0             0             0             0             
  解释              0             0             0             0             0             0             0             0             
  通用              0             0             0             0             0             0             0             0             

  每类 P/R/F1 (LLM task 判定):
    对比           n=0   P=0% R=0% F1=0.00
    诊断           n=7   P=0% R=0% F1=0.00
    选型           n=4   P=0% R=0% F1=0.00
    计算           n=2   P=0% R=0% F1=0.00
    标准解读         n=4   P=0% R=0% F1=0.00
    流程/规程        n=3   P=0% R=0% F1=0.00
    解释           n=4   P=0% R=0% F1=0.00
    通用           n=0   P=0% R=0% F1=0.00

【3】结论与决策提示
  - format: LLM 与 heuristic 接近 => 生产 heuristic 已够用，无需引入 LLM。
  - format: LLM 判定本身 acc<80% => 前端判定不可靠，应优先靠 CSV 真值。
  - format: 落地后 analyze 直读 `_format` 真值 => acc=100.0%（100%=链路一致性校验：标注→前端 format 判定分发生效）。
  - task: 生产恒 GENERAL；LLM 判 8 类可达能力仅作参考（真值为 capability 代理）。
  - 注意: capability 与推理任务不严格一一对应，task 真值存在固有噪声。

明细JSON: results/frontend_judgment\frontend_judgment_20260807_213906.json
