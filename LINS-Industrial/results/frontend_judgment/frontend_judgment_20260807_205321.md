==========================================================================
任务一 前端任务判定量化（Analyzer ≥ format/task 判定）
抽样 40 题 / 全量 2049 题 | 模型 deepseek-chat | 20260807_205321
==========================================================================

【0】系统走查：生产 TaskAnalyzer 当前实际行为
  - task 维度：analyze() 恒返回 GENERAL（中性化单点），LLM 不调用。
      task 分布: {'general': 40}  → task 判定【零启用】
  - format 维度：pipeline 未传 format；analyze() 内 normalize_format('',q) 走 heuristic。
      format 分布: {'QA': 37, 'Calculation': 3}
  - mean_confidence=0.950; LLM 调用=False（0=纯确定性，零延迟）
  - 结论：前端判定（尤其 task）生产中为【中性化/降级】状态。

【1】format(题型) 判定准确率（gt=CSV `_format`）
                   acc      加权F1
  heuristic(生产)   30.0%     17.0%
  LLM 直判          70.0%     66.6%

  混淆矩阵(行=真值, 列=预测; heuristic/生产):
  真实\预测         问答题           填空题           选择题           计算题           
  问答题           11            0             0             1             
  填空题           12            0             0             1             
  选择题           8             0             0             0             
  计算题           6             0             0             1             

  分层 acc(heuristic → LLM):
    选型与替代            heur=25%   llm=88%
    故障诊断与排查          heur=40%   llm=40%
    安全合规与风险控制        heur=40%   llm=60%
    标准规范与术语          heur=33%   llm=50%
    工艺原理与参数影响        heur=29%   llm=86%
    质量计量与检测          heur=20%   llm=80%
    工程计算与估算          heur=25%   llm=75%
    diff:medium      heur=50%   llm=83%
    diff:hard        heur=15%   llm=54%
    diff:easy        heur=11%   llm=67%

【2】task(推理任务) 判定准确率（gt=capability→TaskType 专家映射代理）
                   acc      加权F1
  生产(恒GENERAL)     0.0%      0.0%
  LLM 判8类         57.5%     57.6%

  LLM task 混淆矩阵(行=真值代理, 列=预测):
  真实\预测           comparison    diagnosis     selection     calculation   standard_interpretationprocedure     explanation   general       
  对比              0             0             0             0             0             0             0             0             
  诊断              0             4             2             1             1             1             1             0             
  选型              0             0             8             0             0             0             0             0             
  计算              1             0             1             2             0             0             0             0             
  标准解读            0             0             1             1             4             0             0             0             
  流程/规程           0             1             1             0             0             2             1             0             
  解释              1             0             2             1             0             0             3             0             
  通用              0             0             0             0             0             0             0             0             

  每类 P/R/F1 (LLM task 判定):
    对比           n=0   P=0% R=0% F1=0.00
    诊断           n=10  P=80% R=40% F1=0.53
    选型           n=8   P=53% R=100% F1=0.70
    计算           n=4   P=40% R=50% F1=0.44
    标准解读         n=6   P=80% R=67% F1=0.73
    流程/规程        n=5   P=67% R=40% F1=0.50
    解释           n=7   P=60% R=43% F1=0.50
    通用           n=0   P=0% R=0% F1=0.00

【3】结论与决策提示
  - format: LLM 判定明显优于生产 heuristic => 前端 format 判定有被浪费的容量。
  - format: LLM 判定本身 acc<80% => 前端判定不可靠，应优先靠 CSV 真值。
  - task: 生产恒 GENERAL；LLM 判 8 类可达能力仅作参考（真值为 capability 代理）。
  - 注意: capability 与推理任务不严格一一对应，task 真值存在固有噪声。

明细JSON: results/frontend_judgment\frontend_judgment_20260807_205321.json
