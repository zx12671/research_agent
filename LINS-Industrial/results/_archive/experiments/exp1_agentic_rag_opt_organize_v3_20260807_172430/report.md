# Exp1 Agentic RAG 优化组实验报告

- **模式**: Agentic RAG (Task-Aware) + optimization=`organize_v3`
- **优化点**: hybrid(BM25-jieba, sparse_w=0.2) + 多视角改写
- **运行 ID**: exp1_agentic_rag_opt_organize_v3_20260807_172430
- **时间**: 2026-08-07 17:25:03
- **样本数**: 1

## 总体结果

| 指标 | 值 |
|------|-----|
| 总分 (raw) | 1/3 (33.33%) |
| 总分 (adjusted) | 1.0/3 (33.33%) |
| 平均分 (raw) | 1.00/3.0 |
| 平均分 (adjusted) | 1.00/3.0 |
| SV 违规 | 0/1 (0.0%) |
| 总耗时 | 31.1s |
| 平均耗时 | 31.1s |

## 语义检索指标

| 指标 | 值 |
|------|-----|
| Semantic Hit@1 | 1.0000 |
| Semantic Hit@3 | 1.0000 |
| Semantic Hit@5 | 1.0000 |
| Semantic Hit@10 | 1.0000 |
| Semantic MRR | 1.0000 |
| Semantic NDCG | 2.4500 |

## 按难度

| 难度 | 样本数 | 平均分 | Violations |
|------|--------|--------|------------|
| easy | 1 | 1.00 | 0 |

## 按能力

| 能力 | 样本数 | 平均分 | Violations |
|------|--------|--------|------------|
| 选型与替代 | 1 | 1.00 | 0 |
