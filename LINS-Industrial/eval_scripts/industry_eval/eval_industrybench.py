"""
eval_industrybench.py: IndustryBench 论文级评估脚本

对标论文: https://arxiv.org/abs/2506.19875

评估模式:
  - quick (开卷): 直接注入 knowledge_text -> 用 scorer 打 0-3 分
  - rag (检索增强): 通过检索器获取知识 -> scorer 评分
  - closed_book (闭卷): 无外部知识 -> scorer 评分

评估协议:
  1. 0-3 分原始正确性评分 (ScoringRubric + Judge LLM)
  2. 安全违规 (SV) 检查
  3. SV 调整: adjusted_score = 0 if SV else raw_score
  4. 分层统计: 按能力(7类)/行业(10类)/难度(3级)
"""
import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any

# ===== 路径配置 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')

if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)
if METRICS_DIR not in sys.path:
    sys.path.insert(0, METRICS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== 环境变量 =====
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_KEY:
    print("[ERROR] 请设置 DEEPSEEK_API_KEY 环境变量")
    print("[INFO]   PowerShell: $env:DEEPSEEK_API_KEY='<DEEPSEEK_API_KEY_FROM_ENV>'")
    print("[INFO]   CMD:        set DEEPSEEK_API_KEY=<DEEPSEEK_API_KEY_FROM_ENV>")
    sys.exit(1)

# ===== 导入模块 =====
from metrics.industrybench_scorer import (
    EvalSample, FinalEvalResult, IndustryBenchScorer, RuleBasedScorer
)


# ============================================================
# 1. 数据加载
# ============================================================

def load_industrybench_data(csv_path: str, num_samples: Optional[int] = None) -> List[EvalSample]:
    """加载 IndustryBench 数据集为 EvalSample 列表"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")
    
    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = EvalSample(
                id=row.get('id', '').strip(),
                question=row.get('question', '').strip(),
                ref_answer=row.get('answer', '').strip(),
                knowledge_text=row.get('knowledge_text', '').strip(),
                difficulty=row.get('difficulty', '').strip().lower(),
                capability=row.get('capability', '').strip(),
                industry_primary=row.get('industry_primary', '').strip(),
                domain=row.get('domain', '').strip(),
            )
            # 过滤无效数据
            if sample.question and sample.ref_answer:
                samples.append(sample)
    
    if num_samples:
        samples = samples[:num_samples]
    
    print(f"[INFO] 加载了 {len(samples)} 个有效样本")
    return samples


# ============================================================
# 2. LINS 模型包装
# ============================================================

class LINSModelWrapper:
    """LINS 模型包装器，提供统一接口"""
    
    def __init__(self, deepseek_key: str):
        self.deepseek_key = deepseek_key
        self.lins = None
        self._init_lins()
    
    def _init_lins(self):
        """初始化 LINS"""
        from model.model_LINS import LINS
        self.lins = LINS(
            LLM_name='deepseek-chat',
            assistant_LLM_name='deepseek-chat',
            retriever_name='BGE',
            DeepSeek_keys=self.deepseek_key,
            database_name='none'  # 不使用外部数据库
        )
    
    def answer_quick(self, question: str, knowledge_text: str) -> str:
        """quick 模式：直接注入知识，不检索"""
        prompt = f"""请根据以下知识内容回答问题：

## 知识内容：
{knowledge_text}

## 问题：
{question}

## 要求：
1. 仅基于上述知识内容回答
2. 如果知识内容不足以回答问题，请说明
3. 请包含引用编号如 [1], [2]（如果适用）"""
        
        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=prompt,
            topk=3,
            if_PRA=False,
            if_SKA=False,
            if_QDA=False,
            if_PCA=False,
            recall_top_k=10
        )
        return response
    
    def answer_closed_book(self, question: str) -> str:
        """闭卷模式：仅依赖模型自身知识"""
        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=question,
            topk=1,
            if_PRA=False,
            if_SKA=False,
            if_QDA=False,
            if_PCA=False,
            recall_top_k=1
        )
        return response
    
    def answer_rag(self, question: str) -> str:
        """RAG 模式：检索增强"""
        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=question + "\nPlease include citation numbers like [1], [2] in your answer.",
            topk=5,
            if_PRA=True,
            if_SKA=False,
            if_QDA=False,
            if_PCA=False,
            recall_top_k=50
        )
        return response


# ============================================================
# 3. 评估 Runner（支持评分 + 报告）
# ============================================================

def run_evaluation(
    mode: str,
    num_samples: int,
    scorer_mode: str = "rule",  # "rule" 或 "llm"
    model_name: str = "DeepSeek-LINS",
) -> Dict[str, Any]:
    """
    完整评估流程
    
    Args:
        mode: quick / rag / closed_book
        num_samples: 评估样本数
        scorer_mode: rule=基于规则快速评分, llm=调用 Judge LLM 评分
        model_name: 模型名称（用于报告标识）
    
    Returns:
        summary dict
    """
    print("\n" + "=" * 70)
    print(f"IndustryBench 论文级评估 | mode={mode} | samples={num_samples}")
    print("=" * 70)
    
    # 1. 加载数据
    csv_path = os.path.join(DATA_DIR, 'huggingface_dataset.csv')
    samples = load_industrybench_data(csv_path, num_samples)
    
    # 2. 初始化模型
    print(f"\n[初始化] LINS 模型 (mode={mode})...")
    model = LINSModelWrapper(DEEPSEEK_KEY)
    
    # 3. 初始化评分器
    if scorer_mode == "llm":
        scorer = IndustryBenchScorer(judge_model="deepseek-chat", api_key=DEEPSEEK_KEY)
        print(f"[评分器] Judge LLM: deepseek-chat")
    else:
        scorer = None
        print(f"[评分器] 基于规则的快速评分 (RuleBased)")
    
    # 4. 执行评估
    results = []
    total = len(samples)
    start_ts = time.time()
    
    for idx, sample in enumerate(samples):
        qid = sample.id or str(idx + 1)
        print(f"\n  [{idx+1}/{total}] ID={qid} | {sample.difficulty} | {sample.capability[:6]}... | {sample.question[:40]}...")
        
        # 4a. 获取模型回答
        try:
            t0 = time.time()
            if mode == "quick":
                model_answer = model.answer_quick(sample.question, sample.knowledge_text)
            elif mode == "closed_book":
                model_answer = model.answer_closed_book(sample.question)
            else:  # rag
                model_answer = model.answer_rag(sample.question)
            elapsed = time.time() - t0
            print(f"      回答耗时: {elapsed:.1f}s | 回答长度: {len(model_answer)}")
        except Exception as e:
            print(f"      ❌ 回答失败: {str(e)[:100]}")
            model_answer = f"[ERROR] {str(e)}"
        
        # 4b. 评分
        if scorer_mode == "llm" and scorer:
            # LLM 评分（含 SV 检查）
            try:
                final_result = scorer.evaluate_one(sample, model_answer, qid)
                print(f"      Raw={final_result.raw_score} | SV={final_result.has_violation} | Adj={final_result.adjusted_score:.1f}")
            except Exception as e:
                print(f"      ⚠️ LLM 评分失败: {str(e)[:80]}, 使用规则评分")
                raw = RuleBasedScorer.rule_based_score(sample.question, sample.ref_answer, model_answer)
                sv = RuleBasedScorer.check_safety_simple(sample.knowledge_text, model_answer)
                final_result = FinalEvalResult(
                    sample_id=qid, raw_score=raw, has_violation=sv,
                    adjusted_score=0.0 if sv else float(raw),
                    model_answer=model_answer, ref_answer=sample.ref_answer,
                    question=sample.question, difficulty=sample.difficulty,
                    capability=sample.capability, industry=sample.industry_primary,
                    domain=sample.domain, raw_explanation="rule_based_fallback"
                )
            results.append(final_result)
        else:
            # 规则评分
            raw = RuleBasedScorer.rule_based_score(sample.question, sample.ref_answer, model_answer)
            sv = RuleBasedScorer.check_safety_simple(sample.knowledge_text, model_answer)
            final_result = FinalEvalResult(
                sample_id=qid, raw_score=raw, has_violation=sv,
                adjusted_score=0.0 if sv else float(raw),
                model_answer=model_answer, ref_answer=sample.ref_answer,
                question=sample.question, difficulty=sample.difficulty,
                capability=sample.capability, industry=sample.industry_primary,
                domain=sample.domain
            )
            results.append(final_result)
            print(f"      Raw={raw} | SV={sv} | Adj={final_result.adjusted_score:.1f}")
    
    # 5. 计算汇总
    total_elapsed = time.time() - start_ts
    
    if scorer:
        summary = scorer.get_summary(results)
        scorer.save_results(os.path.join(RESULTS_DIR, f"industrybench_eval_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), results)
    else:
        # 手动汇总
        raw_scores = [r.raw_score for r in results]
        adj_scores = [r.adjusted_score for r in results]
        import numpy as np
        from collections import Counter, defaultdict
        
        summary = {
            "total_samples": len(results),
            "raw_mean": float(np.mean(raw_scores)),
            "raw_median": float(np.median(raw_scores)),
            "raw_std": float(np.std(raw_scores)),
            "adjusted_mean": float(np.mean(adj_scores)),
            "adjusted_median": float(np.median(adj_scores)),
            "adjusted_std": float(np.std(adj_scores)),
            "sv_stats": {
                "sv_rate": sum(1 for r in results if r.has_violation) / len(results),
                "sv_count": sum(1 for r in results if r.has_violation),
                "delta": float(np.mean(adj_scores)) - float(np.mean(raw_scores)),
            },
            "raw_distribution": dict(Counter(raw_scores)),
            "adjusted_distribution": dict(Counter(adj_scores)),
        }
    
    # 6. 打印报告
    print(f"\n{'='*70}")
    print(f"🎯 评估完成！")
    print(f"  模式: {mode} | 样本: {num_samples}")
    print(f"  总耗时: {total_elapsed:.1f}s | 平均: {total_elapsed/max(total,1):.2f}s/样本")
    print(f"  Raw Mean: {summary['raw_mean']:.4f}")
    print(f"  Adj Mean (SV): {summary['adjusted_mean']:.4f}")
    print(f"  Δ Delta: {summary['sv_stats']['delta']:.4f}")
    print(f"  SV Rate: {summary['sv_stats']['sv_rate']:.4f} ({summary['sv_stats']['sv_count']}/{total})")
    
    # 分层打印
    if 'by_difficulty' in summary:
        print(f"\n📊 按难度分层:")
        for name, stats in sorted(summary['by_difficulty'].items(), key=lambda x: -x[1]['count']):
            print(f"  {name:<10} n={stats['count']:4d} Raw={stats['raw_mean']:.4f} Adj={stats['adjusted_mean']:.4f} SV={stats['sv_rate']:.4f}")
    
    if 'by_capability' in summary:
        print(f"\n📊 按能力分层:")
        for name, stats in sorted(summary['by_capability'].items(), key=lambda x: -x[1]['count']):
            print(f"  {name:<16} n={stats['count']:4d} Raw={stats['raw_mean']:.4f} Adj={stats['adjusted_mean']:.4f} SV={stats['sv_rate']:.4f}")
    
    if 'by_industry' in summary:
        print(f"\n📊 按行业分层:")
        for name, stats in sorted(summary['by_industry'].items(), key=lambda x: -x[1]['count']):
            print(f"  {name:<16} n={stats['count']:4d} Raw={stats['raw_mean']:.4f} Adj={stats['adjusted_mean']:.4f} SV={stats['sv_rate']:.4f}")
    
    print(f"\n{'='*70}")
    
    # 保存结果
    output_path = os.path.join(RESULTS_DIR, f"industrybench_eval_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    output = {
        "config": {"mode": mode, "scorer": scorer_mode, "model": model_name, "judge": "deepseek-chat" if scorer_mode == "llm" else "rule_based"},
        "summary": summary,
        "results": [
            {
                "sample_id": r.sample_id, "raw_score": r.raw_score, "has_violation": r.has_violation,
                "adjusted_score": r.adjusted_score, "difficulty": r.difficulty,
                "capability": r.capability, "industry": r.industry,
                "model_answer_preview": r.model_answer[:200] if r.model_answer else "",
                "ref_answer": r.ref_answer,
            }
            for r in results
        ],
        "timing": {"total_seconds": total_elapsed, "avg_per_sample": total_elapsed / max(total, 1)},
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVED] {output_path}")
    return summary


# ============================================================
# 4. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="IndustryBench 论文级评估")
    parser.add_argument("--mode", choices=["quick", "rag", "closed_book", "all"], default="quick",
                        help="评估模式: quick(开卷), rag(检索), closed_book(闭卷), all(全部)")
    parser.add_argument("--num", type=int, default=10, help="评估样本数")
    parser.add_argument("--scorer", choices=["rule", "llm"], default="rule",
                        help="评分方式: rule(基于规则快速), llm(调用 Judge LLM)")
    parser.add_argument("--model", default="DeepSeek-LINS",
                        help="模型名称（用于报告标识）")
    
    args = parser.parse_args()
    
    modes = ["quick", "rag", "closed_book"] if args.mode == "all" else [args.mode]
    
    for mode in modes:
        print(f"\n\n>>> 开始模式: {mode}")
        run_evaluation(
            mode=mode,
            num_samples=args.num,
            scorer_mode=args.scorer,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
