"""
exp4_ablation.py: 实验四 - 消融实验 (多智能体 Agent 效果评估)

评估 LINS 框架中不同 Agent 模块对回答质量的影响。

消融方案:
  - baseline (none): 所有 Agent 关闭，仅基础检索 + 生成
  - +PRA: 启用 Passage Relevant Agent (段落相关性过滤)
  - +SKA: 启用 Self-Knowledge Agent (自知识检查)
  - +QDA: 启用 Question Decomposition Agent (问题分解)
  - +PCA: 启用 Passage Coherence Agent (段落连贯性优化)
  - full: 所有 Agent 启用

评估指标:
  - 平均原始分 (0-3)
  - 安全违规率
  - 平均耗时
  - 不同 Agent 组合的性能差异

输出:
  - JSON 详细结果 (含各配置对比)
  - Markdown 格式报告 (含消融分析表)
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter, defaultdict

# ===== 路径配置 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')

for p in [PROJECT_ROOT, LINS_MAIN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics.industrybench_scorer import (
    EvalSample, FinalEvalResult, RuleBasedScorer
)


# ============================================================
# 1. 消融配置定义
# ============================================================

ABLATION_CONFIGS = {
    "baseline": {
        "name": "基线 (无 Agent)",
        "desc": "所有 Agent 关闭",
        "topk": 5,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 50,
    },
    "+PRA": {
        "name": "PRA (段落相关性)",
        "desc": "仅启用 Passage Relevant Agent",
        "topk": 5,
        "if_PRA": True,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 50,
    },
    "+SKA": {
        "name": "SKA (自知识检查)",
        "desc": "仅启用 Self-Knowledge Agent",
        "topk": 5,
        "if_PRA": False,
        "if_SKA": True,
        "if_QDA": False,
        "if_PCA": False,
        "recall_top_k": 50,
    },
    "+QDA": {
        "name": "QDA (问题分解)",
        "desc": "仅启用 Question Decomposition Agent",
        "topk": 5,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": True,
        "if_PCA": False,
        "recall_top_k": 50,
    },
    "+PCA": {
        "name": "PCA (段落连贯性)",
        "desc": "仅启用 Passage Coherence Agent",
        "topk": 5,
        "if_PRA": False,
        "if_SKA": False,
        "if_QDA": False,
        "if_PCA": True,
        "recall_top_k": 50,
    },
    "full": {
        "name": "全部 Agent",
        "desc": "所有 Agent 启用",
        "topk": 5,
        "if_PRA": True,
        "if_SKA": True,
        "if_QDA": True,
        "if_PCA": True,
        "recall_top_k": 50,
    },
}


# ============================================================
# 2. LINS 模型包装器
# ============================================================

class AblationLINSModel:
    """LINS 模型包装器 - 支持不同 Agent 组合"""

    def __init__(self, deepseek_key: str):
        self.deepseek_key = deepseek_key
        self.lins = None
        self._init_lins()

    def _init_lins(self):
        from model.model_LINS import LINS
        self.lins = LINS(
            LLM_name='deepseek-chat',
            assistant_LLM_name='deepseek-chat',
            retriever_name='BGE',
            DeepSeek_keys=self.deepseek_key,
            database_name='none'
        )

    def answer_with_config(self, question: str, config: Dict) -> str:
        """按指定 Agent 配置生成回答"""
        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=question,
            topk=config["topk"],
            if_PRA=config["if_PRA"],
            if_SKA=config["if_SKA"],
            if_QDA=config["if_QDA"],
            if_PCA=config["if_PCA"],
            recall_top_k=config["recall_top_k"],
        )
        return response


# ============================================================
# 3. 数据加载
# ============================================================

def load_data(csv_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """加载评估数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": row.get('id', '').strip(),
                "question": row.get('question', '').strip(),
                "ref_answer": row.get('answer', '').strip(),
                "knowledge_text": row.get('knowledge_text', '').strip(),
                "difficulty": row.get('difficulty', '').strip().lower(),
                "capability": row.get('capability', '').strip(),
                "industry_primary": row.get('industry_primary', '').strip(),
            })

    if num_samples:
        samples = samples[:num_samples]

    print(f"[INFO] 加载了 {len(samples)} 个样本")
    return samples


# ============================================================
# 4. 实验四核心函数
# ============================================================

def run_ablation_experiment(
    ablation_configs: List[str] = None,
    num_samples: int = 30,
    deepseek_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行消融实验

    Args:
        ablation_configs: 要测试的消融配置列表 (默认全部)
        num_samples: 每个配置评估的样本数
        deepseek_key: API Key
        output_dir: 输出目录
        run_id: 运行ID

    Returns:
        summary dict
    """
    if deepseek_key is None:
        deepseek_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not deepseek_key:
        raise ValueError("未设置 DEEPSEEK_API_KEY")

    if ablation_configs is None:
        ablation_configs = list(ABLATION_CONFIGS.keys())

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments', f'exp4_ablation_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"📊 实验四: 消融实验 | 配置数={len(ablation_configs)} | 样本数={num_samples}")
    print("=" * 70)

    # 1. 加载数据
    csv_path = os.path.join(DATA_DIR, 'huggingface_dataset.csv')
    all_samples = load_data(csv_path, num_samples)
    print(f"  待测配置: {', '.join(ablation_configs)}")

    # 2. 初始化
    print(f"\n[初始化] LINS 模型 & 评分器...")
    model = AblationLINSModel(deepseek_key)
    scorer = RuleBasedScorer()

    # 3. 对每个配置运行评估
    all_results = {}

    for config_key in ablation_configs:
        config = ABLATION_CONFIGS[config_key]
        config_name = config["name"]

        print(f"\n{'─' * 70}")
        print(f"🔬 测试配置: {config_name} | {config['desc']}")
        print(f"    PRA={config['if_PRA']} SKA={config['if_SKA']} "
              f"QDA={config['if_QDA']} PCA={config['if_PCA']}")

        config_results = []
        raw_scores = []
        sv_count = 0
        total_time = 0.0

        for idx, sample in enumerate(all_samples):
            start_time = time.time()

            # 3a. 生成回答
            answer = model.answer_with_config(sample["question"], config)

            elapsed = time.time() - start_time
            total_time += elapsed

            # 3b. 评分 (Rule-based)
            score_result = scorer.score(sample["question"], answer, sample["ref_answer"])
            sv_result = scorer.check_safety_violation(answer, sample["knowledge_text"])

            has_sv = sv_result.has_violation if sv_result else False
            adjusted = 0.0 if has_sv else float(score_result.raw_score)

            config_results.append({
                "sample_id": sample["id"],
                "raw_score": score_result.raw_score,
                "has_violation": has_sv,
                "adjusted_score": adjusted,
                "time_seconds": elapsed,
                "difficulty": sample["difficulty"],
                "capability": sample["capability"],
                "answer_preview": answer[:200] if answer else "",
            })
            raw_scores.append(score_result.raw_score)
            if has_sv:
                sv_count += 1

            if (idx + 1) % 10 == 0:
                avg = sum(raw_scores[-10:]) / min(10, len(raw_scores))
                print(f"  [{idx+1}/{len(all_samples)}] avg_score={avg:.2f} | sv={sv_count}")

        # 3c. 汇总当前配置
        config_summary = {
            "config_key": config_key,
            "config_name": config_name,
            "config_params": {
                "if_PRA": config["if_PRA"],
                "if_SKA": config["if_SKA"],
                "if_QDA": config["if_QDA"],
                "if_PCA": config["if_PCA"],
                "topk": config["topk"],
            },
            "avg_raw_score": sum(raw_scores) / max(len(raw_scores), 1),
            "sv_count": sv_count,
            "sv_rate": sv_count / max(len(raw_scores), 1),
            "total_time_seconds": total_time,
            "avg_time_seconds": total_time / max(len(raw_scores), 1),
            "score_distribution": dict(Counter(raw_scores)),
            "results": config_results,
        }
        all_results[config_key] = config_summary

        print(f"\n  ✅ {config_name} 完成:")
        print(f"     平均分: {config_summary['avg_raw_score']:.3f} / 3.0")
        print(f"     安全违规: {sv_count}/{len(raw_scores)} ({config_summary['sv_rate']*100:.1f}%)")
        print(f"     平均耗时: {config_summary['avg_time_seconds']:.1f}s")

    # 4. 全局对比
    print(f"\n{'=' * 70}")
    print("📊 消融实验对比汇总")
    print(f"{'=' * 70}")
    print(f"{'配置':<20} {'平均分':<10} {'SV率':<10} {'耗时(s)':<10}")
    print(f"{'─' * 50}")

    comparison = {}
    for key, summary in all_results.items():
        print(f"{ABLATION_CONFIGS[key]['name']:<20} "
              f"{summary['avg_raw_score']:.3f}    "
              f"{summary['sv_rate']*100:.1f}%     "
              f"{summary['avg_time_seconds']:.1f}")
        comparison[key] = {
            "name": ABLATION_CONFIGS[key]["name"],
            "avg_raw_score": summary["avg_raw_score"],
            "sv_rate": summary["sv_rate"],
            "avg_time_seconds": summary["avg_time_seconds"],
            "score_distribution": summary["score_distribution"],
        }

    # 5. 保存
    output = {
        "experiment": "exp4_ablation",
        "config": {
            "ablation_configs_tested": ablation_configs,
            "num_samples_per_config": num_samples,
        },
        "comparison": comparison,
        "detailed_results": all_results,
    }

    json_path = os.path.join(output_dir, "exp4_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] 结果已保存至: {json_path}")

    # Markdown 报告
    md_path = os.path.join(output_dir, "exp4_ablation_report.md")
    _save_md_report(md_path, comparison, ablation_configs)
    print(f"[SAVED] 报告已保存至: {md_path}")

    return output


def _save_md_report(path: str, comparison: Dict, config_order: List[str]):
    """保存 Markdown 消融报告"""
    lines = []
    lines.append(f"# 实验四: 消融实验报告")
    lines.append(f"")
    lines.append(f"## 配置对比总表")
    lines.append(f"")
    lines.append(f"| 配置 | 平均分 (0-3) | SV率 | 平均耗时 | 分数分布 |")
    lines.append(f"|------|-------------|------|---------|---------|")

    for key in config_order:
        if key in comparison:
            c = comparison[key]
            dist = ", ".join([f"{s}:{c2}" for s, c2 in sorted(c['score_distribution'].items())])
            lines.append(f"| {c['name']} | {c['avg_raw_score']:.3f} | {c['sv_rate']*100:.1f}% | "
                        f"{c['avg_time_seconds']:.1f}s | {dist} |")

    lines.append(f"")
    lines.append(f"## 消融分析")
    lines.append(f"")
    lines.append(f"### 性能对比")
    lines.append(f"")

    if "baseline" in comparison:
        baseline_score = comparison["baseline"]["avg_raw_score"]
        lines.append(f"- **基线平均分**: {baseline_score:.3f}")
        for key in config_order:
            if key in comparison and key != "baseline":
                diff = comparison[key]["avg_raw_score"] - baseline_score
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"- **{comparison[key]['name']}**: {comparison[key]['avg_raw_score']:.3f} "
                            f"({arrow}{abs(diff):+.3f} vs baseline)")

    lines.append(f"")
    lines.append(f"### 效率对比")
    lines.append(f"")

    for key in config_order:
        if key in comparison:
            c = comparison[key]
            bar_len = int(c['avg_time_seconds'] / 5)
            bar = "█" * bar_len
            lines.append(f"- **{c['name']}**: {bar} {c['avg_time_seconds']:.1f}s")

    lines.append(f"")
    lines.append(f"### 分数分布雷达")
    lines.append(f"")

    for key in config_order:
        if key in comparison:
            c = comparison[key]
            dist = c['score_distribution']
            total = sum(dist.values())
            bars = []
            for score in [0, 1, 2, 3]:
                count = dist.get(score, 0)
                pct = count / max(total, 1) * 100
                bar_len = max(1, int(pct / 5))
                bars.append(f"{score}分:{'█' * bar_len} {count}")
            lines.append(f"- **{c['name']}**: {' | '.join(bars)}")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*由 LINS-Industrial 实验管线自动生成*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验四: 消融实验")
    parser.add_argument("--configs", nargs="+",
                        choices=list(ABLATION_CONFIGS.keys()),
                        default=list(ABLATION_CONFIGS.keys()),
                        help="要测试的消融配置")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="每个配置的评估样本数")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()

    result = run_ablation_experiment(
        ablation_configs=args.configs,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
