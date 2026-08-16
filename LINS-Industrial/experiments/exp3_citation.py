"""
exp3_citation.py: 实验三 - 引用准确性评估

评估 LLM 在回答中的引用准确性，使用 Link-Eval 方法。

评估指标:
  - 引用精确率 (Citation Precision): 引用是否有效指向真实文档
  - 引用召回率 (Citation Recall): 真实文档是否被充分引用
  - F1 分数: 精确率和召回率的调和平均
  - 格式合规率: 引用格式是否正确 [N]

评测流程:
  1. 加载 IndustryBench 测试题
  2. LLM 生成带引用的回答
  3. 使用 IndustrialLinkEval 提取引用
  4. 计算引用指标

输出:
  - JSON 详细结果
  - Markdown 格式报告
"""

import os
import sys
import json
import csv
import re
import time
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any, Set, Tuple
from collections import defaultdict

# ===== 路径配置 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')

for p in [PROJECT_ROOT, LINS_MAIN_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.industrial_linkeval import IndustrialLinkEval


# ============================================================
# 1. 数据加载
# ============================================================

def load_citation_data(csv_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """加载引用评估数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": row.get('id', '').strip(),
                "question": row.get('question', '').strip(),
                "knowledge_text": row.get('knowledge_text', '').strip(),
                "capability": row.get('capability', '').strip(),
                "industry_primary": row.get('industry_primary', '').strip(),
            })

    if num_samples:
        samples = samples[:num_samples]

    print(f"[INFO] 加载了 {len(samples)} 个引用评估样本")
    return samples


# ============================================================
# 2. 知识文本切块（模拟检索到的文档块）
# ============================================================

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """将知识文本切分为文档块"""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ['。', '. ', '!', '？', '\n']:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return chunks


# ============================================================
# 3. LINS 模型包装器（精简版）
# ============================================================

class CitationLINSModel:
    """LINS 模型包装器 - 用于引用评估"""

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

    def answer_with_citations(self, question: str, knowledge_text: str) -> str:
        """
        生成带引用的回答

        将 knowledge_text 切块后注入 prompt，要求 LLM 按 [N] 格式引用
        """
        chunks = chunk_text(knowledge_text)
        if not chunks:
            return self.lins.MAIRAG(
                question=question,
                topk=1,
                if_PRA=False, if_SKA=False, if_QDA=False, if_PCA=False,
                recall_top_k=1
            )[0]

        # 构造带有块编号的知识上下文
        knowledge_ctx = "\n\n".join([
            f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)
        ])

        prompt = f"""请根据以下知识内容回答问题，并在答案中使用 [N] 格式引用对应的知识块。

## 知识内容：
{knowledge_ctx}

## 问题：
{question}

## 要求：
1. 仅基于上述知识内容回答
2. 每个关键信息必须引用来源，格式为 [编号]
3. 如果知识内容不足以回答问题，请说明
4. 请保持回答简洁准确"""

        response, urls, passages, history, sub_qs = self.lins.MAIRAG(
            question=prompt,
            topk=5,
            if_PRA=True, if_SKA=False, if_QDA=False, if_PCA=False,
            recall_top_k=50
        )
        return response


# ============================================================
# 4. 引用质量分析工具
# ============================================================

def analyze_citation_format(answer: str, max_ref: int) -> Dict[str, Any]:
    """
    分析引用格式质量

    Args:
        answer: 模型回答
        max_ref: 最大有效引用编号

    Returns:
        dict with format metrics
    """
    pattern = re.compile(r'\[(\d+)\]')
    matches = pattern.findall(answer)

    if not matches:
        return {
            "citation_count": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "out_of_range": 0,
            "format_valid": True,
        }

    ref_numbers = [int(m) for m in matches]
    valid = [n for n in ref_numbers if 1 <= n <= max_ref]
    invalid = [n for n in ref_numbers if n < 1 or n > max_ref]
    oob = [n for n in ref_numbers if n > max_ref]

    return {
        "citation_count": len(ref_numbers),
        "valid_citations": len(valid),
        "invalid_citations": len(invalid),
        "out_of_range": len(oob),
        "format_valid": True,
    }


# ============================================================
# 5. 实验三核心函数
# ============================================================

def run_citation_experiment(
    num_samples: int = 50,
    chunk_size: int = 200,
    deepseek_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行引用准确性评估实验

    Args:
        num_samples: 评估样本数
        chunk_size: 知识块大小
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

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiments', f'exp3_citation_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"📊 实验三: 引用准确性评估 | 样本数={num_samples}")
    print("=" * 70)

    # 1. 加载数据
    csv_path = os.path.join(DATA_DIR, 'huggingface_dataset.csv')
    samples = load_citation_data(csv_path, num_samples)

    # 2. 初始化
    print(f"\n[初始化] LINS 模型 & LinkEval...")
    model = CitationLINSModel(deepseek_key)
    link_eval = IndustrialLinkEval()

    # 3. 运行评估
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_citation_counts = []
    format_results = []

    detailed_results = []

    for idx, sample in enumerate(samples):
        start_time = time.time()

        # 3a. 生成带引用的回答
        answer = model.answer_with_citations(sample["question"], sample["knowledge_text"])

        # 3b. 将 knowledge_text 切块作为引用目标
        chunks = chunk_text(sample["knowledge_text"], chunk_size=chunk_size)

        # 3c. 使用 LinkEval 提取和分析引用
        statements = link_eval.extract_statements(answer)
        citation_metrics = link_eval.calculate_citation_metrics(statements, chunks)

        # 3d. 分析格式质量
        format_analysis = analyze_citation_format(answer, len(chunks) if chunks else 0)

        elapsed = time.time() - start_time

        all_precisions.append(citation_metrics["citation_precision"])
        all_recalls.append(citation_metrics["citation_recall"])
        all_f1s.append(citation_metrics["f1_score"])
        all_citation_counts.append(format_analysis["citation_count"])

        format_results.append(format_analysis)

        detailed_results.append({
            "sample_id": sample["id"],
            "num_chunks": len(chunks),
            "citation_precision": citation_metrics["citation_precision"],
            "citation_recall": citation_metrics["citation_recall"],
            "f1_score": citation_metrics["f1_score"],
            "citation_count": format_analysis["citation_count"],
            "valid_citations": format_analysis["valid_citations"],
            "invalid_citations": format_analysis["invalid_citations"],
            "answer_preview": answer[:200] if answer else "",
            "time_seconds": elapsed,
        })

        if (idx + 1) % 10 == 0:
            current_f1 = sum(all_f1s) / max(len(all_f1s), 1)
            print(f"  [{idx+1}/{len(samples)}] 完成 | avg_f1={current_f1:.4f} | total_cites={sum(all_citation_counts)}")

    # 4. 汇总统计
    print(f"\n{'─' * 70}")
    print("📈 引用指标汇总")

    avg_precision = sum(all_precisions) / max(len(all_precisions), 1)
    avg_recall = sum(all_recalls) / max(len(all_recalls), 1)
    avg_f1 = sum(all_f1s) / max(len(all_f1s), 1)
    total_citations = sum(all_citation_counts)
    avg_citations = total_citations / max(len(all_citation_counts), 1)
    valid_rate = sum(f["valid_citations"] for f in format_results) / max(total_citations, 1)
    has_citation_count = sum(1 for c in all_citation_counts if c > 0)
    citation_rate = has_citation_count / max(len(all_citation_counts), 1)

    stats = {
        "run_id": run_id,
        "num_samples": len(detailed_results),
        "chunk_size": chunk_size,
        "timestamp": datetime.now().isoformat(),
        "avg_citation_precision": avg_precision,
        "avg_citation_recall": avg_recall,
        "avg_f1_score": avg_f1,
        "total_citations": total_citations,
        "avg_citations_per_sample": avg_citations,
        "valid_citation_rate": valid_rate,
        "citation_usage_rate": citation_rate,
        "samples_with_citations": has_citation_count,
    }

    print(f"  平均引用精确率: {avg_precision:.4f}")
    print(f"  平均引用召回率: {avg_recall:.4f}")
    print(f"  平均 F1 分数:   {avg_f1:.4f}")
    print(f"  总引用数:       {total_citations}")
    print(f"  平均引用/样本:  {avg_citations:.2f}")
    print(f"  有效引用率:     {valid_rate:.2%}")
    print(f"  含引用样本率:   {citation_rate:.2%}")

    # 5. 保存结果
    output = {
        "experiment": "exp3_citation",
        "config": {"num_samples": num_samples, "chunk_size": chunk_size},
        "stats": stats,
        "detailed_results": detailed_results,
    }

    json_path = os.path.join(output_dir, "exp3_citation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] 结果已保存至: {json_path}")

    # Markdown 报告
    md_path = os.path.join(output_dir, "exp3_citation_report.md")
    _save_md_report(md_path, stats)
    print(f"[SAVED] 报告已保存至: {md_path}")

    return output


def _save_md_report(path: str, stats: Dict):
    """保存 Markdown 报告"""
    lines = []
    lines.append(f"# 实验三: 引用准确性评估报告")
    lines.append(f"")
    lines.append(f"**样本数**: {stats['num_samples']}")
    lines.append(f"**chunk_size**: {stats['chunk_size']}")
    lines.append(f"**时间戳**: {stats['timestamp']}")
    lines.append(f"")
    lines.append(f"## 总体引用指标")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 平均引用精确率 | {stats['avg_citation_precision']:.4f} |")
    lines.append(f"| 平均引用召回率 | {stats['avg_citation_recall']:.4f} |")
    lines.append(f"| 平均 F1 分数 | {stats['avg_f1_score']:.4f} |")
    lines.append(f"| 总引用数 | {stats['total_citations']} |")
    lines.append(f"| 平均引用/样本 | {stats['avg_citations_per_sample']:.2f} |")
    lines.append(f"| 有效引用率 | {stats['valid_citation_rate']:.2%} |")
    lines.append(f"| 含引用样本率 | {stats['citation_usage_rate']:.2%} |")
    lines.append(f"")

    # 评分条
    lines.append(f"## 引用质量可视化")
    lines.append(f"")
    for key, label in [("avg_citation_precision", "精确率"), ("avg_citation_recall", "召回率"),
                        ("avg_f1_score", "F1 分数"), ("valid_citation_rate", "有效引用率")]:
        val = stats[key]
        bar_len = int(val * 30)
        bar = "█" * bar_len
        lines.append(f"- **{label}**: {bar} {val:.4f}")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"*由 LINS-Industrial 实验管线自动生成*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验三: 引用准确性评估")
    parser.add_argument("--num_samples", type=int, default=30, help="评估样本数")
    parser.add_argument("--chunk_size", type=int, default=200, help="知识块大小")
    args = parser.parse_args()

    result = run_citation_experiment(
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
    )
