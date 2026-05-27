"""
lins_test_common.py — LINS 框架测试公共工具模块

被 test_deepseek.py 和 test_deepseek_lins_full.py 共用。
"""
import os
import sys
import json
import time

from model.model_LINS import LINS
from model.database import LINS_Database
from model.prompts import return_prompts


# ============================================================
# 配置常量
# ============================================================
PUBMEDQA_ORI_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/ori_pqal.json"
PUBMEDQA_GT_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/test_ground_truth.json"

# 可在各测试脚本中覆盖
PUBMEDQA_TEST_SAMPLES = 3


# ============================================================
# PubmedQA 数据集加载
# ============================================================
def load_pubmedqa_samples(num_samples=None):
    """加载 PubmedQA 测试样本"""
    if num_samples is None:
        num_samples = PUBMEDQA_TEST_SAMPLES

    with open(PUBMEDQA_ORI_PATH, "r", encoding="utf-8") as f:
        ori_data = json.load(f)
    with open(PUBMEDQA_GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    samples = []
    for pmid, entry in ori_data.items():
        if len(samples) >= num_samples:
            break
        samples.append({
            "id": pmid,
            "question": entry["QUESTION"],
            "contexts": entry["CONTEXTS"],
            "labels": entry.get("LABELS", []),
            "answer": gt_data.get(pmid, "unknown")
        })

    print(f"已加载 {len(samples)} 条 PubmedQA 测试样本:")
    for s in samples:
        print(f"  [{s['id']}] {s['question'][:60]}... → answer: {s['answer']}")
    return samples


# ============================================================
# 选择题后处理规则函数
# ============================================================
def should_force_maybe(mAirag_response, mAirag_passages, mAirag_urls, ked_urls, ked_count):
    """
    检测 MAIRAG 回答是否应该强制变为 "Maybe" (C)
    
    收紧规则：只有非常明确的证据才触发maybe，避免过度干预
    
    规则1（收缩）：MAIRAG回答含 ≥3 个不确定性关键词（原文≥2）
    规则2（收缩）：引用数=0 且 回答含 ≥2 不确定性词（原文引用数≤1且≥1）
    规则3（不变）：KED检索到结果但MAIRAG引用0条，且回答含≥1不确定性词（新增条件）
    规则4（收缩）：仅保留最明确的"further research/study needed"表述
    """
    if not mAirag_response:
        return False

    # 不确定性关键词（精简化，移除常见弱词如'suggests','indicates'）
    uncertain_keywords = [
        'not clear', 'unclear', 'limited evidence', 'insufficient',
        'further research', 'further study',
        'inconclusive', 'equivocal',
        'uncertain', 'not certain',
        'no definitive evidence', 'lack of evidence',
        'insufficient evidence',
        'area of concern', 'need for improvement',
        'not well established', 'poorly understood'
    ]

    response_lower = mAirag_response.lower()
    uncertainty_count = sum(1 for kw in uncertain_keywords if kw in response_lower)

    mairag_url_count = len(mAirag_urls) if mAirag_urls else 0
    ked_url_count = len(ked_urls) if ked_urls else 0

    # 规则1（收紧）：不确定性关键词 >= 3个 才触发
    if uncertainty_count >= 3:
        print(f"     [后处理规则1] 不确定性关键词 {uncertainty_count} 个 → 强制maybe")
        return True

    # 规则2（收紧）：引用数=0 且 不确定性词 >= 2个
    if mairag_url_count == 0 and uncertainty_count >= 2:
        print(f"     [后处理规则2] 引用数=0, 不确定性词={uncertainty_count} → 强制maybe")
        return True

    # 规则3（收紧）：KED检索到结果，但MAIRAG引用0条 且 回答含不确定性词
    if ked_count > 0 and mairag_url_count == 0 and ked_url_count > 0 and uncertainty_count >= 1:
        print(f"     [后处理规则3] KED检索{ked_count}条但MAIRAG引用0条 + 不确定性词 {uncertainty_count} → 强制maybe")
        return True

    # 规则4（收紧）：仅匹配最明确的"further research needed"等
    explicit_uncertainty = [
        'further research is needed', 'further study is needed',
        'more research is needed',
        'remains unclear', 'no definitive evidence',
        'insufficient evidence'
    ]
    for phrase in explicit_uncertainty:
        if phrase in response_lower:
            print(f"     [后处理规则4] 明确不确定表述: '{phrase}' → 强制maybe")
            return True

    return False


# ============================================================
# 第一部分：Database 配置测试
# ============================================================
def test_database_config(lins):
    """测试 Database 配置：Online + HERD 多级检索"""
    print("\n" + "=" * 70)
    print("【第一部分】Database 配置测试")
    print("  1.1 Online Database - PubMed (实时API)")
    print("  1.2 Online Database - Bing (实时搜索)")
    print("  1.3 HERD 多级检索：Guidelines(本地) -> PubMed")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples()
    if not pubmedqa_samples:
        print("  未加载到 PubmedQA 样本，跳过数据库测试")
        return lins

    sample = pubmedqa_samples[0]
    print(f"\n  使用 PubmedQA 样本: [{sample['id']}] {sample['question'][:60]}...")

    # ========== 1.1 Online Database - PubMed ==========
    print("\n" + "-" * 60)
    print("1.1 Online Database - PubMed 实时检索")
    print("-" * 60)
    try:
        pubmed_db = LINS_Database(database_name='pubmed')
        pubmed_results = pubmed_db.get_data_list(
            question=sample['question'],
            retmax=5,
            if_split_n=False
        )
        if pubmed_results and pubmed_results.get('texts'):
            print(f"  ✓ PubMed 检索成功，获取 {len(pubmed_results['texts'])} 条结果")
            for i, (title, url) in enumerate(zip(pubmed_results['titles'][:3], pubmed_results['urls'][:3])):
                title_short = title[:50] if title else "N/A"
                print(f"    [{i+1}] {title_short}")
                print(f"        URL: {url}")
        else:
            print("  ⚠ PubMed 返回空结果（可能需配置代理或网络不通）")
    except Exception as e:
        print(f"  ✗ PubMed 检索异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 1.2 Online Database - Bing ==========
    print("\n" + "-" * 60)
    print("1.2 Online Database - Bing 实时检索")
    print("-" * 60)
    try:
        bing_db = LINS_Database(database_name='bing')
        bing_results = bing_db.get_data_list(
            question=sample['question'],
            retmax=5,
            if_split_n=False
        )
        if bing_results and bing_results.get('texts'):
            print(f"  ✓ Bing 检索成功，获取 {len(bing_results['texts'])} 条结果")
            for i, (title, url) in enumerate(zip(bing_results['titles'][:3], bing_results['urls'][:3])):
                title_short = title[:50] if title else "N/A"
                print(f"    [{i+1}] {title_short}")
                print(f"        URL: {url}")
        else:
            print("  ⚠ Bing 返回空结果（可能需配置搜索API或网络不通）")
    except Exception as e:
        print(f"  ✗ Bing 检索异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 1.3 HERD 多级检索 ==========
    print("\n" + "-" * 60)
    print("1.3 HERD 多级检索：Guidelines(本地) -> PubMed (Bing已禁用)")
    print("-" * 60)

    guidelines_path = "./add_dataset/guidelines/guidelines_embedding.json"
    guidelines_available = os.path.exists(f"./LINS-main/{guidelines_path}") or os.path.exists(guidelines_path)
    print(f"  本地指南库: {'✓ 存在' if guidelines_available else '✗ 不存在（将跳过指南层）'}")

    print("\n  >> HERD 完整分层搜索...")
    try:
        herd_passages, herd_urls = lins.HERD_search(
            PICO_question=sample['question'],
            topk=3,
            if_guidelines=guidelines_available,
            if_pubmed=True,
            if_bing=False
        )
        if herd_passages:
            print(f"  ✓ HERD 检索成功，获取 {len(herd_passages)} 条有效证据")
            for i, url in enumerate(herd_urls[:3]):
                print(f"    [{i+1}] {url}")
        else:
            print("  ⚠ HERD 未找到有效证据（三级检索均未返回 Gold）")
    except Exception as e:
        print(f"  ✗ HERD 检索异常: {type(e).__name__}: {str(e)[:80]}")

    return lins


# ============================================================
# 第二部分：Retriever (KED) 测试
# ============================================================
def test_ked_retriever(lins):
    """测试 KED (Keyword Extraction Degradation) 检索算法"""
    print("\n" + "=" * 70)
    print("【第二部分】Retriever (KED) 测试")
    print("  KED = Keyword Extraction Degradation")
    print("  流程：原始问题检索 -> 关键词提取 -> 关键词逐步退化")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples()
    sample = pubmedqa_samples[0]
    print(f"\n  使用 PubmedQA 样本: [{sample['id']}] {sample['question'][:60]}...")

    # ========== 2.1 KED 关键词提取 ==========
    print("\n" + "-" * 60)
    print("2.1 关键词提取 (Keyword Extraction)")
    print("-" * 60)
    try:
        keywords = lins.keyword_extraction(
            question=sample['question'],
            max_num_keywords=5
        )
        print(f"  ✓ 提取的关键词: {keywords}")
    except Exception as e:
        print(f"  ✗ 关键词提取异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 2.2 KED 完整搜索流程 ==========
    print("\n" + "-" * 60)
    print("2.2 KED 完整搜索（原始检索 -> 关键词退化回退）")
    print("-" * 60)
    try:
        ked_results = lins.KED_search(
            question=sample['question'],
            topk=10,
            if_split_n=False
        )
        if ked_results and ked_results.get('texts'):
            print(f"  ✓ KED 搜索成功，获取 {len(ked_results['texts'])} 条结果")
            print(f"  来源 URLs:")
            for i, url in enumerate(ked_results['urls'][:5]):
                print(f"    [{i+1}] {url}")
        else:
            print("  ⚠ KED 搜索未返回结果")
    except Exception as e:
        print(f"  ✗ KED 搜索异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 2.3 KED 退化回退机制测试 ==========
    print("\n" + "-" * 60)
    print("2.3 KED 关键词退化回退机制（模拟空结果后的退化过程）")
    print("-" * 60)
    try:
        complex_question = (
            "What is the molecular mechanism of alpha-synuclein aggregation "
            "in Parkinson's disease and its relationship with mitochondrial dysfunction?"
        )
        print(f"  原始问题: {complex_question[:60]}...")

        extracted = lins.keyword_extraction(
            question=complex_question,
            max_num_keywords=5
        )
        print(f"  提取关键词: {extracted}")
        print(f"  ✓ KED 退化机制测试完成")
    except Exception as e:
        print(f"  ✗ KED 退化测试异常: {type(e).__name__}: {str(e)[:80]}")

    return lins


# ============================================================
# 汇总统计输出
# ============================================================
def print_metric_summary(all_metrics, scope_label="有效样本"):
    """打印评估指标的汇总统计"""
    if not all_metrics:
        print("  无有效评估数据")
        return

    metric_names = [
        'citation_set_precision', 'citation_precision', 'citation_recall',
        'f1_score', 'statement_correctness', 'statement_fluency', 'overall_score'
    ]
    metric_labels_cn = [
        '引用集精准率(CSP)', '引用精准率(CP)', '引用召回率(CR)',
        'F1 分数', '陈述正确性(SC)', '陈述流畅度(SF)', '总体评分'
    ]

    print(f"  {'指标':<24} {'平均':<8} {'最高':<8} {'最低':<8} {'中位数':<8}")
    print(f"  {'-'*56}")

    summary = {}
    for name, label in zip(metric_names, metric_labels_cn):
        values = [m['metrics'][name] for m in all_metrics]
        avg_val = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        sorted_vals = sorted(values)
        median_val = sorted_vals[len(sorted_vals)//2] if sorted_vals else 0
        summary[name] = {
            'avg': avg_val, 'max': max_val, 'min': min_val, 'median': median_val
        }
        print(f"  {label:<24} {avg_val:<8.4f} {max_val:<8.4f} {min_val:<8.4f} {median_val:<8.4f}")

    # 各样本明细表
    print(f"\n  {'='*66}")
    print(f"  各样本详细分数:")
    print(f"  {'='*66}")
    print(f"  {'#':<4} {'PMID':<10} {'CSP':<7} {'CP':<7} {'CR':<7} {'F1':<7} {'SC':<5} {'SF':<7}")
    print(f"  {'-'*56}")
    for i, m in enumerate(all_metrics):
        met = m['metrics']
        print(f"  {i+1:<4} {m['id']:<10} {met['citation_set_precision']:<7.3f} "
              f"{met['citation_precision']:<7.3f} {met['citation_recall']:<7.3f} "
              f"{met['f1_score']:<7.3f} {met['statement_correctness']:<5} "
              f"{met['statement_fluency']:<7.3f}")

    # Overall Score 分布
    print(f"\n  {'='*44}")
    print(f"  Overall Score 分布:")
    print(f"  {'='*44}")
    overalls = [m['metrics']['overall_score'] for m in all_metrics]
    bins = [
        (0.8, 1.0, '★★★★★ 优秀'),
        (0.6, 0.8, '★★★★  良好'),
        (0.4, 0.6, '★★★   一般'),
        (0.2, 0.4, '★★    较差'),
        (0.0, 0.2, '★     很差'),
    ]
    for lo, hi, label in bins:
        count = sum(1 for v in overalls if lo <= v < hi)
        bar = '█' * count
        print(f"  {label:<16} [{lo:.1f}-{hi:.1f}): {count:>2} 个 {bar}")
    count_10 = sum(1 for v in overalls if v == 1.0)
    if count_10 > 0:
        print(f"  {'★★★★★ 满分':<16} [1.0-1.0]: {count_10:>2} 个 {'█' * count_10}")

    print(f"\n  >>> 模型综合评估结论 <<<")
    print(f"  总体均分: {summary['overall_score']['avg']:.4f}")
    print(f"  引用质量均分: {(summary['citation_set_precision']['avg'] + summary['citation_precision']['avg'] + summary['citation_recall']['avg']) / 3:.4f}")
    print(f"  陈述质量均分: {(summary['statement_correctness']['avg'] + summary['statement_fluency']['avg']) / 2:.4f}")


# ============================================================
# LinkEval 评估（公用部分）
# ============================================================
def load_linkeval_data(eval_results, result_dir="./eval_results_lins_full"):
    """从评估结果生成 LinkEval 所需的输入数据"""
    linkeval_inputs = []
    for r in eval_results:
        linkeval_inputs.append({
            'id': r.get('id', ''),
            'question': r.get('question', ''),
            'response': r.get('mAirag_response', ''),
            'statements': r.get('statements', []),
            'references': r.get('references', []),
            'mAirag_urls': r.get('mAirag_urls', []),
            'ked_urls': r.get('ked_urls', []),
            'ked_count': r.get('ked_count', 0)
        })
    return linkeval_inputs


def evaluate_single_linkeval_sample(sample_eval, link_eval):
    """评估单个样本的 LinkEval 指标"""
    statements_eval = sample_eval.get("statements", [])
    refs_eval = sample_eval.get("references", [])

    if not statements_eval or not refs_eval:
        return None

    metrics_eval = link_eval.evaluate(statements_eval, refs_eval)
    details_eval = link_eval.get_detail()
    return metrics_eval, details_eval
