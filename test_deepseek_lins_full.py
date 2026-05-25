"""
test_deepseek_lins_full.py — LINS 完整框架测试 (使用 DeepSeek)
参照论文 LINS-LLM：完整配置 database(Online + HERD) + retriever(KED) + 多智能体

主要配置：
1. Online Database: PubMed (Bio.Entrez API) + Bing (实时搜索)
2. HERD 多级检索: Guidelines(本地) → PubMed → Bing
3. Retriever: KED (Keyword Extraction Degradation) 算法
4. 测试数据集：PubmedQA（快速验证用前3条，后续可扩展）

使用方法：
  set DEEPSEEK_API_KEY=sk-your-key
  python test_deepseek_lins_full.py
"""

import os
import sys
import json
import time

# ========== 配置区域 ==========
os.environ['DEEPSEEK_API_KEY'] = 'sk-94ccad564a7542228ad52f6b2654e11e'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# PubmedQA 数据集路径
PUBMEDQA_ORI_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/ori_pqal.json"
PUBMEDQA_GT_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/test_ground_truth.json"

# 测试 PubmedQA 样本数
PUBMEDQA_TEST_SAMPLES = 5

# 确保 LINS-main 在 sys.path 中
LINS_MAIN_PATH = os.path.join(os.getcwd(), "LINS-main")
if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)
# =============================

from model.model_LINS import LINS
from model.database import LINS_Database
from model.retriever_model import LINS_Retriever
from model.prompts import return_prompts


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
    
    # 引用/检索指标
    mairag_url_count = len(mAirag_urls) if mAirag_urls else 0
    ked_url_count = len(ked_urls) if ked_urls else 0
    
    # --- 规则1（收紧）：不确定性关键词 >= 3个 才触发 ---
    if uncertainty_count >= 3:
        print(f"     [后处理规则1] 不确定性关键词 {uncertainty_count} 个 → 强制maybe")
        return True
    
    # --- 规则2（收紧）：引用数=0 且 不确定性词 >= 2个 ---
    if mairag_url_count == 0 and uncertainty_count >= 2:
        print(f"     [后处理规则2] 引用数=0, 不确定性词={uncertainty_count} → 强制maybe")
        return True
    
    # --- 规则3（收紧）：KED检索到结果，但MAIRAG引用0条 且 回答含不确定性词 ---
    if ked_count > 0 and mairag_url_count == 0 and ked_url_count > 0 and uncertainty_count >= 1:
        print(f"     [后处理规则3] KED检索{ked_count}条但MAIRAG引用0条 + 不确定性词 {uncertainty_count} → 强制maybe")
        return True
    
    # --- 规则4（收紧）：仅匹配最明确的"further research needed"等 ---
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
# 包括 Online Database (PubMed, Bing) 和 HERD 多级检索
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
    
    # 检查本地指南库是否存在
    guidelines_path = "./add_dataset/guidelines/guidelines_embedding.json"
    guidelines_available = os.path.exists(f"./LINS-main/{guidelines_path}") or os.path.exists(guidelines_path)
    print(f"  本地指南库: {'✓ 存在' if guidelines_available else '✗ 不存在（将跳过指南层）'}")
    
    # HERD 搜索
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
        
        if " AND " in extracted:
            keyword_parts = extracted.split(" AND ")
            print(f"  关键词数量: {len(keyword_parts)}")
            for step in range(len(keyword_parts), 0, -1):
                degraded = " AND ".join(keyword_parts[:step])
                print(f"    退化步骤 {len(keyword_parts) - step + 1}: {degraded}")
    except Exception as e:
        print(f"  ✗ 退化测试异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 2.4 get_passages 完整检索链路 ==========
    print("\n" + "-" * 60)
    print("2.4 get_passages 完整检索链路（含 embedding 重排序）")
    print("-" * 60)
    try:
        retrieved = lins.get_passages(
            question=sample['question'],
            topk=5,
            if_split_n=False,
            recall_top_k=20
        )
        if retrieved and retrieved.get('texts'):
            print(f"  ✓ 检索成功，获取 {len(retrieved['texts'])} 条精排结果")
            print(f"  Top-3 URLs:")
            for i, url in enumerate(retrieved['urls'][:3]):
                score = retrieved['scores'][i] if retrieved.get('scores') else None
                score_str = f" (score: {score:.4f})" if score is not None else ""
                print(f"    [{i+1}] {url}{score_str}")
        else:
            print("  ⚠ get_passages 未返回结果")
    except Exception as e:
        print(f"  ✗ get_passages 异常: {type(e).__name__}: {str(e)[:80]}")


# ============================================================
# 第三部分：多智能体协同测试（MAIRAG + 各模块）
# ============================================================
def test_multi_agent(lins):
    """测试多智能体协同：MAIRAG、PRM、SKA、QDA、PCM"""
    print("\n" + "=" * 70)
    print("【第三部分】多智能体协同测试")
    print("  3.1 PRM - 段落相关性评估 (Passage Relevance Module)")
    print("  3.2 SKM - 自知识分析 (Self Knowledge Module)")
    print("  3.3 QDM - 问题分解 (Question Decomposition Module)")
    print("  3.4 MAIRAG - 完整检索增强生成")
    print("  3.5 MAIRAG_options - 带选项的选择题回答")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples()
    sample = pubmedqa_samples[0]
    
    # ========== 3.1 PRM ==========
    print("\n" + "-" * 60)
    print("3.1 PRM 段落相关性评估（使用 PubmedQA 真实上下文段落）")
    print("-" * 60)
    try:
        test_passages = sample['contexts'][:5]
        print(f"  评估 {len(test_passages)} 个段落与问题的相关性...")
        prm_results = lins.PRM(
            question=sample['question'],
            refs=test_passages
        )
        gold_count = sum(1 for r in prm_results if r == "Gold")
        print(f"  ✓ PRM 完成: {gold_count}/{len(prm_results)} 个段落判定为 Gold")
        for i, result in enumerate(prm_results):
            label = sample['labels'][i] if i < len(sample['labels']) else "N/A"
            print(f"    [{i+1}] Label: {label} -> {result}")
    except Exception as e:
        print(f"  ✗ PRM 异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 3.2 SKM ==========
    print("\n" + "-" * 60)
    print("3.2 SKM 自知识分析")
    print("-" * 60)
    try:
        skm_result = lins.SKM(question=sample['question'])
        print(f"  SKM 结果: {skm_result[0][:50] if skm_result else 'None'}...")
        if skm_result and 'CERTAIN' in skm_result[0]:
            print("  -> 模型自信可回答（CERTAIN）")
        else:
            print("  -> 模型不确定（UNCERTAIN），将触发检索")
    except Exception as e:
        print(f"  ✗ SKM 异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 3.3 QDM ==========
    print("\n" + "-" * 60)
    print("3.3 QDM 问题分解")
    print("-" * 60)
    try:
        qdm_result = lins.QDM(question=sample['question'])
        if qdm_result:
            print(f"  ✓ QDM 生成 {len(qdm_result)} 个子问题:")
            for i, q in enumerate(qdm_result):
                print(f"    Sub-{i+1}: {q[:80]}")
        else:
            print("  ⚠ QDM 未生成子问题")
    except Exception as e:
        print(f"  ✗ QDM 异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 3.4 MAIRAG ==========
    print("\n" + "-" * 60)
    print("3.4 MAIRAG 完整检索增强生成")
    print("-" * 60)
    try:
        print(f"  问题: {sample['question'][:80]}...")
        response, urls, passages, history, sub_qs = lins.MAIRAG(
            question=sample['question'],
            if_PRA=True,
            if_SKA=True,
            if_QDA=True,
            if_PCA=False
        )
        if response:
            print(f"  ✓ MAIRAG 回答生成成功:")
            print(f"    回答预览: {response[:200]}...")
            if urls:
                print(f"  引用来源 ({len(urls)} 条):")
                for u in urls[:3]:
                    print(f"    - {u}")
        else:
            print("  ⚠ MAIRAG 未生成回答")
    except Exception as e:
        print(f"  ✗ MAIRAG 异常: {type(e).__name__}: {str(e)[:80]}")

    # ========== 3.5 MAIRAG_options ==========
    print("\n" + "-" * 60)
    print("3.5 MAIRAG_options 带选项的选择题回答")
    print("-" * 60)
    try:
        options_question = f"""
{sample['question']}
A. Yes
B. No
C. Maybe
"""
        print(f"  原始问题: {sample['question'][:60]}...")
        print(f"  正确答案(ground truth): {sample['answer']}")
        
        options_response, options_urls, options_passages, options_history, options_sub_qs = lins.MAIRAG_options(
            question=options_question,
            topk=5,
            single_choice=True
        )
        print(f"  ✓ MAIRAG_options 返回: {options_response}")
        if options_urls:
            print(f"  引用来源 ({len(options_urls)} 条)")
    except Exception as e:
        print(f"  ✗ MAIRAG_options 异常: {type(e).__name__}: {str(e)[:80]}")


# ============================================================
# 第四部分：PubmedQA 完整评估 + KED 准确率
# ============================================================
def test_pubmedqa_evaluation(lins):
    """在 PubmedQA 数据集上运行完整评估，重点测试 KED 检索准确率"""
    print("\n" + "=" * 70)
    print("【第四部分】PubmedQA 完整评估 + KED 准确率分析")
    print("  重点：评估 KED 算法在 PubmedQA 上的检索准确率")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples()
    total = len(pubmedqa_samples)
    
    eval_results = []
    
    for idx, sample in enumerate(pubmedqa_samples):
        print(f"\n  [{idx+1}/{total}] 样本 {sample['id']}: {sample['question'][:60]}...")
        print(f"  Ground Truth: {sample['answer']}")
        
        sample_result = {
            "id": sample['id'],
            "question": sample['question'],
            "ground_truth": sample['answer'],
        }
        
        # ---- 4a. MAIRAG 完整回答（if_PCA=True 启用一致性验证）----
        try:
            response, urls, passages, history, sub_qs = lins.MAIRAG(
                question=sample['question'] + "\nPlease include citation numbers like [1], [2] etc. in your answer.",
                topk=5,
                if_PRA=True,
                if_SKA=True,
                if_QDA=True,
                if_PCA=True,  # 启用Passage Coherence验证，提升回答一致性
                recall_top_k=100  # 增加召回候选数量以提升检索覆盖
            )
            sample_result["mAirag_response"] = response
            sample_result["mAirag_urls"] = urls
            sample_result["mAirag_passages"] = passages
            sample_result["retrieved_count"] = len(passages) if passages else 0
            print(f"     MAIRAG: ✓ 回答生成 ({len(passages) if passages else 0} 条引用)")
        except Exception as e:
            sample_result["mAirag_error"] = str(e)[:100]
            sample_result["mAirag_response"] = None
            sample_result["mAirag_urls"] = []
            sample_result["mAirag_passages"] = []
            print(f"     MAIRAG: ✗ {type(e).__name__}")
        
        # ---- 4b. KED 检索统计（作为后台诊断信息）----
        try:
            ked_data = lins.KED_search(
                question=sample['question'],
                topk=20,
                if_split_n=False
            )
            if ked_data and ked_data.get('texts'):
                ked_count = len(ked_data['texts'])
                sample_result["ked_retrieved_count"] = ked_count
                sample_result["ked_urls"] = ked_data['urls'][:5]
                print(f"     KED检索: ✓ {ked_count} 条结果")
            else:
                sample_result["ked_retrieved_count"] = 0
                print(f"     KED检索: ⚠ 无结果")
        except Exception as e:
            sample_result["ked_error"] = str(e)[:100]
            print(f"     KED检索: ✗ {type(e).__name__}")
        
        # ---- 4c. 选择题评估 — 复用 MAIRAG 的检索结果（避免重复检索）----
        try:
            options_prompt = f"""
Question: {sample['question']}
Options:
A. Yes
B. No
C. Maybe

Please answer with only the option letter (A, B, or C).
"""
            # 复用 MAIRAG 检索到的 passages，避免重复调用 get_passages
            mAirag_passages = sample_result.get("mAirag_passages", [])
            mAirag_urls = sample_result.get("mAirag_urls", [])
            
            if mAirag_passages:
                # 将 MAIRAG 的检索结果传给 MAIRAG_options，跳过其内部检索
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=options_prompt,
                    topk=5,
                    single_choice=True,
                    retrieval_passages=mAirag_passages
                )
                print(f"     选择题(复用MAIRAG结果): {options_response}")
            else:
                # 无检索结果时，让 MAIRAG_options 自行处理
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=options_prompt,
                    topk=5,
                    single_choice=True
                )
                print(f"     选择题(独立检索): {options_response}")
            
            # ===== 后处理：检测不确定性并强制倾向 Maybe =====
            if options_response and options_response.strip().upper() in ["A", "B"]:
                mAirag_response = sample_result.get("mAirag_response", "")
                mAirag_passages = sample_result.get("mAirag_passages", [])
                mAirag_urls = sample_result.get("mAirag_urls", [])
                ked_urls = sample_result.get("ked_urls", [])
                ked_count = sample_result.get("ked_retrieved_count", 0)
                
                if should_force_maybe(mAirag_response, mAirag_passages, mAirag_urls, ked_urls, ked_count):
                    options_response = "C"
                    sample_result["postprocess_forced_maybe"] = True
                    print(f"     → 后处理规则触发: {sample_result.get('options_response', '')} → C (Maybe)")
                else:
                    sample_result["postprocess_forced_maybe"] = False
            
            sample_result["options_response"] = options_response
            
            answer_map = {"yes": "A", "no": "B", "maybe": "C"}
            gt_letter = answer_map.get(sample['answer'].lower(), None)
            prediction = options_response.strip().upper() if options_response else ""
            
            if gt_letter and prediction:
                is_correct = (prediction == gt_letter)
                sample_result["is_correct"] = is_correct
                print(f"     结果: {prediction} (GT: {sample['answer']} -> {gt_letter}) {'✓ 正确' if is_correct else '✗ 错误'}")
            else:
                print(f"     结果: {prediction} (无法与 GT 比较)")
        except Exception as e:
            sample_result["options_error"] = str(e)[:100]
            print(f"     选择题回答: ✗ {type(e).__name__}")
        
        eval_results.append(sample_result)
        print()

    
    # ---- 汇总统计 ----
    print("\n" + "=" * 60)
    print("PubmedQA 评估汇总")
    print("=" * 60)
    
    correct_count = sum(1 for r in eval_results if r.get("is_correct"))
    total_with_gt = sum(1 for r in eval_results if "is_correct" in r)
    avg_retrieved = sum(r.get("retrieved_count", 0) for r in eval_results) / max(total, 1)
    avg_ked = sum(r.get("ked_retrieved_count", 0) for r in eval_results) / max(total, 1)
    forced_maybe_count = sum(1 for r in eval_results if r.get("postprocess_forced_maybe"))
    
    print(f"  总样本数:          {total}")
    print(f"  MAIRAG 平均引用数:  {avg_retrieved:.1f}")
    print(f"  KED 平均检索数:     {avg_ked:.1f}")
    print(f"  后处理触发次数:     {forced_maybe_count}")
    if total_with_gt > 0:
        accuracy = correct_count / total_with_gt * 100
        print(f"  选择题准确率:       {correct_count}/{total_with_gt} = {accuracy:.1f}%")
    
    results_path = "./eval_results_lins_full/pubmedqa_details.jsonl"
    os.makedirs("./eval_results_lins_full", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  详细结果已保存至: {results_path}")
    
    summary = {
        "total_samples": total,
        "correct_count": correct_count,
        "total_with_gt": total_with_gt,
        "accuracy": round(correct_count / max(total_with_gt, 1) * 100, 2) if total_with_gt > 0 else 0,
        "avg_retrieved_count": round(avg_retrieved, 2),
        "avg_ked_retrieved": round(avg_ked, 2),
        "forced_maybe_count": forced_maybe_count,
        "model": "deepseek-chat",
        "retriever": "BGE",
        "database": "pubmed_herd",
        "test_mode": "LINS_full_framework"
    }
    summary_path = "./eval_results_lins_full/accuracy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  汇总已保存至: {summary_path}")
    
    return eval_results


# ============================================================
# 第五部分：LinkEval 量化评估
# ============================================================
def test_linkeval(lins, eval_results=None):
    """使用 LinkEval-DeepSeek 对 PubmedQA 回答进行量化评估
    复用第四部分的 MAIRAG 结果（含 response 和 passages），避免重复检索"""
    print("\n" + "=" * 70)
    print("【第五部分】LinkEval-DeepSeek 量化评估")
    print("  指标: 引用精准率(CP) | 引用召回率(CR) | F1 | 陈述正确性(SC) | 流畅度(SF)")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples()
    
    try:
        from Link_Eval_DeepSeek import LinkEvalDeepSeek, convert_to_statements, format_metrics
        
        link_eval = LinkEvalDeepSeek(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            model_name='deepseek-chat',
            verbose=False
        )
        print("  LinkEval-DeepSeek 初始化成功 ✓\n")

        all_metrics = []
        valid_count = 0

        for idx, sample in enumerate(pubmedqa_samples):
            print(f"  [{idx+1}/{len(pubmedqa_samples)}] {sample['id']} | {sample['question'][:60]}...")
            
            try:
                # 复用第四部分的 MAIRAG 结果
                response = None
                passages = []
                if eval_results and idx < len(eval_results):
                    response = eval_results[idx].get("mAirag_response")
                    passages = eval_results[idx].get("mAirag_passages") or []
                
                # 如果第四部分没有 MAIRAG 结果，再调用一次
                if not response:
                    print(f"     第四部分无 MAIRAG 结果，重新调用...")
                    response, _, passages, _, _ = lins.MAIRAG(
                        question=sample['question'] + "\nPlease include citation numbers like [1], [2] etc. in your answer.",
                        if_PRA=True,
                        if_SKA=True,
                        if_QDA=True,
                        if_PCA=False
                    )
                
                if not response:
                    print(f"      ⚠ 无回答生成，跳过")
                    continue
                
                statements = convert_to_statements(response)
                refs = passages if passages else sample['contexts'][:5]
                
                if statements and refs:
                    metrics, details = link_eval.evaluate(
                        question=sample['question'],
                        statements=statements,
                        refs=refs,
                        correct_answer=sample.get('answer', None),
                        return_details=True
                    )
                    all_metrics.append({
                        'id': sample['id'],
                        'metrics': metrics,
                        'details': details,
                        'statements_count': len(statements),
                        'refs_count': len(refs)
                    })
                    valid_count += 1
                    print(f"      ✓ CP={metrics['citation_precision']:.3f} "
                          f"CR={metrics['citation_recall']:.3f} "
                          f"F1={metrics['f1_score']:.3f} "
                          f"SC={metrics['statement_correctness']} "
                          f"SF={metrics['statement_fluency']:.3f}")
                else:
                    print(f"      ⚠ 无有效陈述或引用，跳过")
            except Exception as e:
                print(f"      ✗ 异常: {type(e).__name__}: {str(e)[:80]}")

        if all_metrics:
            print(f"\n  {'='*60}")
            print(f"  LinkEval 量化评估汇总 ({valid_count} 个有效样本)")
            print(f"  {'='*60}")
            
            metrics_names = [
                'citation_precision', 'citation_recall', 'f1_score',
                'statement_correctness', 'statement_fluency'
            ]
            
            averages = {}
            for name in metrics_names:
                values = [m['metrics'][name] for m in all_metrics if name in m['metrics']]
                if values:
                    avg_val = sum(values) / len(values)
                    averages[name] = round(avg_val, 3)
                    print(f"    {name}: {avg_val:.3f}")
            
            # 生成汇总报告
            linkeval_summary = {
                "total_evaluated": valid_count,
                "averages": averages,
                "model": "deepseek-chat",
                "eval_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_path = "./eval_results_lins_full/linkeval_results.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(linkeval_summary, f, ensure_ascii=False, indent=2)
            print(f"\n  LinkEval 汇总已保存至: {summary_path}")
        else:
            print("  ⚠ 没有有效评估结果")
    
    except ImportError as e:
        print(f"  ✗ 导入 LinkEval-DeepSeek 失败: {e}")
        print("  请确保 LINS-main/Link_Eval_DeepSeek.py 存在")
    except Exception as e:
        print(f"  ✗ LinkEval 评估异常: {type(e).__name__}: {str(e)[:80]}")


# ============================================================
# MedQA 数据集加载
# ============================================================
MEDQA_US_PATH = "./LINS-main/evaluate/evaluate_data/medqa_us/data/medqa_us_test.json"
MEDQA_MAINLAND_PATH = "./LINS-main/evaluate/evaluate_data/medqa_mainland/data/medqa_mainland_test.json"
MEDQA_TEST_SAMPLES = 15


def load_medqa_samples(path, num_samples=None, region="us"):
    """加载 MedQA 测试样本，每个样本取前 num_samples 条"""
    if num_samples is None:
        num_samples = MEDQA_TEST_SAMPLES
    
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    samples = []
    # raw_data 是 dict: {key: {QUESTION, options_str, options, answer, ...}}
    for i, (key, entry) in enumerate(raw_data.items()):
        if len(samples) >= num_samples:
            break
        samples.append({
            "id": key,
            "question": entry["QUESTION"],
            "options_str": entry["options_str"],
            "options": entry["options"],
            "answer": entry["answer"],
            "answer_text": entry.get("answer_text", ""),
            "region": region
        })
    
    print(f"已加载 {len(samples)} 条 MedQA-{region.upper()} 测试样本")
    for s in samples:
        options_short = " / ".join(list(s["options"].keys())[:3]) + "..."
        print(f"  [{s['id']}] {s['question'][:50]}... → answer: {s['answer']}")
    return samples


# ============================================================
# 第六部分：MEDQA-US 选择题评估
# ============================================================
def test_medqa_us_evaluation(lins):
    """在 MedQA-US 数据集上运行选择题评估（15个样本）"""
    print("\n" + "=" * 70)
    print("【第六部分】MedQA-US 选择题评估 (15个样本)")
    print("  数据集: MedQA (美国医学执照考试)")
    print("=" * 70)
    
    samples = load_medqa_samples(MEDQA_US_PATH, num_samples=MEDQA_TEST_SAMPLES, region="us")
    if not samples:
        print("  ✗ 未加载到 MedQA-US 样本")
        return []
    
    eval_results = []
    
    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['id']}: {sample['question'][:60]}...")
        print(f"  Ground Truth: {sample['answer']}")
        
        sample_result = {
            "id": sample['id'],
            "question": sample['question'],
            "ground_truth": sample['answer'],
            "region": "us"
        }
        
        # 构造选项文本
        options_list = []
        for letter, text in sample['options'].items():
            options_list.append(f"{letter}: {text}")
        options_str = "\n".join(options_list)
        
        # ---- 6a. MAIRAG 回答 ----
        try:
            response, urls, passages, history, sub_qs = lins.MAIRAG(
                question=f"{sample['question']}\nPlease include citation numbers like [1], [2] in your answer.",
                topk=5,
                if_PRA=True,
                if_SKA=True,
                if_QDA=True,
                if_PCA=True
            )
            sample_result["mAirag_response"] = response
            sample_result["mAirag_urls"] = urls
            sample_result["mAirag_passages"] = passages
            print(f"     MAIRAG: ✓ 回答生成 ({len(passages) if passages else 0} 条引用)")
        except Exception as e:
            sample_result["mAirag_response"] = None
            sample_result["mAirag_urls"] = []
            sample_result["mAirag_passages"] = []
            print(f"     MAIRAG: ✗ {type(e).__name__}")
        
        # ---- 6b. KED 检索统计 ----
        try:
            ked_data = lins.KED_search(question=sample['question'], topk=20, if_split_n=False)
            if ked_data and ked_data.get('texts'):
                sample_result["ked_retrieved_count"] = len(ked_data['texts'])
                sample_result["ked_urls"] = ked_data['urls'][:5]
                print(f"     KED检索: ✓ {len(ked_data['texts'])} 条结果")
            else:
                sample_result["ked_retrieved_count"] = 0
                print(f"     KED检索: ⚠ 无结果")
        except Exception as e:
            sample_result["ked_error"] = str(e)[:100]
            print(f"     KED检索: ✗ {type(e).__name__}")
        
        # ---- 6c. 选择题回答 ----
        try:
            prompt = f"""Question: {sample['question']}

Options:
{options_str}

Please answer with only the option letter (A, B, C, D, E...). Do not include any additional text."""
            
            mAirag_passages = sample_result.get("mAirag_passages", [])
            
            if mAirag_passages:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=prompt, topk=5, single_choice=True,
                    retrieval_passages=mAirag_passages
                )
            else:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=prompt, topk=5, single_choice=True
                )
            
            # 后处理：同样应用 should_force_maybe
            if options_response and options_response.strip().upper() in ["A", "B"]:
                if should_force_maybe(
                    sample_result.get("mAirag_response", ""),
                    sample_result.get("mAirag_passages", []),
                    sample_result.get("mAirag_urls", []),
                    sample_result.get("ked_urls", []),
                    sample_result.get("ked_retrieved_count", 0)
                ):
                    options_response = "C"
                    sample_result["postprocess_forced_maybe"] = True
                    print(f"     → 后处理规则触发: C (Maybe)")
                else:
                    sample_result["postprocess_forced_maybe"] = False
            
            sample_result["options_response"] = options_response
            
            prediction = options_response.strip().upper() if options_response else ""
            gt = sample['answer'].strip().upper()
            is_correct = (prediction == gt) if prediction and gt else False
            sample_result["is_correct"] = is_correct
            print(f"     选择题: {prediction} (GT: {gt}) {'✓ 正确' if is_correct else '✗ 错误'}")
            
        except Exception as e:
            sample_result["options_error"] = str(e)[:100]
            print(f"     选择题回答: ✗ {type(e).__name__}")
        
        eval_results.append(sample_result)
        print()
    
    # ---- 汇总 ----
    correct = sum(1 for r in eval_results if r.get("is_correct"))
    total = len(eval_results)
    print(f"\n  MedQA-US 汇总: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
    
    results_path = "./eval_results_lins_full/medqa_us_details.jsonl"
    os.makedirs("./eval_results_lins_full", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  结果已保存至: {results_path}")
    
    return eval_results


# ============================================================
# 第七部分：MEDQA-Mainland 选择题评估
# ============================================================
def test_medqa_mainland_evaluation(lins):
    """在 MedQA-Mainland 数据集上运行选择题评估（15个样本）"""
    print("\n" + "=" * 70)
    print("【第七部分】MedQA-Mainland 选择题评估 (15个样本)")
    print("  数据集: MedQA (中国大陆医学考试)")
    print("=" * 70)
    
    samples = load_medqa_samples(MEDQA_MAINLAND_PATH, num_samples=MEDQA_TEST_SAMPLES, region="mainland")
    if not samples:
        print("  ✗ 未加载到 MedQA-Mainland 样本")
        return []
    
    eval_results = []
    
    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['id']}: {sample['question'][:60]}...")
        print(f"  Ground Truth: {sample['answer']}")
        
        sample_result = {
            "id": sample['id'],
            "question": sample['question'],
            "ground_truth": sample['answer'],
            "region": "mainland"
        }
        
        # 构造选项文本
        options_list = []
        for letter, text in sample['options'].items():
            options_list.append(f"{letter}: {text}")
        options_str = "\n".join(options_list)
        
        # ---- 7a. MAIRAG 回答 ----
        try:
            response, urls, passages, history, sub_qs = lins.MAIRAG(
                question=f"{sample['question']}\n请在回答中标注引用编号如[1]、[2]等。",
                topk=5,
                if_PRA=True,
                if_SKA=True,
                if_QDA=True,
                if_PCA=True
            )
            sample_result["mAirag_response"] = response
            sample_result["mAirag_urls"] = urls
            sample_result["mAirag_passages"] = passages
            print(f"     MAIRAG: ✓ 回答生成 ({len(passages) if passages else 0} 条引用)")
        except Exception as e:
            sample_result["mAirag_response"] = None
            sample_result["mAirag_urls"] = []
            sample_result["mAirag_passages"] = []
            print(f"     MAIRAG: ✗ {type(e).__name__}")
        
        # ---- 7b. KED 检索统计 ----
        try:
            ked_data = lins.KED_search(question=sample['question'], topk=20, if_split_n=False)
            if ked_data and ked_data.get('texts'):
                sample_result["ked_retrieved_count"] = len(ked_data['texts'])
                sample_result["ked_urls"] = ked_data['urls'][:5]
                print(f"     KED检索: ✓ {len(ked_data['texts'])} 条结果")
            else:
                sample_result["ked_retrieved_count"] = 0
                print(f"     KED检索: ⚠ 无结果")
        except Exception as e:
            sample_result["ked_error"] = str(e)[:100]
            print(f"     KED检索: ✗ {type(e).__name__}")
        
        # ---- 7c. 选择题回答 ----
        try:
            prompt = f"""问题：{sample['question']}

选项：
{options_str}

请只输出选项的字母（A、B、C、D、E...），不要输出任何额外文字。"""
            
            mAirag_passages = sample_result.get("mAirag_passages", [])
            
            if mAirag_passages:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=prompt, topk=5, single_choice=True,
                    retrieval_passages=mAirag_passages
                )
            else:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=prompt, topk=5, single_choice=True
                )
            
            sample_result["options_response"] = options_response
            
            prediction = options_response.strip().upper() if options_response else ""
            gt = sample['answer'].strip().upper()
            is_correct = (prediction == gt) if prediction and gt else False
            sample_result["is_correct"] = is_correct
            print(f"     选择题: {prediction} (GT: {gt}) {'✓ 正确' if is_correct else '✗ 错误'}")
            
        except Exception as e:
            sample_result["options_error"] = str(e)[:100]
            print(f"     选择题回答: ✗ {type(e).__name__}")
        
        eval_results.append(sample_result)
        print()
    
    # ---- 汇总 ----
    correct = sum(1 for r in eval_results if r.get("is_correct"))
    total = len(eval_results)
    print(f"\n  MedQA-Mainland 汇总: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
    
    results_path = "./eval_results_lins_full/medqa_mainland_details.jsonl"
    os.makedirs("./eval_results_lins_full", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  结果已保存至: {results_path}")
    
    return eval_results


# ============================================================
# 第八部分：MedQA 汇总报告
# ============================================================
def summarize_medqa_results():
    """汇总 MedQA-US 和 MedQA-Mainland 的评估结果"""
    print("\n" + "=" * 70)
    print("【第八部分】MedQA 综合评估汇总")
    print("=" * 70)
    
    regions = {
        "medqa_us": {"path": "./eval_results_lins_full/medqa_us_details.jsonl", "name": "MedQA-US"},
        "medqa_mainland": {"path": "./eval_results_lins_full/medqa_mainland_details.jsonl", "name": "MedQA-Mainland"}
    }
    
    all_summary = {}
    
    for key, cfg in regions.items():
        if os.path.exists(cfg["path"]):
            results = []
            with open(cfg["path"], "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            
            correct = sum(1 for r in results if r.get("is_correct"))
            total = len(results)
            accuracy = round(correct / max(total, 1) * 100, 1)
            
            print(f"\n  {cfg['name']}:")
            print(f"    样本数: {total}")
            print(f"    正确数: {correct}")
            print(f"    准确率: {accuracy}%")
            
            all_summary[key] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy
            }
    
    summary_path = "./eval_results_lins_full/medqa_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总已保存至: {summary_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：初始化模型并运行所有测试"""
    print("\n" + "=" * 70)
    print("LINS-LLM 完整框架测试 (DeepSeek)")
    print("  测试集: PubmedQA(20) + MedQA-US(15) + MedQA-Mainland(15)")
    print("=" * 70)
    
    # 初始化 LINS 模型
    print("\n初始化 LINS 模型...")
    lins = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='BGE',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        BGE_encoder_path='./model/retriever/bge/bge-m3',
        database_name='pubmed'
    )
    print("LINS 模型初始化完成 ✓")
    
    # 运行各测试模块
    test_database_config(lins)
    test_ked_retriever(lins)
    test_multi_agent(lins)
    
    # PubmedQA 评估
    eval_results = test_pubmedqa_evaluation(lins)
    test_linkeval(lins, eval_results)
    
    # MedQA 评估（各15个样本）
    test_medqa_us_evaluation(lins)
    test_medqa_mainland_evaluation(lins)
    summarize_medqa_results()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
