"""
LINS 项目 - DeepSeek API 完整测试脚本
覆盖原项目的核心功能流程链路

使用方式：
  1. 设置环境变量（推荐）：
     set DEEPSEEK_API_KEY=sk-your-deepseek-api-key
     python test_deepseek.py

  2. 或直接修改下方 DEEPSEEK_API_KEY 的值
"""

import os
import sys
import json

# ========== 配置区域 ==========
# 方式一：在这里直接填写你的 DeepSeek API Key（取消注释并填入）
os.environ['DEEPSEEK_API_KEY'] = 'sk-ed0d07bcdbfe4d9e99f8653760954022'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# =============================

from model.model_LINS import LINS

# ============================================================
# PubmedQA 数据集加载工具
# 使用原开源项目使用的标准评估数据集
# ============================================================
PUBMEDQA_ORI_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/ori_pqal.json"
PUBMEDQA_GT_PATH = "./LINS-main/evaluate/evaluate_data/pubmedqa/data/test_ground_truth.json"

# ========== 评估配置 ==========
# 在这里修改 PubmedQA 测试样本数，所有用到的地方会自动更新
PUBMEDQA_TEST_SAMPLES = 3
# =============================

def load_pubmedqa_samples(num_samples=None):
    """
    从 PubmedQA 数据集加载测试样本。
    
    参数：
        num_samples: int，加载的样本数量，默认为 None
                     会自动使用全局变量 PUBMEDQA_TEST_SAMPLES 的值
    
    返回：
        samples: list[dict]，每个包含：
            - id: PubMed ID
            - question: 问题文本
            - contexts: 来自PubMed的上下文段落列表
            - labels: 段落标签（BACKGROUND/RESULTS等）
            - answer: ground truth 答案（yes/no/maybe）
    """
    if num_samples is None:
        num_samples = PUBMEDQA_TEST_SAMPLES
    
    # 加载数据集
    with open(PUBMEDQA_ORI_PATH, "r", encoding="utf-8") as f:
        ori_data = json.load(f)
    with open(PUBMEDQA_GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    # 构建样本列表
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
        print(f"      上下文段落数: {len(s['contexts'])}")
    return samples



# ============================================================
# 方案一：仅使用 DeepSeek 做对话（不需要检索器，快速验证）
# ============================================================
def test_basic_chat():
    print("=" * 60)
    print("测试 1：基础对话（仅 DeepSeek，无需检索器）")
    print("=" * 60)

    lins = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='none',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        database_name='none'
    )

    # 多轮对话
    response, history = lins.chat(question="What is BCR-ABL1?", history=None)
    print("\nChat response:\n", response)

    # 第二轮对话
    response, history = lins.chat(question="What diseases is it associated with?", history=history)
    print("\nFollow-up response:\n", response)

    return lins


# ============================================================
# 方案二：DeepSeek + 本地 BGE 检索器（完整功能链测试）
# ============================================================
def test_with_retriever():
    print("=" * 60)
    print("测试 2：DeepSeek + 本地 BGE 检索器（完整功能链路）")
    print("注意：需要先下载 BGE 模型到 ./model/retriever/bge/bge-m3")
    print("=" * 60)

    lins = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='BGE',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        BGE_encoder_path='./model/retriever/bge/bge-m3',
        database_name='pubmed'
    )

    # ---------- 2.1 基础多轮对话 ----------
    print("\n" + "=" * 60)
    print("2.1 基础多轮对话")
    print("=" * 60)
    response, history = lins.chat(question="What is BCR-ABL1?", history=None)
    print("\nChat response:\n", response)
    response, history = lins.chat(question="What diseases is it associated with?", history=history)
    print("\nFollow-up response:\n", response)

    # 加载 PubmedQA 数据集中的真实样本（数量由 PUBMEDQA_TEST_SAMPLES 控制）
    pubmedqa_samples = load_pubmedqa_samples()

    # ---------- 2.2 原 MAIRAG 完整功能测试 ----------
    print("\n" + "=" * 60)
    print("2.2 MAIRAG 完整功能（含 PRM 段落相关性评估）")
    print("=" * 60)
    # 使用 PubmedQA 数据中的第一个问题
    sample = pubmedqa_samples[0]
    print(f"使用 PubmedQA 样本 [{sample['id']}]: {sample['question'][:80]}...")
    # 启用所有 MAIRAG 内部模块：PRA(PRM), SKA(SKM), QDA(QDM), PCA(PCM)
    response, urls, passages, history, sub_qs = lins.MAIRAG(
        question=sample['question'],
        if_PRA=True,
        if_SKA=True,
        if_QDA=True,
        if_PCA=True
    )
    print("\nMAIRAG response:\n", response)
    print("URLs:", urls)
    if sub_qs:
        print("Sub-questions:", sub_qs)

    # ---------- 2.3 PRM（段落相关性评估）独立测试 ----------
    print("\n" + "=" * 60)
    print("2.3 PRM 段落相关性评估独立测试（使用 PubmedQA 真实段落）")
    print("=" * 60)
    # 使用 PubmedQA 数据集中的真实上下文段落
    # 第一个样本的 CONTEXTS 中包含多个来自 PubMed 的真实段落
    # 它们都与问题直接相关，预期 PRM 会将它们标注为 "Gold"
    sample2 = pubmedqa_samples[1]
    test_passages = sample2['contexts']
    print(f"使用 PubmedQA 样本 [{sample2['id']}]: {sample2['question'][:60]}...")
    print(f"上下文字段落数: {len(test_passages)}")
    
    prm_results = lins.PRM(
        question=sample2['question'],
        refs=test_passages
    )
    print("PRM results (expected: Gold for relevant PubMed passages):")
    for i, result in enumerate(prm_results):
        label = sample2['labels'][i] if i < len(sample2['labels']) else "N/A"
        print(f"  Passage {i+1} [{label}]: {result}")
        print(f"    Text preview: {test_passages[i][:80]}...")

    # ---------- 2.4 SKA（自知识分析） + QDA（问题分解）降级路径测试 ----------
    print("\n" + "=" * 60)
    print("2.4 SKA 自知识分析 + QDA 问题分解（降级路径）")
    print("=" * 60)

    # SKA 独立测试
    skm_result = lins.SKM(
        question="What is the normal body temperature for a healthy adult human?"
    )
    print("SKM result (expected: CERTAIN or UNCERTAIN):", skm_result)

    # QDA 独立测试
    qdm_result = lins.QDM(
        question="What are the treatment options for Parkinson's disease?"
    )
    print("QDM sub-questions:")
    for i, q in enumerate(qdm_result):
        print(f"  Sub-q{i+1}: {q}")

    # SKA_QDA 降级路径（当 MAIRAG 无检索结果时的回退）
    print("\n>> 测试 SKA_QDA 降级路径（无数据库回退）")
    # 创建一个无数据库的临时 LINS 实例来模拟检索失败场景
    lins_no_db = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='none',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        database_name='none'
    )
    response, urls, passages, history, sub_qs = lins_no_db.SKA_QDA(
        question="What is the capital of France?",
        if_SKA=True,
        if_QDA=True
    )
    print("SKA_QDA response (expected: model's own knowledge):\n", response)

    # ---------- 2.5 KED 关键词回退降级测试 ----------
    print("\n" + "=" * 60)
    print("2.5 KED 关键词搜索 + 关键词回退降级测试")
    print("=" * 60)
    print(">> 测试 KED_search（正常检索）")
    results = lins.KED_search(
        question="Parkinson's disease treatment dopamine",
        topk=5
    )
    if results and results.get('texts'):
        print(f"KED found {len(results['texts'])} passages")
        print("URLs:", results['urls'][:3])
    else:
        print("KED returned no results (expected if PubMed is unavailable)")

    # 模拟关键词提取
    print("\n>> 测试 keyword_extraction（KED回退机制的核心）")
    keywords = lins.keyword_extraction(
        question="What is the role of alpha-synuclein in Parkinson's disease?",
        max_num_keywords=3
    )
    print("Extracted keywords:", keywords)

    # ---------- 2.6 HERD 分层搜索测试 ----------
    print("\n" + "=" * 60)
    print("2.6 HERD 分层搜索（指南→PubMed→Bing 三层递进）")
    print("=" * 60)
    print(">> 测试 HERD_search（跳过指南，使用 PubMed+Bing）")
    herd_passages, herd_urls = lins.HERD_search(
        PICO_question="For patients with Parkinson's disease, does levodopa improve motor function?",
        topk=3,
        if_guidelines=False,  # 跳过指南（需要本地数据）
        if_pubmed=True,
        if_bing=True
    )
    if herd_passages:
        print(f"HERD found {len(herd_passages)} passages")
        print("URLs:", herd_urls)
    else:
        print("HERD returned no results (expected if network limited)")

    # # PICO 生成测试（HERD和AEBMP的前置功能）
    print("\n>> 测试 PICO 生成")
    pico_result = lins.PICO(
        patient_information="A 65-year-old male with Parkinson's disease, experiencing motor fluctuations.",
        clinical_question="Should levodopa be used to improve motor function?"
    )
    print("PICO result:\n", pico_result)

    # ---------- 2.7 AEBMP 循证医学（PICO）测试 ----------
    print("\n" + "=" * 60)
    print("2.7 AEBMP 循证医学（基于PICO的完整循证问答）")
    print("=" * 60)
    aebmp_response, aebmp_urls, aebmp_passages, aebmp_history, pico_q = lins.AEBMP(
        patient_information="A 65-year-old male with early-stage Parkinson's disease. "
                           "He has mild tremors and bradykinesia. No significant comorbidities.",
        clinical_question="Should prasinezumab be used to slow motor progression?",
        topk=3,
        if_SKM=True,
        if_QDA=True,
        if_guidelines=False  # 跳过指南（需要本地数据）
    )
    print("AEBMP response:\n", aebmp_response)
    print("PICO question:", pico_q)
    if aebmp_urls:
        print("URLs:", aebmp_urls)

    # # ---------- 2.8 Medical_Entity_Extraction 医学实体提取测试 ----------
    print("\n" + "=" * 60)
    print("2.8 Medical_Entity_Extraction 医学实体提取")
    print("=" * 60)
    patient_text = (
        "The patient is a 65-year-old male with a 3-year history of Parkinson's disease. "
        "He currently takes levodopa/carbidopa 25/100mg three times daily. "
        "Recent MRI shows mild cortical atrophy. He has hypertension treated with lisinopril. "
        "He reports worsening tremors and bradykinesia over the past 3 months."
    )
    entities = lins.Medical_Entity_Extraction(text=patient_text, max_extraction_number=4)
    print("Extracted entities:", entities)
    if 'entity_list' in entities:
        print("Entity list:", entities['entity_list'])

    # ---------- 2.9 Medical_Text_Explanation 医学实体解释测试 ----------
    print("\n" + "=" * 60)
    print("2.9 Medical_Text_Explanation 医学实体解释")
    print("=" * 60)
    entity_list, entity_urls, entity_retrieved_passages, entity_explanations = lins.Medical_Text_Explanation(
        text=patient_text,
        max_extraction_number=3,
        entity_list=entities.get('entity_list', []),
        topk=2
    )
    print("(可看到每个实体的解释已在上方打印)")

    # ---------- 2.10 MAIRAG_options 带选项的单选问答测试 ----------
    print("\n" + "=" * 60)
    print("2.10 MAIRAG_options 带选项的单选问答测试")
    print("=" * 60)

    # ---- 2.10a 正常检索+选项回答 ----
    print(">> 2.10a 正常检索+选项回答")
    options_response, options_urls, options_passages, options_history, options_sub_qs = lins.MAIRAG_options(
        question=(
            "Which of the following medications is the first-line treatment for early Parkinson's disease?\n"
            "A. Haloperidol\nB. Levodopa/carbidopa\nC. Metoprolol\nD. Donepezil"
        ),
        topk=5,
        if_pubmed=True,
        single_choice=True
    )
    print("MAIRAG_options response:", options_response)
    if options_urls:
        print("URLs:", options_urls[:2])
    if options_sub_qs:
        print("Sub-questions:", options_sub_qs)

    # ---- 2.10b 使用 history 跳过检索 ----
    print("\n>> 2.10b 使用 history 跳过检索（模拟多轮对话中的选择题）")
    # 先有对话历史
    _, hist = lins.chat(question="Hello, I am interested in Parkinson's disease treatments.", history=None)
    options_response2, urls2, passages2, history2, sub_qs2 = lins.MAIRAG_options(
        question="Which of the following is the most common first-line treatment?\nA. Levodopa\nB. Aspirin\nC. Insulin\nD. Warfarin",
        history=hist,
        single_choice=True
    )
    print("With-history response:", options_response2)

    # ---- 2.10c SKA+QDA 降级路径 ----
    print("\n>> 2.10c 测试迭代降级（SKM+QDM回退）")
    # 使用 retriever_name='none' + database_name='none' 强制走 SKM→QDM 降级路径
    # （不实际检索数据库，仅测试模型自身知识+CERTAIN/UNCERTAIN判定逻辑）
    lins_local = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='none',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        database_name='none'
    )
    # 先测试 SKM CERTAIN 路径（简单常识问题应有CERTAIN）
    options_response3, urls3, passages3, history3, sub_qs3 = lins_local.MAIRAG_options(
        question="What is the boiling point of water at sea level?\nA. 90°C\nB. 100°C\nC. 110°C\nD. 120°C",
        topk=3,
        single_choice=True
    )
    print("SKA fallback response:", options_response3)

    # ---------- 2.12 Medical_Order_QA 医嘱问答测试 ----------

    print("\n" + "=" * 60)
    print("2.12 Medical_Order_QA 医嘱问答")
    print("=" * 60)
    order_response, order_passages, order_urls, order_history = lins.Medical_Order_QA(
        patient_information=patient_text,
        question="Should the dose of levodopa be increased given the worsening symptoms?",
        topk=3,
        if_PRA=True,
        if_SKA=True,
        if_QDA=True
    )
    print("Medical Order QA response:\n", order_response)
    if order_urls:
        print("URLs:", order_urls)

    # ---------- 2.13 MOEQA 综合问答（实体提取+解释+QA一体化） ----------
    print("\n" + "=" * 60)
    print("2.13 MOEQA 综合问答（实体提取+解释+QA一体化）")
    print("=" * 60)
    moeqa_result = lins.MOEQA(
        patient_information=patient_text,
        explain_text=patient_text,
        question="What lifestyle modifications can help manage Parkinson's symptoms?",
        max_extraction_number=3,
        expla_topk=2,
        QA_topk=3,
        if_explanation=True,
        if_QA=True,
        QA_if_PRA=True,
        QA_if_SKA=True,
        QA_if_QDA=True
    )
    print("MOEQA entity explanations:")
    if isinstance(moeqa_result, tuple):
        explanation_part, qa_part = moeqa_result
        print("  Entities:", explanation_part.get('entity_list', []))
        print("  QA Response:", qa_part.get('QA_response', 'N/A'))
    else:
        print("  Result keys:", list(moeqa_result.keys()))

    # ---------- 2.14 错误处理与边界情况测试 ----------
    print("\n" + "=" * 60)
    print("2.14 错误处理与边界情况测试")
    print("=" * 60)

    print(">> 2.14a 空文本输入")
    try:
        empty_result = lins.Medical_Entity_Extraction(text="", max_extraction_number=3)
        print("  Empty text result:", empty_result)
    except Exception as e:
        print(f"  Empty text raised: {type(e).__name__}: {e}")

    print("\n>> 2.14b 无效数据库名称")
    try:
        lins_bad_db = LINS(
            LLM_name='deepseek-chat',
            assistant_LLM_name='deepseek-chat',
            retriever_name='BGE',
            DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
            BGE_encoder_path='./model/retriever/bge/bge-m3',
            database_name='invalid_database_xyz'
        )
    except Exception as e:
        print(f"  Bad db name raised: {type(e).__name__}: {e}")

    print("\n>> 2.14c PRM 空引用列表（边界情况）")
    try:
        prm_empty = lins.PRM(question="What is diabetes?", refs=[])
        print("  PRM empty refs result:", prm_empty)
    except ValueError as e:
        print(f"  PRM empty refs correctly raised ValueError: {e}")

    print("\n>> 2.14d GRM 空引用列表（边界情况）")
    try:
        grm_empty = lins.GRM(question="What is diabetes?", refs=[])
        print("  GRM empty refs result:", grm_empty)
    except ValueError as e:
        print(f"  GRM empty refs correctly raised ValueError: {e}")

    print("\n>> 2.14e get_passages 无数据库时返回 None（优雅退化）")
    # 使用 database=None 模拟数据库不可用
    passages_none = lins.get_passages(question="test", database=None)
    print(f"  get_passages(database=None) = {passages_none}")

    print("\n>> 2.14f KED_search 无效数据库返回 None")
    ked_none = lins.KED_search(question="test", database=None)
    print(f"  KED_search(database=None) = {ked_none}")

    print("\n>> 2.14g MAIRAG_options 迭代次数耗尽（itera_num > 3）")
    # 无法直接传 itera_num，我们测试检索失败→SKM非CERTAIN→QDM→最终"None"路径
    # 这里用无数据库实例测试
    options_iters, _, _, _, _ = lins_local.MAIRAG_options(
        question="What is the most rare disease?\nA. Common cold\nB. Diabetes\nC. Hypertension\nD. AIDS",
        topk=3,
        single_choice=True
    )
    print(f"  MAIRAG_options fallback exhausted = {options_iters}")

    print("\n>> 2.14h 长文本超过 max_length 截断")
    long_text = "Parkinson's disease is a neurodegenerative disorder. " * 1000  # 约 50KB
    try:
        entity_long = lins.Medical_Entity_Extraction(text=long_text, max_extraction_number=2)
        print("  Long text entity extraction succeeded (model may truncate internally)")
    except Exception as e:
        print(f"  Long text raised: {type(e).__name__}: {e}")

    print("\n>> 2.14i deepseek-reasoner 作为 assistant（非 chat 模型）的退化")
    # 原项目 ModelNotFoundException → 替换为更合适的异常
    # reasoner 不支持作为 assistant（它只返回推理标记），测试退化逻辑
    lins_bad_assistant = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-reasoner',  # 不支持的多轮对话
        retriever_name='none',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        database_name='none'
    )
    try:
        prm_bad = lins_bad_assistant.PRM(question="What is cancer?", refs=["Cancer is a disease."])
        print(f"  Reasoner as assistant PRM result: {prm_bad}")
    except Exception as e:
        print(f"  Reasoner as assistant raised: {type(e).__name__}: {e}")

    # ---------- 2.15 LinkEval-DeepSeek 多维度量化评估 ----------
    print("\n" + "=" * 60)
    print("2.15 LinkEval-DeepSeek 模型量化评估（对齐 LINS 论文指标）")
    print("=" * 60)
    print("评估维度: Citation Set Precision | Citation Precision | Citation Recall")
    print("         | Statement Correctness | Statement Fluency | Overall Score")
    print("=" * 60)
    print("流程: 对每个 PubmedQA 样本 → MAIRAG 生成回答 → evaluate 评估5项指标 → 汇总")

    try:
        import sys, importlib, os
        # 确保 LINS-main 在路径中
        lins_main_path = os.path.dirname(os.path.abspath(__file__))
        if lins_main_path not in sys.path:
            sys.path.insert(0, lins_main_path)
        eval_mod = importlib.import_module('Link_Eval_DeepSeek')
        LinkEvalDeepSeek = eval_mod.LinkEvalDeepSeek
        convert_to_statements = eval_mod.convert_to_statements
        format_metrics = eval_mod.format_metrics

        # 初始化评估器
        link_eval = LinkEvalDeepSeek(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            model_name='deepseek-chat',
            verbose=False  # 不打印单个样本细节，汇总时统一展示
        )
        print("  LinkEval-DeepSeek 初始化成功 ✓（使用 DeepSeek 替代 T5-11B NLI + UniEval）\n")

        # ---- 2.15a 批量评估 PubmedQA 样本（模型评价核心）----
        total_samples = len(pubmedqa_samples)
        print("-" * 60)
        print(f">> 2.15a 批量评估 PubmedQA 样本的 MLA 回答质量（共 {total_samples} 个样本）")
        print("-" * 60)

        all_metrics = []
        valid_count = 0
        skip_count = 0

        for idx, sample_eval in enumerate(pubmedqa_samples):
            print(f"\n  [{idx+1}/{total_samples}] {sample_eval['id']} | {sample_eval['question'][:70]}...")
            try:
                # 用 MAIRAG 生成带引用的回答
                response_eval, urls_eval, passages_eval, history_eval, sub_qs_eval = lins.MAIRAG(
                    question=sample_eval['question'],
                    if_PRA=True,
                    if_SKA=True,
                    if_QDA=True,
                    if_PCA=True
                )

                statements_eval = convert_to_statements(response_eval)
                refs_eval = passages_eval if passages_eval else sample_eval['contexts'][:5]

                if statements_eval and refs_eval:
                    metrics_eval, details_eval = link_eval.evaluate(
                        question=sample_eval['question'],
                        statements=statements_eval,
                        refs=refs_eval,
                        correct_answer=sample_eval.get('answer', None),
                        return_details=True
                    )
                    all_metrics.append({
                        'id': sample_eval['id'],
                        'metrics': metrics_eval,
                        'details': details_eval,
                        'statements_count': len(statements_eval),
                        'refs_count': len(refs_eval)
                    })
                    valid_count += 1
                    print(f"      ✓ CSP={metrics_eval['citation_set_precision']:.3f} "
                          f"CP={metrics_eval['citation_precision']:.3f} "
                          f"CR={metrics_eval['citation_recall']:.3f} "
                          f"SC={metrics_eval['statement_correctness']} "
                          f"SF={metrics_eval['statement_fluency']:.3f}")
                else:
                    print(f"      ⚠ 跳过（无有效陈述或引用）")
                    skip_count += 1
            except Exception as e:
                print(f"      ✗ 异常: {type(e).__name__}: {str(e)[:80]}")
                skip_count += 1

        # ---- 2.15b 汇总统计（最终评价结果）----
        print("\n" + "=" * 70)
        print(f">> 2.15b {valid_count}个有效样本量化评估汇总")
        print("=" * 70)
        print(f"  有效评估: {valid_count} / {total_samples}  |  跳过: {skip_count}\n")

        if all_metrics:
            metric_names = [
                'citation_set_precision', 'citation_precision', 'citation_recall',
                'f1_score', 'statement_correctness', 'statement_fluency', 'overall_score'
            ]
            metric_labels_cn = [
                '引用集精准率(CSP)', '引用精准率(CP)', '引用召回率(CR)',
                'F1 分数', '陈述正确性(SC)', '陈述流畅度(SF)', '总体评分'
            ]

            # --- 汇总统计表 ---
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

            # --- 每个样本的详细结果表 ---
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

            # --- 总体评分分布 ---
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

    except ImportError as e:
        print(f"  ImportError: {e}")
        print("  提示: pip install openai httpx")
        print("  跳过 LinkEval 评估")
    except Exception as e:
        import traceback
        print(f"  LinkEval-DeepSeek 评估异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("  LinkEval 测试继续...")

    return lins



# ============================================================
# 方案三：DeepSeek Reasoner（推理模型）
# ============================================================
def test_reasoner():
    print("=" * 60)
    print("测试 3：DeepSeek Reasoner（推理模型）")
    print("=" * 60)

    lins = LINS(
        LLM_name='deepseek-reasoner',
        assistant_LLM_name='deepseek-reasoner',
        retriever_name='none',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        database_name='none'
    )

    response, history = lins.chat(
        question="A 65-year-old male with Parkinson's disease presents with worsening tremors. "
                 "What factors should be considered in treatment adjustment?",
        history=None
    )
    print("\nReasoner response:\n", response)


# ============================================================
# 方案四：DeepSeek Reasoner + BGE 检索器（推理+检索增强）
# ============================================================
def test_reasoner_with_retriever():
    print("=" * 60)
    print("测试 4：DeepSeek Reasoner + BGE 检索器（推理+检索增强）")
    print("=" * 60)

    lins = LINS(
        LLM_name='deepseek-reasoner',
        assistant_LLM_name='deepseek-reasoner',
        retriever_name='BGE',
        DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
        BGE_encoder_path='./model/retriever/bge/bge-m3',
        database_name='pubmed'
    )

    # 使用 MAIRAG 做检索增强推理
    response, urls, passages, history, sub_qs = lins.MAIRAG(
        question="For Parkinson's disease, what is the mechanism of action of prasinezumab?",
        if_PRA=True,
        if_SKA=True,
        if_QDA=True
    )
    print("\nReasoner MAIRAG response:\n", response)
    print("URLs:", urls)


if __name__ == "__main__":
    # 检查 API Key
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量！")
        print("请先设置：")
        print("  set DEEPSEEK_API_KEY=sk-your-deepseek-api-key")
        exit(1)

    print("DeepSeek API Key 已找到，开始测试...\n")

    # ========== 选择要运行的测试 ==========
    # 取消注释即可运行

    # 方案一：基础对话（无需检索器和数据库）
    # test_basic_chat()

    # 方案二：完整功能链路（需要 BGE 模型 + PubMed）
    # 包括：基础对话、MAIRAG、PRM、SKA、QDA、KED、HERD、AEBMP
    test_with_retriever()

    # 方案三：Reasoner 推理模型（无需检索器和数据库）
    # test_reasoner()

    # 方案四：Reasoner + 检索增强（需要 BGE 模型）
    # test_reasoner_with_retriever()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
