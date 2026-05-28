"""
LINS 项目 - DeepSeek API 快速验证脚本

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
os.environ['DEEPSEEK_API_KEY'] = 'sk-94ccad564a7542228ad52f6b2654e11e'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 确保 LINS-main 在 sys.path 中
LINS_MAIN_PATH = os.path.dirname(os.path.abspath(__file__))
if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)
# =============================

from model.model_LINS import LINS

# ========== 评估配置 ==========
PUBMEDQA_TEST_SAMPLES = 3
# =============================

from lins_test_common import (
    load_pubmedqa_samples,
    test_database_config,
    test_ked_retriever,
    print_metric_summary,
    should_force_maybe,
    load_linkeval_data,
    evaluate_single_linkeval_sample,
)


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

    response, history = lins.chat(question="What is BCR-ABL1?", history=None)
    print("\nChat response:\n", response)

    response, history = lins.chat(question="What diseases is it associated with?", history=history)
    print("\nFollow-up response:\n", response)

    return lins


# ============================================================
# 方案二：DeepSeek + 本地 BGE 检索器（完整功能链测试）
# ============================================================
def test_with_retriever():
    print("=" * 60)
    print("测试 2：DeepSeek + 本地 BGE 检索器（完整功能链路）")
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

    pubmedqa_samples = load_pubmedqa_samples(PUBMEDQA_TEST_SAMPLES)

    # ---------- 2.2 原 MAIRAG 完整功能测试 ----------
    print("\n" + "=" * 60)
    print("2.2 MAIRAG 完整功能（含 PRM 段落相关性评估）")
    print("=" * 60)
    sample = pubmedqa_samples[0]
    print(f"使用 PubmedQA 样本 [{sample['id']}]: {sample['question'][:80]}...")
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

    # ---------- 2.3 PRM 段落相关性评估独立测试 ----------
    print("\n" + "=" * 60)
    print("2.3 PRM 段落相关性评估独立测试（使用 PubmedQA 真实段落）")
    print("=" * 60)
    sample2 = pubmedqa_samples[1]
    test_passages = sample2['contexts']
    print(f"使用 PubmedQA 样本 [{sample2['id']}]: {sample2['question'][:60]}...")
    print(f"上下文字段落数: {len(test_passages)}")

    prm_results = lins.PRM(
        question=sample2['question'],
        refs=test_passages
    )
    print("PRM results:")
    for i, result in enumerate(prm_results):
        label = sample2['labels'][i] if i < len(sample2['labels']) else "N/A"
        print(f"  Passage {i+1} [{label}]: {result}")
        print(f"    Text preview: {test_passages[i][:80]}...")

    # ---------- 2.4 SKA + QDA 降级路径测试 ----------
    print("\n" + "=" * 60)
    print("2.4 SKA 自知识分析 + QDA 问题分解（降级路径）")
    print("=" * 60)

    skm_result = lins.SKM(
        question="What is the normal body temperature for a healthy adult human?"
    )
    print("SKM result:", skm_result)

    qdm_result = lins.QDM(
        question="What are the treatment options for Parkinson's disease?"
    )
    print("QDM sub-questions:")
    for i, q in enumerate(qdm_result):
        print(f"  Sub-q{i+1}: {q}")

    # ---------- 2.5-2.7 多智能体流程 ----------
    print("\n" + "=" * 60)
    print("2.5-2.7 多智能体（PRA + SKA + QDA + KED）")
    print("=" * 60)
    sample3 = pubmedqa_samples[2]
    print(f"使用 PubmedQA 样本 [{sample3['id']}]: {sample3['question'][:80]}...")

    # PRA 段落相关性评估
    print("\n2.5 PRA 段落相关性评估")
    pra_results = lins.PRM(
        question=sample3['question'],
        refs=sample3['contexts']
    )
    gold_passages = [
        sample3['contexts'][i] for i, r in enumerate(pra_results)
        if r in ['Gold', 'Gold with error']
    ]
    print(f"  相关段落（Gold）: {len(gold_passages)} / {len(pra_results)}")

    # SKA 自知识分析
    print("\n2.6 SKA 自知识分析")
    ska_result = lins.SKM(question=sample3['question'])
    uncertainty = ska_result == 'UNCERTAIN'
    print(f"  SKM: {ska_result} → {'需要检索' if uncertainty else '无需检索'}")

    # KED 关键词退化检索
    print("\n2.7 KED 关键词退化检索")
    ked_results = lins.KED_search(
        question=sample3['question'],
        topk=8,
        if_split_n=False
    )
    if ked_results and ked_results.get('texts'):
        print(f"  KED 检索结果: {len(ked_results['texts'])} 条")
    else:
        print("  KED 未返回结果")

    # ---------- 2.8 QDA 问题分解降级 ----------
    print("\n" + "=" * 60)
    print("2.8 QDA 问题分解降级")
    print("=" * 60)
    qda_result = lins.QDM(question=sample3['question'])
    print(f"  问题分解为 {len(qda_result)} 个子问题:")
    for i, q in enumerate(qda_result):
        print(f"    Sub-q{i+1}: {q}")

    # ---------- 2.9 MAIRAG（含 QDA 降级） ----------
    print("\n" + "=" * 60)
    print("2.9 MAIRAG 完整流程（含 QDA 降级）")
    print("=" * 60)
    response, urls, passages, history, sub_qs = lins.MAIRAG(
        question=sample3['question'],
        if_PRA=True,
        if_SKA=True,
        if_QDA=True,
        if_PCA=True
    )
    print("\nMAIRAG response:\n", response)
    print("URLs:", urls)

    # ---------- 2.10 HERD 多级检索 ----------
    print("\n" + "=" * 60)
    print("2.10 HERD 多级检索")
    print("=" * 60)
    herd_passages, herd_urls = lins.HERD_search(
        PICO_question=sample3['question'],
        topk=3,
        if_guidelines=False,
        if_pubmed=True,
        if_bing=False
    )
    print(f"HERD 检索结果: {len(herd_passages)} passages")

    # ---------- 2.11 AEBMP （证据质量评估）----------
    print("\n" + "=" * 60)
    print("2.11 AEBMP 证据质量评估")
    print("=" * 60)
    try:
        aebmp_results = lins.AEBMP_infer(
            question=sample3['question'],
            passages=passages[:3] if passages else []
        )
        print(f"AEBMP 评估完成: {aebmp_results}")
    except Exception as e:
        print(f"AEBMP 评估异常: {type(e).__name__}: {str(e)[:80]}")

    # ---------- 2.12 AEBMP 推荐 ----------
    print("\n" + "=" * 60)
    print("2.12 AEBMP 推荐")
    print("=" * 60)
    try:
        aebmp_response = lins.AEBMP(
            question=sample3['question'],
            passages=passages[:3] if passages else []
        )
        print(f"AEBMP recommendation:\n{aebmp_response[:200]}")
    except Exception as e:
        print(f"AEBMP 异常: {type(e).__name__}: {str(e)[:80]}")

    # ---------- 2.13 PubmedQA 前向评估 ----------
    print("\n" + "=" * 60)
    print("2.13 PubmedQA 前向评估（无 LinkEval）")
    print("=" * 60)
    eval_results = []
    for s in pubmedqa_samples:
        print(f"\n  [{s['id']}] {s['question'][:60]}...")
        try:
            response_before, urls_before, passages_before, history_before, sub_qs_before = lins.MAIRAG(
                question=s['question'],
                if_PRA=True,
                if_SKA=True,
                if_QDA=True,
                if_PCA=True
            )

            ked_result = lins.KED_search(question=s['question'], topk=10)
            ked_urls = ked_result.get('urls', []) if ked_result else []
            ked_count = len(ked_urls)

            answer = 'yes'
            if response_before:
                response_lower = response_before.lower()
                if any(word in response_lower for word in ['yes', 'true', 'indeed', 'is associated']):
                    answer = 'yes'
                elif any(word in response_lower for word in ['no', 'not', 'cannot']):
                    answer = 'no'
                elif any(word in response_lower for word in ['maybe', 'unclear', 'uncertain']):
                    answer = 'maybe'

            has_statement = False
            statements = []
            references = []
            try:
                from model.retriever_model import LINS_Retriever
                if lins.retriever is not None or hasattr(lins, 'retriever'):
                    retriever_obj = lins.retriever if hasattr(lins, 'retriever') else None
                    if retriever_obj is not None:
                        rank_results = retriever_obj.rerank_passages(
                            question=s['question'],
                            passages=passages_before if passages_before else []
                        )
                        if rank_results:
                            has_statement = True
            except Exception:
                pass

            if 'maybe' in response_before.lower():
                if should_force_maybe(response_before, passages_before, urls_before, ked_urls, ked_count):
                    answer = 'maybe'

            is_correct = (answer == s['answer']) if answer else False

            eval_results.append({
                'id': s['id'],
                'question': s['question'],
                'answer': answer,
                'gt': s['answer'],
                'is_correct': is_correct,
                'mAirag_response': response_before,
                'urls': urls_before,
                'statements': statements,
                'references': references,
                'mAirag_urls': urls_before,
                'ked_urls': ked_urls,
                'ked_count': ked_count,
            })

            status = '✓' if is_correct else '✗'
            print(f"    回答: {answer} (GT: {s['answer']}) {status}")
        except Exception as e:
            print(f"    ✗ 异常: {type(e).__name__}: {str(e)[:80]}")

    correct = sum(1 for r in eval_results if r.get('is_correct'))
    total = len(eval_results)
    print(f"\n  PubmedQA 汇总: {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    # ---------- 2.14 LinkEval 引用评估（使用 LinkEval-DeepSeek） ----------
    print("\n" + "=" * 60)
    print("2.14 LinkEval 引用评估（LinkEval-DeepSeek）")
    print("=" * 60)
    try:
        from Link_Eval_DeepSeek import LinkEvalDeepSeek, convert_to_statements

        link_eval = LinkEvalDeepSeek(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            model_name='deepseek-chat',
            verbose=False
        )

        all_metrics = []
        valid_count = 0
        skip_count = 0

        for i, r in enumerate(eval_results):
            print(f"  [{i+1}/{len(eval_results)}] {r['id']}...")
            
            # 确保 sample_eval 包含 question 字段（LinkEvalDeepSeek 需要）
            sample_eval = {
                'id': r['id'],
                'question': r.get('question', ''),
                'statements': r.get('statements', []),
                'references': r.get('references', [])
            }
            
            # 如果 statements 为空但 response 不为空，尝试从 response 提取
            if not sample_eval['statements'] and r.get('mAirag_response'):
                statements = convert_to_statements(r['mAirag_response'])
                sample_eval['statements'] = statements
                # 同时尝试用 passages 作为 references
                if not sample_eval['references'] and r.get('urls'):
                    sample_eval['references'] = r.get('urls', [])
            
            result = evaluate_single_linkeval_sample(sample_eval, link_eval)
            if result:
                metrics_eval, details_eval = result
                all_metrics.append({
                    'id': sample_eval['id'],
                    'metrics': metrics_eval,
                    'details': details_eval,
                    'statements_count': len(sample_eval['statements']),
                    'refs_count': len(sample_eval['references'])
                })
                valid_count += 1
                print(f"      ✓ CP={metrics_eval['citation_precision']:.3f} "
                      f"CR={metrics_eval['citation_recall']:.3f} "
                      f"F1={metrics_eval['f1_score']:.3f} "
                      f"SC={metrics_eval['statement_correctness']} "
                      f"SF={metrics_eval['statement_fluency']:.3f}")
            else:
                print(f"      ⚠ 跳过（无有效陈述或引用）")
                skip_count += 1

        print(f"\n  LinkEval-DeepSeek: {valid_count} 有效 / {len(eval_results)} 总样本  |  跳过: {skip_count}")
        if all_metrics:
            print_metric_summary(all_metrics)

    except ImportError as e:
        print(f"  ImportError: {e}")
        print("  请确保 LINS-main/Link_Eval_DeepSeek.py 存在")
        print("  跳过 LinkEval 评估")
    except Exception as e:
        import traceback
        print(f"  LinkEval 异常: {type(e).__name__}: {e}")
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
# 方案四：DeepSeek Reasoner + BGE 检索器
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

    response, urls, passages, history, sub_qs = lins.MAIRAG(
        question="For Parkinson's disease, what is the mechanism of action of prasinezumab?",
        if_PRA=True,
        if_SKA=True,
        if_QDA=True
    )
    print("\nReasoner MAIRAG response:\n", response)
    print("URLs:", urls)


if __name__ == "__main__":
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量！")
        print("请先设置：")
        print("  set DEEPSEEK_API_KEY=sk-your-deepseek-api-key")
        exit(1)

    print("DeepSeek API Key 已找到，开始测试...\n")

    # 取消注释即可运行

    # 方案一：基础对话（无需检索器和数据库）
    # test_basic_chat()

    # 方案二：完整功能链路（需要 BGE 模型 + PubMed）
    test_with_retriever()

    # 方案三：Reasoner 推理模型（无需检索器和数据库）
    # test_reasoner()

    # 方案四：Reasoner + 检索增强（需要 BGE 模型）
    # test_reasoner_with_retriever()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
