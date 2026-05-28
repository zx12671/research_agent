"""
LINS 项目 - DeepSeek 完整框架评估脚本

覆盖测试集：
  - PubmedQA (20 个样本)
  - MedQA-US (15 个样本)
  - MedQA-Mainland (15 个样本)
"""
import os
import sys
import json
import time

# ========== 配置区域 ==========
os.environ['DEEPSEEK_API_KEY'] = 'sk-94ccad564a7542228ad52f6b2654e11e'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

LINS_MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LINS-main'))
if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)
# =============================

from model.model_LINS import LINS

# ========== 评估配置 ==========
DEFAULT_PUBMEDQA_SAMPLES = 20
DEFAULT_MEDQA_SAMPLES = 15
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
# MedQA 数据集加载
# ============================================================
MEDQA_US_PATH = "./LINS-main/evaluate/evaluate_data/medqa_us/data/medqa_us_test.json"
MEDQA_MAINLAND_PATH = "./LINS-main/evaluate/evaluate_data/medqa_mainland/data/medqa_mainland_test.json"


def load_medqa_samples(path, num_samples=None, region="us"):
    """加载 MedQA 测试样本"""
    if num_samples is None:
        num_samples = DEFAULT_MEDQA_SAMPLES

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
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
        print(f"  [{s['id']}] {s['question'][:50]}... -> answer: {s['answer']}")
    return samples


# ============================================================
# PubmedQA 评估（含 KED + 后处理）
# ============================================================
def test_pubmedqa_evaluation(lins, num_samples=None):
    """在 PubmedQA 数据集上运行完整评估，重点测试 KED 检索准确率"""
    if num_samples is None:
        num_samples = DEFAULT_PUBMEDQA_SAMPLES

    print("\n" + "=" * 70)
    print("【第四部分】PubmedQA 完整评估 + KED 准确率分析")
    print("  重点：评估 KED 算法在 PubmedQA 上的检索准确率")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples(num_samples)
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

        # ---- 4a. MAIRAG 完整回答 ----
        try:
            response, urls, passages, history, sub_qs = lins.MAIRAG(
                question=sample['question'] + "\nPlease include citation numbers like [1], [2] etc. in your answer.",
                topk=5,
                if_PRA=True,
                if_SKA=True,
                if_QDA=True,
                if_PCA=True,
                recall_top_k=100
            )
            sample_result["mAirag_response"] = response
            sample_result["mAirag_urls"] = urls
            sample_result["mAirag_passages"] = passages
            sample_result["retrieved_count"] = len(passages) if passages else 0
            print(f"     MAIRAG: OK 回答生成 ({len(passages) if passages else 0} 条引用)")
        except Exception as e:
            sample_result["mAirag_error"] = str(e)[:100]
            sample_result["mAirag_response"] = None
            sample_result["mAirag_urls"] = []
            sample_result["mAirag_passages"] = []
            print(f"     MAIRAG: FAIL {type(e).__name__}")

        # ---- 4b. KED 检索统计 ----
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
                print(f"     KED检索: OK {ked_count} 条结果")
            else:
                sample_result["ked_retrieved_count"] = 0
                print(f"     KED检索: WARN 无结果")
        except Exception as e:
            sample_result["ked_error"] = str(e)[:100]
            print(f"     KED检索: FAIL {type(e).__name__}")

        # ---- 4c. 选择题评估 ----
        try:
            options_prompt = f"""Question: {sample['question']}
Options:
A. Yes
B. No
C. Maybe

Please answer with only the option letter (A, B, or C)."""
            mAirag_passages = sample_result.get("mAirag_passages", [])

            if mAirag_passages:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=options_prompt, topk=5, single_choice=True,
                    retrieval_passages=mAirag_passages
                )
                print(f"     选择题(复用MAIRAG结果): {options_response}")
            else:
                options_response, _, _, _, _ = lins.MAIRAG_options(
                    question=options_prompt, topk=5, single_choice=True
                )
                print(f"     选择题(独立检索): {options_response}")

            # 后处理
            if options_response and options_response.strip().upper() in ["A", "B"]:
                mAirag_response = sample_result.get("mAirag_response", "")
                ked_urls = sample_result.get("ked_urls", [])
                ked_count = sample_result.get("ked_retrieved_count", 0)

                if should_force_maybe(mAirag_response, mAirag_passages, urls, ked_urls, ked_count):
                    options_response = "C"
                    sample_result["postprocess_forced_maybe"] = True
                    print(f"     -> 后处理规则触发: C (Maybe)")
                else:
                    sample_result["postprocess_forced_maybe"] = False

            sample_result["options_response"] = options_response

            answer_map = {"yes": "A", "no": "B", "maybe": "C"}
            gt_letter = answer_map.get(sample['answer'].lower(), None)
            prediction = options_response.strip().upper() if options_response else ""

            if gt_letter and prediction:
                is_correct = (prediction == gt_letter)
                sample_result["is_correct"] = is_correct
                print(f"     结果: {prediction} (GT: {sample['answer']} -> {gt_letter}) {'CORRECT' if is_correct else 'WRONG'}")
            else:
                print(f"     结果: {prediction} (无法与 GT 比较)")
        except Exception as e:
            sample_result["options_error"] = str(e)[:100]
            print(f"     选择题回答: FAIL {type(e).__name__}")

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
# LinkEval 量化评估（使用自定义 LinkEval-DeepSeek 模块）
# ============================================================
def test_linkeval(lins, eval_results=None, num_samples=None):
    """使用 LinkEval-DeepSeek 对 PubmedQA 回答进行量化评估"""
    if num_samples is None:
        num_samples = DEFAULT_PUBMEDQA_SAMPLES

    print("\n" + "=" * 70)
    print("【第五部分】LinkEval-DeepSeek 量化评估")
    print("  指标: 引用精准率(CP) | 引用召回率(CR) | F1 | 陈述正确性(SC) | 流畅度(SF)")
    print("=" * 70)

    pubmedqa_samples = load_pubmedqa_samples(num_samples)

    try:
        from Link_Eval_DeepSeek import LinkEvalDeepSeek, convert_to_statements, format_metrics

        link_eval = LinkEvalDeepSeek(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            model_name='deepseek-chat',
            verbose=False
        )
        print("  LinkEval-DeepSeek 初始化成功 OK\n")

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

                # 如果第四部分没有，重新调用
                if not response:
                    print(f"     第四部分无 MAIRAG 结果，重新调用...")
                    response, _, passages, _, _ = lins.MAIRAG(
                        question=sample['question'] + "\nPlease include citation numbers like [1], [2] etc. in your answer.",
                        if_PRA=True, if_SKA=True, if_QDA=True, if_PCA=False
                    )

                if not response:
                    print(f"     WARN 无回答生成，跳过")
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
                    print(f"     OK CP={metrics['citation_precision']:.3f} "
                          f"CR={metrics['citation_recall']:.3f} "
                          f"F1={metrics['f1_score']:.3f} "
                          f"SC={metrics['statement_correctness']} "
                          f"SF={metrics['statement_fluency']:.3f}")
                else:
                    print(f"     WARN 无有效陈述或引用，跳过")
            except Exception as e:
                print(f"     FAIL {type(e).__name__}: {str(e)[:80]}")

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
            print("  WARN 没有有效评估结果")

    except ImportError as e:
        print(f"  FAIL 导入 LinkEval-DeepSeek 失败: {e}")
        print("  请确保 LINS-main/Link_Eval_DeepSeek.py 存在")
    except Exception as e:
        print(f"  FAIL LinkEval 评估异常: {type(e).__name__}: {str(e)[:80]}")


# ============================================================
# 通用选择题评估（MedQA-US / MedQA-Mainland）
# ============================================================
def _evaluate_multiple_choice(lins, samples, result_dir, region_label):
    """通用选择题评估流程（MedQA-US 和 MedQA-Mainland 共用）"""
    if not samples:
        print(f"  FAIL 未加载到 {region_label} 样本")
        return []

    eval_results = []

    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] {sample['id']}: {sample['question'][:60]}...")
        print(f"  Ground Truth: {sample['answer']}")

        sample_result = {
            "id": sample['id'],
            "question": sample['question'],
            "ground_truth": sample['answer'],
            "region": sample.get('region', region_label)
        }

        # 构造选项文本
        options_list = [f"{letter}: {text}" for letter, text in sample['options'].items()]
        options_str = "\n".join(options_list)

        # ---- a. MAIRAG 回答 ----
        try:
            response, urls, passages, history, sub_qs = lins.MAIRAG(
                question=f"{sample['question']}\nPlease include citation numbers like [1], [2] in your answer.",
                topk=5, if_PRA=True, if_SKA=True, if_QDA=True, if_PCA=True
            )
            sample_result["mAirag_response"] = response
            sample_result["mAirag_urls"] = urls
            sample_result["mAirag_passages"] = passages
            print(f"     MAIRAG: OK 回答生成 ({len(passages) if passages else 0} 条引用)")
        except Exception as e:
            sample_result["mAirag_response"] = None
            sample_result["mAirag_urls"] = []
            sample_result["mAirag_passages"] = []
            print(f"     MAIRAG: FAIL {type(e).__name__}")

        # ---- b. KED 检索统计 ----
        try:
            ked_data = lins.KED_search(question=sample['question'], topk=20, if_split_n=False)
            if ked_data and ked_data.get('texts'):
                sample_result["ked_retrieved_count"] = len(ked_data['texts'])
                sample_result["ked_urls"] = ked_data['urls'][:5]
                print(f"     KED检索: OK {len(ked_data['texts'])} 条结果")
            else:
                sample_result["ked_retrieved_count"] = 0
                print(f"     KED检索: WARN 无结果")
        except Exception as e:
            sample_result["ked_error"] = str(e)[:100]
            print(f"     KED检索: FAIL {type(e).__name__}")

        # ---- c. 选择题回答 ----
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

            # 后处理
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
                    print(f"     -> 后处理规则触发: C (Maybe)")
                else:
                    sample_result["postprocess_forced_maybe"] = False

            sample_result["options_response"] = options_response

            prediction = options_response.strip().upper() if options_response else ""
            gt = sample['answer'].strip().upper()
            is_correct = (prediction == gt) if prediction and gt else False
            sample_result["is_correct"] = is_correct
            print(f"     选择题: {prediction} (GT: {gt}) {'CORRECT' if is_correct else 'WRONG'}")

        except Exception as e:
            sample_result["options_error"] = str(e)[:100]
            print(f"     选择题回答: FAIL {type(e).__name__}")

        eval_results.append(sample_result)
        print()

    # ---- 汇总 ----
    correct = sum(1 for r in eval_results if r.get("is_correct"))
    total = len(eval_results)
    print(f"\n  {region_label} 汇总: {correct}/{total} = {correct/max(total,1)*100:.1f}%")

    return eval_results


# ============================================================
# MedQA-US 选择题评估
# ============================================================
def test_medqa_us_evaluation(lins, num_samples=None):
    """在 MedQA-US 数据集上运行选择题评估"""
    if num_samples is None:
        num_samples = DEFAULT_MEDQA_SAMPLES

    print("\n" + "=" * 70)
    print("【MedQA-US】选择题评估")
    print("=" * 70)
    samples = load_medqa_samples(MEDQA_US_PATH, num_samples=num_samples, region="us")

    eval_results = _evaluate_multiple_choice(lins, samples, "./eval_results_lins_full", "medqa_us")

    correct_path = "./eval_results_lins_full/medqa_us_details.jsonl"
    os.makedirs("./eval_results_lins_full", exist_ok=True)
    with open(correct_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  结果已保存至: {correct_path}")
    return eval_results


# ============================================================
# MedQA-Mainland 选择题评估
# ============================================================
def test_medqa_mainland_evaluation(lins, num_samples=None):
    """在 MedQA-Mainland 数据集上运行选择题评估"""
    if num_samples is None:
        num_samples = DEFAULT_MEDQA_SAMPLES

    print("\n" + "=" * 70)
    print("【MedQA-Mainland】选择题评估")
    print("=" * 70)
    samples = load_medqa_samples(MEDQA_MAINLAND_PATH, num_samples=num_samples, region="mainland")

    eval_results = _evaluate_multiple_choice(lins, samples, "./eval_results_lins_full", "medqa_mainland")

    correct_path = "./eval_results_lins_full/medqa_mainland_details.jsonl"
    os.makedirs("./eval_results_lins_full", exist_ok=True)
    with open(correct_path, "w", encoding="utf-8") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  结果已保存至: {correct_path}")
    return eval_results


# ============================================================
# MedQA 汇总报告
# ============================================================
def summarize_medqa_results():
    """汇总 MedQA-US 和 MedQA-Mainland 的评估结果"""
    print("\n" + "=" * 70)
    print("【MedQA 综合评估汇总】")
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
    print("LINS 模型初始化完成 OK")

    # 运行各测试模块
    test_database_config(lins)
    test_ked_retriever(lins)

    # PubmedQA 评估（20样本）
    eval_results = test_pubmedqa_evaluation(lins, DEFAULT_PUBMEDQA_SAMPLES)
    test_linkeval(lins, eval_results, DEFAULT_PUBMEDQA_SAMPLES)

    # MedQA 评估（各15样本）
    test_medqa_us_evaluation(lins, DEFAULT_MEDQA_SAMPLES)
    test_medqa_mainland_evaluation(lins, DEFAULT_MEDQA_SAMPLES)
    summarize_medqa_results()

    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量！")
        print("请先设置：")
        print("  set DEEPSEEK_API_KEY=sk-your-deepseek-api-key")
        exit(1)

    main()
