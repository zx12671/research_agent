"""
准确率（Accuracy）评测脚本
Zero-shot 设置，不使用任何提示工程（如 Few-shot、CoT）

数据集：MedMCQA（使用 dev.json）、PubMedQA、MedQA-US、MedQA-Mainland
模型：DeepSeek (deepseek-chat)
"""

import json
import os
import re
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
import httpx

# ==================== 配置区域 ====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"

# 数据集路径
DATASET_DIR = r"C:\Users\duodu\Desktop\数据集"
MEDMCQA_DEV_PATH = os.path.join(DATASET_DIR, "MEDMCQA", "dev.json")
PUBMEDQA_ORI_PATH = r"c:\Users\duodu\Desktop\LINS\LINS-main\evaluate\evaluate_data\pubmedqa\data\ori_pqal.json"
PUBMEDQA_GT_PATH = r"c:\Users\duodu\Desktop\LINS\LINS-main\evaluate\evaluate_data\pubmedqa\data\test_ground_truth.json"
MEDQA_US_PATH = r"c:\Users\duodu\Desktop\LINS\LINS-main\evaluate\evaluate_data\medqa_us\data\medqa_us_test.json"
MEDQA_MAINLAND_PATH = r"c:\Users\duodu\Desktop\LINS\LINS-main\evaluate\evaluate_data\medqa_mainland\data\medqa_mainland_test.json"

# 结果保存路径
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
# =================================================


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self, api_key=None, model_name="deepseek-chat"):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
        if not api_key:
            raise ValueError("DeepSeek API key is required.")
        http_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", http_client=http_client)
        self.model_name = model_name

    def chat(self, messages, max_tokens=256, temperature=0.0):
        """调用 DeepSeek 对话 API"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  调用失败 (第{attempt+1}/3次): {e}")
                time.sleep(2 ** attempt)
        return ""


# ==================== 答案提取 ====================

def extract_answer(response_text, valid_choices=None):
    """
    从模型回复中提取答案字母。
    处理各种格式：
      - "B"
      - "(B)"
      - "The answer is B"
      - "Answer: (B)"
      - "The correct answer is **(C) Vitamin B12**"
      - "The capital of France is Paris. Answer: (B)"
      - "Let me break this down..."（CoT 风格，答案在文本后半部分）
    """
    if valid_choices is None:
        valid_choices = {"A", "B", "C", "D", "E", "F", "G", "H"}

    text = response_text.strip()
    if not text:
        return None

    # 预处理：移除 "Let's break this down step by step" 等 CoT 前缀
    text_clean = re.sub(
        r'(Let\s+(\w+\s+){0,3}(break\s+down|analyze|consider|examine|look\s+at|think\s+about).{0,200}?)'
        r'(?=The\s+(correct\s+)?answer|Answer\s*:|[A-Z][a-z]+\s+is\s+[A-H]\b)',
        '', text, flags=re.IGNORECASE | re.DOTALL
    )

    # 1. 直接匹配单个字母
    if text in valid_choices:
        return text

    # 2. 匹配 "Final Answer: X" (最高优先级)
    for t in [text, text_clean]:
        match = re.search(r'[Ff]inal\s*[Aa]nswer\s*:\s*\(?([A-H])\)?', t)
        if match and match.group(1) in valid_choices:
            return match.group(1)

    # 3. 匹配 "Answer: X" 或 "answer: X"
    for t in [text, text_clean]:
        match = re.search(r'[Aa]nswer\s*:\s*\(?([A-H])\)?', t)
        if match and match.group(1) in valid_choices:
            return match.group(1)

    # 4. 匹配 "is **(X)**" 或 "is (X)" 模式
    for t in [text, text_clean]:
        match = re.search(r'(?:is|:)\s*\*\*\(?([A-H])\)?\*\*', t)
        if match and match.group(1) in valid_choices:
            return match.group(1)

    # 5. 取最后一个 "(X)" 模式（在完整文本中操作）
    matches = list(re.finditer(r'\(([A-H])\)', text))
    if matches:
        # 排除掉 "Step (X)" 或 "(X) The" 这种推理前缀
        for m in reversed(matches):
            choice = m.group(1)
            if choice in valid_choices:
                # 检查上下文，确保不是 "Step (X)" 推理步骤
                pre = text[max(0, m.start()-10):m.start()]
                if not re.search(r'(Step|Stages|Phase|Part)\s*$', pre, re.IGNORECASE):
                    return choice
        # 如果所有括号字母都被排除，用最后一个
        last = matches[-1].group(1)
        if last in valid_choices:
            return last

    # 6. 匹配 "**(X)**" 加粗模式
    matches = list(re.finditer(r'\*\*\(?([A-H])\)?\*\*', text))
    if matches:
        last_match = matches[-1].group(1)
        if last_match in valid_choices:
            return last_match

    # 7. 找文本末尾 30% 部分中的第一个有效字母（排除 Step 干扰）
    third_len = len(text) // 3
    tail = text[-third_len:] if third_len > 0 else text
    for ch in tail:
        if ch in valid_choices:
            return ch

    # 8. 最后手段：找文本中的第一个有效字母
    for ch in text:
        if ch in valid_choices:
            return ch

    return None


# ==================== 数据加载 ====================

def load_medmcqa(path, max_samples=-1):
    """
    加载 MedMCQA dev.json 数据。
    cop 是 1-based 的正确答案位置 (1=A, 2=B, 3=C, 4=D)
    只保留 single choice 的题目（多选题不纳入评测，因为准确率需明确单一答案）
    """
    samples = []
    idx_map = {1: "A", 2: "B", 3: "C", 4: "D"}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("choice_type") != "single":
                continue
            if "cop" not in data or data["cop"] is None:
                continue

            question = data["question"]
            options = {
                "A": data.get("opa", ""),
                "B": data.get("opb", ""),
                "C": data.get("opc", ""),
                "D": data.get("opd", ""),
            }
            answer_idx = idx_map.get(data["cop"], "")
            if not answer_idx:
                continue

            options_str = ""
            for k in ["A", "B", "C", "D"]:
                if options[k]:
                    options_str += f"\n({k}): {options[k]}"

            prompt = f"Question: {question}{options_str}\nAnswer:"

            samples.append({
                "prompt": prompt,
                "question": question[:200],
                "answer_idx": answer_idx,
                "id": data.get("id", ""),
            })

            if max_samples > 0 and len(samples) >= max_samples:
                break

    return samples


def load_pubmedqa(ori_path, gt_path, max_samples=-1):
    """加载 PubMedQA 数据。答案: yes(A), no(B), maybe(C)"""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(ori_path, "r", encoding="utf-8") as f:
        ori_data = json.load(f)

    answer_map = {"yes": "A", "no": "B", "maybe": "C"}
    samples = []

    for qid, value in ori_data.items():
        if qid not in gt_data:
            continue
        answer_text = gt_data[qid]
        answer_idx = answer_map.get(answer_text, "")
        if not answer_idx:
            continue

        prompt = f"Question: {value['QUESTION']}\n(A): yes\n(B): no\n(C): maybe\nAnswer:"
        samples.append({
            "prompt": prompt,
            "question": value['QUESTION'][:200],
            "answer_idx": answer_idx,
            "id": qid,
        })
        if max_samples > 0 and len(samples) >= max_samples:
            break

    return samples


def load_medqa(path, max_samples=-1):
    """加载 MedQA-US 或 MedQA-Mainland 数据（格式相同）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for qid, value in data.items():
        prompt = f"Question: {value['QUESTION']}{value['options_str']}\nAnswer:"
        samples.append({
            "prompt": prompt,
            "question": value['QUESTION'][:200],
            "answer_idx": value["answer"],
            "id": qid,
        })
        if max_samples > 0 and len(samples) >= max_samples:
            break

    return samples


# ==================== 评测执行 ====================

def evaluate_dataset(client, dataset_name, samples):
    """对单个数据集进行评测"""
    correct = 0
    total = 0
    results = []

    # 确定有效选项集
    valid_choices = {"A", "B", "C"} if dataset_name == "PubMedQA" else {"A", "B", "C", "D", "E", "F", "G", "H"}

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"样本数: {len(samples)}")
    print(f"{'='*60}")

    for i, sample in enumerate(tqdm(samples, desc=dataset_name)):
        messages = [{"role": "user", "content": sample["prompt"]}]
        response = client.chat(messages)
        pred_idx = extract_answer(response, valid_choices)
        is_correct = (pred_idx == sample["answer_idx"])
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "id": sample.get("id", i),
            "question": sample["question"],
            "answer_idx": sample["answer_idx"],
            "pred_idx": pred_idx,
            "response_raw": response[:300],
            "is_correct": is_correct,
        })

        if total % 50 == 0:
            tqdm.write(f"  [{dataset_name}] {total}/{len(samples)}, 当前准确率: {correct/total:.4f} ({correct}/{total})")

    accuracy = correct / total if total > 0 else 0
    print(f"\n[{dataset_name}] 最终准确率: {accuracy:.4f} ({correct}/{total})")
    return {"dataset": dataset_name, "total": total, "correct": correct, "accuracy": accuracy, "results": results}


def save_results(summary, all_results):
    """保存评测结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for dataset_name, result in all_results.items():
        # 保存汇总
        path = os.path.join(RESULTS_DIR, f"{dataset_name.lower().replace('-','_')}_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": result["accuracy"], "correct": result["correct"], "total": result["total"]},
                      f, ensure_ascii=False, indent=2)
        # 保存详情
        detail_path = os.path.join(RESULTS_DIR, f"{dataset_name.lower().replace('-','_')}_details.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for r in result["results"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总
    summary_path = os.path.join(RESULTS_DIR, "accuracy_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": DEEPSEEK_MODEL,
            "setting": "zero-shot, no prompt engineering",
            "results": {s["dataset"]: {"accuracy": s["accuracy"], "correct": s["correct"], "total": s["total"]}
                       for s in summary},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {RESULTS_DIR}")


def print_summary(summary):
    """打印汇总结果"""
    print(f"\n{'='*60}")
    print(f"准确率评测汇总 (Zero-shot)")
    print(f"{'='*60}")
    print(f"{'数据集':<22} {'总数':<8} {'正确':<8} {'准确率':<10}")
    print("-"*50)
    for s in summary:
        print(f"{s['dataset']:<22} {s['total']:<8} {s['correct']:<8} {s['accuracy']:<10.4f}")
    print("-"*50)
    if summary:
        total_all = sum(s["total"] for s in summary)
        correct_all = sum(s["correct"] for s in summary)
        print(f"{'总计':<22} {total_all:<8} {correct_all:<8} {correct_all/total_all:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Accuracy Evaluation")
    parser.add_argument("--datasets", nargs="+", default=["medmcqa", "pubmedqa", "medqa_us", "medqa_mainland"],
                        choices=["medmcqa", "pubmedqa", "medqa_us", "medqa_mainland"],
                        help="要评测的数据集")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="每个数据集的最大样本数 (-1 表示全部)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：每个数据集只跑 5 条")
    args = parser.parse_args()

    client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, model_name=DEEPSEEK_MODEL)
    all_results = {}
    summary = []

    dataset_loaders = {
        "medmcqa": ("MedMCQA", lambda: load_medmcqa(MEDMCQA_DEV_PATH, args.max_samples)),
        "pubmedqa": ("PubMedQA", lambda: load_pubmedqa(PUBMEDQA_ORI_PATH, PUBMEDQA_GT_PATH, args.max_samples)),
        "medqa_us": ("MedQA-US", lambda: load_medqa(MEDQA_US_PATH, args.max_samples)),
        "medqa_mainland": ("MedQA-Mainland", lambda: load_medqa(MEDQA_MAINLAND_PATH, args.max_samples)),
    }

    for key in args.datasets:
        name, loader = dataset_loaders[key]

        # 检查文件是否存在
        path_map = {
            "medmcqa": MEDMCQA_DEV_PATH,
            "pubmedqa": PUBMEDQA_ORI_PATH,
            "medqa_us": MEDQA_US_PATH,
            "medqa_mainland": MEDQA_MAINLAND_PATH,
        }
        if not os.path.exists(path_map[key]):
            print(f"警告: {name} 数据集文件未找到: {path_map[key]}")
            continue

        samples = loader()
        if args.debug:
            samples = samples[:5]
            print(f"[调试模式] {name}: 只加载 {len(samples)} 条样本")

        if len(samples) == 0:
            print(f"警告: {name} 没有有效样本")
            continue

        result = evaluate_dataset(client, name, samples)
        all_results[name] = result
        summary.append(result)

    print_summary(summary)
    save_results(summary, all_results)


if __name__ == "__main__":
    main()
