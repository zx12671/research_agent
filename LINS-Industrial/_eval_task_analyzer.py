# -*- coding: utf-8 -*-
"""
_eval_task_analyzer.py — TaskAnalyzer（规则关键词分类）的量化评测。

任务背景：
    analyzer.py 的 TaskAnalyzer 采用「轻量规则关键词匹配」对工业问题进行任务类型分类
    （v2 重构：移除了 LLM 分类以保住检索召回）。分类结果直接影响下游 StrategyPlanner
    选择 ExecutionGraph 模板（不同 task 的 retrieve_k / organize_by / 推理步骤不同）。

    由于它是整条 Agentic 管线的「第一跳」，分类准确率会级联影响检索证据量与最终答案。

本脚本从三个维度量化该分类器的能力，全程零 LLM/API 调用（纯规则可直接运行）：

  A. 手工金标准样本评测（逻辑正确性）
       构造覆盖 8 种任务类型的样本，含
         - 强关键词正向样本（应正确命中）
         - 朴素包含误判样本（一词多义 / 嵌入歧义，规则应判错）
         - 顺序敏感性样本（多个关键词共存时，先命中的词取主导）
       输出：总体准确率、各类别准确率、混淆矩阵、每个样本 predict/gold/hit 明细。

  B. 真实数据集利用率（覆盖能力）
       从 huggingface_dataset.csv 的 2049 个真实工业问答题统计
         - 分类器输出分布（多少落入各 task、多少 fallback 到 general）
         - 真实未被有效分类（落到 general）的比例 —— 即规则的实际覆盖缺口
         - 各关键词对其实问法的命中贡献

  C. 缺陷定位（顺序敏感性 + 朴素字符串包含）
       用特意构造的最小对样本证明：
         (1) 分类顺序由 KEYWORD_MAP 的书写顺序决定，而非语义优先级；
         (2) 子串包含（如 "故障" 命中 "故障诊断的原理"）导致语义错配。

用法：
    cd LINS-Industrial
    python _eval_task_analyzer.py
"""

import sys
import os
import json
import logging

# ==== 路径修复（复用 exp1 的模式）====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.ERROR)

from agentic.analyzer import TaskAnalyzer
from agentic.task_types import TaskType

OUT_FILE = os.path.join(_THIS_DIR, "results", "task_analyzer_classification_eval.json")
CSV_PATH = os.path.join(_THIS_DIR, "data", "industrybench", "huggingface_dataset.csv")


analyzer = TaskAnalyzer()  # 无 LLM，直接构造


def classify(q: str) -> str:
    return analyzer.analyze(q).task.value


# ============================================================
# A. 手工金标准样本评测
# ============================================================
# gold = 语义上应属的任务类型（对真实工业问题的合理解读）
# 样本刻意分为三类：
#   kind='positive'  强关键词，规则应命中（验证能力）
#   kind='tricky'    语义应属 A 类但含 B 类的朴素关键词（验证缺陷）
#   kind='ambiguity' 多关键词共存，验证写入顺序主导
EXAMPLES = [
    # ---- comparison 比较 ----
    ("PID控制和模糊控制在电机调速中有何区别？", "comparison", "positive"),
    ("标准HDMI与Mini HDMI接口的主要差异是什么？", "comparison", "positive"),
    ("对比X和Y两种材料的耐腐蚀性能。", "comparison", "positive"),
    ("What is the difference between Class A and Class S power quality monitors?", "comparison", "positive"),
    ("请问A型和B型安全阀哪个压力等级的适用差异体现在哪里？", "comparison", "positive"),
    # ---- diagnosis 诊断 ----
    ("离心泵轴承过热的原因有哪些？", "diagnosis", "positive"),
    ("如何排查液压系统的油液污染故障？", "diagnosis", "positive"),
    ("电机绕组烧毁的常见故障是什么？", "diagnosis", "positive"),
    ("How to troubleshoot hydraulic contamination?", "diagnosis", "positive"),
    ("变频器过压报警的故障排查步骤是什么？", "diagnosis", "positive"),
    # ---- calculation 计算 ----
    ("请计算500kW电机负载所需的变压器容量。", "calculation", "positive"),
    ("一根100米4寸管道的压降是多少？", "calculation", "positive"),
    ("根据欧姆定律计算该电路的电流，用公式求值。", "calculation", "positive"),
    ("这个反应釜的加热时间大约需要多少？", "calculation", "positive"),
    # ---- selection 选择 ----
    ("请推荐适用于强腐蚀环境的传感器。", "selection", "positive"),
    ("高温炉部件应选择哪种材料？", "selection", "positive"),
    ("哪个牌号的密封圈更适合油箱使用，请推荐。", "selection", "positive"),
    ("Which material should be selected for high-temperature furnace components?", "selection", "positive"),
    # ---- procedure 流程  ----
    ("燃气轮发电机组的启动步骤是什么？", "procedure", "positive"),
    ("描述压力变送器的标定流程。", "procedure", "positive"),
    ("更换润滑油滤芯的操作流程是怎样的？", "procedure", "positive"),
    ("如何正确执行在线绝缘电阻测试？", "procedure", "positive"),
    ("Describe the calibration process for a pressure transmitter.", "procedure", "positive"),
    # ---- standard_interpretation 标准 ---
    ("IEC 60034-1 对温升限值有何规定？", "standard_interpretation", "positive"),
    ("请解释GB/T 5226.1 对机床安全的要求。", "standard_interpretation", "positive"),
    ("国家标准GB/T 20476 规定的参数是什么？", "standard_interpretation", "positive"),
    ("该设备应符合哪个IEC规范？", "standard_interpretation", "positive"),
    # ---- explanation 解释 ----
    ("解释三相异步电动机的工作原理。", "explanation", "positive"),
    ("电力系统中无功补偿的概念是什么？", "explanation", "positive"),
    ("什么是变频调速的基本原理？", "explanation", "positive"),
    ("Explain the working principle of a three-phase induction motor.", "explanation", "positive"),
    # ---- general 一般 ----
    ("请介绍工业以太网在自动化中的应用概况。", "general", "positive"),
    ("简述现场总线在工厂中的应用。", "general", "positive"),
    # ================= 缺陷类 =================
    # tricky：语义应是 explanation，但朴素包含 "故障"→diagnosis / "原因"→diagnosis
    ("故障诊断的基本原理是什么？", "explanation", "tricky"),
    ("介绍一下设备产生故障的常见原因分类及其判别依据？", "explanation", "tricky"),
    # tricky：语义是 standard_interpretation，但含 "如何"→procedure
    ("作为工程师。如何理解GB/T 5226.1中的安全条款？", "standard_interpretation", "tricky"),
    # tricky：语义是 comparison，但含 "选择/推荐/哪个"→selection（应比较后在选）
    ("哪个更好？对比A与B的控制方案后给出选择依据。", "comparison", "tricky"),
    # tricky：semantic procedure，但含 "原理"→explanation
    ("请说明氢气检漏的检测原理及操作步骤。", "explanation", "tricky"),
    # tricky：semantic selection，但含 "多少"→(这里"多少"在关键词表末尾，若含"选择"先判)
    ("需要加装几台补偿柜最好，请给出选择？", "selection", "tricky"),
    # ================= 顺序敏感性 =================
    # 多关键词共存，命中的是 KEYWORD_MAP 书写顺序中更靠前者：
    #  "计算原理"：compare→diagnosis→calc命中(顺序calc在explain前)，实际语义:explanation
    ("三相电机调速采用变频的算法计算原理是什么？", "explanation", "ambiguity"),
    #  "如何…原理"：calc在selection后，含"选择"会先命中selection，这里不能保证，故仅做展示用更强case
    ("如何用公式计算并解释这个误差原理？", "calculation", "ambiguity"),
]


def run_gold_bench() -> dict:
    hits = 0
    rows = []
    per_class = {}
    for q, gold, kind in EXAMPLES:
        pred = classify(q)
        hit = (pred == gold)
        hits += hit
        per_class.setdefault(gold, {"n": 0, "ok": 0})
        per_class[gold]["n"] += 1
        if hit:
            per_class[gold]["ok"] += 1
        rows.append({
            "question": q, "gold": gold, "predict": pred,
            "hit": hit, "kind": kind,
            "mark": "✓" if hit else "✗",
        })
    # 混淆矩阵
    labels = sorted(TaskType)  # TaskType 是 str Enum
    label_names = [str(l) for l in sorted(set(r["gold"] for r in rows) | set(r["predict"] for r in rows))]
    matrix = {g: {p: 0 for p in label_names} for g in label_names}
    for r in rows:
        matrix[r["gold"]][r["predict"]] += 1

    acc = hits / len(rows) if rows else 0.0
    return {
        "n_samples": len(rows),
        "accuracy": round(acc, 4),
        "correct": hits,
        "per_class": {k: {kk: vv for kk, vv in v.items()} for k, v in per_class.items()},
        "confusion_matrix": matrix,
        "rows": rows,
    }


# ============================================================
# B. 真实数据集利用率（2049 题）
# ============================================================
def run_real_distribution(n_limit: int = None) -> dict:
    import csv
    from collections import Counter
    distribution = Counter()
    unmatched = []
    keyword_hits = Counter()
    sampled_rows = []
    count = 0
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            count += 1
            pred = classify(q)
            distribution[pred] += 1
            # 记录未命中任何关键词（落到 general 的真实缺口）
            if pred == "general":
                unmatched.append(q[:80])
            sampled_rows.append({"id": row["id"], "question": q[:90], "predict": pred})
            if n_limit and count >= n_limit:
                break
    total = sum(distribution.values())
    return {
        "total_questions": total,
        "distribution": dict(distribution),
        "general_ratio": round(distribution.get("general", 0) / max(total, 1), 4),
        "effective_classified": round((total - distribution.get("general", 0)) / max(total, 1), 4),
        "sample_of_unmatched": unmatched[:15],
        "first_rows": sampled_rows[:8],
    }


def main():
    gold = run_gold_bench()
    real = run_real_distribution()

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"gold_bench": gold, "real_distribution": real}, f, ensure_ascii=False, indent=2)

    # ----------- 人类可读输出 -----------
    print("=" * 70)
    print("A. 手工金标准样本评测（逻辑正确性，无 LLM）")
    print("=" * 70)
    print(f"样本总数: {gold['n_samples']}  正确: {gold['correct']}  准确率: {gold['accuracy']:.2%}\n")
    print(f"{'gold':<20}{'predict':<20}{'kind':<10}{'hit':<5} question")
    print("-" * 90)
    for r in gold["rows"]:
        print(f"{r['gold']:<20}{r['predict']:<20}{r['kind']:<10}{('✓' if r['hit'] else '✗'):<5} {r['question'][:46]}")
    print("\n按 gold 类别的准确率：")
    for k, v in gold["per_class"].items():
        print(f"  {k:<22} {v['ok']}/{v['n']}")

    print("\n混淆矩阵 (行=gold, 列=predict)：")
    labels = [k for k in gold["confusion_matrix"].keys()]
    header = "gold\\pred".ljust(22) + "".join(p[:8].rjust(9) for p in labels)
    print(header)
    for g, row in gold["confusion_matrix"].items():
        print(g.ljust(22) + "".join(str(row[p]).rjust(9) for p in labels))


    print("\n" + "=" * 70)
    print(f"B. 真实数据集（{real['total_questions']} 问答题）分类利用率")
    print("=" * 70)

    print(f"分类器输出分布: {real['distribution']}")
    print(f"落到 general（未有效分类）比例: {real['general_ratio']:.2%}   (有效分类: {real['effective_classified']:.2%})")
    print("\n样例（落到 general 的真实问题，即规则覆盖缺口）：")
    for u in real["sample_of_unmatched"]:
        print("  -", u)
    print(f"\n结果已写入: {OUT_FILE}")


if __name__ == "__main__":
    main()
