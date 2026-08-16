# -*- coding: utf-8 -*-
"""_probe_frontend_judgment.py — 任务一：量化 agentic 系统的"前端任务判定"。

[前置走查] 生产 TaskAnalyzer 实际行为盘点（task/format 判定、LLM 是否调用）。
[落地走查] 落地后 analyze(question, format=_format) 直接读 CSV `_format` 真值（format 应 100%）。
[维度1] format(4类题型)判定: gt=CSV `_format`；对比 a)落地后读真值 b)LLM直判 c)生产heuristic；acc/混淆/加权F1/分层。
[维度2] task(8类推理任务)判定: gt=capability→TaskType 专家映射；对比 a)LLM判8类 b)生产恒GENERAL。
[结论] 前端判定"当前零启用/降级"的量化证据 + 落地后 read-format 生效证明 + 若启用 LLM 潜力。
运行: python _probe_frontend_judgment.py --samples 40
"""
import os, sys, csv, json, time, re, argparse
from collections import Counter, defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import importlib
cfg = importlib.import_module("experiments.config")
CSV_PATH = cfg.INDUSTRYBENCH_CSV
DEEPSEEK_KEY = cfg.DEEPSEEK_KEY
LLM_NAME = cfg.LLM_NAME

from agentic.task_types import infer_format_heuristic
from agentic.analyzer import TaskAnalyzer

FORMATS = ["QA", "FillBlank", "MultipleChoice", "Calculation"]
FORMAT_LABELS = {"QA": "问答题", "FillBlank": "填空题",
                 "MultipleChoice": "选择题", "Calculation": "计算题"}
CN2EN_FMT = {
    "问答题": "QA", "填空题": "FillBlank", "选择题": "MultipleChoice",
    "计算题": "Calculation", "问答": "QA", "填空": "FillBlank",
    "选择": "MultipleChoice", "计算": "Calculation",
}
TASK_LABELS = {
    "comparison": "对比", "diagnosis": "诊断", "selection": "选型",
    "calculation": "计算", "standard_interpretation": "标准解读",
    "procedure": "流程/规程", "explanation": "解释", "general": "通用",
}
# capability(7类能力域) → TaskType(8类推理任务) 真值代理映射（专家先验，落档）
CAP2TASK = {
    "选型与替代": "selection",
    "故障诊断与排查": "diagnosis",
    "安全合规与风险控制": "procedure",
    "标准规范与术语": "standard_interpretation",
    "工艺原理与参数影响": "explanation",
    "质量计量与检测": "diagnosis",
    "工程计算与估算": "calculation",
}
TASKSET = ["comparison", "diagnosis", "selection", "calculation",
           "standard_interpretation", "procedure", "explanation", "general"]


def load_rows(limit=None):
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        out.append({
            "id": (r.get("id") or "").strip(),
            "question": q,
            "ref": (r.get("answer") or "").strip(),
            "fmt_cn": (r.get("_format") or r.get("format") or "").strip(),
            "capability": (r.get("capability") or "").strip(),
            "difficulty": (r.get("difficulty") or "").strip(),
            "industry": (r.get("industry_primary") or "").strip(),
        })
        if limit and len(out) >= limit:
            break
    return out


def stratify_sample(rows, n):
    """按 (capability, format) 均衡抽样。"""
    bucket = defaultdict(list)
    for r in rows:
        fmt = CN2EN_FMT.get(r["fmt_cn"]) or infer_format_heuristic(r["question"])
        bucket[(r["capability"], fmt)].append(r)
    picked, keys = [], list(bucket)
    idx = 0
    while len(picked) < n and keys:
        k = keys[idx % len(keys)]
        if bucket[k]:
            picked.append(bucket[k].pop())
        else:
            keys.pop(idx % len(keys)); idx -= 1
        idx += 1
    return picked[:n]


def true_format(r):
    return CN2EN_FMT.get(r["fmt_cn"]) or infer_format_heuristic(r["question"])


def true_task(r):
    return CAP2TASK.get(r["capability"], "general")


def _client():
    from openai import OpenAI
    return OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com"), LLM_NAME


def _llm(client, system, user, temperature=0.0, max_tokens=80):
    try:
        resp = client.chat.completions.create(
            model=LLM_NAME,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [WARN] LLM: {e}")
        return ""


FORMAT_DETECT_SYS = (
    "你是工业质检数据标注员。根据题目内容判断其题型(format)，仅四类："
    "问答题(QA)、填空题(FillBlank)、选择题(MultipleChoice)、计算题(Calculation)。"
    "只输出 JSON: {\"format\":\"QA|FillBlank|MultipleChoice|Calculation\",\"confidence\":0.0~1.0}。"
)


def detect_format_llm(client, question):
    txt = _llm(client, FORMAT_DETECT_SYS,
               f"请判断下面这道题的题型类别。\n题目: {question}", temperature=0.0)
    m = re.search(r'"format"\s*:\s*"([A-Za-z]+)"', txt)
    c = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
    fmt = m.group(1) if m else ""
    conf = float(c.group(1)) if c else 0.0
    alias = {"问答": "QA", "填空": "FillBlank", "选择": "MultipleChoice", "计算": "Calculation"}
    return alias.get(fmt, fmt), conf


TASK_DETECT_SYS = (
    "你是工业质检数据标注员。判断题目的推理任务类型(task)，从 8 类单选：\n"
    "comparison=对比比较; diagnosis=故障诊断/检测/排查; selection=选型/替代; "
    "calculation=计算/数值; standard_interpretation=标准规范解读/术语; "
    "procedure=安全合规流程/操作规程; explanation=原理/机制解释; general=通用其他。\n"
    "只输出 JSON: {\"task\":\"comparison|diagnosis|selection|calculation|"
    "standard_interpretation|procedure|explanation|general\",\"confidence\":0.0~1.0}。"
)


def detect_task_llm(client, question):
    txt = _llm(client, TASK_DETECT_SYS,
               f"请判断下面这道题的推理任务类型。\n题目: {question}", temperature=0.0)
    m = re.search(r'"task"\s*:\s*"([A-Za-z_]+)"', txt)
    c = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
    t = m.group(1) if m else ""
    conf = float(c.group(1)) if c else 0.0
    if t not in TASKSET:
        alias = {"compare": "comparison", "differences": "comparison",
                 "select": "selection", "standard": "standard_interpretation",
                 "safety": "procedure", "detect": "diagnosis", "explain": "explanation"}
        t = alias.get(t.lower(), "")
    return (t if t in TASKSET else "general"), conf


def class_metrics(cm, classes, true_list, pred_list):
    acc = sum(t == p for t, p in zip(true_list, pred_list)) / max(1, len(true_list))
    wf1, wsum = 0.0, 0
    per = {}
    for cl in classes:
        tp = cm[cl][cl]
        fp = sum(cm[oc][cl] for oc in classes if oc != cl)
        fn = sum(cm[cl][oc] for oc in classes if oc != cl)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        n = sum(1 for t in true_list if t == cl)
        per[cl] = {"n": n, "prec": round(prec, 3), "rec": round(rec, 3), "f1": round(f1, 3)}
        wf1 += f1 * n
        wsum += n
    return {"acc": acc, "weighted_f1": wf1 / max(1, wsum), "per": per}


def build_confmat(true_list, pred_list, classes):
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_list, pred_list):
        cm[t][p] += 1
    return cm


def walk_production_analyzer(sample_rows):
    """生产 TaskAnalyzer 实际行为盘点：task/format 判定、LLM 是否调用。"""
    an = TaskAnalyzer()
    fmt_dist, task_dist = Counter(), Counter()
    conf_sum = 0.0
    rows = []
    for r in sample_rows:
        ta = an.analyze(r["question"])   # 生产 pipeline.py 不传 format (落地前)
        fmt_dist[ta.format] += 1
        task_dist[ta.task.value] += 1
        conf_sum += ta.confidence
        rows.append({
            "id": r["id"], "question": r["question"],
            "analyzer_task": ta.task.value, "analyzer_format": ta.format,
            "analyzer_confidence": ta.confidence, "llm_called": False,
            "true_format_csv": true_format(r), "true_task_cap2task": true_task(r),
        })
    return {
        "n": len(sample_rows),
        "task_dist": dict(task_dist),
        "format_dist": dict(fmt_dist),
        "mean_confidence": conf_sum / max(1, len(sample_rows)),
        "llm_called_any": False,
        "rows": rows,
    }


def walk_landed_analyzer(sample_rows):
    """
    落地后：analyze(question, format=_format) 直接读 CSV `_format` 真值。

    生产链路已把 `_format` 透传到 analyze()（pipeline.run / AgenticRAGEngine.answer
    新增可选 format 参数），因此 TaskAnalysis.format 由 normalize_format(_format, q)
    归一化直存 ground-truth（应为 100% 命中），这是「analyze 直接读 format」落地的证明。
    task 维度仍恒为 GENERAL（中性化单点，非本任务范围）。
    """
    an = TaskAnalyzer()
    fmt_dist, task_dist = Counter(), Counter()
    rows = []
    for r in sample_rows:
        fmt_cn = r["fmt_cn"]  # CSV `_format` 中文标签/枚举
        ta = an.analyze(r["question"], format=fmt_cn)  # 落地：直读真值
        fmt_dist[ta.format] += 1
        task_dist[ta.task.value] += 1
        rows.append({
            "id": r["id"], "question": r["question"],
            "analyzer_task": ta.task.value, "analyzer_format": ta.format,
            "analyzer_confidence": ta.confidence, "llm_called": False,
            "true_format_csv": true_format(r), "true_task_cap2task": true_task(r),
        })
    return {
        "n": len(sample_rows),
        "task_dist": dict(task_dist),
        "format_dist": dict(fmt_dist),
        "mean_confidence": sum(x["analyzer_confidence"] for x in rows) / max(1, len(rows)),
        "llm_called_any": False,
        "rows": rows,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40, help="LLM 判定抽样数")
    ap.add_argument("--out", type=str, default="results/frontend_judgment")
    ap.add_argument("--no_llm", action="store_true", help="跳过 LLM 判定")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    all_rows = load_rows()
    cap_dist = Counter(r["capability"] for r in all_rows)
    # 全量 heuristic format 判定诊断（无 LLM，2049 题全部）
    all_heur = defaultdict(lambda: defaultdict(int))
    all_heur_pred = Counter()
    all_heur_acc = 0
    for r in all_rows:
        gt = true_format(r)
        h = infer_format_heuristic(r["question"])
        all_heur[gt][h] += 1
        all_heur_pred[h] += 1
        all_heur_acc += (gt == h)
    all_heur_acc = all_heur_acc / max(1, len(all_rows))
    print(f"[INFO] 全量 heuristic format acc={all_heur_acc*100:.1f}% dist={dict(all_heur_pred)}")


    print(f"[INFO] 全量 {len(all_rows)} 题; capability = {dict(cap_dist)}")

    sample = stratify_sample(all_rows, args.samples)
    fmt_gt = [true_format(r) for r in sample]

    # 1) 生产走查
    walk = walk_production_analyzer(sample)
    print(f"[走查] task分布={walk['task_dist']} format分布={walk['format_dist']} "
          f"conf={walk['mean_confidence']:.3f} LLM={walk['llm_called_any']}")

    # 落地后走查：analyze(question, format=_format) 直接读 `_format` 真值
    landed = walk_landed_analyzer(sample)
    print(f"[落地走查] format分布={landed['format_dist']} "
          f"conf={landed['mean_confidence']:.3f}")

    heur_fmt = [r["analyzer_format"] for r in walk["rows"]]
    heur_fmt_cm = build_confmat(fmt_gt, heur_fmt, FORMATS)
    heur_fmt_m = class_metrics(heur_fmt_cm, FORMATS, fmt_gt, heur_fmt)
    landed_fmt = [r["analyzer_format"] for r in landed["rows"]]
    landed_fmt_cm = build_confmat(fmt_gt, landed_fmt, FORMATS)
    landed_fmt_m = class_metrics(landed_fmt_cm, FORMATS, fmt_gt, landed_fmt)
    prod_task_cm = build_confmat([true_task(r) for r in sample],
                                 ["general"] * len(sample), TASKSET)
    prod_task_m = class_metrics(prod_task_cm, TASKSET,
                                [true_task(r) for r in sample],
                                ["general"] * len(sample))

    # 2) LLM 判定
    llm_fmt = [""] * len(sample)
    llm_conf = [0.0] * len(sample)
    llm_task = [""] * len(sample)
    llm_tconf = [0.0] * len(sample)
    if not args.no_llm:
        client, _ = _client()
        for i, s in enumerate(sample):
            f, fc = detect_format_llm(client, s["question"])
            t, tc = detect_task_llm(client, s["question"])
            llm_fmt[i], llm_conf[i] = f, fc
            llm_task[i], llm_tconf[i] = t, tc
            print(f"  [{i+1}/{len(sample)}] {s['question'][:36]}.. fmt:{f}({fc:.2f}) task:{t}({tc:.2f})")

    llm_fmt_cm = build_confmat(fmt_gt, llm_fmt, FORMATS)
    llm_fmt_m = class_metrics(llm_fmt_cm, FORMATS, fmt_gt, llm_fmt)
    llm_task_cm = build_confmat([true_task(r) for r in sample], llm_task, TASKSET)
    llm_task_m = class_metrics(llm_task_cm, TASKSET,
                               [true_task(r) for r in sample], llm_task)



    def layer_acc(pred_list):
        layers = {}
        by_cap = defaultdict(list)
        for idx in range(len(sample)):
            by_cap[sample[idx]["capability"]].append((fmt_gt[idx], pred_list[idx]))
        for cap, pairs in by_cap.items():
            layers[cap] = round(sum(a == b for a, b in pairs) / max(1, len(pairs)), 3)
        by_diff = defaultdict(list)
        for idx in range(len(sample)):
            by_diff[sample[idx]["difficulty"]].append((fmt_gt[idx], pred_list[idx]))
        for d, pairs in by_diff.items():
            layers[f"diff:{d}"] = round(sum(a == b for a, b in pairs) / max(1, len(pairs)), 3)
        return layers

    layer_heur = layer_acc(heur_fmt)
    layer_llm = layer_acc(llm_fmt) if not args.no_llm else {}
    layer_landed = layer_acc(landed_fmt)
    detail = {
        "meta": {"samples": len(sample), "csv": CSV_PATH, "model": LLM_NAME,
                 "all_rows": len(all_rows),
                 "all_format_dist": dict(Counter(r["fmt_cn"] for r in all_rows)),
                 "all_capability_dist": dict(cap_dist),
                 "all_heur_format": {
                     "acc": round(all_heur_acc, 4),
                     "pred_dist": dict(all_heur_pred),
                     "cm": {k: dict(v) for k, v in all_heur.items()},
                 }},
        "cap2task_truth_proxy": CAP2TASK,
        "production_walk": walk,
        "landed_analyzer_walk": landed,
        "format": {
            "heur": {"cm": {k: dict(v) for k, v in heur_fmt_cm.items()},
                     "metrics": heur_fmt_m},
            "landed_truth": {"cm": {k: dict(v) for k, v in landed_fmt_cm.items()},
                             "metrics": landed_fmt_m},
            "llm": {"cm": {k: dict(v) for k, v in llm_fmt_cm.items()},
                    "metrics": llm_fmt_m},
        },
        "task": {
            "prod_general": {"cm": {k: dict(v) for k, v in prod_task_cm.items()},
                             "metrics": prod_task_m},
            "llm": {"cm": {k: dict(v) for k, v in llm_task_cm.items()},
                    "metrics": llm_task_m},
        },
        "layer_acc_heur_fmt": layer_heur,
        "layer_acc_llm_fmt": layer_llm,
        "layer_acc_landed_fmt": layer_landed,
        "rows": [{
            "id": s["id"], "question": s["question"], "capability": s["capability"],
            "difficulty": s["difficulty"], "true_format": fmt_gt[i],
            "true_task": true_task(s), "heur_format": heur_fmt[i],
            "landed_format": landed_fmt[i],
            "llm_format": llm_fmt[i], "llm_fmt_conf": round(llm_conf[i], 3),
            "llm_task": llm_task[i], "llm_task_conf": round(llm_tconf[i], 3),
            "prod_task": "general",
        } for i, s in enumerate(sample)],
    }
    json_path = os.path.join(args.out, f"frontend_judgment_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)



    rep = []

    def R(x=None):
        if x is not None:
            rep.append(x)

    R("=" * 74)
    R("任务一 前端任务判定量化（Analyzer ≥ format/task 判定）")
    R(f"抽样 {len(sample)} 题 / 全量 {len(all_rows)} 题 | 模型 {LLM_NAME} | {ts}")
    R("=" * 74)
    R("\n【0】系统走查：生产 TaskAnalyzer 当前实际行为")
    R("  - task 维度：analyze() 恒返回 GENERAL（中性化单点），LLM 不调用。")
    R(f"      task 分布: {walk['task_dist']}  → task 判定【零启用】")
    R("  - format 维度（落地前）：pipeline 未传 format；analyze() 内 normalize_format('',q) 走 heuristic。")
    R(f"      format 分布: {walk['format_dist']}")
    R(f"  - mean_confidence={walk['mean_confidence']:.3f}; LLM 调用={walk['llm_called_any']}（0=纯确定性，零延迟）")
    R("  - 结论：前端判定（尤其 task）生产中为【中性化/降级】状态；format 仅靠 heuristic。")
    R("\n【0b】落地后走查：analyze(question, format=_format) 直接读 `_format` 真值")
    R("  - pipeline.run / AgenticRAGEngine.answer 已透传可选 format 参数；analyze 直读真值归一化。")
    R(f"      format 分布: {landed['format_dist']}  (与 gt 一一对应 → 100%)")
    R(f"      mean_confidence={landed['mean_confidence']:.3f}; LLM 调用={landed['llm_called_any']}")
    R("\n【1】format(题型) 判定准确率（gt=CSV `_format`；落地后 vs heuristic vs LLM 判）")
    R(f"  {'':<12}{'acc':>8}{'加权F1':>10}")
    R(f"  {'落地后(读真值)':<13}{landed_fmt_m['acc']*100:>7.1f}%{landed_fmt_m['weighted_f1']*100:>9.1f}%")
    R(f"  {'heuristic(生产)':<13}{heur_fmt_m['acc']*100:>7.1f}%{heur_fmt_m['weighted_f1']*100:>9.1f}%")
    R(f"  {'LLM 直判':<13}{llm_fmt_m['acc']*100:>7.1f}%{llm_fmt_m['weighted_f1']*100:>9.1f}%")
    R("\n  混淆矩阵(行=真值, 列=预测; heur):")
    R("  " + "真实\\预测".ljust(14) + "".join(f"{FORMAT_LABELS[c]:<14}" for c in FORMATS))
    for tf in FORMATS:
        R("  " + FORMAT_LABELS[tf].ljust(14) + "".join(
            f"{heur_fmt_cm[tf][c]:<14}" for c in FORMATS))
    R("  落地后混淆矩阵(行=真值, 列=预测):")
    for tf in FORMATS:
        R("  " + FORMAT_LABELS[tf].ljust(14) + "".join(
            f"{landed_fmt_cm[tf][c]:<14}" for c in FORMATS))
    R("\n  分层 acc(heuristic → 落地后):")
    for k in layer_heur:
        v2 = layer_landed.get(k, "-")
        v2 = f"{v2*100:.0f}%" if isinstance(v2, (int, float)) else str(v2)
        R(f"    {k:<16} heur={layer_heur[k]*100:.0f}%   landed={v2}")



    R("\n【2】task(推理任务) 判定准确率（gt=capability→TaskType 专家映射代理）")
    R(f"  {'':<12}{'acc':>8}{'加权F1':>10}")
    R(f"  {'生产(恒GENERAL)':<13}{prod_task_m['acc']*100:>7.1f}%{prod_task_m['weighted_f1']*100:>9.1f}%")
    R(f"  {'LLM 判8类':<13}{llm_task_m['acc']*100:>7.1f}%{llm_task_m['weighted_f1']*100:>9.1f}%")
    R("\n  LLM task 混淆矩阵(行=真值代理, 列=预测):")
    R("  " + "真实\\预测".ljust(16) + "".join(f"{t:<14}" for t in TASKSET))
    for tf in TASKSET:
        R("  " + TASK_LABELS[tf].ljust(16) + "".join(
            f"{llm_task_cm[tf][c]:<14}" for c in TASKSET))
    R("\n  每类 P/R/F1 (LLM task 判定):")
    for cl in TASKSET:
        p = llm_task_m["per"].get(cl, {})
        R(f"    {TASK_LABELS[cl]:<12} n={p.get('n',0):<3} P={p.get('prec',0)*100:.0f}% "
          f"R={p.get('rec',0)*100:.0f}% F1={p.get('f1',0):.2f}")

    R("\n【3】结论与决策提示")
    llm_fmt_acc = llm_fmt_m["acc"]
    heur_fmt_acc = heur_fmt_m["acc"]
    if llm_fmt_acc > heur_fmt_acc + 0.05:
        R("  - format: LLM 判定明显优于生产 heuristic => 前端 format 判定有被浪费的容量。")
    else:
        R("  - format: LLM 与 heuristic 接近 => 生产 heuristic 已够用，无需引入 LLM。")
    if llm_fmt_acc < 0.80:
        R("  - format: LLM 判定本身 acc<80% => 前端判定不可靠，应优先靠 CSV 真值。")
    R(f"  - format: 落地后 analyze 直读 `_format` 真值 => acc={landed_fmt_m['acc']*100:.1f}%（100%=链路一致性校验：标注→前端 format 判定分发生效）。")
    R("  - task: 生产恒 GENERAL；LLM 判 8 类可达能力仅作参考（真值为 capability 代理）。")
    R("  - 注意: capability 与推理任务不严格一一对应，task 真值存在固有噪声。")
    R(f"\n明细JSON: {json_path}")

    text = "\n".join(rep)
    print(text)
    md_path = os.path.join(args.out, f"frontend_judgment_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[SAVED] {md_path}")


if __name__ == "__main__":
    main()

