"""
_ab_format_first.py — Format-First 答题定制 A/B 量化实验

目的: 在"检索内容保持一致"的严格控制下，量化"按题目类型(format)定制答题提示" vs "统一答题提示"
对答案质量的增益。衡量两个层面:

  题型来源 --format_source:
    - annotated (默认): 管线B 的题型直接在题型划分处读取 CSV 已标注的 `_format` 真值，
      不调用 LLM 判定。此时【A】节正确率应为 100%，用于校验"标注→答题"链路一致性，
      核心诉求是量化【B】节"题型已知时 format 定制答题"的真实增益。
    - llm (对比): 管线B 用 LLM 判定题目 format(问答题/填空题/选择题/计算题) + 置信度，
      与标注字段做混淆矩阵，输出准确率 + 加权 F1，衡量前端判定可靠性。

  答题贴合度 (后端收益, A vs B):
     两条管线使用完全相同的检索结果 (同一批 hybrid_retrieve 文档、相同 k、相同顺序)，
     仅 differ 在答题 prompt 是否 format-定制。
     用 RuleBasedScorer 量化答案贴合度:
       - rule_based_score (0-3 规则覆盖分)
       - compute_coverage (关键词覆盖)
       - compute_entity_coverage (实体覆盖)
     另加 format 结构命中检测:
       - 填空题: 答案是否含冗余引导语("答案是/答案:"/"是:") → 冗余=贴合差
       - 选择题: 答案是否给出 A/B/C 选项结构
       - 计算题: 答案是否含数值 + 单位

管线:
  管线A(基线):  Question → hybrid_retrieve(同参) → 统一答题提示 → LLM → ansA
  管线B(format):  Question → 题型划分(读标注 _format / 或 --format_source=llm 判定)
                  → hybrid_retrieve(同参,同一批文档) → format定制答题提示 → LLM → ansB

控制变量:
  - 检索完全同源 (B 与 A 用同一批 docs, 构造 evidence 字符串并交给 LLM, 不由模型重检索)
  - 除"是否 format 定制 prompt"(及题型来源)外无其他差异。

运行:
  python LINS-Industrial/_ab_format_first.py --samples 16               # 默认读标注 _format
  python LINS-Industrial/_ab_format_first.py --samples 16 --format_source llm  # LLM判定对比
"""


import os
import sys
import csv
import json
import time
import argparse
import re
from collections import Counter, defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR  # this script lives under LINS-Industrial/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datetime import datetime

# --- DeepSeek client ---
from openai import OpenAI

# --- config (相对 _THIS_DIR) ---
import importlib.util


def _load_config():
    """加载 experiments/config 常量，避免依赖 sys.path 拼接。"""
    cfg_mod = importlib.import_module("experiments.config")
    return (
        getattr(cfg_mod, "INDUSTRYBENCH_CSV", None),
        getattr(cfg_mod, "DEEPSEEK_KEY", None),
        getattr(cfg_mod, "LLM_NAME", "deepseek-chat"),
    )


CSV_PATH, DEEPSEEK_KEY, LLM_NAME = _load_config()

FORMATS = ["QA", "FillBlank", "MultipleChoice", "Calculation"]
FORMAT_LABELS = {
    "QA": "问答题",
    "FillBlank": "填空题",
    "MultipleChoice": "选择题",
    "Calculation": "计算题",
}
CN2EN = {
    "问答题": "QA",
    "填空题": "FillBlank",
    "选择题": "MultipleChoice",
    "计算题": "Calculation",
    "问答": "QA",
    "填空": "FillBlank",
    "选择": "MultipleChoice",
    "计算": "Calculation",
}


# ---------------------------------------------------------------------------
# 1. 数据加载: 按 format 分层取样
# ---------------------------------------------------------------------------
def load_samples(num_per_format: int):
    rows_by_format = defaultdict(list)
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qf = (row.get("_format") or row.get("format") or "").strip()
            # 数据集 _format 列是中文标签("问答题"等), 统一映射到枚举
            qf = CN2EN.get(qf) or infer_format_heuristic(row.get("question", ""))
            if qf in FORMATS:
                rows_by_format[qf].append(row)


    samples = []
    for qf in FORMATS:
        pool = rows_by_format.get(qf, [])
        picked = pool[:num_per_format]
        for row in picked:
            samples.append({
                "id": row.get("id", "").strip(),
                "question": row.get("question", "").strip(),
                "ref": row.get("answer", "").strip(),
                "format": qf,
                "capability": row.get("capability", "").strip(),
                "difficulty": row.get("difficulty", "").strip(),
            })
    return samples


def infer_format_heuristic(q: str) -> str:
    q = q.strip()
    if "___" in q or "____" in q or "＿" in q:
        return "FillBlank"
    if re.match(r"^\s*(A[\.\、]|A\)|A\s)", q) or "A." in q[:20]:
        return "MultipleChoice"
    if any(kw in q for kw in ["多少", "计算", "数值", "参数", "mm", "kW", "A", "V", "Hz", "功率", "电压", "电流"]):
        return "Calculation"
    return "QA"


# ---------------------------------------------------------------------------
# 2. 检索器 (复用在 format×capability 矩阵中验证过的模式)
# ---------------------------------------------------------------------------
class _HybridRetriever:
    def __init__(self):
        from retrieval.retriever import OpenDomainRetriever
        self.r = OpenDomainRetriever(project_root=_PROJECT_ROOT)
        manifest = os.path.join(_PROJECT_ROOT, "knowledge_corpus", "manifest.json")
        if os.path.exists(manifest):
            self.r.load_from_manifest(manifest_path=manifest)
        else:
            self.r.load_from_manifest()
        self.ready = True
        print(f"[INFO] HybridRetriever 就绪: {len(self.r.chunks)} chunks")

    def retrieve(self, query: str, k: int = 10):
        # A/B 同源检索: hybrid_retrieve 为 query 驱动, 无 capability 过滤参数
        result = self.r.hybrid_retrieve(query, k=k)
        return result.chunks




# ---------------------------------------------------------------------------
# 3. LLM 辅助
# ---------------------------------------------------------------------------
def _llm(client, system, user, temperature=0.2, max_tokens=800):
    try:
        resp = client.chat.completions.create(
            model=LLM_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return ""


FORMAT_DETECT_SYS = (
    "你是工业质检数据标注员。根据题目内容判断其题目类型，只输出 JSON，"
    "格式: {\"format\":\"QA|FillBlank|MultipleChoice|Calculation\",\"confidence\":0.0~1.0}。"
)


def detect_format_llm(client, question):
    user = f"请判断下面这道题的题目类型。\n题目: {question}"
    txt = _llm(client, FORMAT_DETECT_SYS, user, temperature=0.0)
    m = re.search(r"\"format\"\s*:\s*\"([A-Za-z]+)\"", txt)
    c = re.search(r"\"confidence\"\s*:\s*([0-9.]+)", txt)
    fmt = m.group(1) if m else ""
    conf = float(c.group(1)) if c else 0.0
    # 归一化到 FORMUL labels
    alias = {"MultipleChoice": "MultipleChoice", "FillBlank": "FillBlank",
             "Calculation": "Calculation", "QA": "QA", "问答": "QA",
             "填空": "FillBlank", "选择": "MultipleChoice", "计算": "Calculation"}
    fmt = alias.get(fmt, fmt)
    return fmt, conf


# --- 答题提示 (format-定制) ---
FORMAT_ANSWER_PROMPTS = {
    "QA": (
        "请基于【参考依据】直接用中文准确、完整地回答问题。"
        "先给出结论，再补充必要的解释。不要编造资料以外的信息。"
    ),
    "FillBlank": (
        "这是一道填空题。请根据【参考依据】直接给出填空处的准确内容，"
        "只输出该具体数值/短语本身，不要加\"答案是\"\"答案：\"\"'\"等任何引导语，"
        "不要换行解释，除非空内就是一句话。"
    ),
    "MultipleChoice": (
        "这是一道选择题。请根据【参考依据】给出正确选项(如\"A\")及其内容，"
        "并简短说明理由。如选项中多个正确或均不正确请明确说明。"
    ),
    "Calculation": (
        "这是一道计算题。请根据【参考依据】给出计算过程和最终结果，"
        "结果需带单位，并说明所用公式/关键参数。"
    ),
}
UNIFORM_ANSWER_SYS = (
    "你是工业领域技术专家。请根据【参考依据】回答用户问题，"
    "答案需准确、完整、条理清晰。若依据不足请说明。"
)


def build_evidence(docs, k=10):
    """把检索文档列表拼成稳定 evidence 字符串 (与既有 probe 格式一致)。"""
    parts = []
    for i, d in enumerate(docs[:k]):
        content = getattr(d, "content", d.get("content", "") if isinstance(d, dict) else "")
        ind = getattr(d, "industry", "") if not isinstance(d, dict) else d.get("industry", "")
        cap = getattr(d, "capability", "") if not isinstance(d, dict) else d.get("capability", "")
        parts.append(f"[{i+1}] (score≈){getattr(d,'score','') if not isinstance(d,dict) else d.get('score','')} [{ind}/{cap}]\n{content}")
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# 4. 量化评估
# ---------------------------------------------------------------------------
def quant_answer(sample, prediction):
    """返回 rule score + 覆盖指标 + format 结构命中。"""
    from metrics.industrybench_scorer import RuleBasedScorer
    out = {}
    out["rule_score"] = RuleBasedScorer.rule_based_score(
        sample["question"], sample["ref"], prediction
    )
    out["cov"] = RuleBasedScorer.compute_coverage(sample["ref"], prediction) or 0.0
    out["ent_cov"] = RuleBasedScorer.compute_entity_coverage(sample["ref"], prediction) or 0.0


    qf = sample["format"]
    pred = (prediction or "").strip()
    redundant = bool(re.search(r"答案为|答案是|答案为|答案[:：]", pred))
    out["verbose_redundant"] = int(redundant)

    if qf == "FillBlank":
        # 期望简洁回填: 无引导语 且 不含换行解释
        out["struct_hit"] = int((not redundant) and len(pred.splitlines()) <= 2)
    elif qf == "MultipleChoice":
        out["struct_hit"] = int(bool(re.search(r"\b[A-D]\b", pred)))
    elif qf == "Calculation":
        out["struct_hit"] = int(bool(re.search(r"\d", pred)))
    else:
        out["struct_hit"] = int(len(pred) >= 10)
    return out


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16, help="总样本数(按 format 均分)")
    ap.add_argument("--k", type=int, default=10, help="检索 top-k")
    ap.add_argument("--out", type=str, default="results/ab_format_first")
    ap.add_argument("--format_source", type=str, choices=["annotated", "llm"],
                    default="annotated",
                    help="管线B的题型来源: annotated=直接读CSV已标注 `_format`(默认), llm=用LLM判定")
    args = ap.parse_args()


    num_per = max(1, args.samples // 4)
    samples = load_samples(num_per)
    print(f"[INFO] 样本 {len(samples)} 条, 每 format {num_per}")
    print(f"[INFO] CSV: {CSV_PATH}")

    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    retriever = _HybridRetriever()

    # format 判定结果
    detect_rows = []
    # A/B 结果
    ab_rows = []
    start = time.time()
    for idx, s in enumerate(samples):
        print(f"\n[{idx+1}/{len(samples)}] [{s['format']}] {s['question'][:60]}")

        # ---- 统一检索 (A/B 同源) ----
        docs = retriever.retrieve(s["question"], k=args.k)

        evidence = build_evidence(docs, args.k)

        # ---- 管线A: 统一答题 ----
        ansA = _llm(client, UNIFORM_ANSWER_SYS,
                    f"【参考依据】\n{evidence}\n\n【问题】\n{s['question']}",
                    temperature=0.2, max_tokens=1200)

        # ---- 管线B: format 先行 (定制答题) ----
        #        题型来源:
        #          annotated (默认) = 直接读 CSV 已标注 `_format` 真值
        #          llm            = 用 LLM 判定 (保留对比用)
        if args.format_source == "annotated":
            fmt_det, conf = s["format"], 1.0
        else:
            fmt_det, conf = detect_format_llm(client, s["question"])
        sysB = FORMAT_ANSWER_PROMPTS.get(fmt_det, UNIFORM_ANSWER_SYS)
        ansB = _llm(client, sysB,
                    f"【参考依据】\n{evidence}\n\n【问题】\n{s['question']}",
                    temperature=0.2, max_tokens=1200)


        detect_rows.append({
            "id": s["id"], "capability": s["capability"],
            "true_format": s["format"], "pred_format": fmt_det, "confidence": conf,
        })

        qA = quant_answer(s, ansA)
        qB = quant_answer(s, ansB)
        ab_rows.append({
            "id": s["id"], "question": s["question"],
            "format": s["format"], "capability": s["capability"],
            "ansA": ansA, "ansB": ansB,
            "A": qA, "B": qB, "det_format": fmt_det, "conf": conf,
        })
        print(f"    A(rule={qA['rule_score']}) vs B(rule={qB['rule_score']}) | det={fmt_det}({conf:.2f})")

    elapsed = time.time() - start
    print(f"\n[INFO] 总耗时 {elapsed:.1f}s ({elapsed/max(1,len(samples)):.1f}s/样本)")

    # ---- 汇总: format 判定精度 ----
    conf_mat = defaultdict(lambda: defaultdict(int))
    for r in detect_rows:
        conf_mat[r["true_format"]][r["pred_format"]] += 1
    correct = sum(1 for r in detect_rows if r["pred_format"] == r["true_format"])
    acc = correct / max(1, len(detect_rows))
    # 加权 F1 (macro over true formats, 按出现次数加权)
    macro_f1, wsum = 0.0, 0.0
    for tf in FORMATS:
        tp = conf_mat[tf][tf]
        fp = sum(conf_mat[oft][tf] for oft in FORMATS if oft != tf)
        fn = sum(conf_mat[tf][oft] for oft in FORMATS if oft != tf)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        w = sum(r["true_format"] == tf for r in detect_rows)
        macro_f1 += f1 * w
        wsum += w
    weighted_f1 = macro_f1 / max(1, wsum)

    # ---- 汇总: A/B 对比 ----
    def avg(diffs_key):
        return sum(r[diffs_key] for r in ab_rows) / max(1, len(ab_rows))

    A_rule = avg_rule = sum(r["A"]["rule_score"] for r in ab_rows) / max(1, len(ab_rows))
    B_rule = sum(r["B"]["rule_score"] for r in ab_rows) / max(1, len(ab_rows))
    A_cov = sum(r["A"]["cov"] for r in ab_rows) / max(1, len(ab_rows))
    B_cov = sum(r["B"]["cov"] for r in ab_rows) / max(1, len(ab_rows))
    A_ent = sum(r["A"]["ent_cov"] for r in ab_rows) / max(1, len(ab_rows))
    B_ent = sum(r["B"]["ent_cov"] for r in ab_rows) / max(1, len(ab_rows))
    A_struct = sum(r["A"]["struct_hit"] for r in ab_rows) / max(1, len(ab_rows))
    B_struct = sum(r["B"]["struct_hit"] for r in ab_rows) / max(1, len(ab_rows))
    A_verb = sum(r["A"]["verbose_redundant"] for r in ab_rows) / max(1, len(ab_rows))
    B_verb = sum(r["B"]["verbose_redundant"] for r in ab_rows) / max(1, len(ab_rows))

    # 分层 by format
    layer = {}
    for qf in FORMATS:
        sub = [r for r in ab_rows if r["format"] == qf]
        if not sub:
            continue
        layer[qf] = {
            "n": len(sub),
            "A_rule": sum(r["A"]["rule_score"] for r in sub) / len(sub),
            "B_rule": sum(r["B"]["rule_score"] for r in sub) / len(sub),
            "A_struct": sum(r["A"]["struct_hit"] for r in sub) / len(sub),
            "B_struct": sum(r["B"]["struct_hit"] for r in sub) / len(sub),
            "A_verb": sum(r["A"]["verbose_redundant"] for r in sub) / len(sub),
            "B_verb": sum(r["B"]["verbose_redundant"] for r in sub) / len(sub),
        }

    # ---- 输出报告 ----
    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = []
    rep = report.append
    rep("=" * 72)
    rep("Format-First 答题定制 A/B 量化报告")
    rep(f"样本: {len(samples)}, 检索k={args.k}, 模型={LLM_NAME}, 耗时{elapsed:.0f}s")
    rep("=" * 72)
    if args.format_source == "annotated":
        rep("\n【A】题型来源: 直接读取 CSV 已标注 `_format` 真值 (非LLM判定)")
        rep("  说明: 题型划分阶段直接采用数据集 `_format` 列标注, 不走 LLM;")
        rep("  此处正确率=100% 属预期的链路一致性校验 (标注→答题题型分发生效)。")
    else:
        rep("\n【A】Format 判定精度 (LLM判定 vs 标注 _format)")
        rep(f"  总体准确率: {acc*100:.1f}% ({correct}/{len(detect_rows)})")
        rep(f"  加权F1:     {weighted_f1*100:.1f}%")
        rep("  混淆矩阵 (行=真实, 列=预测):")
        rep("    " + "真实行→" + "".join(f"{FORMAT_LABELS[c]:<12}" for c in FORMATS))
        for tf in FORMATS:
            rep("    " + f"{FORMAT_LABELS[tf]:<12}" + "".join(f"{conf_mat[tf][c]:<12}" for c in FORMATS))




    rep("\n【B】答题贴合度 (A=统一提示, B=format定制), 检索同源")
    rep(f"  {'指标':<22}{'A(基线)':>10}{'B(format)':>12}{'Δ':>8}")
    def row(name, a, b):
        rep(f"  {name:<22}{a*100 if name!='rule(0-3)' else a:>10.3f}"
            f"{b*100 if name!='rule(0-3)' else b:>12.3f}"
            f"{((b-a)*100 if name!='rule(0-3)' else b-a):>+8.3f}")
    rep(f"  {'rule_score(0-3)':<22}{A_rule:>10.3f}{B_rule:>12.3f}{B_rule-A_rule:>+8.3f}")
    rep(f"  {'关键词覆盖 cov%':<22}{A_cov*100:>9.1f}{B_cov*100:>11.1f}{(B_cov-A_cov)*100:>+8.1f}")
    rep(f"  {'实体覆盖 ent_cov%':<22}{A_ent*100:>9.1f}{B_ent*100:>11.1f}{(B_ent-A_ent)*100:>+8.1f}")
    rep(f"  {'结构命中 struct%':<22}{A_struct*100:>9.1f}{B_struct*100:>11.1f}{(B_struct-A_struct)*100:>+8.1f}")
    rep(f"  {'冗余引导语 verb%':<22}{A_verb*100:>9.1f}{B_verb*100:>11.1f}{(B_verb-A_verb)*100:>+8.1f}")

    rep("\n【C】按题型分层 (rule 分 / struct% / verb%)")
    for qf in FORMATS:
        if qf not in layer:
            continue
        L = layer[qf]
        rep(f"  {FORMAT_LABELS[qf]:<10} n={L['n']:<3} "
            f"rule A={L['A_rule']:.2f}→B={L['B_rule']:.2f}  "
            f"struct A={L['A_struct']*100:.0f}%→B={L['B_struct']*100:.0f}%  "
            f"verb A={L['A_verb']*100:.0f}%→B={L['B_verb']*100:.0f}%")

    rep("\n【D】结论解读提示")
    rep("  - 若 B.rule / esp. FillBlank verb↓ & struct↑ 明显 → format 定制答题有实质增益")
    rep("  - 若 A≈B → format 只影响表述风格, 不影响内容分; 则 format 定为低价值前端")
    rep("  - format 判定若 acc<80% → 即使后端有增益, 建议用句法/heuristic 或融合真值, 不依赖 LLM")

    text = "\n".join(report)
    print("\n" + text)

    # 写文件
    with open(os.path.join(args.out, f"ab_format_first_{ts}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, f"ab_rows_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(ab_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, f"detect_rows_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(detect_rows, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.out}/")


if __name__ == "__main__":
    main()
