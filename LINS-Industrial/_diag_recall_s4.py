# -*- coding: utf-8 -*-
"""
_diag_recall_s4.py — S4 检索阶段专项排查（分而治之 · 阶段4）。

目标（对齐 docs/stagewise_debug_plan.md 的 S4）：
  判定: semantic Hit@1 >= 0.7 且各 capability 差异 <20%（否则继续查召回）。

输出（results/s4_recall/）：
  A) 方法对比  single / multi / hybrid  k=10：主口径 cov@10(句子覆盖率)/good% + 交叉参照
     hit@1/3(整段IoU)、MRR、NDCG（官方 RetrievalEvaluator）。
  B) 按 capability 分层：cov@10 / good% / hit@1 与"问题占比/chunk占比"对照；能力间差异判定。
  C) 按 industry 分层：hit@1 / cov@10。
  D) k-sweep（hybrid, k=5/10/20/50）：验证 k 增大是"真实拉回更多 GT"还是"仅稀释 top 分"。
  E) format 正交性：说明（来自 _probe_format_misread 结论）。

口径说明（对齐仓库 recall_metrics.py P0 规范）：
  主验收 = GT 句子覆盖率 sent_coverage()（避免整段 IoU 对长 GT 的系统性低估，见
  docs/agentic_recall_completeness_correction.md / agentic_recall_entrance_diagnosis.md）；
  整段 IoU hit@k 仅作交叉参照，不作结论盖章。

Ground truth: 每题自带 knowledge_text；相关 chunk = 与 kt 的汉字 bigram IoU >= IOU_TH。
用法:
  python _diag_recall_s4.py --samples 70 [--full] [--methods single,multi,hybrid]
"""
import os, sys, csv, json, time, random, argparse
from collections import Counter, defaultdict

# 修复 Windows GBK 控制台无法打印 emoji 的问题（输出重定向为 UTF-8）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
MANIFEST = os.path.join(_LINS, "knowledge_corpus", "manifest.json")
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
OUT = os.path.join(_LINS, "results", "s4_recall")

IOU_TH = 0.15          # 整段 IoU 命中阈值（交叉参照口径）
GOOD_COV = 0.5         # 句子覆盖率\"拼齐答案\"阈值（主验收口径，与 recall_metrics.GOOD_COV 对齐）
CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]

sys.path.insert(0, _LINS)
from functools import lru_cache

# 统一口径模块（P0 规范）：主验收 = GT 句子覆盖率；整段 IoU 仅作交叉参照
from retrieval.recall_metrics import sent_coverage, doc_hit_position


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def first_hit_rank(chunks, kt):
    """【交叉参照·整段 IoU】首个与 kt 整段 IoU 命中的 chunk 位次(1-based)，无则 None。"""
    for i, c in enumerate(chunks, 1):
        content = getattr(c, "content", "") or ""
        if iou(kt, content) >= IOU_TH:
            return i
    return None


def cover_at_k(chunks, kt):
    """【主验收口径】给定 top-k 的 GT 句子覆盖率 cov(0~1)；采用 recall_metrics.complete。
    cov>=GOOD_COV(0.5) 视为该题\"拼得齐答案\"。"""
    cov, _, _ = sent_coverage(kt, chunks)
    return cov, (cov >= GOOD_COV)


class RelEvaluator:
    """按 RetrievalEvaluator 官方口径评估（recall@k / MRR / NDCG），
    相关集合 = knowledge_text 的 IoU 命中 chunk_id 集合。"""

    def __init__(self, k_values=(1, 3, 10)):
        from eval_scripts.industrial_linkeval.retrieval_evaluator import RetrievalEvaluator
        self.ev = RetrievalEvaluator(k_values=list(k_values))

    def per_query(self, chunks, relevant_ids):
        rids = [c.chunk_id for c in chunks]
        return self.ev.evaluate(rids, set(relevant_ids))

    def batch(self, queries):
        rid_sets = [r for r, _ in queries]
        rel_sets = [set(r) for _, r in queries]
        return self.ev.evaluate_batch(rid_sets, rel_sets)


def load_questions():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stratify_sample(rows, per_cap):
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap in CAPS:
        picked = by_cap.get(cap, [])
        if not picked:
            continue
        sample += random.sample(picked, min(per_cap, len(picked)))
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=70, help="总样本数(按 capability 均分)")
    ap.add_argument("--per_cap", type=int, default=None, help="每 capability 抽样数(覆盖 --samples)")
    ap.add_argument("--methods", type=str, default="single,multi,hybrid")
    ap.add_argument("--ksweep", type=str, default="5,10,20,50")
    ap.add_argument("--out", type=str, default=OUT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    methods = [m.strip() for m in args.methods.split(",")]
    ksweep = [int(x) for x in args.ksweep.split(",")]
    per_cap = args.per_cap or max(1, args.samples // 7 + (1 if args.samples % 7 else 0))

    rows_full = load_questions()
    sample = stratify_sample(rows_full, per_cap)
    print(f"[样本] {len(sample)} 题; capability 分布={dict(Counter(r['capability'] for r in sample))}")

    # 语料层 同能力/同行业 chunk 占比
    c_cap, c_ind = Counter(), Counter()
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d = json.loads(s)
            c_cap[d.get("capability", "") or ""] += 1
            c_ind[d.get("industry", "") or ""] += 1
    q_cap = Counter((r.get("capability") or "") for r in sample)

    from retrieval.retriever import OpenDomainRetriever
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=MANIFEST)
    except Exception as e:
        print("manifest 加载失败，尝试默认:", e)
        retr.load_from_manifest()
    print(f"[检索器] {len(retr.chunks)} chunks 就绪")
    retr.retrieve("预热", k=1)   # 触发模型加载后再计时

    ev = RelEvaluator()

    rows_detail = []
    agg = {}
    for m in methods:
        agg[m] = {cap: {"n": 0, "hit1": 0, "hit3": 0, "hit10": 0,
                        "mrr": 0.0, "ndcg": 0.0, "same_cap": 0, "cap_slots": 0,
                        "cov10": 0.0, "good10": 0}
                  for cap in CAPS}
    kagg = {"n": 0, "hit": {k: 0 for k in ksweep}}

    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        cap = r.get("capability", ""); ind = r.get("industry_primary", "")
        if not q or not kt:
            continue
        detail = {"id": r.get("id", ""), "cap": cap, "ind": ind,
                  "q": q[:60], "methods": {}}
        for m in methods:
            try:
                if m == "single":
                    res = retr.retrieve(q, k=10, use_ked=True)
                elif m == "multi":
                    res = retr.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion")
                else:
                    res = retr.hybrid_retrieve(q, k=10, use_ked=True)
            except Exception as e:
                print(f"  [{m}] {idx} err: {e}")
                continue
            chunks = getattr(res, "chunks", []) or []
            rel_ids = [c.chunk_id for c in chunks if iou(kt, c.content) >= IOU_TH]
            rank = first_hit_rank(chunks, kt)
            cov10, good10 = cover_at_k(chunks, kt)
            a = agg[m][cap]
            a["n"] += 1
            if rank and rank <= 1: a["hit1"] += 1
            if rank and rank <= 3: a["hit3"] += 1
            if rank and rank <= 10: a["hit10"] += 1
            if rank: a["mrr"] += 1.0 / rank
            rq = ev.per_query(chunks, rel_ids)
            a["ndcg"] += rq.ndcg
            a["same_cap"] += sum(1 for c in chunks if c.capability == cap)
            a["cap_slots"] += len(chunks)
            a["cov10"] += cov10
            a["good10"] += int(good10)
            detail["methods"][m] = {"rank": rank, "n_rel_in_pool": len(rel_ids),
                                    "cov10": round(cov10, 3), "good10": bool(good10)}
        try:
            for k in ksweep:
                res = retr.hybrid_retrieve(q, k=k, use_ked=True)
                rank = first_hit_rank(res.chunks, kt)
                if rank and rank <= k:
                    kagg["hit"][k] += 1
            kagg["n"] += 1
        except Exception as e:
            print(f"  [ksweep] {idx} err: {e}")
        rows_detail.append(detail)
        if idx % 10 == 0:
            print(f"  .. 进度 {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    # ---- 聚合汇总 ----
    m0 = methods[0]
    present = {c: agg[m0][c] for c in CAPS if agg[m0][c]["n"] > 0}
    hits1 = [a["hit1"] / a["n"] for a in present.values()]
    overall1 = sum(hits1) / len(present) if present else 0.0
    spread = (max(hits1) - min(hits1)) if hits1 else 1.0

    rep = []
    R = rep.append
    R("=" * 80)
    R(f"S4 检索阶段专项排查 | 样本 {len(sample)} 题 (每cap≈{per_cap}) | IOU={IOU_TH} | {ts}")
    R("=" * 80)

    R("\n【A】方法对比 (k=10)：single / multi / hybrid")
    R("  主口径: cov@10(句子覆盖率) / good%(cov>=0.5拼齐率)；交叉参照: hit@1/3(整段IoU), MRR, NDCG")
    R(f"  {'方法':<8}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'cov@10':>8}{'good%':>8}{'MRR':>8}{'NDCG':>8}")
    for m in methods:
        a = {c: agg[m][c] for c in CAPS}
        n = sum(x["n"] for x in a.values())
        hit1 = sum(x["hit1"] for x in a.values()) / max(1, n)
        hit3 = sum(x["hit3"] for x in a.values()) / max(1, n)
        mrr = sum(x["mrr"] for x in a.values()) / max(1, n)
        ndcg = sum(x["ndcg"] for x in a.values()) / max(1, n)
        cov10 = sum(x["cov10"] for x in a.values()) / max(1, n)
        good = sum(x["good10"] for x in a.values()) / max(1, n) * 100
        R(f"  {m:<8}{n:>4}{hit1*100:>7.1f}%{hit3*100:>7.1f}%{cov10*100:>7.1f}%{good:>7.0f}%{mrr*100:>7.1f}%{ndcg*100:>7.1f}%")

    R(f"\n【B】按 capability 分层 (方法={m0}, 主口径=cov@10)")
    R(f"  {'能力':<14}{'样本':>4}{'hit@1':>7}{'cov@10':>8}{'good%':>7}{'同能力率':>9}{'问题占比':>8}{'chunk占比':>8}")
    for cap in CAPS:
        a = agg[m0].get(cap)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        same = a["same_cap"] / max(1, a["cap_slots"])
        qp = q_cap.get(cap, 0) / max(1, len(sample)) * 100
        cp = c_cap.get(cap, 0) / max(1, sum(c_cap.values())) * 100
        R(f"  {cap:<14}{n:>4}{a['hit1']/n*100:>6.0f}%{a['cov10']/n*100:>7.1f}%"
          f"{a['good10']/n*100:>6.0f}%{same*100:>8.0f}%{qp:>7.0f}%{cp:>7.0f}%")
    covs = [a["cov10"] / a["n"] for a in present.values()]
    goods = [a["good10"] / a["n"] for a in present.values()]
    R(f"\n  overall({m0}): hit@1={overall1*100:.1f}% | cov@10均值={sum(covs)/len(covs)*100 if covs else 0:.1f}% | "
      f"good%(cov>=0.5)={sum(goods)/len(goods)*100 if goods else 0:.1f}% | "
      f"能力间hit@1 max-min={spread*100:.1f}%")
    # 主判定：主口径 cov 与交叉参照 hit@1 均未达阈值视为继续查召回
    cov_mean = sum(covs) / len(covs) if covs else 0.0
    R(f"  判定(整段IoU口径 hit@1>=0.7 且 差异<20%): "
      f"{'[PASS]' if (overall1 >= 0.7 and spread < 0.20) else '[FAIL] 继续查召回'}")
    R(f"  主口径参照: 句子覆盖率 cov@10={cov_mean*100:.1f}%（补全度，与 agentic_recall_completeness_correction 同口径，用于定位\"召回 vs 补全\")")

    R("\n【C】按 industry 分层 (hit@1 / hit@10)")
    ind_agg = defaultdict(lambda: {"n": 0, "h1": 0, "h10": 0})
    for d in rows_detail:
        if m0 not in d["methods"]:
            continue
        rk = d["methods"][m0].get("rank")
        ia = ind_agg[d["ind"] or "(空)"]
        ia["n"] += 1
        if rk and rk <= 1: ia["h1"] += 1
        if rk and rk <= 10: ia["h10"] += 1
    for ind, ia in sorted(ind_agg.items(), key=lambda x: -x[1]["n"]):
        R(f"  {ind:<20} n={ia['n']:<3} hit@1={ia['h1']/ia['n']*100:>5.0f}%  hit@10={ia['h10']/ia['n']*100:>5.0f}%")

    R("\n【D】k-sweep (hybrid)：GT 累计命中 5/10/20/50")
    R("  " + "".join(f"  @{k:<5}" for k in ksweep))
    R("  " + "".join(f"  {kagg['hit'][k]/max(1,kagg['n'])*100:>4.0f}%" for k in ksweep) +
      f"  (n={kagg['n']})")
    R("  -- 若 @20 明显 > @10 且 @10<0.7，说明 GT 需更大候选池(召回缺口在排序而非池内)")

    R("\n【E】format 正交性核验（来自 _probe_format_misread 结论）")
    R("  -- format 身份可检索(近原文内容 100%)；真实瓶颈是问题措辞与 format 标注语义失配，")
    R("  -- format 仅作二级过滤字段，不并入检索主轴。")

    text = "\n".join(rep)
    print(text)
    md_path = os.path.join(args.out, f"s4_recall_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    json_path = os.path.join(args.out, f"s4_rows_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows_detail, "kagg": kagg,
                   "agg_by_method_byn_cap": {m: {c: dict(agg[m][c]) for c in CAPS}
                                             for m in methods}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {md_path}")


if __name__ == "__main__":
    main()

