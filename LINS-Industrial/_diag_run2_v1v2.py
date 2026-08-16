# -*- coding: utf-8 -*-
"""_diag_run2_v1v2.py — 落地前对比：run2 v1 vs v2 的 recall 指标（与 S4 完全同口径）。

在**同一批样本**上，对同一 ranker 分别跑 v1 / v2，输出与 `_diag_recall_run2.py` 一致
的 S4 口径：主口径 cov@10 + good%；交叉 hit@1/hit@3/MRR/NDCG；附 v2 回捞块进 top-k 累计。
注意：LLM 打分在 v2 下需 max_tokens=2000（否则长候选被截断、回捞块丢分）。

用法:
  python _diag_run2_v1v2.py --per_cap 1 --pool 50            # 零成本 Sim（快）
  python _diag_run2_v1v2.py --per_cap 1 --pool 50 --llm      # 真实 DeepSeek(max_tokens=2000)
"""
import csv, os, sys, time, random, argparse, json
from collections import Counter
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["TRANSFORMERS_OFFLINE"] = "1"
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

from _diag_recall_run2 import (
    iou, first_hit_rank, cover_at_k, RelEvaluator,
    stratify_sample, load_questions, LLMRelevanceRanker, SimRelevanceRanker,
)
from retrieval.retriever import OpenDomainRetriever

MANIFEST = os.path.join(_LINS, "knowledge_corpus", "manifest.json")
OUT_DIR = os.path.join(_LINS, "results", "run2_v1v2")
CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cap", type=int, default=1)
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="用真实 DeepSeek(默认Sim)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    sample = stratify_sample(load_questions(), args.per_cap)
    print(f"[样本] {len(sample)} 题; cap={dict(Counter(r['capability'] for r in sample))}")

    from experiments.config import LLM_NAME, DEEPSEEK_KEY, register_paths
    register_paths()
    retr = OpenDomainRetriever(project_root=_LINS)
    retr.load_from_manifest(manifest_path=MANIFEST)
    retr.retrieve("预热", k=1)

    ranker = (LLMRelevanceRanker(api_key=DEEPSEEK_KEY, model=LLM_NAME, max_tokens=2000)
              if args.llm else SimRelevanceRanker(retr))

    agg = {m: {cap: {"n": 0, "hit1": 0, "hit3": 0, "hit10": 0, "mrr": 0.0,
                     "ndcg": 0.0, "cov10": 0.0, "good10": 0, "bf_in": 0}
               for cap in CAPS} for m in ["v1", "v2", "v3", "v4"]}
    ev = RelEvaluator()
    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q, kt, cap = r.get("question", ""), r.get("knowledge_text", ""), r.get("capability", "")
        if not q or not kt:
            continue
        meth = {
            "v1": lambda: retr.two_stage_retrieve(q, k=10, pool_k=args.pool,
                                                  n_anchor=args.n_anchor, ranker=ranker),
            "v2": lambda: retr.two_stage_retrieve_v2(q, k=10, pool_k=args.pool,
                                                     n_anchor=args.n_anchor, ranker=ranker),
            "v3": lambda: retr.two_stage_retrieve_v3(q, k=10, pool_k=args.pool,
                                                     n_anchor=args.n_anchor, ranker=ranker),
            "v4": lambda: retr.two_stage_retrieve_v4(q, k=10, pool_k=args.pool,
                                                     n_anchor=args.n_anchor, ranker=ranker),
        }
        import re
        for m, fn in meth.items():
            res = fn()
            _accum(agg, m, cap, res, kt, ev)
            mm = re.search(r"bf_in_topk=(\d+)", res.query_expanded)
            if mm:
                agg[m][cap]["bf_in"] += int(mm.group(1))
        if args.llm:
            print(f"  .. id={r['id']} (累计{time.time()-t0:.0f}s)")
        elif idx % 3 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    _report(agg, sample, args, ts, ranker)




def _accum(a, m, cap, res, kt, ev):
    chunks = list(getattr(res, "chunks", [])) or []
    rel_ids = [c.chunk_id for c in chunks if iou(kt, getattr(c, "content", "") or "") >= 0.15]
    rank = first_hit_rank(chunks, kt)
    cov10, good10 = cover_at_k(chunks, kt)
    x = a[m][cap]
    x["n"] += 1
    if rank and rank <= 1: x["hit1"] += 1
    if rank and rank <= 3: x["hit3"] += 1
    if rank and rank <= 10: x["hit10"] += 1
    if rank: x["mrr"] += 1.0 / rank
    x["ndcg"] += ev.per_query(chunks, rel_ids).ndcg
    x["cov10"] += cov10
    x["good10"] += int(good10)


def _report(agg, sample, args, ts, ranker):
    METHODS = ["v1", "v2", "v3", "v4"]
    R = []
    R.append("=" * 94)
    R.append(f"run2 v1/v2/v3/v4 | 样本 {len(sample)} 题({args.per_cap}/cap) | "
             f"pool={args.pool} anchor={args.n_anchor}(v3/v4=去重文档篇数) | "
             f"ranker={'LLM' if args.llm else 'Sim'} | {ts}")
    R.append("=" * 88)
    R.append("\n【A】方法对比 (k=10) —— 主口径 cov@10 + good%；交叉 hit/MRR/NDCG")
    R.append(f"  {'方法':<6}{'n':>4}{'cov@10':>8}{'good%':>7}{'hit@1':>7}{'hit@3':>7}"
             f"{'MRR':>7}{'NDCG':>8}  (回捞进top10合计)")
    for m in METHODS:
        n = sum(agg[m][c]["n"] for c in CAPS)
        if n == 0:
            continue
        cov = sum(agg[m][c]["cov10"] for c in CAPS) / n
        good = sum(agg[m][c]["good10"] for c in CAPS) / n
        h1 = sum(agg[m][c]["hit1"] for c in CAPS) / n
        h3 = sum(agg[m][c]["hit3"] for c in CAPS) / n
        mrr = sum(agg[m][c]["mrr"] for c in CAPS) / n
        ndcg = sum(agg[m][c]["ndcg"] for c in CAPS) / n
        bf = sum(agg[m][c]["bf_in"] for c in CAPS)
        R.append(f"  {m:<6}{n:>4}{cov*100:>7.1f}%{good*100:>6.0f}%{h1*100:>6.0f}%"
                 f"{h3*100:>6.0f}%{mrr*100:>6.1f}%{ndcg*100:>7.1f}%  {bf:>4}")

    R.append("\n【B】按 capability 分层 (cov@10 / hit@1)")
    R.append(f"  {'能力':<12}{'n':>4}{'v1':>13}{'v2':>13}{'v3':>13}{'v4':>13}")
    for cap in CAPS:
        if not any(agg[m][cap]["n"] for m in METHODS):
            continue
        l = f"  {cap:<12}"
        for m in METHODS:
            a = agg[m][cap]
            if a["n"] == 0:
                l += " " * 13
            else:
                l += f"  {a['cov10']/a['n']*100:>4.0f}%/{a['hit1']/a['n']*100:>3.0f}%"
        R.append(l)

    n = max(1, sum(agg["v1"][c]["n"] for c in CAPS))
    R.append("\n【C】差值与 v1 比 (PP)")
    for mm in ["v2", "v3", "v4"]:
        def _d(key):
            return (sum(agg[mm][c][key] for c in CAPS) - sum(agg["v1"][c][key] for c in CAPS)) / n
        R.append(f"  {mm}-v1: Δcov={_d('cov10')*100:+.1f}pp Δgood={_d('good10')*100:+.1f}pp "
                 f"Δhit1={_d('hit1')*100:+.1f}pp Δhit3={_d('hit3')*100:+.1f}pp "
                 f"ΔMRR={_d('mrr')*100:+.1f}pp ΔNDCG={_d('ndcg')*100:+.1f}pp")

    text = "\n".join(R)
    print(text)
    md = os.path.join(OUT_DIR, f"run2_v1v2_{ts}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    js = os.path.join(OUT_DIR, f"run2_v1v2_{ts}.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "agg_by_cap":
                   {m: {c: dict(agg[m][c]) for c in CAPS} for m in METHODS}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {md}")
    print(f"[SAVED] {js}")


if __name__ == "__main__":
    main()

