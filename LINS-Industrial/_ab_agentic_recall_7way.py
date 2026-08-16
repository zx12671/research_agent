# -*- coding: utf-8 -*-
"""_ab_agentic_recall_7way.py — 真实 agentic 链路检索方法 7 路 recall 定量对比。

在同一批样本上，用与生产 agentic 完全一致的底层检索方法，输出统一的 recall 指标表。

方法（= 生产 retrievable 通路，k=10）：
  single: retr.retrieve(q,k=10,use_ked=True)            —— base 纯 top10
  multi : multi_query_retrieve(strategy='fusion')        —— 切逗号子查询 + fusion
  hybrid: hybrid_retrieve(q,k=10,use_ked=True)           —— dense(KED)+BM25 → RRF
  v1    : two_stage_retrieve        (回捞块 rel 恒 0, 补全进不了 top-k)
  v2    : two_stage_retrieve_v2     (回捞块吃 rel, 3 锚全吃 bonus 早期版)
  v3    : two_stage_retrieve_v3     (去重 3 篇、3 篇全吃 +0.5 anchor_bonus)
  v4    : two_stage_retrieve_v4     (去重 3 篇、bonus 只给 top1 —— 生产默认)

指标与 _diag_recall_run2.py 完全一致：主口径 cov@10 + good%；交叉 hit@1/3/10、
MRR、NDCG；相关集合 = chunk 与 knowledge_text bigram IoU>=IOU_TH。

用法:
  python _ab_agentic_recall_7way.py --n_total 50 --sim      # Sim ranker（零成本快）
  python _ab_agentic_recall_7way.py --n_total 50 --llm      # 真实 DeepSeek
  python _ab_agentic_recall_7way.py --n_total 50 --llm --only v1 v4
输出: results/agentic_recall_7way/agentic_recall_7way_{ts}.md + .json
"""
import os, sys, json, time, random, argparse
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from _diag_recall_run2 import (
    load_questions, stratify_sample, RelEvaluator, IOU_TH,
    LLMRelevanceRanker, SimRelevanceRanker,
)
from retrieval.retriever import OpenDomainRetriever

MANIFEST = os.path.join(_LINS, "knowledge_corpus", "manifest.json")
OUT_DIR = os.path.join(_LINS, "results", "agentic_recall_7way")
CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]
ALL_METHODS = ["single", "multi", "hybrid", "v1", "v2", "v3", "v4"]


def _dispatch(retr, m, q, args, ranker):
    if m == "single":
        return retr.retrieve(q, k=10, use_ked=True).chunks
    if m == "multi":
        return retr.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion").chunks
    if m == "hybrid":
        return retr.hybrid_retrieve(q, k=10, use_ked=True).chunks
    kw = dict(k=10, pool_k=args.pool, n_anchor=args.n_anchor, ranker=ranker)
    fn = {"v1": retr.two_stage_retrieve, "v2": retr.two_stage_retrieve_v2,
          "v3": retr.two_stage_retrieve_v3, "v4": retr.two_stage_retrieve_v4}[m]
    return fn(q, **kw).chunks


def _cum(agg, m, cap, chunks):
    """与 _diag_recall_run2._accumulate 同口径的指标累计（chunks 实际是 res.chunks）。"""
    from _diag_recall_run2 import iou, first_hit_rank, cover_at_k
    kt = agg["__kt__"]
    chunks = list(chunks) or []
    rel_ids = [c.chunk_id for c in chunks if iou(kt, getattr(c, "content", "") or "") >= IOU_TH]
    rank = first_hit_rank(chunks, kt)
    cov10, good10 = cover_at_k(chunks, kt)
    a = agg[m][cap]
    a["n"] += 1
    if rank and rank <= 1: a["hit1"] += 1
    if rank and rank <= 3: a["hit3"] += 1
    if rank and rank <= 10: a["hit10"] += 1
    if rank: a["mrr"] += 1.0 / rank
    a["ndcg"] += agg["__ev__"].per_query(chunks, rel_ids).ndcg
    a["cov10"] += cov10
    a["good10"] += int(good10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=50, help="总样本数（cap 均衡抽取）")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--sim", action="store_true", help="Sim ranker（零成本）")
    ap.add_argument("--llm", action="store_true", help="真实 DeepSeek ranker")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*", choices=ALL_METHODS, default=None,
                    help="只跑指定方法，缺省全部 7 路")
    args = ap.parse_args()

    methods = args.only or ALL_METHODS
    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # cap 均衡抽取到 n_total（每 cap 取向上取整再截断）
    rows_full = load_questions()
    per_cap = -(-args.n_total // len(CAPS))
    sample = stratify_sample(rows_full, per_cap)[:args.n_total]
    print(f"[样本] {len(sample)} 题; cap={dict(Counter(r['capability'] for r in sample))}")

    from experiments.config import LLM_NAME, DEEPSEEK_KEY, register_paths
    register_paths()

    retr = OpenDomainRetriever(project_root=_LINS)
    retr.load_from_manifest(manifest_path=MANIFEST)
    retr.retrieve("预热", k=1)
    print(f"[检索器] {len(retr.chunks)} chunks")

    assert args.sim or args.llm, "需指定 --sim 或 --llm"
    ranker = (LLMRelevanceRanker(api_key=DEEPSEEK_KEY, model=LLM_NAME, max_tokens=2000)
              if args.llm else SimRelevanceRanker(retr))

    agg = {m: {cap: {"n": 0, "hit1": 0, "hit3": 0, "hit10": 0, "mrr": 0.0,
                     "ndcg": 0.0, "cov10": 0.0, "good10": 0} for cap in CAPS}
           for m in methods}
    agg["__ev__"] = RelEvaluator()

    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q, kt, cap = r.get("question", ""), r.get("knowledge_text", ""), r.get("capability", "")
        if not q or not kt:
            continue
        agg["__kt__"] = kt
        for m in methods:
            try:
                chunks = _dispatch(retr, m, q, args, ranker)
            except Exception as e:
                print(f"  [{m}] err: {e!r}")
                continue
            _cum(agg, m, cap, chunks)
        if idx % max(1, args.n_total // 5) == 0 or args.llm:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")
    del agg["__ev__"], agg["__kt__"]

    _report(agg, methods, sample, args, ts)


def _report(agg, methods, sample, args, ts):
    R = []
    R.append("=" * 100)
    R.append(f"agentic 检索 7 路 recall | 样本 {len(sample)} 题 | pool={args.pool} "
             f"anchor={args.n_anchor} | ranker={'LLM' if args.llm else 'Sim'} | "
             f"IOU={IOU_TH} | {ts}")
    R.append("=" * 100)
    R.append("\n【A】方法对比 (k=10) —— 主口径 cov@10 + good%；交叉 hit/MRR/NDCG")
    R.append(f"  {'方法':<8}{'n':>4}{'cov@10':>8}{'good%':>7}{'hit@1':>7}{'hit@3':>7}"
             f"{'hit@10':>8}{'MRR':>7}{'NDCG':>8}")

    def _row(m):
        n = sum(agg[m][c]["n"] for c in CAPS)
        if n == 0:
            return None

        def _v(key):
            return sum(agg[m][c][key] for c in CAPS) / n
        return (f"  {m:<8}{n:>4}{_v('cov10')*100:>7.1f}%{_v('good10')*100:>6.0f}%"
                f"{_v('hit1')*100:>6.0f}%{_v('hit3')*100:>6.0f}%{_v('hit10')*100:>7.0f}%"
                f"{_v('mrr')*100:>6.1f}%{_v('ndcg')*100:>7.1f}%")

    for m in methods:
        row = _row(m)
        if row:
            R.append(row)

    R.append("\n【B】按 capability 分层 (cov@10 / hit@1)")
    cols = [m for m in methods if any(agg[m][c]["n"] for c in CAPS)]
    R.append("  " + f"{'能力':<12}{'n':>4} " + " ".join(f"{m[:7]:>11}" for m in cols))
    for cap in CAPS:
        if sum(agg[m][cap]["n"] for m in methods) == 0:
            continue
        line = f"  {cap:<12}"
        for m in cols:
            a = agg[m][cap]
            if a["n"] == 0:
                line += " " * 11
            else:
                line += f"  {a['cov10']/a['n']*100:>5.0f}/{a['hit1']/a['n']*100:>3.0f}"
        R.append(line)

    if "v1" in methods:
        n = max(1, sum(agg["v1"][c]["n"] for c in CAPS))
        R.append("\n【C】差值与 v1 比 (PP)")

        def _d(m, key):
            return (sum(agg[m][c][key] for c in CAPS) - sum(agg["v1"][c][key] for c in CAPS)) / n
        for mm in [m for m in methods if m != "v1"]:
            R.append(f"  {mm}-v1: Δcov={_d(mm,'cov10')*100:+.1f}pp "
                     f"Δgood={_d(mm,'good10')*100:+.1f}pp "
                     f"Δhit1={_d(mm,'hit1')*100:+.1f}pp Δhit10={_d(mm,'hit10')*100:+.1f}pp "
                     f"ΔMRR={_d(mm,'mrr')*100:+.1f}pp ΔNDCG={_d(mm,'ndcg')*100:+.1f}pp")

    text = "\n".join(R)
    print(text)
    md = os.path.join(OUT_DIR, f"agentic_recall_7way_{ts}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    js = os.path.join(OUT_DIR, f"agentic_recall_7way_{ts}.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "IOU": IOU_TH,
                   "agg_by_cap": {m: {c: dict(agg[m][c]) for c in CAPS} for m in methods}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {md}\n[SAVED] {js}")


if __name__ == "__main__":
    main()

