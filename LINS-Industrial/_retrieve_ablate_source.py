# -*- coding: utf-8 -*-
"""
retrieve 召回增益归因 + 句子覆盖率诊断。

背景:
 - 上一轮 `_retrieve_single_vs_hybrid.py` 用「整段 IoU>=0.20」判命中，发现
   hybrid(单查询,dense+BM25) 的 hit@1/@3 高于 pure dense。
 - docs/agentic_recall_completeness_correction.md 指出整段 IoU 口径严重低估，
   主张用「GT 句子覆盖率」；并称 BM25 在中文语料上零增益。
 - 本脚本用控制变量法把「hit@@ 提升」归因到 BM25 与 dense 扩容(top-20) 两个来源，
   同时用句子覆盖率口径重新诊断「top-50 外丢失」的真实构成(A/B/C 三类)。

四组对照 (同一 30 题, K=10/20/50):
  A. single       = retrieve(q, use_ked=True)      # 现状, dense k
  B. hybrid       = hybrid_retrieve(sparse=True)   # 落地: dense(max(20)) + BM25 RRF
  C. hybrid_nospar = hybrid_retrieve(sparse=False) # 去 BM25, 只看 dense扩容+RRF
  D. dense_k50    = dense_search(k)                # 放大 dense 池, 诊断 recall 入口

输出: 句覆盖率(mean/median/cov>=50%) x 入口 x K; 归因拆分; 逐题 A/B/C 归类。
"""
import os
import csv
import json
import time
import random
import re
import statistics
from collections import Counter, defaultdict

from retrieval.recall_metrics import sent_coverage, get_content as getc

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7
TH_SHOW = 0.5

def sample_questions():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    s = []
    for cap, lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP, len(lst)))
    return s[:N]


def main():
    sample = sample_questions()
    print(f"[样本] {len(sample)} 题 | cap: {dict(Counter(r.get('capability') for r in sample))}")
    from retrieval.retriever import OpenDomainRetriever
    r = OpenDomainRetriever(); r.load_from_manifest()
    HYB = dict(use_ked=True, use_multi_query=False, sparse_pool=50)

    methods_meta = ["single", "hybrid", "hybrid_nospar", "dense_k50"]
    cov = {m: {k: [] for k in (10, 20, 50)} for m in methods_meta}
    per_q = []

    t0 = time.time()
    def chunks_for(m, q, k):
        if m == "single":
            return list(r.retrieve(q, k=k, use_ked=True).chunks)
        if m == "hybrid":
            return list(r.hybrid_retrieve(q, k=k, **HYB, use_sparse=True).chunks)
        if m == "hybrid_nospar":
            return list(r.hybrid_retrieve(q, k=k, **HYB, use_sparse=False).chunks)
        if m == "dense_k50":
            return list(r.dense_search(q, k=k).chunks)
        return []

    for idx, rr in enumerate(sample, 1):
        q, kt = rr.get("question", ""), rr.get("knowledge_text", "")
        if not q or not kt:
            continue
        for m in methods_meta:
            for k in (10, 20, 50):
                cov[m][k].append(sent_coverage(kt, chunks_for(m, q, k))[0])

        c10 = sent_coverage(kt, chunks_for("hybrid", q, 10))
        c50 = sent_coverage(kt, chunks_for("hybrid", q, 50))
        n_sent = c10[2]
        if n_sent >= 60:
            cls = "A超长GT"
        elif c50[0] <= 0.02:
            cls = "B真hard-miss"
        else:
            cls = "C常见弱覆盖"
        per_q.append({
            "q": q[:60], "cap": rr.get("capability", ""),
            "n_sent": n_sent, "cov_k10": round(c10[0], 3),
            "cov_k50": round(c50[0], 3), "in_top50": c50[0] > 0.0, "cls": cls,
        })
        if idx % 10 == 0 or idx == len(sample):
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    def summary(covs):
        n = len(covs)
        over = sum(1 for c in covs if c >= TH_SHOW)/n
        return (statistics.mean(covs), statistics.median(covs), over, n)

    print("\n==== GT 句子覆盖率 (入口 x K; mean/med/cov>=50%) ====")
    print(f"{'入口':<14}{'K=10':>22}{'K=20':>22}{'K=50':>22}")
    for m in methods_meta:
        line = f"{m:<14}"
        for k in (10, 20, 50):
            mn, md, ov, _ = summary(cov[m][k])
            line += f"{mn:.2f}/{md:.2f}/{ov:.0%}".rjust(22)
        print(line)

    print("\n==== 归因: K=10 句覆盖率均值提升拆分 ====")
    base = statistics.mean(cov["single"][10])
    hb = statistics.mean(cov["hybrid"][10])
    hn = statistics.mean(cov["hybrid_nospar"][10])
    print(f"  single        = {base:.3f}")
    print(f"  hybrid        = {hb:.3f}   (+{hb-base:+.3f})")
    print(f"  hybrid_nospar = {hn:.3f}   (+{hn-base:+.3f} <- dense扩容贡献)")
    print(f"  BM25 单独贡献 = {hb-hn:+.3f}")



    print("\n==== 逐题归类 (top50 外 / cov10<50%) ====")
    cls_agg = defaultdict(lambda: {"n": 0, "top50_out": 0, "cov10_low": 0})
    for p in per_q:
        c = cls_agg[p["cls"]]
        c["n"] += 1
        if not p["in_top50"]:
            c["top50_out"] += 1
        if p["cov_k10"] < TH_SHOW:
            c["cov10_low"] += 1
    print(f"{'类别':<14}{'n':>4}{'top50外':>8}{'cov10<50%':>10}")
    for cls, c in cls_agg.items():
        print(f"{cls:<14}{c['n']:>4}{c['top50_out']:>8}{c['cov10_low']:>10}")

    print("\n==== top50 外题目明细 ====")
    for p in per_q:
        if not p["in_top50"]:
            print(f"  [{p['cls']}] cov10={p['cov_k10']:.2f} cov50={p['cov_k50']:.2f} "
                  f"n_sent={p['n_sent']} {p['q']}")

    out = os.path.join(_LINS, "results", "diag_recall_ablate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "cov": {m: {str(k): cov[m][k] for k in (10, 20, 50)} for m in methods_meta},
            "per_q": per_q,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[已写] {out}")


if __name__ == "__main__":
    main()

