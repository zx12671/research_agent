# -*- coding: utf-8 -*-
"""
A/B：jieba 中文分词重做 BM25 后，三重入口 × K 的召回/补全度收敛表。

背景（docs/agentic_recall_completeness_correction.md）：
  旧 BM25 用「中文整段 + 2/3字字符滑窗 n-gram」分词，在中文工业语料上实测
  bm25_only 句子覆盖率恒为 0.00（K=10/20/50），混合检索结果=纯 dense，BM25 通道
  对候选池零增益 -> "召回上限基本由 dense 决定"。
本次改造（retrieval/retriever.py::_tokenize）：
  中文改为 jieba 精确模式切【词】，保留字母数字正则（型号/标准号/数值参数）。
  本脚本验证改造后的三入口，回答两个问题：
    Q1  BM25 通道是否"复活"（bm25_only 从全 0 -> 捞出可覆盖 GT 句子的新块）？
    Q2  hybrid 对 top-10/50 覆盖率是否有实质提升（vs 纯 dense 现状）？

三入口：
  single     = retrieve(q, k, use_ked=True)             # 纯 dense（生产默认）
  hybrid     = hybrid_retrieve(dense + BM25jieba -> RRF) # 落地形态
  bm25_only  = bm25_search(q, top50) -> top-k           # 纯 BM25，诊断通道

主口径 = retrieval.recall_metrics.sent_coverage（GT 句子覆盖率）；
交叉参照 = 整段 IoU hit@k。

用法：
  python _ab_bm25_jieba.py [--n 30] [--seed 7]
"""
import os
import csv
import json
import time
import random
import argparse
import statistics
from collections import Counter, defaultdict

from retrieval.recall_metrics import sent_coverage, doc_hit_position, get_content, coverage_summary
from retrieval.retriever import OpenDomainRetriever

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
DOC_IOU_TH = 0.20   # 旧口径交叉参照阈值
GOOD_COV = 0.5      # "拼齐答案"阈值
TOPK = [1, 3, 5, 10]
COV_KS = [10, 20, 50]
HYBRID_KW = dict(use_ked=True, use_multi_query=False, use_sparse=True, sparse_pool=50)


def sample_rows(n, seed):
    random.seed(seed)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    s = []
    for cap, lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP, len(lst)))
    return s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sample = sample_rows(args.n, args.seed)
    print(f"[样本] {len(sample)} 题 | cap: {dict(Counter(r.get('capability') for r in sample))}")

    r = OpenDomainRetriever()
    r.load_from_manifest()

    methods = ["single", "hybrid", "bm25_only"]
    cov = {m: {k: [] for k in COV_KS} for m in methods}
    agg = {m: {"n": 0, "hit": {kk: 0 for kk in TOPK}} for m in methods}
    cap_cov10 = {m: defaultdict(list) for m in methods}
    new_hit_by_sparse = {"n_bm25_adds_hit": 0, "n_bm25_only_hit": 0, "n_total": 0}
    bm25_only_hit_ids = []

    def chunks_for(m, q, k):
        if m == "single":
            return list(r.retrieve(q, k=k, use_ked=True).chunks)
        if m == "hybrid":
            return list(r.hybrid_retrieve(q, k=k, **HYBRID_KW).chunks)
        if m == "bm25_only":
            # 纯 BM25：取 top-k 的 chunk_id，转成 RetrievedChunk 对象供 sent_coverage 消费
            sp = r.bm25_search(q, k=max(k, 50))
            return [r._chunk_to_obj(cid, sc, i + 1) for i, (cid, sc) in enumerate(sp[:k])]
        return []

    t0 = time.time()
    for idx, row in enumerate(sample, 1):
        q, kt = row.get("question", ""), row.get("knowledge_text", "")
        if not q or not kt:
            continue
        cap = row.get("capability") or ""
        for m in methods:
            c10 = chunks_for(m, q, 10)
            for k in COV_KS:
                cov[m][k].append(sent_coverage(kt, chunks_for(m, q, k))[0])
            cvg10 = cov[m][10][-1]
            cap_cov10[m][cap].append(cvg10)
            # 交叉参照：整段 IoU hit@k
            hr = doc_hit_position(c10, kt, DOC_IOU_TH)
            if c10:
                agg[m]["n"] += 1
                for kk in TOPK:
                    if hr and hr <= kk:
                        agg[m]["hit"][kk] += 1

        # Q1 复活信号：bm25_only 在 top-50 是否覆盖到至少 1 句 GT
        b50 = chunks_for("bm25_only", q, 50)
        cov_b50 = sent_coverage(kt, b50)[0]
        bm25_only_hit_ids.append((q[:50], round(cov_b50, 3)))
        if cov_b50 > 0:
            new_hit_by_sparse["n_bm25_only_hit"] += 1
        new_hit_by_sparse["n_total"] += 1

        # Q2 新块信号：hybrid top-10 相比 single top-10，是否多盖了 GT 句子
        s10 = chunks_for("single", q, 10)
        h10 = chunks_for("hybrid", q, 10)
        _, n_s, _ = sent_coverage(kt, s10)
        _, n_h, _ = sent_coverage(kt, h10)
        if n_h > n_s:
            new_hit_by_sparse["n_bm25_adds_hit"] += 1

        if idx % 5 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    print("\n==== Q1 收敛表：GT 句子覆盖率 (入口 x K；mean/med/cov>=50%) ====")
    print(f"{'入口':<12}" + "".join(f"{'K='+str(k):>26}" for k in COV_KS))
    for m in methods:
        line = f"{m:<12}"
        for k in COV_KS:
            s = coverage_summary(cov[m][k])
            line += f"{s['mean']:.3f}/{s['median']:.3f}/{s['good_rate']:.0%}".rjust(26)
        print(line)

    print("\n==== Q2 BM25 通道复活/增益信号 ====")
    bm_hit = new_hit_by_sparse["n_bm25_only_hit"]
    tot = new_hit_by_sparse["n_total"]
    print(f"  bm25_only 在 K=50 覆盖>=1句GT 的题: {bm_hit}/{tot} "
          f"({bm_hit/tot:.0%}  ; 旧实现为 0%)")
    bm_add = new_hit_by_sparse["n_bm25_adds_hit"]
    print(f"  hybrid top-10 比 single top-10 多盖 GT 句子的题: {bm_add}/"
          f"{tot}  (<- BM25 引入新块的证据)")

    print("\n==== 交叉参照：整段 IoU hit@k (top-10) ====")
    print(f"{'入口':<12}" + "".join(f"{('hit@'+str(kk)):>10}" for kk in TOPK) + f"{'n':>6}")
    for m in methods:
        a = agg[m]; line = f"{m:<12}"
        for kk in TOPK:
            line += f"{(a['hit'][kk]/a['n'] if a['n'] else 0):>10.0%}"
        line += f"{a['n']:>6}"
        print(line)

    print("\n==== 按 capability 分层：cov@10 均值 ====")
    allcaps = sorted({c for m in methods for c in cap_cov10[m]})
    print(f"{'capability':<22}" + "".join(f"{m:>12}" for m in methods))
    for c in allcaps:
        line = f"{c:<22}"
        for m in methods:
            v = cap_cov10[m].get(c, [])
            line += f"{(statistics.mean(v) if v else float('nan')):>12.3f}"
        print(line)

    out = os.path.join(_LINS, "results", "ab_bm25_jieba.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "cov": {m: {str(k): cov[m][k] for k in COV_KS} for m in methods},
            "bm25_signal": new_hit_by_sparse,
            "bm25_only_hit_ids": bm25_only_hit_ids,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[已写] {out}")


if __name__ == "__main__":
    main()
