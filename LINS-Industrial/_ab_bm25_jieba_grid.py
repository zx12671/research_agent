# -*- coding: utf-8 -*-
"""
sparse_weight 网格：找 BM25-jieba 融合的"只补不扰"权重甜点。

背景（_ab_bm25_jieba.py 结论）：
  jieba 让 BM25 通道"复活"，但 hybrid 目前是温和正增益(+0.008 cov@10) + 少量
  前段扰动(质量计量cov@10 -0.040 退化、bm25_only h@10 40% < dense 47%)。
  根因疑似 sparse_weight=0.8 让 BM25 冲入前段过猛，扰动 dense 排序。

本脚本对 sparse_weight ∈ {0.8, 0.6, 0.4, 0.2} 分别跑 hybrid，并对照 single 基线，
输出一张"只补不扰"判据收敛表：
  - 主口径：cov@10/20/50 均值（hybrid 全档应 >= single，且越高越好）
  - "扰"判据：质量计量(capability)cov@10 退化 >=0、h@10 不下降、hybrid hit@10>=single
  - 新块信号：hybrid top-10 比 single 多盖 GT 句子的题数
据此选一个"增益保持 + 扰动消除"的权重。

dense 与 bm25_only 与权重无关，只算一份作为基线；hybrid 每权重独立重算。
用法：
  python _ab_bm25_jieba_grid.py [--n 30] [--seed 7] [--weights 0.8 0.6 0.4 0.2]
"""
import os
import csv
import json
import time
import random
import argparse
import statistics
from collections import Counter, defaultdict

from retrieval.recall_metrics import sent_coverage, doc_hit_position, coverage_summary
from retrieval.retriever import OpenDomainRetriever

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
DOC_IOU_TH = 0.20
GOOD_COV = 0.5
TOPK = [10]            # 只看 top-10 的命中（organizer 消费端）
COV_KS = [10, 20, 50]
DENSE_W = 1.0


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
    ap.add_argument("--weights", nargs="+", type=float, default=[0.8, 0.6, 0.4, 0.2])
    args = ap.parse_args()
    WTS = [round(w, 2) for w in args.weights]

    sample = sample_rows(args.n, args.seed)
    print(f"[样本] {len(sample)} 题 | cap: {dict(Counter(r.get('capability') for r in sample))}")
    print(f"[网格] sparse_weight = {WTS}")

    r = OpenDomainRetriever()
    r.load_from_manifest()

    all_keys = ["single", "bm25_only"] + [f"hybrid{wt}" for wt in WTS]
    cov = {tk: {k: [] for k in COV_KS} for tk in all_keys}
    hit10 = {tk: 0 for tk in all_keys}
    cap10 = defaultdict(lambda: defaultdict(list))
    adds = defaultdict(int)          # 入口 -> top-10 多盖 GT 句的题数

    def chunks_for(tk, q, k):
        if tk == "single":
            return list(r.retrieve(q, k=k, use_ked=True).chunks)
        if tk == "bm25_only":
            sp = r.bm25_search(q, k=max(k, 50))
            return [r._chunk_to_obj(cid, sc, i + 1) for i, (cid, sc) in enumerate(sp[:k])]
        w = float(tk[len("hybrid"):])
        return list(r.hybrid_retrieve(
            q, k=k, use_ked=True, use_multi_query=False, use_sparse=True,
            sparse_pool=50, dense_weight=DENSE_W, sparse_weight=w,
        ).chunks)

    t0 = time.time()
    for idx, row in enumerate(sample, 1):
        q, kt = row.get("question", ""), row.get("knowledge_text", "")
        if not q or not kt:
            continue
        cap = row.get("capability") or ""
        single10 = chunks_for("single", q, 10)
        for k in COV_KS:
            cov["single"][k].append(sent_coverage(kt, chunks_for("single", q, k))[0])
        hr = doc_hit_position(single10, kt, DOC_IOU_TH)
        if hr is not None and hr <= 10:
            hit10["single"] += 1
        cap10[cap]["single"].append(cov["single"][10][-1])
        _, n_s, _ = sent_coverage(kt, single10)
        for k in COV_KS:
            cov["bm25_only"][k].append(sent_coverage(kt, chunks_for("bm25_only", q, k))[0])
        for wt in WTS:
            tk = f"hybrid{wt}"
            h10 = chunks_for(tk, q, 10)
            for k in COV_KS:
                cov[tk][k].append(sent_coverage(kt, chunks_for(tk, q, k))[0])
            cap10[cap][tk].append(cov[tk][10][-1])
            hh = doc_hit_position(h10, kt, DOC_IOU_TH)
            if hh is not None and hh <= 10:
                hit10[tk] += 1
            _, n_h, _ = sent_coverage(kt, h10)
            if n_h > n_s:
                adds[tk] += 1
        if idx % 5 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    # —— 输出：只补不扰判据收敛表 ——
    n = len(sample)
    base10 = statistics.mean(cov["single"][10])
    print("\n==== sparse_weight 网格：'只补不扰'判据（vs single 基线）====")
    hdr = (f"{'入口':<10}{'cov@10':>8}{'cov@20':>8}{'cov@50':>8}{'dCov10':>8}"
           f"{'h@10':>6}{'顶新块':>6}{'质量计量':>10}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'single':<10}{statistics.mean(cov['single'][10]):>8.3f}"
          f"{statistics.mean(cov['single'][20]):>8.3f}{statistics.mean(cov['single'][50]):>8.3f}"
          f"{'—':>8}{hit10['single']:>6}{'—':>6}{'base':>10}")
    for wt in WTS:
        tk = f"hybrid{wt}"
        m10 = statistics.mean(cov[tk][10])
        d10 = m10 - base10
        qm = statistics.mean(cap10["质量计量与检测"].get(tk, [0])) \
            - statistics.mean(cap10["质量计量与检测"].get("single", [0]))
        print(f"{tk:<10}{m10:>8.3f}{statistics.mean(cov[tk][20]):>8.3f}"
              f"{statistics.mean(cov[tk][50]):>8.3f}{d10:>+8.3f}"
              f"{hit10[tk]:>6}{adds[tk]:>6}{qm:>+.3f}")

    print("\n==== 按 capability 的 cov@10（看退化侧）====")
    caps = sorted(cap10.keys())
    print(f"{'capability':<22}{'single':>9}" + "".join(f"{('w'+str(w)):>9}" for w in WTS))
    for c in caps:
        line = f"{c:<22}"
        for tk in ["single"] + [f"hybrid{w}" for w in WTS]:
            vals = cap10[c].get(tk, [])
            line += f"{(statistics.mean(vals) if vals else float('nan')):>9.3f}"
        print(line)

    out = os.path.join(_LINS, "results", "ab_bm25_jieba_grid.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "weights": WTS,
            "cov": {tk: {str(k): cov[tk][k] for k in COV_KS} for tk in all_keys},
            "hit10": hit10,
            "adds_top10": dict(adds),
            "cap_cov10": {c: {tk: cap10[c].get(tk, []) for tk in all_keys} for c in caps},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[已写] {out}")


if __name__ == "__main__":
    main()


