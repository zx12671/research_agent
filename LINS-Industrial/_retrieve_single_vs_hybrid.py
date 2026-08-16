# -*- coding: utf-8 -*-
"""
Retriever 环节量化对比：单查询纯 Dense(生产现状) VS 单查询 Hybrid(dense+BM25→RRF)。
【P0 口径已切换】—— 评估口径统一到 retrieval/recall_metrics.py，主指标为句子覆盖率。

口径说明（为何以句子覆盖率为准）：
 - 旧口径 = 整段 bigram IoU>=0.20 判"命中"，会被超长 GT 稀释，系统性低估召回，
   曾误得出"约 50% GT 在 top-50 外"的结论（见 docs/agentic_recall_entrance_diagnosis.md）。
 - 新主口径 = GT 句子覆盖率 sent_coverage()：这题 top-K 拼齐了 GT 答案的百分之几(0~1)。
 - 整段 IoU 命中仅作【并行交叉参照】（doc_hit_position），不单独作结论。

输出：
 - 主表：单查询 pure-dense vs 单查询 hybrid(dense+BM25,无multi_query) 的
         cov@10 / cov@20 / cov@50(句子覆盖) + 平均覆盖率 + "拼齐>=50%"比例
 - 交叉参照：旧口径 hit@1/3/5/10(top-10) 与 top-50 累积命中(整段 IoU)
 - 按 capability 分层：cov@10（哪类提升/退化）
 - score 分布（dense 分数 vs hybrid RRF 分数，供 downstream/organizer 参考）

用法:
  python _retrieve_single_vs_hybrid.py [--n 30] [--seed 7]
"""
import os
import csv
import time
import random
import argparse
import statistics
from collections import Counter, defaultdict

from retrieval.recall_metrics import (
    sent_coverage, doc_hit_position, get_content, coverage_summary,
)
from retrieval.retriever import OpenDomainRetriever

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")

PER_CAP = 5           # 每 capability 抽 N 题
DOC_IOU_TH = 0.20     # 旧口径(整段 IoU)阈值——仅交叉参照
GOOD_COV = 0.5        # "拼齐答案"判定阈值
TOPK = [1, 3, 5, 10]          # 旧口径 hit@k（并列展示）
WIDER = [1, 5, 10, 20, 50]    # 旧口径 top-50 累积
COV_KS = [10, 20, 50]         # 新口径句覆盖率的 K 档
# hybrid 固定单查询、不开 multi_query（与本次需求一致）
HYBRID_KW = dict(use_ked=True, use_multi_query=False, use_sparse=True, sparse_pool=50)


def sample_rows(n, seed):
    random.seed(seed)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap, lst in sorted(by_cap.items()):
        sample += random.sample(lst, min(PER_CAP, len(lst)))
    return sample[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="总样本数上限")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sample = sample_rows(args.n, args.seed)
    n_total = len(sample)
    print(f"[样本] {n_total} 题 | capability 分布: {dict(Counter(r.get('capability') for r in sample))}")

    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    methods = ["single", "hybrid"]

    # —— 新口径(主)：句子覆盖率 cov@K ——
    cov = {m: {k: [] for k in COV_KS} for m in methods}
    # —— 旧口径(交叉参照)：整段 IoU hit@k、top-50 累积 ——
    agg = {m: {"n": 0, "hit": {kk: 0 for kk in TOPK}} for m in methods}
    wider = {m: {"n": 0, "hit": {kk: 0 for kk in WIDER}} for m in methods}
    scores = {m: [] for m in methods}
    cap_cov10 = {m: defaultdict(list) for m in methods}
    examples = {m: [] for m in methods}

    def chunks_for(m, q, k):
        if m == "single":
            return list(retriever.retrieve(q, k=k, use_ked=True).chunks)
        return list(retriever.hybrid_retrieve(q, k=k, **HYBRID_KW).chunks)

    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        if not q or not kt:
            continue
        for m in methods:
            c10 = list(chunks_for(m, q, 10))
            scores[m] += [float(c.score) for c in c10]

            # 新口径：cov@10/20/50
            for k in COV_KS:
                cov[m][k].append(sent_coverage(kt, chunks_for(m, q, k))[0])
            cvg10 = cov[m][10][-1]

            # 旧口径(交叉参照)：hit@k + top-50 累积
            hr = doc_hit_position(c10, kt, DOC_IOU_TH)
            if c10:
                agg[m]["n"] += 1
                for kk in TOPK:
                    if hr and hr <= kk:
                        agg[m]["hit"][kk] += 1
            if hr is None and len(examples[m]) < 6:
                examples[m].append((q[:70], [get_content(c)[:30] for c in c10[:3]]))
            cap = r.get("capability") or ""
            cap_cov10[m][cap].append(cvg10)

            r50 = chunks_for(m, q, 50)
            hr50 = doc_hit_position(r50, kt, DOC_IOU_TH)
            wider[m]["n"] += 1
            for kk in WIDER:
                if hr50 and hr50 <= kk:
                    wider[m]["hit"][kk] += 1

        if idx % 10 == 0 or idx == n_total:
            print(f"  .. {idx}/{n_total}  ({time.time()-t0:.0f}s)")

    _print_report(cov, agg, wider, scores, cap_cov10, examples,
                  methods, n_total, TOPK, WIDER, COV_KS, GOOD_COV)



def _print_report(cov, agg, wider, scores, cap_cov10, examples,
                  methods, n_total, TOPK, WIDER, COV_KS, GOOD_COV):
    print("\n=========== 【主口径 · P0】GT 句子覆盖率 cov@K ===========")
    print(f"{'方法':<10}{'n':>5}" + "".join(f"cov@{kk}".rjust(12) for kk in COV_KS)
          + "".join(f"{'mean':>8}{'med':>7}{'cov>=50%':>10}"))
    for m in methods:
        line = f"{m:<10}{n_total:>5}"
        for k in COV_KS:
            s = coverage_summary(cov[m][k], GOOD_COV)
            line += (f"{s['mean']:.2f}/{s['median']:.2f}/{s['good_rate']:.0%}").rjust(12)
        s10 = coverage_summary(cov[m][10], GOOD_COV)
        line += f"{s10['mean']:>8.2f}{s10['median']:>7.2f}{s10['good_rate']:>9.0%}"
        print(line)
    print("  (读法: mean/med/cov>=50% —— 覆盖率均值/中位/\"拼齐>=50%\"的题占比)")

    def pct(a, kk):
        n = a["n"] or 1
        return a["hit"][kk] / n * 100

    print("\n=========== 【交叉参照 · 旧口径】整段 IoU hit@1/3/5/10 ===========")
    print(f"{'方法':<10}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>9}")
    for m in methods:
        a = agg[m]
        print(f"{m:<10}{a['n']:>4}" + "".join(f"{pct(a, kk):>7.1f}%" for kk in TOPK))

    print("\n=========== 【交叉参照 · 旧口径】top-50 累积命中 ===========")
    print(f"{'方法':<10}{'n':>4}" + "".join(f"@{kk:<6}" for kk in WIDER))
    for m in methods:
        a = wider[m]; n = a["n"] or 1
        print(f"{m:<10}{a['n']:>4}" + "".join(f"{a['hit'][kk]/n*100:>5.0f}%  " for kk in WIDER))

    print("\n=========== 按 capability：cov@10 均值 ===========")
    all_caps = [c for mm in methods for c in cap_cov10[mm]]
    cap_order = [c for c in ["选型与替代", "标准规范与术语", "工艺原理与参数影响",
                             "安全合规与风险控制", "质量计量与检测",
                             "故障诊断与排查", "工程计算与估算"] if c in all_caps]
    print(f"{'capability':<16}{'n':>4}{'single_cov10':>14}{'hybrid_cov10':>14}{'delta':>8}")
    for cap in cap_order:
        sv = statistics.mean(cap_cov10["single"].get(cap, [0.0]))
        hv = statistics.mean(cap_cov10["hybrid"].get(cap, [0.0]))
        n = len(cap_cov10["single"].get(cap, []))
        print(f"{cap:<16}{n:>4}{sv:>14.2f}{hv:>14.2f}{hv-sv:>+8.2f}")

    print("\n=========== score 分布（top-10 内所有 chunk 分数） ===========")
    for m in methods:
        v = scores[m]
        if v:
            import numpy as np
            v = np.array(v)
            print(f"{m:<10} n={len(v)} min={v.min():.3f} p50={np.percentile(v,50):.3f} "
                  f"mean={v.mean():.3f} p90={np.percentile(v,90):.3f} max={v.max():.3f}")

    print("\n=========== 整段 IoU 口径下 top-10 内未命中 GT 的题（前 3 例，旧口径参照） ===========")
    for m in methods:
        print(f"  [{m}]")
        for q, tops in examples[m][:3]:
            print(f"    Q: {q}")
            for t in tops:
                print(f"       - {t}")
            print()


if __name__ == "__main__":
    main()

