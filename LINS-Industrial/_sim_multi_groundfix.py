# -*- coding: utf-8 -*-
"""
_sim_multi_groundfix.py —— 离线模拟：给 multi/hybrid RRF 融合加"原 query 保底"，
验证是否消除 multi 相对 single 的负增益（不改 retriever，先验证再落地）。

根因（_calibrate_multi_gate.py）：子查询 top-1 相似度无法可靠区分价值(分布重叠)，
"质量门控"被数据证伪。改"原查询保底"：融合时把原始完整 query 的 dense 结果
作为最高权重成员参与 RRF。

方法（k=10, use_ked=True, 口径=sent_coverage@10 + 整段IoU hit@1）：
  A single  /  B multi现状  /  C multi+原query保底(权重 2:1)
验证 C >= A 且 C > B。

输出：results/s4_recall/multi_groundfix_sim.json
"""
import os, sys, json, random, time, csv, argparse
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
from retrieval.recall_metrics import sent_coverage, get_content
from retrieval.retriever import OpenDomainRetriever

CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]
IOU_TH = 0.15
GOOD_COV = 0.5


def grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def rrf_merge_weighted(dense_lists, weights, k):
    acc, obj = {}, {}
    for w, cl in zip(weights, dense_lists):
        for c in cl:
            acc[c.chunk_id] = acc.get(c.chunk_id, 0.0) + w / (60 + c.rank)
            obj.setdefault(c.chunk_id, c)
    order = sorted(acc.items(), key=lambda x: x[1], reverse=True)[:k]
    return [obj[c] for c, _ in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()

    by_cap = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap in CAPS:
        picked = by_cap.get(cap, [])
        if picked:
            sample.extend(random.sample(picked, min(args.per_cap, len(picked))))

    agg = {"A_single": {"cov": [], "good": 0}, "B_multi": {"cov": [], "good": 0},
           "C_fix": {"cov": [], "good": 0}, "hit1": {"A": 0, "B": 0, "C": 0}, "n": 0}
    rows = []
    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        sa = retr.retrieve(q, k=10, use_ked=True).chunks
        covA = sent_coverage(kt, sa)[0]; gA = covA >= GOOD_COV
        rankA = next((i + 1 for i, c in enumerate(sa) if iou(kt, get_content(c)) >= IOU_TH), None)
        sb = retr.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion").chunks
        covB = sent_coverage(kt, sb)[0]; gB = covB >= GOOD_COV
        rankB = next((i + 1 for i, c in enumerate(sb) if iou(kt, get_content(c)) >= IOU_TH), None)
        subs = retr.ked.decompose(q) if len(retr.ked.decompose(q)) > 1 else [q]
        lists = [retr.dense_search(q, k=20, use_ked=True).chunks]
        weights = [2.0]
        for sq in subs:
            if sq.strip() == q.strip():
                continue
            lists.append(retr.dense_search(sq, k=20, use_ked=True).chunks)
            weights.append(1.0)
        sc = rrf_merge_weighted(lists, weights, 10)
        covC = sent_coverage(kt, sc)[0]; gC = covC >= GOOD_COV
        rankC = next((i + 1 for i, c in enumerate(sc) if iou(kt, get_content(c)) >= IOU_TH), None)

        agg["A_single"]["cov"].append(covA); agg["A_single"]["good"] += int(gA)
        agg["B_multi"]["cov"].append(covB); agg["B_multi"]["good"] += int(gB)
        agg["C_fix"]["cov"].append(covC); agg["C_fix"]["good"] += int(gC)
        for arm, rk in (("A", rankA), ("B", rankB), ("C", rankC)):
            if rk and rk == 1:
                agg["hit1"][arm] += 1
        agg["n"] += 1
        rows.append({"id": r.get("id"), "cap": r.get("capability"),
                     "single_cov": round(covA, 3), "multi_cov": round(covB, 3),
                     "fix_cov": round(covC, 3)})
        if idx % 10 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    n = agg["n"]
    stat = lambda x: sum(x) / len(x)
    print("\n=== multi 原query保底 离线模拟 (n=%d) ===" % n)
    print(f"  {'臂':<12}{'cov@10':>8}{'good%':>8}{'hit@1':>8}")
    for arm, key in (("A single", "A_single"), ("B multi现状", "B_multi"), ("C multi+保底", "C_fix")):
        c = agg[key]; h = agg["hit1"][arm[0]]
        print(f"  {arm:<12}{stat(c['cov'])*100:>7.1f}%{c['good']/n*100:>7.0f}%{h/n*100:>7.0f}%")
    print("  判读: C>=A 且 C>B → 保底法消除负增益且不劣于 single")
    out = os.path.join(_LINS, "results", "s4_recall", "multi_groundfix_sim.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": {arm: {"cov10": stat(agg[key]["cov"]),
                                 "good": agg[key]["good"] / n, "hit1": agg["hit1"][arm[0]] / n}
                           for arm, key in (("A_single", "A_single"), ("B_multi", "B_multi"), ("C_fix", "C_fix"))},
               "n": n, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
