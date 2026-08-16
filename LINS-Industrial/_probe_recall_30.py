# -*- coding: utf-8 -*-
"""
30 样本召回率实测：单查询 / 多查询 / 混合查询 三模式对比。
指标: hit@1/3/5/10（汉字 bigram IoU >= 0.20 判定命中 knowledge_text）
另统计 GT 在 top-1/5/10/20/50 的累积命中（回答'GT 到底进没进候选池'）。
"""
import os
import csv
import random
import traceback
from collections import Counter
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5          # 每 capability 抽 5 题，7 类共最多 35，取 30 内
N = 30
IOU_TH = 0.20
TOPK = [1, 3, 5, 10]
WIDER = [1, 5, 10, 20, 50]
SEED = 7

@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)

def first_hit(chunks, kt):
    """返回首个命中 GT 的位次(1-based)，未命中返回 None。"""
    for i, c in enumerate(chunks, 1):
        content = getattr(c, "content", None) or (c.get("content") if isinstance(c, dict) else "")
        if iou(kt, content) >= IOU_TH:
            return i
    return None

def main():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r.get("capability") or "", []).append(r)
    sample = []
    for cap, lst in sorted(by_cap.items()):
        sample += random.sample(lst, min(PER_CAP, len(lst)))
    sample = sample[:N]
    print(f"[样本] 抽取 {len(sample)} 题")
    print("capability 分布:", dict(Counter(r.get("capability") for r in sample)))

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    try:
        retriever.load_from_manifest()
    except Exception as e:
        print("manifest load err:", e)
        raise

    # 探测可用方法
    has = {m: hasattr(retriever, m) for m in
           ["retrieve", "multi_query_retrieve", "hybrid_retrieve", "dense_search", "bm25_search"]}
    print("[API 探测]", has)

    methods = ["single", "multi", "hybrid"]
    agg = {m: {"n": 0, "hit": {kk: 0 for kk in TOPK}} for m in methods}
    wider_agg = {m: {"n": 0, "hit": {kk: 0 for kk in WIDER}} for m in methods}
    miss_examples = {m: [] for m in methods}

    for idx, r in enumerate(sample, 1):
        q = r.get("question", "")
        kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        results = {}
        try:
            results["single"] = retriever.retrieve(q, k=10, use_ked=True)
        except Exception as e:
            print(f"  single err: {e}")
        try:
            results["multi"] = retriever.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion")
        except Exception as e:
            print(f"  multi err: {e}")
        try:
            results["hybrid"] = retriever.hybrid_retrieve(q, k=10, use_ked=True)
        except Exception as e:
            print(f"  hybrid err: {e}")

        for m, res in results.items():
            chunks = getattr(res, "chunks", [])
            if not chunks:
                continue
            agg[m]["n"] += 1
            hr = first_hit(chunks, kt)
            for kk in TOPK:
                if hr and hr <= kk:
                    agg[m]["hit"][kk] += 1
            if hr is None:
                miss_examples[m].append(q[:50])
            # wider: 放大检索候选池，看 GT 在 top-50 内的位次
            try:
                if m == "single":
                    res50 = retriever.retrieve(q, k=50, use_ked=True)
                elif m == "multi":
                    res50 = retriever.multi_query_retrieve(q, k=50, use_ked=True, strategy="fusion")
                else:
                    res50 = retriever.hybrid_retrieve(q, k=50, use_ked=True)
                c50 = getattr(res50, "chunks", [])
                hr50 = first_hit(c50, kt)
                wider_agg[m]["n"] += 1
                for kk in WIDER:
                    if hr50 and hr50 <= kk:
                        wider_agg[m]["hit"][kk] += 1
            except Exception as e:
                wider_agg[m]["n"] += 1  # 保持分母一致但记做未命中 top50
                print(f"  {m} wider(err): {e}")

        if idx % 5 == 0:
            print(f"  .. 进度 {idx}/{len(sample)}")

    print("\n==== top-10 窗口内：三模式 hit@1/3/5/10 ====")
    print(f"{'方法':<8}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>9}")
    for m in methods:
        a = agg[m]; n = a["n"] or 1
        line = f"{m:<8}{a['n']:>4}"
        for kk in TOPK:
            line += f"{a['hit'][kk] / n * 100:>7.1f}%"
        print(line)

    print("\n==== 候选池放大到 50：GT 累积命中（回答 GT 进没进池） ====")
    print(f"{'方法':<8}{'n':>4}" + "".join(f"@{kk:<5}" for kk in WIDER))
    for m in methods:
        a = wider_agg[m]; n = a["n"] or 1
        line = f"{m:<8}{a['n']:>4}"
        for kk in WIDER:
            line += f"{a['hit'][kk] / n * 100:>6.0f}%  "
        print(line)

    print("\n==== top-10 内未命中 GT 的题 (每人最多3例) ====")
    for m in methods:
        print(f"  [{m}]")
        for ex in miss_examples[m][:3]:
            print(f"    - {ex}")

if __name__ == "__main__":
    main()
