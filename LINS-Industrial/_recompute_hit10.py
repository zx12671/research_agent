# -*- coding: utf-8 -*-
"""
用可靠判据重算 L0 检索真实命中率。

问题：_diag_recall_entrance2 / _probe_evidence_loss 用"整段 knowledge_text vs 单块 字形 IoU>=TH"
判命中。但 knowledge_text 是改写/压缩的答案段落，整段对单块 IoU 天然低 -> 判 miss 可能是假 miss
（检索其实命中了改写答案的来源知识块）。

本脚本改用**句子级存在性**判据：
  对每题，用 query 检索 top-10；把 knowledge_text 分句，统计这些句子在 top-10 的 content 拼接里
  有多大的"整句子串命中率"（sentence-level coverage）。若覆盖 >= COVER_TH(默认0.5) => 判检索成功。

对比三种检索：纯dense(query+ked) / pure bm25 / hybrid。输出真实 hit@10（按句子覆盖）。
"""
import os, sys, io, re, csv
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
N = 30
SEED = 7
COVER_TH = 0.5


def split_sentences(text):
    parts = re.split(r"[。！？\n；;]", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 4]


def cover(sents, haystack):
    """答案句在拼接文本里的整句子串命中率"""
    hit = 0
    for s in sents:
        if s in haystack:
            hit += 1
    return hit / len(sents) if sents else 0.0


def soft_cover(sents, chunks, th=0.45):
    """软命中：答案句与任一 top-10 chunk 的 2gram Jaccard >= th（容忍改写）。"""
    from collections import Counter as _C
    hit = 0
    for s in sents:
        gs = _grams(s)
        matched = False
        for c in chunks:
            gc = _grams(c)
            if not gs or not gc:
                continue
            inter = len(gs & gc)
            inter = len(gs & gc)
            if inter / (len(gs) + len(gc) - inter) >= th:
                matched = True
                break
        if matched:
            hit += 1
    return hit / len(sents) if sents else 0.0


def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}


def main():
    import random
    from collections import defaultdict
    random.seed(SEED)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["capability"]].append(r)
    sample = []
    for cap, lst in sorted(by_cap.items()):
        sample += random.sample(lst, min(PER_CAP, len(lst)))
    sample = sample[:N]
    print(f"样本 = {len(sample)} 题")

    from retrieval.retriever import OpenDomainRetriever
    rt = OpenDomainRetriever()
    rt.load_from_manifest()

    agg = {m: {"n": 0, "success": 0, "cover_sum": 0.0, "soft_succ": 0, "soft_sum": 0.0}
           for m in ["dense", "bm25", "hybrid"]}
    print(f"{'id':>5} {'cap':8} {'dense_cov':>9} {'bm25_cov':>9} {'hyb_cov':>8}")
    for r in sample:
        q, kt = r.get("question"), r.get("knowledge_text")
        rid = r.get("id")
        sents = split_sentences(kt or "")
        if not q or not kt or not sents:
            continue
        res = {}
        res["dense"] = [getattr(c, "content", "") for c in rt.dense_search(q, k=10).chunks]
        bq = rt.bm25_search(q, k=10)
        bm25_contents = []
        for cid, _sc in bq:
            ch = rt.chunks.get(cid, {})
            bm25_contents.append(ch.get("content", ""))
        res["bm25"] = bm25_contents
        res["hybrid"] = [getattr(c, "content", "") for c in rt.hybrid_retrieve(q, k=10).chunks]

        line = f"{rid:>5} {r['capability'][:8]:8}"
        for m in agg:
            chunks_c = res[m]
            hay = "".join(chunks_c)
            cov = cover(sents, hay)
            scov = soft_cover(sents, chunks_c)
            agg[m]["n"] += 1
            agg[m]["cover_sum"] += cov
            agg[m]["soft_sum"] += scov
            if cov >= COVER_TH:
                agg[m]["success"] += 1
            if scov >= COVER_TH:
                agg[m]["soft_succ"] += 1
            line += f" {cov*100:>7.0f}%"
        print(line)

    print("\n==== 真实 hit@10（按答案句覆盖>=%.0f%%）====" % (COVER_TH * 100))
    for m in agg:
        n = agg[m]["n"] or 1
        print(f"  {m:6}: 硬句覆盖成功 {agg[m]['success']}/{n} = {agg[m]['success']/n*100:.0f}%, "
              f"平均覆盖 {agg[m]['cover_sum']/n*100:.0f}% | "
              f"软2gram命中成功 {agg[m]['soft_succ']}/{n} = {agg[m]['soft_succ']/n*100:.0f}%, "
              f"平均软覆盖 {agg[m]['soft_sum']/n*100:.0f}%")


if __name__ == "__main__":
    main()
