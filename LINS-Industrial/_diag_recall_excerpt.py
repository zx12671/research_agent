# -*- coding: utf-8 -*-
"""
金标准人工抽验：对"疑缺库"题，打印 GT(knowledge_text) 全文 vs 语义最相似块(取自 dense(kt) top1)。
判据：看一眼"库里最接近这块答案的 chunk"到底有多像答案。
  - 若最相似块与答案核心实体/语义高度吻合只是字形不同 -> 之前"缺库"是漏测(应归入"在库"),
  - 若最相似块与答案几乎无关 / 只是泛泛相关 -> "缺库"坐实。
"""
import os, csv, random
from functools import lru_cache
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, IOU_HIT_TH, SEED = 5, 30, 0.20, 7

@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}
def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb)/len(ga | gb)) if (ga and gb) else 0.0
def _as_dict(c):
    if isinstance(c, dict):
        return {"content": c.get("content",""), "id": c.get("id") or c.get("chunk_id")}
    return {"content": getattr(c,"content","") or "", "id": getattr(c,"id",None) or getattr(c,"chunk_id",None)}

def sample_questions():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows: by_cap.setdefault(r.get("capability") or "", []).append(r)
    s = []
    for cap,lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP,len(lst)))
    return s[:N]

def main():
    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever(); retriever.load_from_manifest()
    miss = []
    for r in sample:
        q, kt = r.get("question",""), r.get("knowledge_text","")
        res = retriever.retrieve(q, k=10, use_ked=True)
        chunks = getattr(res,"chunks",[])
        if not any(iou(kt, _as_dict(c)["content"]) >= IOU_HIT_TH for c in chunks):
            miss.append(r)
    sel = ["电磨","二氧化锡","亚甲基蓝","VBDRM","VDRM","制冷压缩机","直线振动筛","GB/T 2423"]
    print(f"[miss 共{len(miss)}题，取抽样核验]\n")
    for r in miss:
        q, kt = r.get("question",""), r.get("knowledge_text",""); cap = r.get("capability","")
        show = not sel or any(k in q for k in sel)
        if not show: continue
        try:
            dr = retriever.dense_search(kt, k=1)
            top = dr.chunks[0]; topd = _as_dict(top)
            sc = iou(kt, topd["content"])
        except Exception as e:
            print("dense err:", e); continue
        print("="*70)
        print(f"Q      : {q[:70]}")
        print(f"cap    : {cap}")
        print("-"*70)
        print(f"GT     : {kt[:220]}")
        print("-"*70)
        print(f"TOP-1  : IoU={sc:.2f}")
        print(f"  {topd['content'][:220]}")
        print("="*70, "\n")

if __name__ == "__main__":
    main()
