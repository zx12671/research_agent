# -*- coding: utf-8 -*-
"""
命中判据修正验证：GT(答案) vs chunk 的匹配, 从"整段IoU"改为"GT逐句 vs chunk局部 最长覆盖"。
目的：回答"之前用全段IoU判miss, 低估了多少真实召回"。

做法：对 30 样本, 取 retriever.retrieve(q,k=10,single) 返回的 top-10，
  对每个 chunk 计算它覆盖 GT 的"最长公共片段覆盖率"(把GT按句切, 每句对chunk算bigram JT，
  任一整句 IoU>=TH_S 即该句被覆盖; 两句以上覆盖=该题"碎片拼接命中")。
对比旧口径(整段IoU>=0.20)的命中率。
"""
import os, csv, random, re
from functools import lru_cache
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7
IOU_FULL_TH = 0.20     # 旧口径整段阈值
IOU_SENT_TH = 0.30     # 句子级命中阈值
MIN_COVERED = 1        # 至少覆盖GT前N句算"碎片命中"(>=2 更严)

@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}
def jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb)/len(ga | gb)) if (ga and gb) else 0.0
def split_sent(text):
    parts = re.split(r"(?<=[。；;！？])\s*", text or "")
    return [p for p in parts if len(p.strip()) >= 6]
def _as_dict(c):
    if isinstance(c, dict):
        return {"content": c.get("content","")}
    return {"content": getattr(c,"content","") or ""}

def sample_questions():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows: by_cap.setdefault(r.get("capability") or "", []).append(r)
    s=[]
    for cap,lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP,len(lst)))
    return s[:N]

def main():
    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever(); retriever.load_from_manifest()

    full_hit = sent_cov = 0
    rows_out = []
    for r in sample:
        q, kt = r.get("question",""), r.get("knowledge_text","")
        cap = r.get("capability","")
        res = retriever.retrieve(q, k=10, use_ked=True)
        chunks = [_as_dict(c)["content"] for c in getattr(res,"chunks",[])]
        # 旧口径
        old = any(jt(kt, c) >= IOU_FULL_TH for c in chunks)
        # 逐句覆盖
        sents = split_sent(kt)
        covered = 0
        for s_c in sents:
            if any(jt(s_c, c) >= IOU_SENT_TH for c in chunks):
                covered += 1
        new_hit = covered >= MIN_COVERED
        full_hit += old
        sent_cov += new_hit
        rows_out.append((q[:28], cap, len(chunks), len(sents), covered, old, new_hit))

    print(f"样本 {len(sample)} 题, top-10(single,k=10)\n")
    print(f"{'Q':30}{'cap':10}{'nSentGT':>8}{'covSent':>8}  {'old整段':>8}{'sentence':>9}")
    print("-"*80)
    for q, cap, nc, ns, cov, old, new in rows_out:
        print(f"{q:30}{cap[:9]:10}{ns:>8}{cov:>8}  {str(old):>8}{str(new):>9}")
    print("-"*80)
    print(f"旧口径(整段IoU) 召回  : {full_hit}/{len(sample)} = {full_hit/len(sample)*100:.0f}%")
    print(f"新口径(句子覆盖) 召回  : {sent_cov}/{len(sample)} = {sent_cov/len(sample)*100:.0f}%")

if __name__ == "__main__":
    main()
