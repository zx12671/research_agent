# -*- coding: utf-8 -*-
"""
召回精确度量 v3 —— GT 句子覆盖率分布（single vs multi 两模式）。

旧口径(整段IoU) 严重低估真实召回(single top-10 从47%->句子口径87%)。
本脚本用"GT逐句被 top-K chunk 覆盖的比例"作为召回/答案补全度的稳健度量：
  coverage = 被top-K覆盖的GT独立句数 / GT总句数。
对比 single / multi 模式 top-10、top-20 的覆盖率分布。
"""
import os, csv, random, re, statistics
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7
IOU_SENT_TH = 0.30
TH_SHOW = 0.5     # 报告"覆盖>=50%"比例
SHOW_LOW = True   # 打印覆盖率低于该值的题

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

def coverage(kt, chunks):
    sents = split_sent(kt)
    if not sents: return 1.0, 0, 1
    cov = sum(1 for sc in sents if any(jt(sc, c) >= IOU_SENT_TH for c in chunks))
    return cov/len(sents), cov, len(sents)

def main():
    import io, os as _os
    out = _os.path.join(_LINS, "results", "diag_recall_coverage.txt")
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    buf = io.StringIO()
    sink = buf

    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever(); retriever.load_from_manifest()

    # 三种召回入口 + 多档 K
    entry_points = []
    for k in (10, 20, 50):
        for label, getter in [
            ("dense+bm25", lambda q,k=k: getattr(retriever.retrieve(q, k=k, use_ked=True), "chunks", [])),
            ("dense_only", lambda q,k=k: getattr(retriever.dense_search(q, k=k), "chunks", [])),
            ("bm25_only", lambda q,k=k: getattr(retriever.bm25_search(q, k=k), "chunks", [])),
        ]:
            entry_points.append((label, k, getter))

    sink.write("== GT 句子覆盖率分布 (top-K 检索) ==\n")
    sink.write(f"{'entry':12}{'k':>4}  {'mean_cov':>8}{'median':>8}{'cov>=50%':>10}{'>0且<50%':>12}\n")
    cov_by = {}
    for label, k, getter in entry_points:
        covs = []
        for r in sample:
            q, kt = r.get("question",""), r.get("knowledge_text","")
            try:
                chunks = [_as_dict(c)["content"] for c in getter(q)]
            except Exception:
                chunks = []
            covs.append(coverage(kt, chunks)[0])
        over = sum(1 for c in covs if c >= TH_SHOW)/len(covs)
        partial = sum(1 for c in covs if 0 < c < TH_SHOW)/len(covs)
        cov_by[(label, k)] = covs
        sink.write(f"{label:12}{k:>4}  {statistics.mean(covs):>8.2f}{statistics.median(covs):>8.2f}{over:>10.0%}{partial:>12.0%}\n")

    sink.write("\n== 低覆盖题明细 (dense+bm25, 各档K下 cov<50%) ==\n")
    for k in (10, 20, 50):
        getter = [g for (l, kk, g) in entry_points if l=="dense+bm25" and kk==k][0]
        sink.write(f"\n--- K={k} ---\n")
        for r in sample:
            q, kt = r.get("question",""), r.get("knowledge_text","")
            cap = r.get("capability","")
            try:
                chunks = [_as_dict(c)["content"] for c in getter(q)]
            except Exception:
                chunks = []
            cvg, c, n = coverage(kt, chunks)
            if cvg < 0.5:
                sink.write(f"  cov={cvg:.2f} ({c}/{n}句) cap={cap}\n    Q: {q[:60]}\n")

    with open(out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    sink.write(f"\n[已写 UTF-8 报告] {out}\n")

if __name__ == "__main__":
    main()
