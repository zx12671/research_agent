# -*- coding: utf-8 -*-
"""
_diag_bclass_allcap.py —— 路径3诊断深化：哪些能力域存在"GT 连 top-50 都不进"的
B类(措辞/编码失配)高发，以及是否与"含标准号"强相关（支持归因到精确标识召回缺失）。

方法：对全部 capability 题，用 single retriever 检索 top-10/top-50，逐题归类：
  B_池外失配(r10=None,r50=None 即 GT 相关块连 top50 都不进)  ← 不是排序而是召回失配
  C_补全度(r10 有但 cov<0.5) / A_排序(r50 有但 r10 无) / OK
并按题面是否含标准号模式(GB/GB-T/JB/...编号)统计 B 类占比 × 标准号相关性。

输出：results/s4_recall/bclass_all_cap.json + 控制台各能力域汇总
"""
import os, sys, json, csv, re
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
STD_PAT = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG|DL|QB|YB|CJ|SN)\s*[-—]?\s*\d+[-—]?\d*|GB|GB/T", re.I)


def grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def main():
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()
    CI = getattr(retr, "chunks", {})

    def hit(cls, kt):
        for i, cid in enumerate(cls, 1):
            if iou(kt, CI[cid].get("content", "")) >= IOU_TH:
                return i
        return None

    by_cap = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        by_cap[r.get("capability") or ""].append(r)

    cap_stat = {}
    all_B = []
    for cap in CAPS:
        items = by_cap.get(cap, [])
        cnt = defaultdict(int); n_B_std = 0; n_B = 0; n = 0
        for r in items:
            q = r.get("question", ""); kt = r.get("knowledge_text", "")
            if not q or not kt:
                continue
            res10 = retr.retrieve(q, k=10, use_ked=True)
            res50 = retr.retrieve(q, k=50, use_ked=True)
            c10 = [c.chunk_id for c in res10.chunks]
            c50 = [c.chunk_id for c in res50.chunks]
            cov10, _, _ = sent_coverage(kt, res10.chunks)
            r10 = hit(c10, kt); r50 = hit(c50, kt)
            if r10 is None and r50 is None:
                cls = "B"
            elif r10 is None:
                cls = "A"
            elif cov10 < GOOD_COV:
                cls = "C"
            else:
                cls = "OK"
            cnt[cls] += 1; n += 1
            if cls == "B":
                n_B += 1
                if STD_PAT.search(q):
                    n_B_std += 1
        cap_stat[cap] = {"n": n, "B": cnt.get("B", 0), "A": cnt.get("A", 0),
                         "C": cnt.get("C", 0), "OK": cnt.get("OK", 0),
                         "B_pct": cnt.get("B", 0) / n if n else 0,
                         "B_std_pct": (n_B_std / n_B) if n_B else 0,
                         "B_std_count": n_B_std, "B_count": n_B}
    print(f"  {'能力域':<14}{'n':>5}{'B池外%':>7}{'C补全%':>7}{'A排序%':>7}{'OK%':>6}{'B中含标准号':>12}")
    for cap, s in cap_stat.items():
        bc = s["B"] / s["n"] if s["n"] else 0
        cc = s["C"] / s["n"] if s["n"] else 0
        ac = s["A"] / s["n"] if s["n"] else 0
        oc = s["OK"] / s["n"] if s["n"] else 0
        print(f"  {cap:<14}{s['n']:>5}{bc*100:>6.0f}%{cc*100:>6.0f}%{ac*100:>6.0f}%{oc*100:>5.0f}%"
              f"{s['B_std_count']}/{s['B_count']} ({s['B_std_pct']*100:.0f}%)")
    out = os.path.join(_LINS, "results", "s4_recall", "bclass_all_cap.json")
    json.dump(cap_stat, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
