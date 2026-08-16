# -*- coding: utf-8 -*-
"""
_probe_bclass_bm25.py —— 路径3 落地前验证：标准规范 B 类(池外失配,尤其含标准号)题，
用 BM25 精确 token 通道能否捞回 GT 相关块？据此判断"精确标识召回"是否可行修法。

方法：
 取所有含标准号 + "B类(r10=r50=None,cov<0.5)" 的标准规范题（BN_std_spec / bclass 判B的样本），
 对每题：
   · bm25 检索 top50 → 看 GT 相关块(整段IoU)是否进入 top50（r_bm25）
   · 若 r_bm25 存在 → BM25 精确通道可补召回（说明 dense 失配但库里有料、BM25 能捞）
   · 若全不存在 → 库里可能根本没有该标准号的 chunk（真缺料），BM25 也救不了
 输出：results/s4_recall/bclass_bm25_probe.json + 摘要
"""
import os, sys, json, csv, re
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
from retrieval.recall_metrics import get_content
from retrieval.retriever import OpenDomainRetriever

IOU_TH = 0.15
STD_PAT = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG)\s*[-—]?\s*\d+[-—]?\d*", re.I)
# 只做含明确标准号编号的题（排除空泛 GB/T 匹配）


def grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def hit_rank(chunks, kt, CI):
    for i, c in enumerate(chunks, 1):
        content = getattr(c, "content", "") or ""
        if iou(kt, content) >= IOU_TH:
            return i
    return None


def main():
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()
    CI = getattr(retr, "chunks", {})
    sample = [r for r in csv.DictReader(
        open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
             encoding="utf-8-sig", newline=""))
        if r.get("capability") == "标准规范与术语"]

    rows = []
    n_B = 0; n_B_recover = 0; n_checked = 0
    for r in sample:
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        # 只关心含明确标准号编号的题
        m = STD_PAT.search(q)
        if not m:
            continue
        # 确认它为 B 类(池外失配)
        res50 = retr.retrieve(q, k=50, use_ked=True)
        r_dense = hit_rank(res50.chunks, kt, CI)
        if r_dense is not None:
            continue  # 非 B 类，跳过
        n_B += 1
        # 试 BM25 精确通道
        bm = retr.bm25_search(q, k=50) if hasattr(retr, "bm25_search") else []
        ids = [cid for cid, _ in bm]
        r_bm = None
        chosen = None
        for ccid in ids:
            if iou(kt, CI[ccid].get("content", "")) >= IOU_TH:
                r_bm = ids.index(ccid) + 1
                chosen = ccid
                break
        if r_bm is not None:
            n_B_recover += 1
        rows.append({"id": r.get("id"), "q": q[:40], "std": m.group(0),
                     "r_dense50": r_dense, "r_bm25": r_bm, "recover": r_bm is not None})
        n_checked += 1

    print(f"含标准号且 B类(池外) 题: {n_B}  | 检查 {n_checked}")
    print(f"BM25 能捞回 GT 的: {n_B_recover} ({n_B_recover/max(1,n_B)*100:.0f}%)")
    print("→ 若 >50%: BM25 精确通道可补 B 类召回，是低成本有效修法；"
          "若 ≈0: 库内可能缺对应标准号 chunk(真缺料)")
    out = os.path.join(_LINS, "results", "s4_recall", "bclass_bm25_probe.json")
    json.dump({"n_B_std": n_B, "n_checked": n_checked, "n_bm25_recover": n_B_recover,
               "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
