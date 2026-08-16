# -*- coding: utf-8 -*-
"""
_diag_bn_std_spec.py —— 路径3前置病灶归因：标准规范"超饱和但命中低"是否真成立、病灶在哪。

核心问题：标准规范 chunk 占 63%，但整段IoU hit@1=50%（低于工艺原理75%）——需判断
这到底是：
  · A 排序问题：GT 相关块进了 top-50 但不在 top-10（rerank 可救）
  · B 措辞失配：GT 相关块连 top-50 都不进（query/embedding 可救）
  · C 补全度：相关块在 top-10 但句子覆盖率<0.5（本就拼不全）
主口径用 sent_coverage + 整段IoU，对标准规范类全量题逐一归类。

用法： python _diag_bn_std_spec.py
输出： results/s4_recall/bn_std_spec_focus.json
"""
import os, sys, json, random, csv
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
from retrieval.recall_metrics import sent_coverage, get_content
from retrieval.retriever import OpenDomainRetriever

CAPS = ["标准规范与术语"]
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


def hit_rank(idx_chunks, kt):
    """返回在 idx_chunks(chunk_id list) 中首个相关 chunk 的位次(1-based)或 None。"""
    for i, cid in enumerate(idx_chunks, 1):
        if iou(kt, CI[cid].get("content", "")) >= IOU_TH:
            return i
    return None


def main():
    random.seed(7)
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()
    global CI
    CI = getattr(retr, "chunks", {})

    # 抽样：每 capability 全部 or per_cap
    by_cap = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap in CAPS:
        sample.extend(by_cap.get(cap, []))   # 标准规范全量（该能力题不多，全拿来）

    rows = []
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        res10 = retr.retrieve(q, k=10, use_ked=True)
        res50 = retr.retrieve(q, k=50, use_ked=True)
        c10 = [c.chunk_id for c in res10.chunks]
        c50 = [c.chunk_id for c in res50.chunks]
        cov10, covered10, n_sent = sent_coverage(kt, res10.chunks)
        r10 = hit_rank(c10, kt)
        r50 = hit_rank(c50, kt)
        # 归类：A 排序(B进top50但>=11) / B 措辞(B连top50都没有) / C 补全度(在top10但cov<0.5)
        if r10 is None and r50 is not None:
            cls = "A_排序(B在top50内,但不在top10)"
        elif r50 is None:
            cls = "B_措辞失配(B连top50都不进)"
        elif r10 is not None and cov10 < GOOD_COV:
            cls = "C_补全度(top10内有B但cov<0.5)"
        else:
            cls = "OK(top10内有B且cov>=0.5)"
        row = {"id": r.get("id"), "cap": cap, "q": q[:35],
               "r10": r10, "r50": r50, "cov10": round(cov10, 3), "n_sent": n_sent, "class": cls}
        rows.append(row)
        print(f"[{idx}] {cls} | id={row['id']} r10={r10} r50={r50} cov10={row['cov10']} n_sent={n_sent} | {q[:30]}")

    cnt = defaultdict(int)
    for x in rows:
        cnt[x["class"]] += 1
    print("\n=== 标准规范·single 病灶归类 (n=%d) ===" % len(rows))
    for k, v in cnt.items():
        print(f"  {k}: {v}  ({v/len(rows)*100:.0f}%)")
    out = os.path.join(_LINS, "results", "s4_recall", "bn_std_spec_focus.json")
    json.dump({"rows": rows, "counts": dict(cnt)}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
