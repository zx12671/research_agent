# -*- coding: utf-8 -*-
"""
召回入口逐题诊断（对 miss 的题）：
回答两个问题：
  A) 存在性：GT(knowledge_text) 对应的 chunk 在 corpus(55095) 里到底存不存在？
      方式：knowledge_text 对全量 chunk 做字形 bigram IoU，取最高者（max_chunk）。
            max_IoU 高 -> 文本在库(存在)；max_IoU 低 -> "没这个文本"强证据。
  B) 匹配信号：若存在，这个 chunk 在 dense(bge相似度) 与 BM25 里分别排第几？
      方式：对 query 全量打分，看目标 chunk 的 rank（排进 top10/50? 在外就说明匹配信号失灵）。

输出逐题: question / max_IoU / 对应 chunk(cap, excerpt) / dense_rank / bm25_rank / 定性结论。
"""
import os
import csv
import random
import time
from functools import lru_cache
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
PER_CAP = 5
N = 30
IOU_HIT_TH = 0.20      # 检索命中判定阈值（与 _probe_recall_30 对齐）
EXIST_TH = 0.30        # 存在性判定：max_IoU>=0.30 -> 文本在库
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

def sample_questions():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r.get("capability") or "", []).append(r)
    sample = []
    for cap, lst in sorted(by_cap.items()):
        sample += random.sample(lst, min(PER_CAP, len(lst)))
    return sample[:N]

def load_corpus_chunks():
    """返回 dict[chunk_id] = (content, capability)。"""
    chunks = {}
    try:
        import json
        with open(CHUNKS, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                d = json.loads(s)
                cid = d.get("id") or d.get("chunk_id") or d.get("doc_id")
                chunks[cid] = (d.get("content", ""), d.get("capability", ""))
    except Exception as e:
        print("corpus chunks load fail:", e)
    return chunks

def find_gt_chunk(kt, corpus):
    """全量扫描找与 kt 字形最匹配的 chunk。返回 (max_iou, chunk_id, content, cap)。"""
    best = (0.0, None, "", "")
    for cid, (content, cap) in corpus.items():
        sc = iou(kt, content) if content else 0.0
        if sc > best[0]:
            best = (sc, cid, content, cap)
    return best

def main():
    sample = sample_questions()
    print(f"[样本] 共 {len(sample)} 题")
    corpus = load_corpus_chunks()
    print(f"[corpus] 加载 {len(corpus)} 个 chunk")
    if not corpus:
        print("!! corpus 空，无法做存在性扫描")
        return

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    # 先确定性找出 top-10 miss（single 模式）
    miss = []
    for r in sample:
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        if not q or not kt:
            continue
        res = retriever.retrieve(q, k=10, use_ked=True)
        chunks = getattr(res, "chunks", [])
        hit = any(iou(kt, getattr(c, "content", "") or "") >= IOU_HIT_TH for c in chunks)
        if not hit:
            miss.append(r)
    print(f"\n[top-10 miss] 共 {len(miss)} 题（将逐题诊断）\n")

    stats = Counter()
    WAIT_TIP = "  (全量 IoU 扫描较慢，请耐心)"

    for idx, r in enumerate(miss, 1):
        q, kt, cap = r.get("question", ""), r.get("knowledge_text", ""), r.get("capability", "")
        print(f"--- [{idx}/{len(miss)}] ---")
        print(f"Q   : {q[:60]}")
        print(f"cap : {cap}")
        print(WAIT_TIP)
        t0 = time.time()
        max_iou, cid, content, ccap = find_gt_chunk(kt, corpus)
        dt = time.time() - t0
        existed = max_iou >= EXIST_TH

        # B) 匹配信号排名：若存在，算 dense/bm25 rank
        dense_rank = bm25_rank = None
        if existed:
            try:
                dn = retriever.dense_search(q, k=55095)
                for i, c in enumerate(dn.chunks, 1):
                    if getattr(c, "id", None) == cid or iou(kt, getattr(c, "content", "")) >= IOU_HIT_TH:
                        dense_rank = i
                        break
            except Exception as e:
                print("  dense_search err:", e)
            try:
                bn = retriever.bm25_search(q, k=55095)
                for i, c in enumerate(bn.chunks, 1):
                    if getattr(c, "id", None) == cid or iou(kt, getattr(c, "content", "")) >= IOU_HIT_TH:
                        bm25_rank = i
                        break
            except Exception as e:
                print("  bm25_search err:", e)

        # 定性结论
        if existed:
            if dense_rank and dense_rank <= 50:
                verdict = "在库+rank前50"
            elif dense_rank and dense_rank is not None:
                verdict = "在库+match弱(dense超50)"
            elif dense_rank is None:
                verdict = "在库+rank未测出"
            out = (f"max_IoU={max_iou:.2f} chunkId={cid} cap={ccap} "
                   f"dense_rank={dense_rank} bm25_rank={bm25_rank} => {verdict}")
        else:
            verdict = "疑似不在库"
            out = f"max_IoU={max_iou:.2f} (阈值{EXIST_TH}) chunkId={cid} cap={ccap} => {verdict}"
        stats[verdict] += 1
        print(f">>> {out}")
        print(f"    (存在性扫描耗时 {dt:.1f}s)\n")

    print("==== 诊断汇总 ====")
    for k, v in stats.items():
        print(f"  {v:>2} 题 -> {k}")

if __name__ == "__main__":
    main()
