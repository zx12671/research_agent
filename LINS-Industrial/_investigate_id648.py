# -*- coding: utf-8 -*-
"""
深入调查"知识在库却检索不到为何" —— 以 id648(电磨小磨头) 为代表性 miss。

步骤：
 1) 从 knowledge_text 分句，定位每个整句子串命中的 chunk_id 集合（这些=答案真实来源块）。
 2) 对这些 chunk 汇总 document_id / source / capability / 数量。
 3) 用 query 跑 纯dense / 纯bm25 / hybrid，看这些"答案源 chunk"被排到什么位次，
    定位检索失败发生在 dense 还是 bm25，还是都在 top-N 之外。
"""
import os, sys, io, re, csv, json
from collections import defaultdict, Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
QID = "648"


def split_sentences(text):
    parts = re.split(r"[。！？\n；;]", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 4]


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    r = next(x for x in rows if x.get("id") == QID)
    q, kt = r["question"], r["knowledge_text"]
    print(f"Q{ QID}: {q[:60]}\nktlen={len(kt)}\n")

    # 建倒排: 前缀 -> 候选 chunk (用每句前若干字预筛, 再精确整句匹配)
    paragraphs = list(open(CHUNKS, encoding="utf-8"))
    corpus = {}
    for line in paragraphs:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        c = d.get("content", "")
        if c:
            corpus[d["chunk_id"]] = d
    print(f"corpus chunks = {len(corpus)}\n")

    # 预筛: 每句前 6 字所命中的 chunk_id 作为疑似答案源
    hit_chunk_ids = set()
    per_sent = []
    for s in split_sentences(kt):
        found = [cid for cid, d in corpus.items() if s in d.get("content", "")]
        hit_chunk_ids.update(found)
        per_sent.append((s[:20], len(found)))
    print(f"句数={len(per_sent)}, 命中的源 chunk 数={len(hit_chunk_ids)}")
    print("\n命中的源 chunk 概览 (document/cap)")
    doc_counter = Counter()
    for cid in hit_chunk_ids:
        d = corpus[cid]
        doc_counter[(d.get("document_id"), d.get("capability"), d.get("source"))] += 1
    for (doc, cap, src), n in doc_counter.most_common():
        print(f"   doc={doc} cap={cap} src={src} chunks={n}")

    # 检索对比
    from retrieval.retriever import OpenDomainRetriever
    rt = OpenDomainRetriever()
    rt.load_from_manifest()

    def score_by(cids, chunks):
        """返回命中的源 chunk 在 chunks 序列中的位次列表(1-based)"""
        ranks = [i + 1 for i, ck in enumerate(chunks)
                 if getattr(ck, "chunk_id", None) in cids
                 or getattr(ck, "id", None) in cids]
        return ranks

    print("\n==== 检索定位（答案源 chunk 被排在哪）====")
    k = 2000
    # dense
    dr = rt.dense_search(q, k=k)
    dense_ranks = score_by(hit_chunk_ids, dr.chunks)
    # bm25
    bq = rt.bm25_search(q, k=k)
    bm25_ids = [cid for cid, sc in bq]
    bm25_ranks = [i + 1 for i, cid in enumerate(bm25_ids) if cid in hit_chunk_ids]
    # hybrid
    hr = rt.hybrid_retrieve(q, k=k)
    hy_ranks = score_by(hit_chunk_ids, hr.chunks)

    dd = dr.chunks[0].score if dr.chunks else None
    def fmt(ranks):
        if not ranks:
            return "MISS (top2000 未命中)"
        return f"命中 {len(ranks)} 个, 首位 {min(ranks)}位, 全集 {sorted(ranks)[:8]}..."
    print(f"  dense  : {fmt(dense_ranks)}")
    print(f"  bm25   : {fmt(bm25_ranks)}")
    print(f"  hybrid : {fmt(hy_ranks)}")
    # 关键: 它们在检索里的最高位
    if dense_ranks:
        print(f"  >> 答案源在 dense 的最佳位次 = {min(dense_ranks)}")
    if bm25_ranks:
        print(f"  >> 答案源在 bm25 的最佳位次 = {min(bm25_ranks)}")

    # 打一个实际命中的源内容片段, 与 query 相关度
    s0 = next((cid for cid, d in corpus.items() if split_sentences(kt) and split_sentences(kt)[0] in d.get('content','')), None)
    if s0:
        print("\n首个答案句对应的源 chunk 内容片段:")
        print("  " + corpus[s0]["content"][:150])


if __name__ == "__main__":
    main()
