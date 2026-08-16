# -*- coding: utf-8 -*-
"""
召回入口逐题诊断 v2 —— 语义级存在性判定。

v1 用纯字形 IOU(0.30) 判"不在库"，但 16 题全部判"疑似不在库"很可疑：
knowledge_text 是标准答案文本，与其承载 chunk 之间可能是"语义等价但字形不同"，
字形 IoU 天然低，不能仅凭它判缺库。

v2 改进：对每个 miss 题做三路证据：
  A) 字形存在性：knowledge_text 对全量 55095 块的最大字形 IoU（放宽到 0.20）。
  B) 语义存在性：以 knowledge_text 本身为 query 做 dense 检索（答案本体 embedding），
     看 top-K 里有没有 IoU>=0.20 的块 / 或最大语义相似度。这定位"这段知识到底在不在库"。
  C) 信号失灵判定：若 B 证明知识在库，再看 query 对该目标块的 dense_rank / bm25_rank，
     看是不是"query 捞不出它"（dense/bm25 信号失灵）。
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
IOU_HIT_TH = 0.20      # 字形命中阈值（与 _probe_recall_30 对齐）
SEED = 7
EMB_K = 20             # 以 knowledge_text 检索的深度

@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)

def _as_dict(chunk):
    """统一 chunk 接口：dict 或对象 -> dict(content=, id=, cap=)。"""
    if isinstance(chunk, dict):
        return {"content": chunk.get("content", ""),
                "id": chunk.get("id") or chunk.get("chunk_id"),
                "cap": chunk.get("capability", "")}
    return {"content": getattr(chunk, "content", "") or "",
            "id": getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None),
            "cap": getattr(chunk, "capability", "") or ""}

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
    import json
    chunks = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d = json.loads(s)
            chunks[d["chunk_id"]] = d
    return chunks

def find_gt_chunk_by_iou(kt, corpus):
    best = (0.0, None, "")
    for cid, d in corpus.items():
        c = d.get("content", "")
        sc = iou(kt, c) if c else 0.0
        if sc > best[0]:
            best = (sc, cid, d.get("capability", ""))
    return best

def main():
    sample = sample_questions()
    print(f"[样本] {len(sample)} 题")
    corpus = load_corpus_chunks()
    print(f"[corpus] {len(corpus)} chunk")

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    # 定位 top-10 miss（single）
    miss = []
    for r in sample:
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        res = retriever.retrieve(q, k=10, use_ked=True)
        chunks = getattr(res, "chunks", [])
        hit = any(iou(kt, _as_dict(c)["content"]) >= IOU_HIT_TH for c in chunks)
        if not hit:
            miss.append(r)
    print(f"[top-10 miss] {len(miss)} 题\n")

    stats = Counter()
    idx = 0
    for r in miss:
        idx += 1
        q, kt, cap = r.get("question", ""), r.get("knowledge_text", ""), r.get("capability", "")
        print(f"--- [{idx}/{len(miss)}] cap={cap} ---")
        print(f"Q   : {q[:55]}")

        # A) 字形存在性（全量）
        t0 = time.time()
        max_iou, cid, ccap = find_gt_chunk_by_iou(kt, corpus)
        print(f"  [A 字形存在性] 全量max_IoU={max_iou:.2f} chunk={cid} cap={ccap}  ({time.time()-t0:.1f}s)")

        # B) 语义存在性：knowledge_text 自检索全量
        #    对全量 54921/55095 逐块 dense 打分太重，退而用 retriever.dense_search(kt, k=EMB_K)
        #    看那 top-K 里有没有与 kt 字形重叠>=0.20 的（即答案本体现在库里的直接证据）
        sem_hit = False
        sem_top_iou = 0.0
        try:
            dr = retriever.dense_search(kt, k=EMB_K)
            for c in dr.chunks:
                cd = _as_dict(c)
                sc = iou(kt, cd["content"])
                sem_top_iou = max(sem_top_iou, sc)
                if sc >= IOU_HIT_TH:
                    sem_hit = True
            print(f"  [B 语义存在性] knowledge_text密集检索 top{EMB_K} 最大IoU={sem_top_iou:.2f} "
                  f"命中={sem_hit}")
        except Exception as e:
            print(f"  [B] dense(kt) err: {e}")

        # 若 B 里没命中，把密检索深度拉到全量（用 bge 全打分的 rank 上限截断，
        # 尝试定位答案在不在；dense_search 支持大 k 则全量）
        if not sem_hit:
            try:
                dr_full = retriever.dense_search(kt, k=2000)
                # 取 top2000 里的最大 IoU，若仍为 0，说明答案文本几乎没有 dense 近邻
                best_in_2000 = max([iou(kt, _as_dict(c)["content"]) for c in dr_full.chunks] or [0.0])
                print(f"  [B*] knowledge_text 扩到 top2000 最大IoU={best_in_2000:.2f}")
                sem_hit = best_in_2000 >= IOU_HIT_TH
            except Exception as e:
                print(f"  [B*] deep err: {e}")

        # C) 信号失灵：仅当语义存在性确认知识在库才检查 query 的 rank
        if sem_hit:
            dense_rank = bm25_rank = None
            try:
                dq = retriever.dense_search(q, k=2000)
                for i, c in enumerate(dq.chunks, 1):
                    if iou(kt, _as_dict(c)["content"]) >= IOU_HIT_TH:
                        dense_rank = i
                        break
            except Exception as e:
                print(f"  [C dense] err: {e}")
            try:
                # bm25_search 返回 [(chunk_id, score), ...]，按得分降序
                bq = retriever.bm25_search(q, k=2000)
                for i, (cid_b, _sc) in enumerate(bq, 1):
                    cd_b = corpus.get(cid_b, {})
                    if iou(kt, cd_b.get("content", "")) >= IOU_HIT_TH:
                        bm25_rank = i
                        break
            except Exception as e:
                print(f"  [C bm25] err: {e}")
            print(f"  [C 信号] query对该目标块 dense_rank={dense_rank} bm25_rank={bm25_rank}")

        # 归类
        if sem_hit:
            verdict = "在库(语义)"
            if dense_rank is not None and dense_rank <= 50:
                verdict += "+dense前50"
            elif dense_rank is not None:
                verdict += "+match弱"
        else:
            verdict = "疑缺库(语义&字形皆低)"
        stats[verdict] += 1
        print(f"  => {verdict}\n")

    print("==== 诊断汇总 ====")
    for k, v in stats.items():
        print(f"  {v:>2} 题 -> {k}")

if __name__ == "__main__":
    main()
