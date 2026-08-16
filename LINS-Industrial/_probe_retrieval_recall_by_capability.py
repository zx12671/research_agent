# -*- coding: utf-8 -*-
"""
临时小测试：按 capability 分层的检索召回率。

目标：验证"知识集比例错位"(标准63.4% vs 选型10.1%/工艺9.2%) 是否真实转化为检索劣化。

方法（抽样的临时小测试）：
  - 每 capability 抽取 N 题（默认 25），总样本可控。
  - 对每题 q 走真实检索 pipeline: retriever.retrieve(q, k=10) (KED+dense+fallback)。
  - Ground truth: 每题自带 knowledge_text（标准答案知识）。
    判定 top-k 是否命中"相关 chunk" —— 用 knowledge_text 与 chunk.content 的汉字 bigram IoU，>=阈值即算命中。
  - 输出按 capability 聚合的 hit@1/3/5/10 + 同能力命中率(top-10中capability==q.capability占比)。
  - 与两类先验对比：问题占比 / chunk占比.
"""
import os
import csv
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
MANIFEST = os.path.join(_LINS, "knowledge_corpus", "manifest.json")

PER_CAP = 25            # 每 capability 抽样题数
K = 10
IOU_TH = 0.15           # bigram IoU 阈值（命中即视为相关）


@lru_cache(maxsize=None)
def _grams(s: str, n: int = 2):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def load_questions():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    return rows, by_cap


def main():
    random.seed(42)
    rows, by_cap = load_questions()
    print("总问题数:", len(rows), "| 能力类:", list(by_cap.keys()))

    # 抽样
    sample = []
    for cap, lst in by_cap.items():
        pick = random.sample(lst, min(PER_CAP, len(lst)))
        for r in pick:
            sample.append((r["capability"], r["question"], r.get("knowledge_text", "")))
        print(f"  抽样 {cap}: {len(pick)} 题 / 该类共 {len(lst)} 题")
    print("样本总数:", len(sample))

    # 加载检索器（复用 55095 条的既有 bge-small FAISS 索引）
    from retrieval.retriever import OpenDomainRetriever
    from retrieval import embedder as _emb_ns
    retriever = OpenDomainRetriever()
    try:
        retriever.load_from_manifest(manifest_path=MANIFEST)
    except Exception as e:
        print("manifest 加载失败:", e)
        retriever.load_from_manifest()

    # 检索 + 打分
    per = defaultdict(lambda: {"n": 0, "cap_hits": 0, "cap_slots": 0,
                               "hit": {1: 0, 3: 0, 5: 0, 10: 0}})
    for cap, q, kt in sample:
        if not q or not kt:
            continue
        res = retriever.retrieve(q, k=K, use_ked=True)
        chunks = res.chunks
        # 每题的 top-K 同能力命中记录
        p = per[cap]
        p["n"] += 1
        p["cap_slots"] += len(chunks)
        for c in chunks:
            if c.capability == cap:
                p["cap_hits"] += 1
        # hit@k 判定
        hit_best = 0
        for i, c in enumerate(chunks, 1):
            if hit_best == 0 and iou(kt, c.content) >= IOU_TH:
                hit_best = i
        for kk in [1, 3, 5, 10]:
            if hit_best and hit_best <= kk:
                p["hit"][kk] += 1

    # 输出
    q_cap = Counter((r.get("capability") or "") for r in rows)
    c_cap = Counter()
    with open(os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                c_cap[json.loads(s).get("capability", "")] += 1
    tot_q = sum(q_cap.values()); tot_c = sum(c_cap.values())

    print("\n==== 按 capability 分层的检索召回率（小样本） ====")
    print(f"{'能力':<14}{'样本':>4}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}"
          f"{'同能力率':>9}{'问题占比':>8}{'chunk占比':>9}")
    for cap in [_ for _ in ["选型与替代", "标准规范与术语", "工艺原理与参数影响",
                            "安全合规与风险控制", "质量计量与检测",
                            "故障诊断与排查", "工程计算与估算"] if _ in per]:
        p = per[cap]
        n = p["n"] or 1
        cap_rate = p["cap_hits"] / p["cap_slots"] if p["cap_slots"] else 0
        qp = q_cap[cap] / tot_q * 100
        cp = c_cap[cap] / tot_c * 100
        print(f"{cap:<14}{n:>4}{p['hit'][1]/n*100:>6.0f}%{p['hit'][3]/n*100:>6.0f}%"
              f"{p['hit'][5]/n*100:>6.0f}%{p['hit'][10]/n*100:>7.0f}%{cap_rate*100:>8.0f}%"
              f"{qp:>7.0f}%{cp:>8.0f}%")


if __name__ == "__main__":
    main()
