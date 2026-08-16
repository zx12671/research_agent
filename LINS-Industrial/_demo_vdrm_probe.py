# -*- coding: utf-8 -*-
r"""
_demo_vdrm_probe.py — 判断"VDRM 题 fail 是 bge-small 模型问题，还是分块/query 粒度问题"。

方法：同一题，对比不同 query 表达在 bge-small 下的 dense rank：
  原 query（长、含场景+术语）：dense rank=562（已知）
  改写 query：去掉冗余、突出唯一判别键"VDRM 不低于1200V 三相整流 晶闸管"
若改写后 dense rank 大幅提前 → 说明杠杆在 query 表达/分块粒度，不在"换更大模型"。
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
sys.path.insert(0, os.path.abspath(os.path.join(_LINS, "..")))
from retrieval.retriever import OpenDomainRetriever
import numpy as np

r = OpenDomainRetriever(project_root=_LINS)
r.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
print(f"[就绪] chunks={len(r.chunks)}")

TARGET = "80775e037f73"  # 含答案的核心 chunk
QUERIES = {
    "原 query (完整长句)":
        "在380V线电压的三相整流电路中，为确保安全裕度，应选择断态重复峰值电压（VDRM）不低于多少伏的普通晶闸管？",
    "改写1 (去场景/聚焦判别键)":
        "三相整流电路 选 VDRM 不低于1200V 的普通晶闸管",
    "改写2 (术语直陈·短查询)":
        "断态重复峰值电压 VDRM 选型 380V 三相整流 晶闸管 1200V",
    "改写3 (标准选型问句)":
        "三相整流电路中普通晶闸管断态重复峰值电压应选多少伏",
}
methods = {"dense": "dense_search", "bm25": "bm25_search"}

def dense_rank(q):
    ids = [c.chunk_id for c in r.dense_search(q, k=2000, use_ked=False).chunks]
    return ids.index(TARGET) + 1 if TARGET in ids else None

def bm25_rank(q):
    ids = [cid for cid, _ in r.bm25_search(q, k=2000)]
    return ids.index(TARGET) + 1 if TARGET in ids else None

print("\n%30s | dense rank | bm25 rank" % "query")
print("-" * 60)
for name, q in QUERIES.items():
    dr = dense_rank(q)
    br = bm25_rank(q)
    print("%28s | %10s | %9s" % (name, dr if dr else "top2000外", br if br else "top2000外"))

# 交叉参照：答案 chunk 在原 query 下的两个 "相似度邻居" 是不是同主题的其他 chunk（是否"抓对主题但排错位"）
qs = list(QUERIES.values())[0]
print("\n== 原 query 下答案 chunk rank 判定 ==")
ids = [c.chunk_id for c in r.dense_search(qs, k=600, use_ked=False).chunks]
print("  答案 chunk 在 top600 dense 内:", "YES rank=" + str(ids.index(TARGET)+1) if TARGET in ids else "NO")
