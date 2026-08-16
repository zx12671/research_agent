# -*- coding: utf-8 -*-
"""精确对比 knowledge_corpus/chunks (55095条) 与 2049 问题 的分类标签集合。

核心结论面向：知识集的类型与问题类型能否匹配。
  1) capability 7类逐值对照（问题数 vs chunk 数）
  2) industry 10类逐值对照
  3) 标签集合完全一致性判断 + 比例偏离量化
"""
import os
import json
import csv
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")

chunks = []
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            chunks.append(json.loads(s))

with open(CSV, encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))

c_cap = Counter((c.get("capability") or "").strip() for c in chunks)
c_ind = Counter((c.get("industry") or "").strip() for c in chunks)
q_cap = Counter((q.get("capability") or "").strip() for q in qs)
q_ind = Counter((q.get("industry_primary") or "").strip() for q in qs)

def table(title, qc, cc):
    print("\n### " + title)
    all_keys = list(dict.fromkeys(list(qc.keys()) + list(cc.keys())))
    print(f"{'标签':<16}{'问题数':>8}{'chunk数':>10}{'问题占比':>9}{'chunk占比':>9}")
    for k in all_keys:
        qv, cv = qc.get(k, 0), cc.get(k, 0)
        qp = qv / sum(qc.values()) * 100 if sum(qc.values()) else 0
        cp = cv / sum(cc.values()) * 100 if sum(cc.values()) else 0
        print(f"{k or '(空)':<16}{qv:>8}{cv:>10}{qp:>8.1f}%{cp:>8.1f}%")
    print(f"标签值集合完全一致:", set(qc.keys()) == set(cc.keys()))
    print(f"问题侧类数={len(qc)} chunk侧类数={len(cc)}")

table("能力维度 capability", q_cap, c_cap)
table("行业维度 industry vs industry_primary", q_ind, c_ind)

# 缺标签覆盖率：每个 capability 的 chunk 是否为 0（即该类型问题是否完全没有对应知识）
print("\n" + "=" * 50)
print("逐能力类型 coverage 检查（chunk数>0?）:")
for k, v in q_cap.most_common():
    cv = c_cap.get(k, 0)
    flag = "OK" if cv > 0 else "!!! 缺该类型知识 !!!"
    print(f"  {k:<16} 问题数={v:>5}  chunk数={cv:>6}  {flag}")
