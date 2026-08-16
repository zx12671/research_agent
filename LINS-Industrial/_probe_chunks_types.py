# -*- coding: utf-8 -*-
import os
import json
import csv
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")

print("chunks:", CHUNKS, "存在:", os.path.exists(CHUNKS), "大小(B):", os.path.getsize(CHUNKS))

rows = []
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print("chunk 条数:", len(rows))

keys = Counter()
for r in rows:
    for k in r.keys():
        keys[k] += 1
print("\nchunk 字段出现频次:")
for k, v in keys.most_common():
    print(f"  {k}: {v}/{len(rows)}")

print("\n首条 chunk(节选):")
print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])

TYPE_CANDIDATES = ["capability", "industry", "industry_primary", "category",
                   "source", "doc_id", "document_id", "domain", "task_type", "type",
                   "capabilities", "kinds"]
print("\n" + "=" * 60)
for col in TYPE_CANDIDATES:
    vals = Counter((r.get(col) or "").strip() for r in rows)
    if vals:
        print(f"\n■ chunk 字段 '{col}' 分布 (top 20)：")
        for k, v in vals.most_common(20):
            print(f"  {k[:40] or '(空)':<42} {v}")

print("\n" + "=" * 60)
print("对照 CSV(2049) 分类维度：\n")
with open(CSV, encoding="utf-8-sig", newline="") as f:
    qs = list(csv.DictReader(f))
q_cap = Counter((r.get("capability") or "").strip() for r in qs)
q_ind = Counter((r.get("industry_primary") or "").strip() for r in qs)

def has_field(name):
    return any(name in r for r in rows)

print("chunks 含 capability 字段:", has_field("capability"),
      "| CSV 能力类数:", len(q_cap), "| chunk capability 不同值数:", len({r.get('capability','') for r in rows}))
print("chunks 含 industry 字段:", has_field("industry") or has_field("industry_primary"),
      "| CSV 行业类数:", len(q_ind), "| chunk industry 不同值数:", len({r.get('industry', r.get('industry_primary','')) for r in rows}))
