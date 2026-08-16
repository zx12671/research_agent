# -*- coding: utf-8 -*-
"""快速探测 chunks 的 industry 字段实际取值集合，与 CSV industry_primary 对照。"""
import os
import csv
import json
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")

c_ind = Counter()
c_csv = Counter()
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            d = json.loads(s)
            k = d.get("industry", "") or "(空)"
            c_ind[k] += 1
with open(CSV, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        k = r.get("industry_primary", "") or "(空)"
        c_csv[k] += 1

print("=== chunks.industry 取值 ===")
for k, v in c_ind.most_common():
    print(f"  {k!r:<16} {v}")
print("\n=== csv.industry_primary 取值 ===")
for k, v in c_csv.most_common():
    print(f"  {k!r:<16} {v}")
