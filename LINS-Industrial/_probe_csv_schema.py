# -*- coding: utf-8 -*-
"""探测 huggingface_dataset.csv 的列结构与各类"问题类型"分布。

回答：数据源里有哪些问题类型？
分别从多个维度输出：
  1) 全部列名（哪些列承载"类型"信息）
  2) capability / question_format(或 _format) / difficulty / industry_primary / domain 分布
  3) 若存在"类别/type/task"类隐含列一并列出
"""
import os
import csv
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_THIS_DIR, "data", "industrybench", "huggingface_dataset.csv")

with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print("文件:", CSV_PATH)
print("行数(含表头下记录):", len(rows))
print("\n全部列名 (n=%d):" % len(fieldnames))
for i, col in enumerate(fieldnames, 1):
    print(f"  [{i}] {col}")

# 各类别维度分布
CAT_COLS = ["capability", "question_format", "_format", "format", "difficulty",
            "industry_primary", "domain", "type", "task_type", "category"]
print("\n" + "=" * 60)
for col in CAT_COLS:
    if col not in fieldnames:
        continue
    print(f"\n■ 列 '{col}' 分布 (top 20)：")
    c = Counter((row.get(col) or "").strip() for row in rows)
    for k, v in c.most_common(20):
        print(f"  {k or '(空)':<30} {v}")
    print(f"  合计: {sum(c.values())}")
