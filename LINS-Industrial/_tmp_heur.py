# -*- coding: utf-8 -*-
import csv
from collections import Counter, defaultdict
from agentic.task_types import infer_format_heuristic
from agentic.analyzer import TaskAnalyzer

cn = {"问答题": "QA", "填空题": "FillBlank", "选择题": "MultipleChoice", "计算题": "Calculation"}
rows = list(csv.DictReader(open("data/industrybench/huggingface_dataset.csv", encoding="utf-8-sig")))
an = TaskAnalyzer()
cm = defaultdict(lambda: defaultdict(int))
cman = defaultdict(lambda: defaultdict(int))
corr = corr_an = n = 0
dist = Counter()
for r in rows:
    q = r["question"]
    gt = cn.get(r.get("_format", ""), "?")
    h = infer_format_heuristic(q)
    a = an.analyze(q).format
    dist[h] += 1
    cm[gt][h] += 1
    cman[gt][a] += 1
    corr += (gt == h)
    corr_an += (gt == a)
    n += 1
print(f"heuristic fmt acc = {100*corr/n:.1f}% ({corr}/{n})  pred dist = {dict(dist)}")
print(f"analyze().format acc = {100*corr_an/n:.1f}%")
print("heuristic 混淆(行=真值):")
for gt in ["QA", "FillBlank", "MultipleChoice", "Calculation"]:
    print(f"  {gt:<16} {dict(cm[gt])}")
