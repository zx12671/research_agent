# -*- coding: utf-8 -*-
"""验证 VDRM 答案是否在库、以及库内表述形式 —— 判断 bge-small 是否该为此类 hard-miss 背锅。"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
CH = r"knowledge_corpus/chunks/industrybench_chunks.jsonl"

# 1) 含"晶闸管"的 chunk 样本
print("== 含[晶闸管]的库内 chunk 样本(前 5) ==")
n = 0
for ln in open(CH, encoding="utf-8"):
    c = json.loads(ln).get("content", "")
    if "晶闸管" in c:
        print("  -", c[:80].replace("\n", " "))
        n += 1
        if n >= 5:
            break

# 2) 关键 term 在全库的出现次数
print("\n== 关键 term 在全库 chunk 出现次数 ==")
for t in ["VDRM", "断态重复峰值电压", "普通晶闸管", "1200", "选型"]:
    cnt = 0
    for ln in open(CH, encoding="utf-8"):
        if t in json.loads(ln).get("content", ""):
            cnt += 1
    print(f"  [{t}] = {cnt}")

# 3) 完整打印含 VDRM 的 chunk（若存在），看答案"1200V 选型"是否落在此处
print("\n== 含[VDRM]或[断态重复峰值电压]的 chunk 完整内容 ==")
shown = 0
for ln in open(CH, encoding="utf-8"):
    c = json.loads(ln).get("content", "")
    if "VDRM" in c or "断态重复峰值电压" in c:
        print("  --- chunk:", json.loads(ln).get("chunk_id"), "---")
        print("  " + c.replace("\n", " | ")[:400])
        shown += 1
        if shown >= 3:
            break
if shown == 0:
    print("  ✗ 全库无 VDRM / 断态重复峰值电压 字样")
