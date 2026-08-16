# -*- coding: utf-8 -*-
"""
format × capability 融合分层召回复实测。
检索只依赖 chunk.capability(同能力率); _format 是题目侧标注(chunk上无format)。
融合方式 = 在 capability 之上按 _format 交叉切片, 检验同一能力下不同题型召回复现实测
是否有额外差异(capability 解释不了的部分), 判断 format 是否纯"输出格式"维 / 也影响"检索诉求"。
"""
import os
import csv
import random
from collections import defaultdict
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CELL = 8
K = 10
IOU_TH = 0.15

@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = s.replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}

def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb: return 0.0
    return len(ga & gb)/len(ga | gb)

def main():
    random.seed(42)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cells = defaultdict(list)
    for r in rows:
        cells[(r.get("_format"), r.get("capability"))].append(r)

    sample = []
    for (fmt, cap), lst in sorted(cells.items()):
        for r in random.sample(lst, min(PER_CELL, len(lst))):
            sample.append((fmt, cap, r["question"], r.get("knowledge_text", "")))
    print(f"format×cap 交叉单元数:{len(cells)} 样本:{len(sample)} (每格≤{PER_CELL})")

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    # 统计桶
    fmt_hit = defaultdict(lambda: {"n":0, "hit":{1:0,3:0,5:0,10:0}})
    cap_hit = defaultdict(lambda: {"n":0, "hit":{1:0,3:0,5:0,10:0}})
    cell = defaultdict(lambda: {"n":0, "hit":0, "cap_hits":0, "slots":0})

    for fmt, cap, q, kt in sample:
        if not q or not kt:
            continue
        res = retriever.retrieve(q, k=K, use_ked=True)
        chunks = res.chunks
        hb = 0
        for i, c in enumerate(chunks, 1):
            if hb == 0 and iou(kt, c.content) >= IOU_TH:
                hb = i
        f = fmt_hit[fmt]; f["n"] += 1
        g = cap_hit[cap]; g["n"] += 1
        c = cell[(fmt, cap)]; c["n"] += 1; c["slots"] += len(chunks)
        for ch in chunks:
            if ch.capability == cap:
                c["cap_hits"] += 1
        for kk in (1,3,5,10):
            if hb and hb <= kk:
                f["hit"][kk] += 1
                g["hit"][kk] += 1
                if kk == 10:
                    c["hit"] += 1


    fmt_order = ["问答题","填空题","选择题","计算题"]
    cap_order = ["选型与替代","标准规范与术语","工艺原理与参数影响","安全合规与风险控制",
                 "质量计量与检测","故障诊断与排查","工程计算与估算"]

    print("\n== 单维 _format 召回 ==")
    print(f"{'题型':<6}{'样本':>5}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}")
    for f in fmt_order:
        d = fmt_hit.get(f)
        if d and d["n"]:
            n = d["n"]
            print(f"{f:<6}{n:>5}{d['hit'][1]/n*100:>6.0f}%{d['hit'][3]/n*100:>6.0f}%"
                  f"{d['hit'][5]/n*100:>6.0f}%{d['hit'][10]/n*100:>7.0f}%")

    print("\n== 单维 capability 召回 ==")
    print(f"{'能力':<14}{'样本':>5}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}")
    n_valid, hits10 = 0, 0
    for cname in cap_order:
        d = cap_hit.get(cname)
        if d and d["n"]:
            n = d["n"]
            n_valid += n; hits10 += d["hit"][10]
            print(f"{cname:<14}{n:>5}{d['hit'][1]/n*100:>6.0f}%{d['hit'][3]/n*100:>6.0f}%"
                  f"{d['hit'][5]/n*100:>6.0f}%{d['hit'][10]/n*100:>7.0f}%")
    print(f"{'总体':<14}{n_valid:>5}{'':>7}{'':>7}{'':>7}{hits10/n_valid*100:>7.0f}%")



    def matrix(key):
        hdr = "fmt\\cap".ljust(10) + "".join(c[:4].rjust(9) for c in cap_order)
        print(f"\n== format × capability {key} ==")
        print(hdr)
        for f in fmt_order:
            line = f.ljust(10)
            for cap in cap_order:
                d = cell.get((f, cap))
                line += eval_expr(d, key).rjust(9)
            print(line)


    def eval_expr(d, key):
        if not d or d["n"] < 3:
            return "·"
        if key == "hit@10":
            return f"{d['hit']/d['n']*100:>6.0f}%"
        return f"{d['cap_hits']/d['slots']*100:>6.0f}%"

    matrix("hit@10")
    matrix("同能力率")

    # format 恒定性检验: 同一cap内 hit@10 的 format 内方差
    print("\n== 同一 capability 下, 不同 format 的 hit@10 差异(σ) ==")
    for cap in cap_order:
        vals = []
        for f in fmt_order:
            d = cell.get((f, cap))
            if d and d["n"] >= 3:
                vals.append(d["hit"]/d["n"]*100)
        if len(vals) >= 2:
            mean = sum(vals)/len(vals)
            var = sum((v-mean)**2 for v in vals)/len(vals)
            import math
            print(f"  {cap:<16} hit@10区间[{min(vals):.0f}%,{max(vals):.0f}%] 内format间σ={math.sqrt(var):.1f}")

if __name__ == "__main__":
    main()
