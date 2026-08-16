# -*- coding: utf-8 -*-
"""
A/B: 证据先验重排（方案二试点）——只动 evidence 重排，绝不下发 Solver。

目标：回答上一轮定性判断——"task 维度对 Solver 无贡献（A/B avg_delta=0）"。
这里转换试用场：把 task/format 分类作为【证据层先验信号】，仅用于 rescore/重排
Evidence，不再进入 Solver。量化它对 ground-truth 召回的得失。

比较三种排序（同一批真实 retrieve(q, k=10)）：
    baseline    : 原检索序（KED+dense+fallback）
    prior_small : 命中期望capability 的 chunk，score += PRIOR_DELTA 后重排
    prior_hard  : 期望capability 分区分到非期望之前（能力硬优先）

期望capability 由 format(题型) + task 关键词(问答题细分) 推断，与 analyzer 的
KEYWORD_MAP / normalize_format 口径对齐，但只烧在证据层。

度量：
    hit@1/3/5/10（AiOU>=IOU_TH 判定命中 knowledge_text）
    ground-truth 命中位次位移：improved / unchanged / worsened（量化先验得失）
    按 capability 分层：观察先验是否把某类能力全局阈值拉高
"""
import os
import csv
import re
import random
from collections import Counter, defaultdict
from functools import lru_cache

_EV = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_EV, "data", "industrybench", "huggingface_dataset.csv")
K = 10
IOU_TH = 0.15
PER_CELL = 6
PRIOR_DELTA = 0.15
SEED = 42

# ---------- format/题型 -> 期望 capability ----------
FMT2CAP = {
    "Calculation": ["工程计算与估算"],
    "MultipleChoice": ["选型与替代"],
    "FillBlank": ["标准规范与术语", "工艺原理与参数影响"],
    "QA": [],  # 见 CAP_SUB 按关键词细分
}
# ---------- task 关键词 -> 期望 capability（问答题细分） ----------
CAP_SUB = [
    (r"选型|选择|推荐|哪个|对比|差异|区别|更适合|适用", "选型与替代"),
    (r"标准|规范|GB|IEC|条款|要求|限值|规定", "标准规范与术语"),
    (r"故障|排查|诊断|失效|过压|烧毁|报警|污染", "故障诊断与排查"),
    (r"计算|多少|容量|压降|电流|电压|加热时间|公式|功率", "工程计算与估算"),
    (r"原理|概念|工作原理|是什么|如何|过程|机制", "工艺原理与参数影响"),
]


def expect_cap(question: str, q_fmt: str = "") -> list:
    """由 format+task 关键词推断期望 capability（仅证据层先验，非分类硬标签）。"""
    fmt = q_fmt
    if fmt in FMT2CAP and FMT2CAP[fmt]:
        return FMT2CAP[fmt]
    for pat, cap in CAP_SUB:
        if re.search(pat, question):
            return [cap]
    return []


# ---------- 命中度量（AiOU） ----------
@lru_cache(maxsize=None)
def _grams(s: str, n: int = 2):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def hit_rank(chunks, kt):
    """返回首个命中 ground-truth 的位次(1-based)，未命中返回 None。"""
    for i, c in enumerate(chunks, 1):
        if iou(kt, c.content) >= IOU_TH:
            return i
    return None


def prior_small(chunks, expect):
    return sorted(chunks, key=lambda c: -(c.score + (PRIOR_DELTA if c.capability in expect else 0.0)))


def prior_hard(chunks, expect):
    in_, out = [], []
    for c in chunks:
        (in_ if c.capability in expect else out).append(c)
    in_.sort(key=lambda c: -c.score)
    out.sort(key=lambda c: -c.score)
    return in_ + out


def main():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 按 (capability, _format) 交叉分层抽样
    cells = defaultdict(list)
    for r in rows:
        cells[(r.get("capability"), r.get("_format"))].append(r)
    sample = []
    for key, lst in sorted(cells.items()):
        for r in random.sample(lst, min(PER_CELL, len(lst))):
            sample.append(r)
    print(f"[样本] 交叉单元={len(cells)} 总量={len(sample)}")

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    methods = ["base", "small", "hard"]
    agg = defaultdict(lambda: Counter())
    delta = defaultdict(lambda: Counter())  # delta: improved/unchanged/worsened per method
    cap_n = defaultdict(int)

    from agentic.task_types import normalize_format

    NO_PRIOR = 0
    GIN = 0
    GIN_N = 0
    for r in sample:
        q = r.get("question", "")
        kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        q_fmt = normalize_format(r.get("_format", ""), q)
        expect = expect_cap(q, q_fmt)
        if not expect:
            NO_PRIOR += 1
        gt = r.get("capability", "")
        if expect:
            GIN_N += 1
            GIN += 1 if gt in expect else 0    # 检验先验前提：真值capability∈期望集合

        res = retriever.retrieve(q, k=K, use_ked=True)
        chunks = res.chunks
        if not chunks:
            continue
        base = hit_rank(chunks, kt)
        cand = {
            "base": chunks,
            "small": prior_small(chunks, expect),
            "hard": prior_hard(chunks, expect),
        }
        base_rank = base or (K + 1)
        cap_n[chunks[0].capability if hasattr(chunks[0], "capability") else ""] += 1
        for m in methods:
            hr = hit_rank(cand[m], kt)
            for kk in [1, 3, 5, 10]:
                if hr and hr <= kk:
                    agg[m]["hit%d" % kk] += 1
            agg[m]["n"] += 1
            if m != "base":
                hr_r = hr or (K + 1)
                if hr_r < base_rank:
                    delta[m]["improved"] += 1
                elif hr_r > base_rank:
                    delta[m]["worsened"] += 1
                else:
                    delta[m]["unchanged"] += 1

    print(f"[未命中期望capability(general先验缺省)] 样本数={NO_PRIOR}")
    if GIN_N:
        print(f"[先验前提检验] 真值capability∈期望集合 占比: {GIN}/{GIN_N} = {GIN / GIN_N * 100:.1f}%")

    print(f"\n{'方法':<8}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>9}"
          f"{'improved':>9}{'unchanged':>10}{'worsened':>9}")
    for m in methods:
        a = agg[m]
        n = a["n"] or 1
        d = delta[m]
        line = f"{m:<8}{n:>4}"
        for kk in [1, 3, 5, 10]:
            line += f"{a['hit%d' % kk] / n * 100:>8.1f}%"
        if m == "base":
            print(line)
        else:
            line += f"{d['improved']:>9}{d['unchanged']:>10}{d['worsened']:>9}"
            print(line)


if __name__ == "__main__":
    main()
