# -*- coding: utf-8 -*-
"""
临时小测试：capability × industry_primary 二维分层的检索召回复实测。

取样：按 (capability, industry) 交叉单元分层抽样，每单元上限 PER_CELL 题(默认5)，
      覆盖所有非空交叉组合，同时兼顾行业维度的代表性。

对每题 q 走真实检索 pipeline: retriever.retrieve(q, k=10) (KED+dense+fallback)。
Ground truth: 每题自带 knowledge_text，top-k 命中"相关chunk"用 汉字bigram IoU>=阈值 判定。
聚合:
  - 按 industry_primary(10类): hit@1/3/5/10 + 同行业率(c.industry==q.industry_primary 占比)
     对照行业问题占比 / 行业chunk占比, 检验行业维度比例错位是否转成召回劣化
  - 按 capability(7类): hit@1/3/5/10 + 同能力率(保留, 便于对照)
  - 交叉矩阵: 每 (cap,ind) 的 hit@10 与样本数(样本>=5 才展示)
"""
import os
import csv
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")

PER_CELL = 5
K = 10
IOU_TH = 0.15


@lru_cache(maxsize=None)
def _grams(s: str, n: int = 2):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def load_chunk_cap_industry():
    caps = Counter(); inds = Counter()
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            d = json.loads(s)
            caps[d.get("capability", "")] += 1
            inds[d.get("industry", "")] += 1
    return caps, inds


def main():
    random.seed(42)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # 交叉单元
    cells = defaultdict(list)
    for r in rows:
        cells[(r.get("capability"), r.get("industry_primary"))].append(r)

    sample = []
    for (cap, ind), lst in sorted(cells.items()):
        for r in random.sample(lst, min(PER_CELL, len(lst))):
            sample.append((cap, ind, r["question"], r.get("knowledge_text", "")))
    print(f"交叉单元数: {len(cells)} | 样本总数: {len(sample)}")

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    try:
        retriever.load_from_manifest()
    except Exception as e:
        print("manifest 加载失败:", e)
        retriever.load_from_manifest()

    # 聚合结构
    cap_agg = defaultdict(lambda: {"n": 0, "cap_hits": 0, "slots": 0, "hit": {1: 0, 3: 0, 5: 0, 10: 0}})
    ind_agg = defaultdict(lambda: {"n": 0, "ind_hits": 0, "slots": 0, "hit": {1: 0, 3: 0, 5: 0, 10: 0}})
    cell_agg = defaultdict(lambda: {"n": 0, "hit10": 0})

    for cap, ind, q, kt in sample:
        if not q or not kt:
            continue
        res = retriever.retrieve(q, k=K, use_ked=True)
        chunks = res.chunks
        hit_best = 0
        for i, c in enumerate(chunks, 1):
            if hit_best == 0 and iou(kt, c.content) >= IOU_TH:
                hit_best = i

        # capability 聚合
        ca = cap_agg[cap]; ca["n"] += 1; ca["slots"] += len(chunks)
        for c in chunks:
            if c.capability == cap:
                ca["cap_hits"] += 1
        # industry 聚合
        ia = ind_agg[ind]; ia["n"] += 1; ia["slots"] += len(chunks)
        for c in chunks:
            if c.industry == ind:
                ia["ind_hits"] += 1
        # 交叉聚合
        ce = cell_agg[(cap, ind)]; ce["n"] += 1
        if hit_best and hit_best <= 10:
            ce["hit10"] += 1

        for kk in [1, 3, 5, 10]:
            if hit_best and hit_best <= kk:
                ca["hit"][kk] += 1
                ia["hit"][kk] += 1

    c_cap, c_ind = load_chunk_cap_industry()
    tot_q = len(rows); tot_c = sum(c_ind.values())
    q_cap = Counter(r.get("capability") for r in rows)
    q_ind = Counter(r.get("industry_primary") for r in rows)

    def row(cap_label, agg, base_col, prior_q, prior_c, name_fmt=lambda i: i):

        n = agg["n"] or 1
        if base_col == "cap":
            rate = agg["cap_hits"] / agg["slots"] if agg["slots"] else 0
        else:
            rate = agg["ind_hits"] / agg["slots"] if agg["slots"] else 0
        print(f"{name_fmt(cap_label):<16}{n:>4}{agg['hit'][1]/n*100:>7.0f}%"
              f"{agg['hit'][3]/n*100:>7.0f}%{agg['hit'][5]/n*100:>7.0f}%"
              f"{agg['hit'][10]/n*100:>8.0f}%{rate*100:>9.0f}%"
              f"{prior_q/tot_q*100:>8.0f}%{prior_c/tot_c*100:>9.0f}%")

    cap_order = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
                 "质量计量与检测", "故障诊断与排查", "工程计算与估算"]
    ind_order = ["机械五金加工", "化工材料涂料", "电子仪表传感", "电工电器电力",
                 "通用跨行业", "冶金钢铁矿产", "能源储能新能源", "安防消防防爆",
                 "包装印刷", "纺织皮革"]

    print("\n==== 按 capability 分层（2D抽样版） ====")
    print(f"{'能力':<14}{'样本':>4}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}"
          f"{'同能力率':>9}{'问题占比':>8}{'chunk占比':>9}")
    for cap in cap_order:
        if cap in cap_agg:
            row(cap, cap_agg[cap], "cap", q_cap[cap], c_cap[cap])

    print("\n==== 按 industry_primary 分层 ====")
    print(f"{'行业':<14}{'样本':>4}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}"
          f"{'同行业率':>9}{'问题占比':>8}{'chunk占比':>9}")
    for ind in ind_order:
        if ind in ind_agg:
            row(ind, ind_agg[ind], "ind", q_ind[ind], c_ind[ind])

    print("\n==== 交叉矩阵 hit@10 (cap x industry), 样本数>=5 显示 ====")
    hdr = f"{'能力/行业':<18}"
    for ind in ind_order:
        hdr += f"{ind[:4]:>8}"
    print(hdr)
    for cap in cap_order:
        line = f"{cap:<18}"
        for ind in ind_order:
            ce = cell_agg.get((cap, ind))
            if ce and ce["n"] >= 5:
                line += f"{ce['hit10']/ce['n']*100:>7.0f}%"
            else:
                line += f"{'·':>8}"
        print(line)


if __name__ == "__main__":
    main()
