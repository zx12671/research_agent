# -*- coding: utf-8 -*-
"""
A/B 验证：Query 端"多视角分解检索"是否真的抬升句级覆盖率？
对比三路检索入口（同一 30 样本, seed=7, 最终 top-10, use_ked=True）：
    A) retrieve(裸单查询)
    B) multi_query_retrieve（ked.decompose 按逗号切子问题 → 逐子检索 → RRF 融合）
    C) hybrid_retrieve（multi-query dense + BM25 sparse → RRF）

度量（完全复用 _diag_recall_sentence.py 口径）：
    - hit_new: coverage>=1 句（句级命中; old整段IoU 已证作废）
    - covMean/covMed: 平均/中位覆盖率
    - 重点关注 C 类(并列复合)与 B 类(hard miss) 在不同入口下的差异，
      用于判断"当前正则拆分(切逗号)是否足够，能否上线"。
"""
import os, csv, random
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7
IOU_SENT_TH = 0.30
MIN_COVERED = 1


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def split_sent(text):
    import re
    parts = re.split(r"(?<=[。；;！？])\s*", text or "")
    return [p for p in parts if len(p.strip()) >= 6]


def _content(c):
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "") or ""


def sample_questions():
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r.get("capability") or "", []).append(r)
    s = []
    for cap, lst in sorted(by_cap.items()):
        s += random.sample(lst, min(PER_CAP, len(lst)))
    return s[:N]


def coverage(sents, chunks):
    covered = 0
    for s_c in sents:
        if any(jt(s_c, c) >= IOU_SENT_TH for c in chunks):
            covered += 1
    total = len(sents)
    return covered, total, (covered / total if total else 0.0)


def main():
    import time
    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    entries = {
        "A_single": lambda q: retriever.retrieve(q, k=10, use_ked=True),
        "B_multiquery": lambda q: retriever.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion"),
        "C_hybrid": lambda q: retriever.hybrid_retrieve(q, k=10, use_ked=True,
                                                        use_multi_query=True, use_sparse=True),
    }
    # 用同一份问题独立跑三路（不同随机不影响检索，仅入参相同即可）
    aggs = {k: [] for k in entries}         # 每题覆盖率
    hitcnt = {k: 0 for k in entries}
    rows = []
    t0 = time.time()
    for r in sample:
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        cap = r.get("capability", "")
        sents = split_sent(kt)
        qshort = q[:26]
        row = {"q": qshort, "cap": cap, "nSentGT": len(sents)}
        covB = 0
        for name, fn in entries.items():
            res = fn(q)
            chunks = [_content(c) for c in getattr(res, "chunks", [])]
            cov, total, ratio = coverage(sents, chunks)
            covB = max(covB, cov)
            aggs[name].append(ratio)
            if cov >= MIN_COVERED:
                hitcnt[name] += 1
            row[name] = f"{cov}/{total}({ratio:.2f})"
        # B类/危险：三路都覆盖不到1句 → 记录为一类
        row["_all_0"] = covB < MIN_COVERED
        rows.append(row)
    elapsed = time.time() - t0

    # ---------- 汇总 ----------
    print(f"A/B 三入口句级覆盖率对比 (N={len(sample)}, seed={SEED}, top-10, use_ked)")
    print(f"耗时 {elapsed:.1f}s\n")
    print(f"{'Q':28}{'cap':10}{'#S':>4}  {'A_single':>16}{'B_multi':>16}{'C_hybrid':>16}  全0")
    print("-" * 105)
    for row in rows:
        flag = " ★" if row["_all_0"] else ""
        print(f"{row['q']:28}{row['cap'][:9]:10}{row['nSentGT']:>4}  "
              f"{row['A_single']:>16}{row['B_multiquery']:>16}{row['C_hybrid']:>16}   {str(row['_all_0']):>5}{flag}")
    print("-" * 105)

    import statistics
    print("【命中率】coverage>=1 句:")
    for k in entries:
        print(f"   {k:14}: {hitcnt[k]}/{N} = {hitcnt[k]/N*100:.0f}%")
    print("\n【覆盖率均值 / 中位数】:")
    for k in entries:
        mean = statistics.mean(aggs[k])
        med = statistics.median(aggs[k])
        print(f"   {k:14}: mean={mean:.3f}  med={med:.3f}")

    # C 类判断：B/C 相对 A 的增益
    gainB = statistics.mean(aggs["B_multiquery"]) - statistics.mean(aggs["A_single"])
    gainC = statistics.mean(aggs["C_hybrid"]) - statistics.mean(aggs["A_single"])
    print("\n【相对单查询的全局覆盖率增益】")
    print(f"   B_multi vs A_single : {gainB:+.4f}")
    print(f"   C_hybrid vs A_single : {gainC:+.4f}")

    # 三路都覆盖0句的题（B类 hard miss 数量）
    all0 = sum(1 for row in rows if row["_all_0"])
    print(f"\n三路均覆盖0句(hard miss)题数: {all0} / {N}")


if __name__ == "__main__":
    main()
