# -*- coding: utf-8 -*-
"""
核验"疑缺库"判定是否为误判。

背景：_diag_recall_entrance2 用"整段 knowledge_text 对单块的字形 IoU<0.20"判"疑缺库"。
但 chunk 是文档按 total_chunks>1 切块的，knowledge_text 又是跨越多个源块的答案段落，
整段对比单块字形重叠天然极低 => 存在系统性误判风险。

本脚本改用"句子级存在性"验证：
  - 把 knowledge_text 按分句切分；
  - 对每个句子在全库找最大字符重叠：
      * 硬证据: 句子整句是某 chunk content 的子串(needle in haystack)
      * 宽证据: 句子相对某 chunk 的字符重叠率 orelap >= 0.5
  - 统计"句子命中率" => 若高(如>=60%)，铁证"其实在库"，原"疑缺库"判错。

用法: python _verify_knowledge_in_corpus.py
"""
import os, sys, io, re, csv, json
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")

# 与诊断一致的 14 个"疑缺库"题的问句特征开头（取 question 片段做定位）
MISS_QS = [
    "电磨（直磨机）在高速使用小磨头",
    "在节水灌溉项目后评价中",
    "在使用SYT-2000数字微压计",
    "在喷涂加工中，为了确保橡胶漆",
    "在制冷压缩机系统维修后",
    "DZSF型直线振动筛",
    "A68无线门铃系统",
    "在GB/T 2423.59",
    "根据GB/T 7350",
    "在GB893.1标准中",
    "在电子测量仪器中，用于表示测量",
    "在纺织材料性能测试中",
    "在采用钼蓝分光光度法",
    "在配制用于容量分析的亚甲基蓝",
]

def split_sentences(text):
    # 按 。！？；\n 切句，去空
    parts = re.split(r"[。！？\n；;]", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 4]


def main():
    # 定位 miss 题
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    print(f"total rows = {len(rows)}")
    miss_rows = []
    for miss_q in MISS_QS:
        target = [r for r in rows if miss_q in (r.get("question") or "")]
        if target:
            miss_rows.append(target[0])
    print(f"定位到 {len(miss_rows)} 个 diagnose-miss 题\n")

    # 载入全库 content
    corpus_contents = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c = d.get("content", "")
            if c:
                corpus_contents.append(c)
    print(f"corpus contents = {len(corpus_contents)}\n")

    print("=" * 80)
    print(f"{'id':>5} {'句数':>4} {'硬命中(子串)':>12} {'宽命中(≥0.5)':>13} {'句命中率':>8}")
    print("-" * 80)
    agg_sent = agg_hard = agg_wide = 0
    for r in miss_rows:
        rid = r.get("id", "")
        kt = r.get("knowledge_text") or ""
        sents = split_sentences(kt)
        if not sents:
            print(f"{rid:>5}  (无分句, ktlen={len(kt)})")
            continue
        n_hard = 0
        n_wide = 0
        for s in sents:
            best = 0.0
            hard = False
            for c in corpus_contents:
                if s in c:                      # 整句是子串 -> 铁证
                    hard = True
                    n_hard += 1
                    break
            if hard:
                continue
            # 宽证据：句子与某个 chunk 的字符 bigram Jaccard >= 0.5（容忍改写/拆分）
            for c in _candidates(s, corpus_contents, max_cand=300):
                if _jaccard2(s, c) >= 0.5:
                    n_wide += 1
                    break
        rate = (n_hard + n_wide) / len(sents) * 100
        agg_sent += len(sents); agg_hard += n_hard; agg_wide += n_wide
        print(f"{rid:>5} {len(sents):>4} {n_hard:>10} {n_wide:>12} {rate:>7.0f}%")

    print("-" * 80)
    total_hit = agg_hard + agg_wide
    print(f"汇总: {agg_sent} 句, 整句子串命中 {agg_hard}, 2gram命中 {agg_wide}, "
          f"句子命中率 {total_hit/agg_sent*100:.0f}%")


def _norm(s):
    return str(s).replace(" ", "").replace("\u3000", "")


def _grams(s, n=2):
    s = _norm(s)
    return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}


def _jaccard2(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / (len(ga) + len(gb) - inter) if inter else 0.0


def _candidates(sentence, corpus, max_cand=300):
    """用句内稳定片段做子串预筛，返回少量候选 chunk，避免全库暴力。"""
    cand = []
    for probe in (sentence[:8], sentence[:6], sentence[:5]):
        if len(probe) < 3:
            continue
        cand = [c for c in corpus if probe in c]
        if cand:
            break
    return cand[:max_cand]


if __name__ == "__main__":
    main()
