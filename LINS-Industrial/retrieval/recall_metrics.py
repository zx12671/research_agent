# -*- coding: utf-8 -*-
"""
recall_metrics.py —— 检索召回评估的【统一口径】模块（P0 落点）。

为什么存在：
   历史上有大量 recall 探针脚本各自内联了"整段 bigram IoU 判定命中"的口径，
   得出"约 50% GT 在 top-50 外"式的系统性低估结论。
   docs/agentic_recall_entrance_diagnosis.md 已证明这是口径失真，而非检索真漏。

P0 规范（本模块落地）：
   1. 检索优化/回退的【主验收口径】= GT 句子覆盖率 sent_coverage()。
      - GT 长文按标点切句，逐句判断是否被 top-K 某 chunk 覆盖(句级 bigram-Jaccard>=th)。
      - 返回 0~1 连续分数：这题 top-K 拼齐了 GT 答案的百分之几。
   2. 整段 IoU 命中(hit@k) 仅保留作【并行交叉参照】，不再作为结论依据。
   3. 鼓励多口径交叉：任何单一字符/词法判据都会在某类题上失真
      (整段低估 / 句级对长句-改写漏报)，应并列呈现而非择一盖章。

用法：
   from retrieval.recall_metrics import sent_coverage, doc_hit_position, coverage_summary
"""
import re
from functools import lru_cache

#: 句级命中阈值（与 docs 对齐；与 completeness_correction 一致取 0.30）
SENT_IOU_TH = 0.30
#: 覆盖"达标"阈值：cov>=该值视为"这题基本拼齐了答案"
GOOD_COV = 0.5


@lru_cache(maxsize=None)
def bigram_grams(s: str, n: int = 2):
    """汉字 bigram（2-gram）字符片段集合，去掉空格，避免英文词间空格稀释。"""
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def bigram_iou(a, b):
    """两段文本的 bigram Jaccard(IoU)，∈[0,1]。空集返回 0。"""
    ga, gb = bigram_grams(a), bigram_grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def split_sent(text: str):
    """按中文/英文句末标点切句，过滤过短片段(>=6 字符才视为有信息量的句)。"""
    parts = re.split(r"(?<=[。；;!?！？])\s*", text or "")
    return [p for p in parts if len(p.strip()) >= 6]


def get_content(c):
    """兼容 RetrievedChunk 对象 / dict：统一取 content 文本。"""
    return getattr(c, "content", None) or (c.get("content") if isinstance(c, dict) else "")


def sent_coverage(kt, chunks, th=SENT_IOU_TH):
    """【主验收口径】GT 句子覆盖率。

    GT 长文按句子拆分，逐句看是否被任一 chunk 覆盖（句级 bigram-Jaccard >= th）。

    Args:
        kt: ground-truth 长文(str)
        chunks: 检索返回的块(RetrievedChunk 或 dict)
        th: 句级命中阈值

    Returns:
        (coverage, covered_n, total_n)  覆盖比例∈[0,1] + 命中句数 + 总句数
    """
    sents = split_sent(kt)
    if not sents:
        return 1.0, 0, 0
    contents = [get_content(ch) for ch in chunks]
    covered = sum(1 for sc in sents
                  if any(bigram_iou(sc, cc) >= th for cc in contents))
    return covered / len(sents), covered, len(sents)


def doc_hit_position(chunks, kt, th=0.20):
    """【并行交叉口径·整段 IoU】首个命中 GT(整段)的位次(1-based)，未命中返回 None。

    仅作参照；不单独作为结论依据（整段口径会系统性低估，见模块 docstring）。
    """
    for i, c in enumerate(chunks, 1):
        content = get_content(c)
        if content and bigram_iou(kt, content) >= th:
            return i
    return None


def sent_covered_by_topk(kt, chunks, th=SENT_IOU_TH):
    """返回被覆盖的 GT 句子 id 列表（供逐题细盯）。"""
    sents = split_sent(kt)
    contents = [get_content(ch) for ch in chunks]
    return [i for i, sc in enumerate(sents)
            if any(bigram_iou(sc, cc) >= th for cc in contents)]


def coverage_summary(covs, good=GOOD_COV):
    """一组句子覆盖率的概要统计。

    Returns:
        dict(mean, median, good_rate, n)
        good_rate: cov>=good 的比例（"拼得齐答案"的题占比）
    """
    import statistics
    n = len(covs)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "good_rate": 0.0, "n": 0}
    return {
        "mean": round(statistics.mean(covs), 4),
        "median": round(statistics.median(covs), 4),
        "good_rate": round(sum(1 for c in covs if c >= good) / n, 4),
        "n": n,
    }

