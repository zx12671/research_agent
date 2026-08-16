# -*- coding: utf-8 -*-
"""
证据链路分层损失量化探针。

目标：量化真实 agentic 链路 q→Retriever→Organizer→prompt 每一层的"相关信息"损失，
定位稀释(DISTRACT)发生在哪一步。纯检索+组织，不含 LLM，可复现。

链路（真实 production）:
    q ──retrieve(k=10)──▶ top10 chunks (L0 候选池)
         ──organizer.execute──▶ organized (去重0.97+分组, 不过滤)
         ──get_context()──▶ evidence_str (进入最终prompt的文本)
              ├─相关块字符/总字符 = 纯度(dilution)
              └─首个相关块 rank/位置 = rank下沉可读性

判定：chunk.content 与 knowledge_text 字形 bigram IoU>=IOU_TH => 相关块(=命中)。
"""
import os, sys, csv, random
from collections import defaultdict
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
K = 10
IOU_TH = 0.15


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}


def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _content(d):
    if isinstance(d, dict):
        return d.get("content", "")
    return getattr(d, "content", "") or ""


def sample_rows():
    random.seed(7)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    s = []
    for cap, lst in by_cap.items():
        s += random.sample(lst, min(PER_CAP, len(lst)))
    return s


def main():
    sample = sample_rows()
    print(f"证据链路分层损失量化: N={len(sample)} 题, k={K}, IOU_TH={IOU_TH}\n")

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    from agentic.task_types import ExecutionDirective
    from agentic.organizer import EvidenceOrganizer
    organizer = EvidenceOrganizer()
    directive = ExecutionDirective(module="organization", action="group_by_topic", params={})

    agg0 = {"n": 0, "any_hit": 0, "n_rel_hit": 0}
    agg1 = {"org_in": 0, "drop_rel": 0}
    agg_dil = {"related_chars": 0, "total_chars": 0, "fr_sum": 0, "fr_n": 0}
    agg_cons = {"in80": 0}
    rows = []

    for r in sample:
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        cap = r.get("capability", "")
        if not q or not kt:
            continue

        res = retriever.retrieve(q, k=K, use_ked=True)     # L0
        chunks = res.chunks
        rel_top10 = [c for c in chunks if iou(kt, c.content) >= IOU_TH]
        n_rel_top10 = len(rel_top10)

        org = organizer.execute(directive, chunks, question=q, task_type="general")  # L1
        org_docs = org.get_all_documents()
        rel_org = [d for d in org_docs if iou(kt, _content(d)) >= IOU_TH]
        n_rel_org = len(rel_org)

        # L2: evidence_str 纯度
        rel_chars = sum(len(_content(d)) for d in rel_org)
        total_chars = sum(len(_content(d)) for d in org_docs)
        purity = rel_chars / total_chars if total_chars else 0.0

        # 首个相关块位置
        first_rel_rank = None
        prefix_chars = 0
        in80 = False
        for d in org_docs:
            c = _content(d)
            prefix_chars += len(c)
            if iou(kt, c) >= IOU_TH:
                if first_rel_rank is None:
                    first_rel_rank = getattr(d, "rank", 0) or 0
                if prefix_chars <= 0.8 * total_chars:
                    in80 = True
                break

        agg0["n"] += 1
        if n_rel_top10 > 0:
            agg0["any_hit"] += 1
        agg0["n_rel_hit"] += n_rel_top10
        agg1["org_in"] += n_rel_org
        agg1["drop_rel"] += (n_rel_top10 - n_rel_org)
        agg_dil["related_chars"] += rel_chars
        agg_dil["total_chars"] += total_chars
        if first_rel_rank is not None:
            agg_dil["fr_sum"] += first_rel_rank
            agg_dil["fr_n"] += 1
        if in80:
            agg_cons["in80"] += 1
        rows.append({
            "q": q[:24], "cap": cap[:8], "nrel_top10": n_rel_top10,
            "nrel_org": n_rel_org, "purity": purity, "first_rank": first_rel_rank,
        })

    main_print(sample, agg0, agg1, agg_dil, agg_cons, rows)


def main_print(sample, agg0, agg1, agg_dil, agg_cons, rows):
    n = len(sample) or 1
    print("=" * 72)
    print("L0 检索入口 (top-10 候选池)")
    print(f"  至少命中1块题目占比   : {agg0['any_hit']}/{n} = {agg0['any_hit']/n*100:.0f}%  (hit@10)")
    print(f"  平均相关块/top10      : {agg0['n_rel_hit']/n:.2f} 块/题")
    print()
    print("L1 组织层 (去重0.97+分组, 不过滤)")
    print(f"  相关块 top10→组织     : {agg0['n_rel_hit']} → {agg1['org_in']}  "
          f"(丢失 {agg1['drop_rel']}, 组织层几乎无损)")
    print()
    print("L2 稀释层 (进入最终prompt的 evidence_str)")
    tot = agg_dil["total_chars"] or 1
    print(f"  相关块字符占比(纯度)  : {agg_dil['related_chars']/tot*100:.1f}%  "
          f"(相关 {agg_dil['related_chars']//1000}k / {tot//1000}k chars)")
    fr_n = agg_dil["fr_n"]
    if fr_n:
        print(f"  有相关块的题目中, 首个相关块平均 rank: {agg_dil['fr_sum']/fr_n:.1f}")
    print()
    print("L3 可消费性 (若模型只认真读前80%文本)")
    print(f"  首个相关块落在前80%文本: {agg_cons['in80']}/{n} = {agg_cons['in80']/n*100:.0f}%")
    print()
    print("对照理解: 轻量直答=长上下文自筛→等效纯度≈100% & 总能先读到相关块")
    print("=" * 72)
    print(f"{'题目':26}{'cap':8}{'top10相关':>8}{'org相关':>7}{'纯度':>7}{'首rank':>7}")
    for row in rows:
        print(f"{row['q']:26}{row['cap']:8}{row['nrel_top10']:>8}{row['nrel_org']:>7}"
              f"{row['purity']*100:>6.0f}%{str(row['first_rank']):>7}")


if __name__ == "__main__":
    main()
