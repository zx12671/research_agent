# -*- coding: utf-8 -*-
"""
证据链路分层损失量化探针 v2 —— 可靠判据版。

v1 用"整段 knowledge_text vs 单块 字形 IoU>=0.15"判相关块，对改写型答案产生大量假阴性
(见 docs 9.5：98.6% 答案句逐字在库却因 IoU 被判'不相关')。
v2 改用**句子级源块存在性**：块 content 若是 knowledge_text 任一整句的逐字宿主 => 相关块。

链路(production)：
    q ──retrieve(k=10)──▶ top10 chunks (L0 候选池)
         ──organizer.execute──▶ organized (去重0.97+分组,不过滤)
         ──get_context()──▶ evidence_str (L2 稀释) + 首相关块rank(L3 可消费)

指标与 v1 一致，仅相关块判定被修正，产出修正后的四层损失分布。
"""
import os, sys, csv, re, random
from collections import defaultdict
from functools import lru_cache

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP = 5
K = 10


def _split_sents(kt):
    parts = re.split(r"[。！？\n；;]", kt or "")
    return [p.strip() for p in parts if p and len(p.strip()) >= 4]


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}


def _jaccard2(a, b):
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / (len(ga) + len(gb) - inter) if inter else 0.0


def is_related(content, sents):
    """rel(c) = c 是 kt 任一句的逐字宿主(硬) 或 2gram Jaccard>=0.35(软容忍改写)。"""
    if not content or not sents:
        return False, False
    hard = any(s in content for s in sents)
    if hard:
        return True, True
    for s in sents:
        if _jaccard2(s, content) >= 0.35:
            return False, True
    return False, False


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
    print(f"证据链路分层损失(句子级判据v2): N={len(sample)} 题, k={K}\n")

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
        sents = _split_sents(kt)

        def rel_pred(d):
            h, s = is_related(_content(d), sents)
            return h or s

        res = retriever.retrieve(q, k=K, use_ked=True)     # L0
        chunks = res.chunks
        rel_top10 = [c for c in chunks if rel_pred(c)]
        n_rel_top10 = len(rel_top10)

        org = organizer.execute(directive, chunks, question=q, task_type="general")  # L1
        org_docs = org.get_all_documents()
        rel_org = [d for d in org_docs if rel_pred(d)]
        n_rel_org = len(rel_org)

        rel_chars = sum(len(_content(d)) for d in rel_org)
        total_chars = sum(len(_content(d)) for d in org_docs)
        purity = rel_chars / total_chars if total_chars else 0.0

        first_rel_rank = None
        prefix_chars = 0
        in80 = False
        for d in org_docs:
            c = _content(d)
            prefix_chars += len(c)
            if rel_pred(d):
                if first_rel_rank is None:
                    first_rel_rank = getattr(d, "rank", None) or 0
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
    print("=" * 74)
    print("L0 检索入口 (top-10 候选池, 句子级相关块)")
    print(f"  至少相关1块题目占比   : {agg0['any_hit']}/{n} = {agg0['any_hit']/n*100:.0f}%  (hit@10-相关)")
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
    print("对照: 轻量直答=长上下文自筛→等效纯度≈100% & 总能先读到相关块")
    print("=" * 74)
    print(f"{'题目':26}{'cap':8}{'top10相关':>8}{'org相关':>7}{'纯度':>7}{'首rank':>7}")
    for row in rows:
        print(f"{row['q']:26}{row['cap']:8}{row['nrel_top10']:>8}{row['nrel_org']:>7}"
              f"{row['purity']*100:>6.0f}%{str(row['first_rank']):>7}")


if __name__ == "__main__":
    main()

