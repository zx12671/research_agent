# -*- coding: utf-8 -*-
"""
L2 重排+截断 A/B 验证探针。

对比 EvidenceOrganizer.execute() 的 4 种 L2 配置对"稀释(纯度)/可读性/相关块保留"的影响，
复用 _probe_evidence_loss2 的句子级相关判据(同一 35 题样本, k=10)。

配置:
  base             : rerank=False, truncate=False   (= 现有线上行为, 应复现纯度≈43%)
  rerank           : rerank=True                     (仅重排, 不删块)
  rerank+trunc6k   : rerank=True, truncate=True, max_chars=6000
  rerank+trunc4k   : rerank=True, truncate=True, max_chars=4000

度量(每题):
  purity    = 相关块字符 / 组织后总字符       (L2 稀释: 越高越好)
  first_rank= 首个相关块在 get_context 输出中的 rank (L3 可读: 越小越好)
  n_rel_in  = 组织后相关块数 / 检索相关块数   (相关块保留率; trunc 是否会误删=关键风险)
  in80      = 首相关块是否落在前80%文本
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

from _probe_evidence_loss2 import sample_rows, _split_sents, is_related, _content


def measure(retriever, organizer, directive, q, kt, sents, **kw):
    res = retriever.retrieve(q, k=10, use_ked=True)
    chunks = res.chunks

    def rel(d):
        h, s = is_related(_content(d), sents)
        return h or s

    n_rel_retrieve = sum(1 for c in chunks if rel(c))
    org = organizer.execute(directive, chunks, question=q, task_type="general", **kw)
    docs = org.get_all_documents()
    n_rel_org = sum(1 for d in docs if rel(d))
    rel_chars = sum(len(_content(d)) for d in docs if rel(d))
    total_chars = sum(len(_content(d)) for d in docs)
    purity = rel_chars / total_chars if total_chars else 0.0

    first_rank = None
    prefix = 0
    in80 = False
    order = [getattr(d, "rank", None) or 0 for d in docs]
    for d in docs:
        c = _content(d)
        prefix += len(c)
        if rel(d):
            first_rank = getattr(d, "rank", None) or 0
            if prefix <= 0.8 * total_chars:
                in80 = True
            break

    return {
        "n_rel_retrieve": n_rel_retrieve, "n_rel_org": n_rel_org,
        "n_docs": len(docs), "purity": purity, "first_rank": first_rank,
        "in80": in80, "total_chars": total_chars,
    }


def main():
    sample = sample_rows()
    print(f"L2 重排+截断 A/B: N={len(sample)} 题, k=10(句子级相关判据)\n")
    from retrieval.retriever import OpenDomainRetriever
    from agentic.task_types import ExecutionDirective
    from agentic.organizer import EvidenceOrganizer
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()
    directive = ExecutionDirective(module="organization", action="group_by_topic", params={})

    configs = {
        "base":           {},
        "rerank":         dict(rerank=True),
        "rerank+top5":    dict(rerank=True, truncate=True, max_chunks=5, max_chars=0),
        "rerank+trunc6k": dict(rerank=True, truncate=True, max_chars=6000),
        "rerank+trunc4k": dict(rerank=True, truncate=True, max_chars=4000),
    }

    agg = {name: {"n": 0, "purity": 0.0, "fr_sum": 0.0, "fr_n": 0,
                  "in80": 0, "rel_drop": 0, "n_rel_retrieve": 0, "n_rel_org": 0,
                  "tot_chars": 0.0}
           for name in configs}

    organizer_cache = {}
    for name in configs:
        organizer_cache[name] = EvidenceOrganizer()

    for r in sample:
        q, kt = r.get("question"), r.get("knowledge_text")
        sents = _split_sents(kt or "")
        if not q or not kt or not sents:
            continue
        # base 复用同一个 organizer 实例以公平比较
        for name, kw in configs.items():
            org = organizer_cache[name]
            m = measure(retriever, org, directive, q, kt, sents, **kw)
            a = agg[name]
            a["n"] += 1
            a["purity"] += m["purity"]
            a["tot_chars"] += m["total_chars"]
            a["n_rel_retrieve"] += m["n_rel_retrieve"]
            a["n_rel_org"] += m["n_rel_org"]
            if m["first_rank"] is not None:
                a["fr_sum"] += m["first_rank"]
                a["fr_n"] += 1
            if m["in80"]:
                a["in80"] += 1
            if m["n_rel_org"] < m["n_rel_retrieve"]:
                a["rel_drop"] += (m["n_rel_retrieve"] - m["n_rel_org"])

    print("=" * 96)
    hdr = f"{'config':16}{'纯度':>7}{'首rank':>7}{'in80':>6}{'相关块保留':>10}{'rel丢失':>7}{'平均字数':>9}"
    print(hdr)
    print("-" * 96)
    for name in configs:
        a = agg[name]
        n = a["n"] or 1
        fr = a["fr_sum"] / a["fr_n"] if a["fr_n"] else float("nan")
        purity = a["purity"] / n * 100
        avg_chars = a["tot_chars"] / n
        print(f"{name:16}{purity:>6.1f}%{fr:>7.1f}{a['in80']/n*100:>5.0f}%"
              f"{a['n_rel_org']}/{a['n_rel_retrieve']:>8}{a['rel_drop']:>7}{avg_chars:>8.0f}")
    print("=" * 96)
    print("说明: 纯度=相关字符占比(越高稀释越小); 首rank越小越好; rel丢失=截断是否误删相关块(应为0)")


if __name__ == "__main__":
    main()
