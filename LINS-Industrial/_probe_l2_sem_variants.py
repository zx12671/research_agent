# -*- coding: utf-8 -*-
"""
L2 语义/能力相关置顶 A/B —— '做准相关信号'版。

上轮 9.7 结论：词法(lex)重排无效/微害(奖励题干复述块)，top5 截断抬纯度但丢 27% 相关块。
本轮把重排信号从'词法'升级为'语义分 + capability 先验'(signal="cap")，看能否顶置真答案源块。

比较(同一 35 题, k=10, 句子级相关判据):
  base     : 原检索序 = organizer 默认(不重排)
  lex      : rerank, signal="lex"   (词法为主, 上轮已见微害, 作对比)
  cap      : rerank, signal="cap"   (语义分+能力先验, 本轮主线)
  cap+top5 : signal="cap" + truncate top5 (看能力先验下截断是否仍丢相关块)

度量(每题):
  purity     : 相关字符占比
  first_rank : 首个相关块 rank
  gt_shift   : 与 base 相比，首个/全部"答案源块"位次 improved/unchanged/worsened
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

from _probe_evidence_loss2 import sample_rows, _split_sents, is_related, _content

def main():
    sample = sample_rows()
    print(f"L2 语义/能力置顶 A/B: N={len(sample)} 题, k=10(句子级判据)\n")
    from retrieval.retriever import OpenDomainRetriever
    from agentic.task_types import ExecutionDirective
    from agentic.organizer import EvidenceOrganizer
    rt = OpenDomainRetriever()
    rt.load_from_manifest()
    directive = ExecutionDirective(module="organization", action="group_by_topic", params={})

    configs = {
        "base":     dict(),
        "lex":      dict(rerank=True, signal="lex"),
        "cap":      dict(rerank=True, signal="cap"),
        "cap+top5": dict(rerank=True, signal="cap", truncate=True, max_chunks=5, max_chars=0),
    }

    agg = {name: {"n": 0, "purity": 0.0, "fr_sum": 0.0, "fr_n": 0, "in80": 0,
                  "tot_chars": 0.0, "n_rel_base": 0, "n_rel_out": 0}
           for name in configs}
    shift = {name: {"improved": 0, "unchanged": 0, "worsened": 0} for name in configs}

    # 每题 base 得到标准序 + 相关块集合；再跑各 variant，比较首个相关块位次
    for r in sample:
        q, kt = r.get("question"), r.get("knowledge_text")
        sents = _split_sents(kt or "")
        if not q or not kt or not sents:
            continue
        res = rt.retrieve(q, k=10, use_ked=True)
        base_chunks = list(res.chunks)

        bdocs = EvidenceOrganizer().execute(
            directive, base_chunks, question=q, task_type="general").get_all_documents()
        base_first_pos = next((pos for pos, d in enumerate(bdocs, 1)
                               if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1])), None)
        b_n_rel = sum(1 for d in bdocs if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1]))

        for name in configs:
            docs = EvidenceOrganizer().execute(
                directive, base_chunks, question=q, task_type="general",
                **configs[name]).get_all_documents()
            rel_chars = sum(len(_content(d)) for d in docs if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1]))
            total = sum(len(_content(d)) for d in docs)
            purity = rel_chars / total if total else 0.0
            first_rank = next((getattr(d, "rank", 0) or 0 for p, d in enumerate(docs, 1)
                               if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1])), None)
            prefix = 0
            in80 = False
            for d in docs:
                prefix += len(_content(d))
                if is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1]:
                    in80 = prefix <= 0.8 * total
                    break
            n_rel_out = sum(1 for d in docs if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1]))

            a = agg[name]
            a["n"] += 1
            a["purity"] += purity
            a["tot_chars"] += total
            a["in80"] += (1 if in80 else 0)
            a["n_rel_base"] += b_n_rel
            a["n_rel_out"] += n_rel_out
            if first_rank is not None:
                a["fr_sum"] += first_rank
                a["fr_n"] += 1
            first_pos = next((pos for pos, d in enumerate(docs, 1)
                              if (is_related(_content(d), sents)[0] or is_related(_content(d), sents)[1])), None)
            if first_pos is not None and base_first_pos is not None and name != "base":
                if first_pos < base_first_pos:
                    shift[name]["improved"] += 1
                elif first_pos > base_first_pos:
                    shift[name]["worsened"] += 1
                else:
                    shift[name]["unchanged"] += 1

    print("=" * 96)
    print(f"{'config':10}{'纯度':>7}{'首rank':>7}{'in80':>6}{'相关块保留':>12}{'rel丢失':>7}{'gt improve':>10}{'worse':>6}")
    print("-" * 96)
    for name in configs:
        a = agg[name]
        n = a["n"] or 1
        fr = a["fr_sum"] / a["fr_n"] if a["fr_n"] else float("nan")
        drop = a["n_rel_base"] - a["n_rel_out"]
        s = shift[name]
        imp = s["improved"]
        print(f"{name:10}{a['purity']/n*100:>6.1f}%{fr:>7.1f}{a['in80']/n*100:>5.0f}%"
              f"{a['n_rel_out']}/{a['n_rel_base']:>11}{drop:>7}{imp:>10}"
              f"{s['worsened']:>6}")
    print("=" * 96)
    print("说明: 纯度越高稀释越小; 首rank越小越好; rel丢失=截断误删相关块; "
          "gt improve/worsen=重排使首个相关块相对base提前/推后的题数")


if __name__ == "__main__":
    main()
