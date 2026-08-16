# -*- coding: utf-8 -*-
"""
_smoke_pairs_1.py —— 重跑前"1 题冒烟"：用回填 ref 在真实 agentic 引擎上跑 base+ef，
确认得分非零（不再退化），且 delta/ref 命中正常，再决定是否全量重跑 35×2。
只跑 2 条 agentic 序列，成本可控。
"""
import os, sys, json

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import DEEPSEEK_KEY, LLM_NAME, register_paths
register_paths()

from metrics.industrybench_scorer import RuleBasedScorer
from _ab_conditional_ef_35 import (
    AgenticRAGEngine, EFAgenticRAGEngine,
    self_sufficiency_signal, is_gap_saturated, TH_SAT,
    load_35_pool, _load_refs, tokens_of_docs, n_extra,
)


def n_docs_top10(ret):
    dids = set()
    for d in (ret.documents[:10] if ret else []):
        if getattr(d, "document_id", None):
            dids.add(d.document_id)
    return len(dids)


def main():
    pool = load_35_pool()
    refs = _load_refs()
    scorer = RuleBasedScorer()
    print(f"Pool N={len(pool)}；refs={len(refs)}；TH_SAT={TH_SAT}")
    print("=" * 60)

    # 随机抽 1 题（固定第一个），打印 ref 命中情况
    item = pool[0]
    q = item["q"]
    ref = refs.get(q, "")
    print(f"样本题[{item['cap']}]: {q[:40]}")
    print(f"  ref_len={len(ref)}  ref前80: {ref[:80]}")
    if not ref:
        print("  !!! ref 仍为空 —— 回填失效，提前中止")
        return

    base_engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    ef_engine = EFAgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)

    ans_b, ret_b = base_engine.answer(q, retrieval_k=10)
    sc_b = scorer.rule_based_score(q, ref, ans_b)
    sig_b = self_sufficiency_signal(ret_b.scores if ret_b else [])
    sig_b["n_docs_top10"] = n_docs_top10(ret_b)
    print(f"\n[base] sc={sc_b} | n_docs={sig_b['n_docs_top10']} | mu_top={sig_b['mu_top']} headroom={sig_b['headroom']}")

    ans_e, ret_e = ef_engine.answer(q, retrieval_k=10)
    sc_e = scorer.rule_based_score(q, ref, ans_e)
    ne = n_extra(ret_e.documents if ret_e else [])
    print(f"[ef]   sc={sc_e} | 回捞块={ne} | n_docs={len(ret_e.documents) if ret_e else 0}")
    print(f"delta={sc_e - sc_b:+d}")

    print("\n结论: " + ("非退化，可全量重跑 35×2" if (sc_b or sc_e) else "得分仍全 0，需停查 ref/评分"))


if __name__ == "__main__":
    main()
