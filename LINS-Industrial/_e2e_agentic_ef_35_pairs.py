# -*- coding: utf-8 -*-
"""
_e2e_agentic_ef_35_pairs.py —— 逆推式协议 第1步：35 题 × 2 条件（base vs ef 无条件回捞）agentic 量化。

背景（决策记录，见 docs/agentic_conditional_ef_ab.md）：
  绝对阈值门控 mu_top<0.25 在 35 题上全部判 SAT（退化），改用逆推式协议：
  只跑“不加回捞 vs 无条件回捞”两次 agentic，拿每条题“回捞实际升/降/持平分”的真值标签，
  再离线评估多个 零-LLM 检索层信号对“该不该回捞”的判别力（第2步 _analyze_gate_predictors_35.py）。

实现：完全复用 _ab_conditional_ef_35 的生产引擎（base=AgenticRAGEngine，ef=EFAgenticRAGEngine），
  仅跑2个条件（跳过 ef_cond），并把每个题的 base-top10 自足度信号 + ef 侧回捞消耗一并落盘。

运行（35×2 完整 agentic，成本高，建议后台 & 盯日志）：
    cd LINS-Industrial
    python _e2e_agentic_ef_35_pairs.py
"""
import os, sys, json, time

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import DEEPSEEK_KEY, LLM_NAME, register_paths
register_paths()

from metrics.industrybench_scorer import RuleBasedScorer
# 复用生产引擎与信号工具（import 不触发 main）
from _ab_conditional_ef_35 import (
    AgenticRAGEngine,
    EFAgenticRAGEngine,
    self_sufficiency_signal,
    is_gap_saturated,
    TH_SAT,
    load_35_pool,
    _load_refs,
    tokens_of_docs,
    n_extra,
)

OUT = os.path.join(_LINS, "results", "ab_ef_pairs_35.json")


def acc(out):
    """增量汇总（在推进途中随时可用），最终 main 末尾再算一次覆盖写盘。"""
    rows = out["rows"]
    up = sum(1 for r in rows if r["delta"] > 0)
    down = sum(1 for r in rows if r["delta"] < 0)
    same = sum(1 for r in rows if r["delta"] == 0)
    return {
        "N": len(rows),
        "up_down_same": [up, down, same],
        "mean_delta": round(sum(r["delta"] for r in rows) / len(rows), 3) if rows else 0.0,
        "thsat_gap_n": sum(1 for r in rows if r["gate_by_thsat"]),
        "thsat_sat_n": sum(1 for r in rows if not r["gate_by_thsat"]),
        "n_extra_total": sum(r["n_extra_ef"] for r in rows),
        "token_total": {"base": sum(r["tokens_base"] for r in rows),
                        "ef": sum(r["tokens_ef"] for r in rows)},
        "time_s_total": {"base": round(sum(r["time_s"]["base"] for r in rows), 1),
                         "ef": round(sum(r["time_s"]["ef"] for r in rows), 1)},
    }


def flush(out):
    """渐增式落盘：每完成一题调用一次，被杀也能保留已完成部分。"""
    out["agg"] = acc(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT)  # 原子替换，避免写一半被截断



def n_docs_top10(ret):
    dids = set()
    for d in (ret.documents[:10] if ret else []):
        if getattr(d, "document_id", None):
            dids.add(d.document_id)
    return len(dids)


def main():
    pool = load_35_pool()
    refs = _load_refs()
    print(f"逆推式协议 Step1 | 35 题×2 条件（base vs ef 无条件回捞）| N={len(pool)}")

    base_engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    ef_engine = EFAgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    scorer = RuleBasedScorer()

    # 断点续跑：若已有结果文件，跳过已完成的题，避免中途被杀后全部重跑。
    out = {"known_th": TH_SAT, "rows": []}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            if prev.get("known_th") == TH_SAT:
                out["rows"] = [r for r in prev.get("rows", []) if r.get("delta") is not None]
                print(f"续跑：已存在 {len(out['rows'])}/35 条，跳过完成题")
        except Exception as e:
            print("读取旧结果失败，从零开始:", e)
    done = {r["q"] for r in out["rows"]}

    for idx, item in enumerate(pool, 1):
        q = item["q"]
        ref = refs.get(q, "")
        if q in done:
            print(f"[{idx}/{len(pool)}] 跳过(已完成): {q[:26]}...")
            continue
        print(f"\n[{idx}/{len(pool)}] ({item['cap']}) {q[:26]}...")


        # ---- base（生产原样 top10）----
        t0 = time.time()
        ans_b, ret_b = base_engine.answer(q, retrieval_k=10)
        tb = time.time() - t0
        sc_b = scorer.rule_based_score(q, ref, ans_b)
        sig_b = self_sufficiency_signal(ret_b.scores if ret_b else [])
        sig_b["n_docs_top10"] = n_docs_top10(ret_b)
        tc_b = tokens_of_docs(ret_b.documents if ret_b else []) + (len(ans_b or "") // 2)
        n_b = len(ret_b.documents) if ret_b else 0

        # ---- ef 无条件（top10 + 整档回捞）----
        t0 = time.time()
        ans_e, ret_e = ef_engine.answer(q, retrieval_k=10)
        te = time.time() - t0
        sc_e = scorer.rule_based_score(q, ref, ans_e)
        ne = n_extra(ret_e.documents if ret_e else [])
        tc_e = tokens_of_docs(ret_e.documents if ret_e else []) + (len(ans_e or "") // 2)
        n_e = len(ret_e.documents) if ret_e else 0

        delta = sc_e - sc_b  # 真值标签：>0 升分 ==0 持平 <0 降分
        print(f"  base sc={sc_b} 证据={n_b} tokens={tc_b} ({tb:.0f}s)")
        print(f"  ef   sc={sc_e} 证据={n_e}(回捞{ne}) tokens={tc_e} ({te:.0f}s)  delta={delta:+d}")
        print(f"       base信号 mu_top={sig_b['mu_top']} headroom={sig_b['headroom']} "
              f"mu_tail={sig_b['mu_tail']} max={sig_b['max_s']} n_docs={sig_b['n_docs_top10']}")

        out["rows"].append({
            "q": q, "cap": item["cap"], "cls": item["cls"],
            # 可在线得到的判别信号（全部来自 base top10，零-LLM）
            "sig_base": {k: sig_b[k] for k in ("mu_top", "mu_tail", "headroom", "max_s", "n_docs_top10")},
            # ef 侧“实际消耗”类信号
            "n_evidence_base": n_b,
            "n_extra_ef": ne,
            "score_base": sc_b,
            "score_ef": sc_e,
            "delta": delta,                # 真值标签
            "gate_by_thsat": int(is_gap_saturated(sig_b, TH_SAT)[0]),  # 绝对阈值门控结论
            "tokens_base": tc_b,
            "tokens_ef": tc_e,
            "time_s": {"base": round(tb, 1), "ef": round(te, 1)},
        })
        flush(out)  # 渐增式落盘：每题完成即写盘，被杀也能续跑

    rows = out["rows"]
    print("\n" + "=" * 60)
    print(f"N={len(rows)} | 升/降/持平={out['agg']['up_down_same']} | mean_delta={out['agg']['mean_delta']}")
    print(f"旧绝对阈值(TH={TH_SAT}) GAP={out['agg']['thsat_gap_n']} SAT={out['agg']['thsat_sat_n']}")
    print(f"回捞块={out['agg']['n_extra_total']} | tokens base={out['agg']['token_total']['base']} ef={out['agg']['token_total']['ef']}")
    print("已写:", OUT)



if __name__ == "__main__":
    main()
