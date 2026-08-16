# -*- coding: utf-8 -*-
"""证据利用能力诊断 v2 (N=35)：找答案(检索覆盖) vs 利用答案(分数传导) 解耦.
复用 e2e_35 同一批样本与候选；读取 e2e_35 输出的 scoring json."""
import os, json

_LINS = os.path.dirname(os.path.abspath(__file__))

def build(retriever, q):
    from _probe_evidence_forward_end2end import build_candidates
    return build_candidates(retriever, q)

def main():
    import importlib
    e2e = importlib.import_module("_probe_evidence_forward_end2end")
    # 与 e2e_35 对齐（PER_CAP=5, N=35），保证 sample_questions 抽 35 题
    e2e.PER_CAP = 5
    e2e.N = 35
    sample = e2e.sample_questions()

    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    jp = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")
    with open(jp, encoding="utf-8") as f:
        e2e_rows = json.load(f)["rows"]
    assert len(e2e_rows) == len(sample), f"数量不一致 {len(e2e_rows)} vs {len(sample)}"

    IOU_SENT_TH = 0.30
    COV_OK = 0.50
    def cov_of(sents, chunks):
        c = sum(1 for sc in sents if any(e2e.jt(sc, t) >= IOU_SENT_TH for t in chunks))
        return c / len(sents) if sents else 0.0

    print(f"证据利用诊断 N={len(sample)} | 找答案OK: cov(ef)>={COV_OK:.0%}(句JT>={IOU_SENT_TH}) "
          f"| 利用失败: ef_score<=base_score\n")

    quad = {"OK_useUp": [], "OK_useFail": [], "OK_low": [], "findFail": []}
    for r, sc in zip(sample, e2e_rows):
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        sents = e2e.split_sent(kt)
        base, efc, _m = build(retriever, q)
        cov_b, cov = cov_of(sents, base), cov_of(sents, efc)
        ef, base_, gain = sc["ef_score"], sc["base_score"], sc["gain"]
        row = dict(q=sc["q"], cap=sc["cap"], cls=sc["cls"], nS=sc["nS"],
                   cov_b=round(cov_b, 2), cov_ef=round(cov, 2),
                   base_score=base_, ef_score=ef, gain=gain)
        if cov < COV_OK:
            quad["findFail"].append(row)
        elif ef <= base_:
            quad["OK_useFail"].append(row)
        elif ef < 2:
            quad["OK_low"].append(row)
        else:
            quad["OK_useUp"].append(row)

    labels = {
        "OK_useUp":   "1. 找答案OK+利用成功(ef>base)  检索+模型双达标",
        "OK_useFail": "2. 找答案OK+利用失败(ef<=base) 证据在但分未升 *",
        "OK_low":     "3. 找答案OK+仍低分(ef<2)      部分利用",
        "findFail":   "4. 找答案未OK(cov<0.5)         检索仍缺",
    }
    order = ["OK_useUp", "OK_useFail", "OK_low", "findFail"]
    for k in order:
        rows = quad[k]
        print(f"\n== {labels[k]}  (n={len(rows)}) ==" if rows else f"\n== {labels[k]}  (n=0 空) ==")
        for x in rows:
            print(f"  [{x['cls']}] {x['q']:24}{x['cap']:8}S{x['nS']:>4} "
                  f"covB={x['cov_b']:.2f} covE={x['cov_ef']:.2f} "
                  f"base={x['base_score']}->ef={x['ef_score']} {x['gain']}")

    n = len(sample)
    print("\n" + "=" * 78)
    for k in order:
        print(f"{k:10} {len(quad[k]):3}  ({len(quad[k])/n:5.1%})")
    ok_has = sum(len(quad[k]) for k in ["OK_useUp", "OK_useFail", "OK_low"])
    use_fail = len(quad["OK_useFail"]) + len(quad["OK_low"])
    print(f"\n证据到位(找答案OK) {ok_has}/{n}; 其中利用不足 {use_fail} "
          f"(纯平/降 {len(quad['OK_useFail'])} + 仍低分 {len(quad['OK_low'])}); 检索仍缺 {len(quad['findFail'])}。")

    out = os.path.join(_LINS, "results", "diag_evidence_utilization_35.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {"n": len(v), "rows": v} for k, v in quad.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {out}")

if __name__ == "__main__":
    main()
