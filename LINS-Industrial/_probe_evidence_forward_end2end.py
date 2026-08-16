# -*- coding: utf-8 -*-
"""
evidence-forward 端到端终验：base(top-10) vs ef_pruned(整档回捞+按query相似度每档取top50)
最小闭环，隔离"候选质量 -> 答案分"变量。

口径（用户拍板）：
  - base     = 仅 retrieve(q, k=10)（生产现状）
  - ef_pruned = 对 top-10 每档回捞全部块 -> 每档内按 jt(块, query) 排序取 top50 -> 合并去重
生成：同一 standard prompt + DeepSeek 轻量直答（一次调用），只换候选。
评分：RuleBasedScorer.rule_based_score(q, ref, pred)（纯规则 0-3 覆盖度）
      + check_safety_simple(knowledge_text) SV 清零（与 exp1 rule 模式一致）。
重点：看 A 类(超长GT>=50句) 答案分能否因 ef_pruned 补全证据而上升。
"""
import os, csv, re, random, time, statistics
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import DEEPSEEK_KEY, LLM_NAME, INDUSTRYBENCH_CSV, register_paths
register_paths()

CSV = INDUSTRYBENCH_CSV or os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 3, 24, 7          # 每能力3题，尽量覆盖多能力；N为上限
MAX_PER_DOC = 50                     # 每档封顶50块
JT_RANK_TH  = 0.02                   # 每档相似度过滤下限（去与query几乎无关的中后段噪声）
SAMPLER_TEMP = 0.2


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def split_sent(text):
    return [p for p in re.split(r"(?<=[。；;！？])\s*", text or "") if len(p.strip()) >= 6]


def _content(c):
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "") or ""


def doc_backfill_pruned(retriever, res, query, max_per_doc=MAX_PER_DOC):
    """每档回捞全部块 -> 按 jt(块, query) 降序取 max_per_doc 并过滤低相关 -> 合并去重。"""
    top = list(getattr(res, "chunks", []))
    top_contents = [_content(c) for c in top]
    got_docs, extra = set(), []
    for c in top:
        did = getattr(c, "document_id", "") or ""
        if not did or did in got_docs:
            continue
        got_docs.add(did)
        same = []
        for cid, data in retriever.chunks.items():
            dd = data.get("document_id") or data.get("doc_id") or ""
            if dd == did:
                same.append(data)
        # 还原顺序并丢弃已在 top 中的块（避免重复计候选）
        same.sort(key=lambda d: d.get("chunk_index", 0))
        scored = []
        for d in same:
            txt = _content(d)
            if not txt or txt in top_contents:
                continue
            scored.append((jt(query, txt), txt))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [t for s, t in scored if s >= JT_RANK_TH][:max_per_doc]
        for t in kept:
            if t not in extra:
                extra.append(t)
    return top_contents, extra


def build_candidates(retriever, q):
    """返回 (base_contents, ef_pruned_contents, meta)。"""
    res = retriever.retrieve(q, k=10, use_ked=True)
    base = _content_list(res)
    topc, extra = doc_backfill_pruned(retriever, res, q)
    meta = {"ntop": len(topc), "nextra": len(extra), "ndoc": len(set(
        getattr(d, "document_id", "") or getattr(d, "doc_id", "") or "" for d in _res_chunks(res))),
        "nchunks_total": len(retriever.chunks)}
    return base, topc + extra, meta


def _content_list(res):
    return [_content(c) for c in getattr(res, "chunks", [])]


def _res_chunks(res):
    return list(getattr(res, "chunks", []))


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


SYSTEM = ("你是一名严谨的工业领域专家。请仅依据给定的知识片段回答工程问题："
          "数值/事实给出明确结论，解释所用依据。若知识不足，明确说'知识不足'。回答要简短准确。")

def ask(llm, model, q, ref, cands):
    ctx = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(cands))
    if not ctx:
        ctx = "（无检索证据）"
    user = (f"问题：{q}\n\n参考知识片段：\n{ctx}\n\n"
            "请给出答案，并简要说明依据哪几个片段。")
    resp = llm.chat.completions.create(
        model=model,
        temperature=SAMPLER_TEMP,
        max_tokens=600,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def main():
    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    from metrics.industrybench_scorer import RuleBasedScorer

    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()
    from openai import OpenAI
    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    scorer = RuleBasedScorer()

    print(f"evidence-forward 端到端: base vs ef_pruned(整档回捞+每档top{MAX_PER_DOC})  N={len(sample)} seed={SEED}")
    print("评分: RuleBasedScorer.rule_based_score 0-3 + SV清零\n")

    t0 = time.time()
    rows, meta = [], dict(nextra=[], ncand_base=[], ncand_ef=[], ndoc=[], ntokens=[])
    for r in sample:
        q, kt, ref = r.get("question", ""), r.get("knowledge_text", ""), r.get("answer", "")
        cap, nS = r.get("capability", ""), len(split_sent(kt))
        cls = "A" if nS >= 50 else "B"
        base, efc, m = build_candidates(retriever, q)
        row = {"q": q[:24], "cap": cap[:8], "cls": cls, "nS": nS}

        scores = {}
        for name, cands in [("base", base), ("ef", efc)]:
            pred = ask(llm, LLM_NAME, q, ref, cands)
            rs = scorer.rule_based_score(q, ref, pred)
            sv = scorer.check_safety_simple(kt, pred) if kt else False
            adj = 0.0 if sv else float(rs)
            scores[name] = (rs, sv, adj)
            row[f"{name}_score"] = rs
            row[f"{name}_adj"] = round(adj, 2)
            row[f"{name}_ncand"] = len(cands)
        # A/B 决策：ef 是否提升
        row["gain"] = "↑" if scores["ef"][2] > scores["base"][2] else ("↓" if scores["ef"][2] < scores["base"][2] else "=")
        row["base_sv"] = scores["base"][1]
        row["ef_sv"] = scores["ef"][1]
        meta["nextra"].append(m["nextra"])
        meta["ncand_base"].append(len(base))
        meta["ncand_ef"].append(len(base) + m["nextra"])
        meta["ndoc"].append(m["ndoc"])
        rows.append(row)
        print(f"  {row['q']:24}{row['cap']:8}{row['cls']}{row['nS']:>4}  base="
              f"{row['base_adj']:4.1f}({'SV' if row['base_sv'] else ''}){row['base_ncand']:>3}| ef="
              f"{row['ef_adj']:4.1f}({'SV' if row['ef_sv'] else ''}){row['ef_ncand']:>4}  {row['gain']}")
    dt = time.time() - t0

    print("\n===== 汇总 (adj分, 均值) =====")
    for cls in ["all", "A", "B"]:
        sub = [r for r in rows if cls == "all" or r["cls"] == cls]
        if not sub:
            continue
        b = [r["base_adj"] for r in sub]; e = [r["ef_adj"] for r in sub]
        up = sum(1 for r in sub if r["gain"] == "↑"); dn = sum(1 for r in sub if r["gain"] == "↓")
        sv_b = sum(1 for r in sub if r["base_sv"]); sv_e = sum(1 for r in sub if r["ef_sv"])
        print(f"[{cls}] n={len(sub)}  base.mean={statistics.mean(b):.3f} ef.mean={statistics.mean(e):.3f}  "
              f"增益: 升{up}/降{dn}/平{len(sub)-up-dn}  SV: base{sv_b}/ef{sv_e}  "
              f"0分率 base{(sum(1 for x in b if x<=0))/len(sub):.0%} ef{(sum(1 for x in e if x<=0))/len(sub):.0%}")
    print(f"\n候选量: base.avg={statistics.mean(meta['ncand_base']):.0f}  ef.avg={statistics.mean(meta['ncand_ef']):.0f}"
          f"  (每档回捞块 avg={int(statistics.mean(meta['nextra']))}, 命中档数 avg={statistics.mean(meta['ndoc']):.1f})")
    print(f"耗时 {dt:.1f}s (共 {2*len(rows)} 次LLM查询)")

    import json
    out = os.path.join(_LINS, "results", "probe_evidence_forward_e2e.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "meta": {k: v for k, v in meta.items()}, "cfg": {
            "MAX_PER_DOC": MAX_PER_DOC, "JT_RANK_TH": JT_RANK_TH, "N": len(rows)}},
            f, ensure_ascii=False, indent=2)
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()
