# -*- coding: utf-8 -*-
"""
降分机理专诊：对比 ef 与 base 的 solver 输入差异，定位降分根因。
聚焦 21 基线里 3 道确认降分题（gain=↓）+ 对照（升分/持平）：
  - 光纤滤料截污（工艺，base3→ef2, n=13）
  - 污水池聚脲防水涂料（工艺，base3→ef2, n=33）
  - 插座端子质量控制（质量，base3→ef2, n=13）
  + 对照：切削表面鳞片(升分0→2)、隧道衬砌(升分2→3)、桥式整流器(持平=2)

方法（与 _probe_evidence_forward_end2end 完全同口径）：
  - base = top-10；ef = top10 + 整档回捞(每档 jt(query,块) 排序 top50, 过滤 th).
  - 对每块标注：src(top10/extra)、true_evid(与该题 standard answer 句级 bigram JT>=0.30 命中句数)、
    qsim=jt(query,块)、chunk_id/document_id。
  - 多次采样(NR=5)跑单轮直答 ask()，解析模型答卷中"引用的 [N] 号片段"，判定用的真证据还是噪声。

归因判定：
  - 引用 true_evid==0 的回捞块 → 【噪声块被误判为正确】
  - 真证据块在 ef 中位置靠前但模型仍答错/引用真证据次数下降 → 【证据太多失焦】
  - ef 与 base 引用真证据次数相当且降分 → 覆盖度量噪声/其它（记为"疑随机"）
"""
import os, re, csv, random, json, time, statistics
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
MAX_PER_DOC = 50
JT_RANK_TH = 0.02
SAMPLER_TEMP = 0.2
NR = 5                          # 每条件采样次数（压随机性）
J22 = "probe_evidence_forward_e2e_21.json"

# 关注的题：q 前缀 → (降分标志, 备注)   前缀来自 21 json rows
FOCUS = {
    "在相同体积下，纤维滤料与传统滤砂相比": ("DEGRADE", "光纤滤料截污 3->2"),
    "在污水池专用聚脲防水防腐涂料的施工中": ("DEGRADE", "聚脲防水涂料 3->2"),
    "插座端子在冲压成型后、电镀前的质量控制": ("DEGRADE", "插座端子 3->2"),
    "在切削塑性材料如不锈钢或铝合金时": ("CTRL_UP", "切削鳞片 0->2"),
    "在安装新45型电压表时": ("CTRL_UP2", "电压表 3->3 持平"),
    "桥式整流器规格标识中的IF(av)参数": ("CTRL_UP3", "桥式整流器 2->2 持平"),
}


@lru_cache(maxsize=None)
def _grams(s, n=2):
    s = str(s).replace(" ", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jt(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def split_sent(text):
    return [p for p in re.split(r"(?<=[。；;！？])\s*", text or "") if len(p.strip()) >= 6]


def _load_rows():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def pick_rows():
    """按 21 json 的 3 降分 + 3 对照 q 前缀，从 CSV 全表精确匹配出完整 row。"""
    import json as _j
    with open(os.path.join(_LINS, "results", J22), encoding="utf-8") as f:
        base21 = _j.load(f)
    deg_qs = [r["q"] for r in base21["rows"] if r.get("gain") == "↓"]
    wanted = {k: v for k, v in FOCUS.items()}
    # 先补全降分题的完整 q(21 json 里只存了24字前缀)
    full_q = {}
    for r in base21["rows"]:
        for pre in list(wanted.keys()):
            if pre in r["q"]:
                full_q[pre] = r
    rows_all = _load_rows()
    out = {}
    for pre, (flag, note) in wanted.items():
        q_prefix = pre
        ref_row = full_q.get(pre)
        q_full = ref_row["q"] if ref_row else None
        match = None
        if q_full:
            match = next((x for x in rows_all if (x.get("question") or "").startswith(q_full)), None)
        if match is None:
            # 降级：按前缀匹配
            match = next((x for x in rows_all if (x.get("question") or "").startswith(q_prefix)), None)
        if match is not None:
            out[q_prefix] = (flag, note, match)
    return out


def build_annotated(retriever, q, ref):
    """返回 (base_list, ef_list)，每个元素含 content,true_evid,qsim,is_top10,chunk_id,doc 标注。"""
    res = retriever.retrieve(q, k=10, use_ked=True)

    top = list(getattr(res, "chunks", []))
    top_contents = [_content(c) for c in top]
    # base 标注
    base = []
    ref_sents = split_sent(ref)
    for c in top:
        txt = _content(c)
        base.append(_annot(txt, ref_sents, q, True, getattr(c, "chunk_id", ""), getattr(c, "document_id", "")))
    # ef = base + 回捞 extra（带标注）
    extra = doc_backfill_pruned_ann(retriever, top, q, ref_sents)
    return base, base + extra


def _annot(txt, ref_sents, q, is_top, cid, doc):
    true_evid = sum(1 for s in ref_sents if jt(s, txt) >= 0.30)
    return {
        "content": txt,
        "true_evid": true_evid,
        "qsim": round(jt(q, txt), 3),
        "is_top10": is_top,
        "chunk_id": cid,
        "doc": doc,
    }


def doc_backfill_pruned_ann(retriever, top, query, ref_sents, max_per_doc=MAX_PER_DOC):
    top_contents = [_content(c) for c in top]
    got_docs, extra = set(), {}
    for c in top:
        did = getattr(c, "document_id", "") or ""
        if not did or did in got_docs:
            continue
        got_docs.add(did)
        same = []
        for cid, data in retriever.chunks.items():
            dd = data.get("document_id") or data.get("doc_id") or ""
            if dd == did:
                same.append((cid, data))
        same.sort(key=lambda d: d[1].get("chunk_index", 0))
        scored = []
        for cid, d in same:
            txt = _content(d)
            if not txt or txt in top_contents:
                continue
            scored.append((jt(query, txt), txt, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [(t, cid) for s, t, cid in scored if s >= JT_RANK_TH][:max_per_doc]
        for t, cid in kept:
            extra.setdefault(t, cid)
    # 按回捞排序 → 标注
    out = []
    for t, cid in extra.items():
        out.append(_annot(t, ref_sents, query, False, cid, ""))
    return out


def _content(c):
    if isinstance(c, dict):
        return c.get("content", "")
    return getattr(c, "content", "") or ""


SYSTEM = ("你是一名严谨的工业领域专家。请仅依据给定的知识片段回答工程问题："

          "数值/事实给出明确结论，解释所用依据。若知识不足，明确说'知识不足'。回答要简短准确。")

def ask(llm, model, q, cands):
    ctx = "\n".join(f"[{i+1}] {t['content']}" for i, t in enumerate(cands))
    if not ctx:
        ctx = "（无检索证据）"
    user = (f"问题：{q}\n\n参考知识片段：\n{ctx}\n\n"
            "请给出答案，并尽量用『[编号]』标注你依据了哪几个片段。")
    resp = llm.chat.completions.create(
        model=model, temperature=SAMPLER_TEMP, max_tokens=600,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def ref_used_indices(answer):
    """抽取答案中引用的片段编号，如 [1] / [3,5]。"""
    idx = set()
    for m in re.finditer(r"\[(\d+)\]", answer):
        idx.add(int(m.group(1)))
    return idx


def mechanism(base, ef, base_preds, ef_preds):
    """根据多采样答案对降分归因。"""
    def agg(preds):
        used_true = 0      # 引用真证据块次数
        used_noise = 0     # 引用噪声(非真证据)块次数
        used_top = 0
        used_extra = 0
        for p in preds:
            for i in ref_used_indices(p):
                if 1 <= i <= len(base):
                    blk = base[i-1]
                    src = "top10"
                elif 1 <= i <= len(ef):
                    blk = ef[i-1]
                    src = "extra" if not blk["is_top10"] else "top10"
                else:
                    continue
                if blk["true_evid"] > 0:
                    used_true += 1
                else:
                    used_noise += 1
                if src == "top10":
                    used_top += 1
                else:
                    used_extra += 1
        return used_true, used_noise, used_top, used_extra

    bt, bn, btop, bextra = agg(base_preds)
    et, en, etop, eextra = agg(ef_preds)
    # 真证据块在 ef 中的总命中（可被利用的上限）
    ef_true_total = sum(1 for b in ef if b["true_evid"] > 0)
    base_true_total = sum(1 for b in base if b["true_evid"] > 0)
    # 判定
    reason = []
    if en > 0 and en > et * 0.5:
        reason.append("NOISE_CITED: 明显引用了噪声(非真证据)块当作依据")
    if ef_true_total > base_true_total and et <= bt:
        reason.append("DISTRACT: 回捞补进了更多真证据块，但模型反而没引用（失焦）")
    if ef_true_total > 0 and et == 0:
        reason.append("UTIL_FAIL: ef 含真证据块但模型一次都没引用")
    if not reason:
        if et < bt:
            reason.append("USE_DROP: 引用真证据次数下降")
        else:
            reason.append("RANDOM_OR_METRIC: 引用未见明显退化，或与覆盖度度量弱相关")
    return {
        "base_true_total": base_true_total, "ef_true_total": ef_true_total,
        "base_used": {"true": bt, "noise": bn, "top10": btop, "extra": bextra, "n_pred": len(base_preds)},
        "ef_used": {"true": et, "noise": en, "top10": etop, "extra": eextra, "n_pred": len(ef_preds)},
        "mechanism": reason,
    }


def main():
    import random
    selected = pick_rows()
    from retrieval.retriever import OpenDomainRetriever
    from metrics.industrybench_scorer import RuleBasedScorer
    from openai import OpenAI

    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()
    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    scorer = RuleBasedScorer()

    print(f"降分机理专诊: N={len(selected)}题 x {NR}采样 x {2}条件 = {len(selected)*NR*2} 次LLM调用\n")
    report = {}
    for pre, (flag, note, row_m) in selected.items():
        q = row_m.get("question", ""); ref = row_m.get("answer", ""); kt = row_m.get("knowledge_text", "")
        base, ef = build_annotated(retriever, q, ref)
        bpreds, epreds = [], []
        bsc, esc = [], []
        for _ in range(NR):
            bp = ask(llm, LLM_NAME, q, base); ep = ask(llm, LLM_NAME, q, ef)
            bpreds.append(bp); epreds.append(ep)
            bsc.append(scorer.rule_based_score(q, ref, bp))
            esc.append(scorer.rule_based_score(q, ref, ep))
        mech = mechanism(base, ef, bpreds, epreds)
        mech["base_scores"] = bsc; mech["ef_scores"] = esc
        mech["base_mean"] = round(statistics.mean(bsc), 2); mech["ef_mean"] = round(statistics.mean(esc), 2)
        # 抽样贴一段代表答案（首个降分采样的 ef 答案）供人工核验
        mech["sample_ef_pred"] = epreds[0][:600]
        mech["sample_base_pred"] = bpreds[0][:400]
        report[pre] = {**{"flag": flag, "note": note, "q": q[:40], "n_base": len(base), "n_ef": len(ef)}, **mech}
        print(f"[{flag}] {note} ({len(base)} -> {len(ef)}块)  base均={statistics.mean(bsc):.2f} ef均={statistics.mean(esc):.2f}")
        print(f"    真证据块 base={mech['base_true_total']} ef={mech['ef_true_total']}  "
              f"引用(真/噪) base={mech['base_used']['true']}/{mech['base_used']['noise']} "
              f"ef={mech['ef_used']['true']}/{mech['ef_used']['noise']}")
        print(f"    -> {mech['mechanism']}\n")

    out = os.path.join(_LINS, "results", "diag_degrade_mechanism.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("已写:", out)


if __name__ == "__main__":
    main()
