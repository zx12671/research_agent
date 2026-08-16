# -*- coding: utf-8 -*-
"""
_ab_organize_util.py — S4→S5 组织消融：固定证据源(S3)下，三档 organizer 对"答案分+证据引用"的传导。

分治定位（用户口径）
--------------------
S1 前端判定 → S2 规划 → S3 检索召回 → S4 组织 → S5 推理生成 → S6 检索质量 → S7 最终答案分
本脚本只改 **S4(organizer 逻辑)**，S3 固定为同一 top-10 候选，测 S5(LLM 作答+引用) 与 S7(答案分) 的传导。

三档（全部用生产组件，零自定义）：
  V1 = PassThroughOrganizer          ：无去重/无修剪，原序全保留 → 测"S4 修剪过度"假设
  V2 = EvidenceOrganizer()           ：生产现状（去重+task分组，不减文档，FAISS 序保留）
  V3 = EvidenceOrganizer(truncate=True, signal='lex', max_chars=6000)
                                     ：L2 去噪声提纯（按 lex 相关排序并从低分端砍）→ 测"S4 噪声稀释"假设

判据（归属判定）：
  - V2 已最优             → S4 组织无瓶颈，缺口在 S3(召回) 或 S5(推理)
  - V1 > V2               → EvidenceOrganizer 修剪过度，丢证据
  - V3 > V2               → 碎片/噪声稀释 prompt(DISTRACT)，去噪声有增益
  - 参考"引用真证据次数"  → 区分"证据在但没利用(UTIL_FAIL/DISTRACT)" vs "检索没捞到(finFail)"

LLM：真实 DeepSeek（与参照组一致）；答案分 = RuleBasedScorer.rule_based_score(0-3)。
用法：python _ab_organize_util.py            # 默认 FOCUS 10 题
"""
import os
import sys
import re
import json
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from experiments.config import register_paths
register_paths()

from experiments.config import DEEPSEEK_KEY, LLM_NAME, INDUSTRYBENCH_CSV, RESULTS_DIR
from retrieval.retriever import OpenDomainRetriever
from agentic.organizer import EvidenceOrganizer
from agentic.task_types import ExecutionDirective
from experiments.ab_organizer_eval import PassThroughOrganizer
from metrics.industrybench_scorer import RuleBasedScorer
import _diag_degrade_mechanism as dm  # 复用 FOCUS/jt/_annot/ref_used_indices/ask

TEMPERATURE = 0.2
MAX_TOKENS = 600

DICT = "group_by_topic"


def _as_chunk_list(res):
    return list(getattr(res, "chunks", []))


def organize_v1(pt, directive, cands, q, task_type="general"):
    ev = pt.execute(directive, cands, question=q, task_type=task_type)
    return pt.get_context(ev), ev


def organize_v2(o2, directive, cands, q, task_type="general"):
    ev = o2.execute(directive, cands, question=q, task_type=task_type)
    return ev.get_context(), ev


def organize_v3(o3, directive, cands, q, task_type="general"):
    ev = o3.execute(directive, cands, question=q, task_type=task_type,
                    rerank=True, truncate=True, signal="lex", max_chars=6000)
    return ev.get_context(), ev




def answer_from_context(llm, model, q, ctx):
    """用 organizer 已格式化好的完整 context 直接提问（真实 DeepSeek）。"""
    user = (f"问题：{q}\n\n参考知识：\n{ctx}\n\n"
            "请仅依据给定知识给出简短准确的答案；若知识不足，明确说'知识不足'。")
    resp = llm.chat.completions.create(
        model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": dm.SYSTEM},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


def build_sample(n_extra=4):
    """FOCUS 10 题 = 6 FOCUS + 分层补 4 题。"""
    sel = dm.pick_rows()
    base10 = []
    for pre, (flag, note, row) in sel.items():
        base10.append({"question": row.get("question", ""),
                       "ref_answer": row.get("answer", ""),
                       "capability": row.get("capability", ""),
                       "_focus": note})
    import csv, random
    rows_all = list(csv.DictReader(open(INDUSTRYBENCH_CSV, encoding="utf-8-sig")))
    used_q = {x["question"] for x in base10}
    pool = [r for r in rows_all if (r.get("question") or "") not in used_q]
    random.seed(7)
    random.shuffle(pool)
    caps, added = set(), []
    for r in pool:
        cap = r.get("capability", "")
        if cap not in caps and len(added) < n_extra:
            caps.add(cap)
            added.append({"question": r.get("question", ""),
                          "ref_answer": r.get("answer", ""),
                          "capability": cap,
                          "_focus": "extra"})
        if len(added) >= n_extra:
            break
    return base10 + added, rows_all


def main():
    from openai import OpenAI
    llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    sample, _rows = build_sample(n_extra=4)
    print(f"[样本] {len(sample)} 题 (6 FOCUS + 4 extra)")

    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    pt = PassThroughOrganizer()
    o2 = EvidenceOrganizer()
    o3 = EvidenceOrganizer()
    directive = ExecutionDirective(module="organization", action=DICT, params={})
    scorer = RuleBasedScorer()

    arms = ["V1_pass", "V2_prod", "V3_purify"]
    S = {a: {"scores": [], "docs": [], "ctx": []} for a in arms}
    rows_out = []
    t0 = time.time()

    for idx, s in enumerate(sample, 1):
        q, ref, cap = s["question"], s["ref_answer"], s["capability"]
        print(f"[{idx}/{len(sample)}] {q[:36]}... cap={cap}")
        cands = _as_chunk_list(retriever.retrieve(q, k=10, use_ked=True))

        jobs = [
            ("V1_pass", "ctx1", pt, "pass", None),
            ("V2_prod", "ctx2", o2, "v2", None),
            ("V3_purify", "ctx3", o3, "v3", None),
        ]
        ctxs = {}
        evs = {}
        for arm, key, org, kind, _ in jobs:
            if kind == "pass":
                ev = pt.execute(directive, cands, question=q, task_type="general")
                ctx = pt.get_context(ev)
            elif kind == "v2":
                ev = o2.execute(directive, cands, question=q, task_type="general")
                ctx = ev.get_context()
            else:
                ev = o3.execute(directive, cands, question=q, task_type="general",
                                rerank=True, truncate=True, signal="lex", max_chars=6000)
                ctx = ev.get_context()
            ctxs[key] = ctx
            evs[arm] = ev

        for arm, key, _, _, _ in jobs:
            ans = answer_from_context(llm, LLM_NAME, q, ctxs[key])
            sc = scorer.rule_based_score(q, ref, ans)
            S[arm]["scores"].append(sc)
            S[arm]["docs"].append(getattr(evs[arm], "organized_count", 0))
            S[arm]["ctx"].append(len(ctxs[key]))
            # 逐档逐题记录
        rows_out.append({
            "q": q, "cap": cap, "focus": s.get("_focus", ""),
            "V1": S["V1_pass"]["scores"][-1], "V1_ctx": len(ctxs["ctx1"]),
            "V2": S["V2_prod"]["scores"][-1], "V2_ctx": len(ctxs["ctx2"]),
            "V3": S["V3_purify"]["scores"][-1], "V3_ctx": len(ctxs["ctx3"]),
        })
        if idx % 2 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    import statistics
    print("\n" + "=" * 70)
    print("  S4→S5 组织消融（固定 S3 top-10）答案分 RuleBased(0-3)")
    print("=" * 70)
    print(f"{'臂':<10}{'mean':>7}{'docs':>7}{'ctx':>7}")
    for a in arms:
        print(f"{a:<10}{statistics.mean(S[a]['scores']):>7.3f}"
              f"{statistics.mean(S[a]['docs']):>7.1f}{statistics.mean(S[a]['ctx']):>7.0f}")
    print("\n逐题分数 (V1_pass/V2_prod/V3_purify):")
    for r in rows_out:
        mark = ""
        if r["V2"] < r["V1"]:
            mark += " [V1>V2 修剪过度?]"
        if r["V2"] < r["V3"]:
            mark += " [V3>V2 噪声稀释?]"
        print(f"  {r['q'][:26]:26} V1={r['V1']} V2={r['V2']} V3={r['V3']}{mark}")

    out = os.path.join(RESULTS_DIR, "ab_organize_util_10.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"arms": {a: {"mean": round(statistics.mean(S[a]['scores']), 3),
                                "docs_mean": round(statistics.mean(S[a]['docs']), 2),
                                "ctx_mean": round(statistics.mean(S[a]['ctx']), 1),
                                "scores": S[a]['scores']} for a in arms},
                   "rows": rows_out}, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()


