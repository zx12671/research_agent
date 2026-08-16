"""
_probe_organize_ab.py — 件① 提纯档（organize_panel）*零-LLM* 结构差异探针。

目的：
    生产装配了点用 RerankTruncOrganizer（execute 默认 rerank=True/truncate=True，
    signal='lex', max_chars=6000 → L2 词法顶置 + 低分端截断），但之前从未按
    A/B 类分层量化它相对 EvidenceOrganizer（无提纯）在【证据输入层】的差异。
    本探针【不打分、不调 LLM】，纯结构度量，先回答一个前置问题：
      "提纯档到底改了什么、对 A 类(GT 超长难覆盖)是否真把相关块顶到前面/砍掉噪声？"

对照（同一 top-10 检索结果）：
    base = EvidenceOrganizer()               # 无提纯（组=呈现，绝不过滤）
    rt   = RerankTruncOrganizer()            # 提纯档（生产默认；词法重排+低分截断）

度量（每题）：
    in_n              top-10 输入块数
    base_n / rt_n     组织后保留块数（rt 截断可能 < base）
    drop_pct          提纯档相比 base 少保留的块占比（截断强度）
    ctx_chars_base/rt get_context() 字符数（提纯档是否真压低了 prompt 体积）
    lex_front_top5    两个档里 top-5 词法相关块占比（_lexical_overlap(q,doc)>=0.5 的块）
    reorder_diff      top-5 集合在两种组织下的不同块数（重排是否生效）

按 cls=A/B 分层汇总，输出均值对比表。

用法：
    python _probe_organize_ab.py
    （默认跑 35 题池；--limit N 可截断样本数用于快速验证脚本本身）
"""
import os
import json
import sys
import argparse
from collections import defaultdict

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import register_paths
register_paths()

from retrieval.retriever import OpenDomainRetriever
from agentic.organizer import EvidenceOrganizer, RerankTruncOrganizer
from agentic.task_types import ExecutionDirective

POOL = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")
OUT = os.path.join(_LINS, "results", "organize_ab_structure_35.json")


def _doc_text(d) -> str:
    return getattr(d, "content", "") or ""


def _lex_hit(org, query, d, th=0.5) -> bool:
    """判断该块与 query 是否'词法相关'（用 organizer 自带的轻量 2-gram 信号）。"""
    if not hasattr(org, "_lexical_overlap"):
        return False
    try:
        return org._lexical_overlap(_doc_text(d), query) >= th
    except Exception:
        return False


def _top5_set(docs):
    return set(_doc_text(d)[:60] for d in docs[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 题（0=全部）用于快速验证")
    ap.add_argument("--max_chars", type=int, default=6000,
                    help="提纯档截断上限（默认 6000=生产现状，几乎不截断；调小如 1200/2000 验证截断是否真正压缩证据）")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="提纯档块数上限（None=不按块数截）")
    args = ap.parse_args()

    pool = json.load(open(POOL, encoding="utf-8"))
    rows = pool["rows"]
    if args.limit > 0:
        rows = rows[:args.limit]
    QS = [(r["q"], r.get("cls", "?")) for r in rows]

    print(f"[probe] 载入 {len(QS)} 题（cls 分布:",
          {c: sum(1 for _, c2 in QS if c2 == c) for c in sorted({c for _, c in QS})}, ")")

    retriever = OpenDomainRetriever()
    try:
        retriever.load_from_manifest(
            manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception as e:
        print("[WARN] 指定 manifest 失败，回退自动发现:", e)
        retriever.load_from_manifest()

    directive = ExecutionDirective(
        module="organization", action="group_by_topic",
        params={"organize_by": "topic", "remove_duplicate": True})

    org_base = EvidenceOrganizer()
    org_rt = RerankTruncOrganizer()

    agg = defaultdict(lambda: {"n": 0, "in_n": [], "base_n": [], "rt_n": [],
                               "drop_pct": [], "ctx_base": [], "ctx_rt": [],
                               "lex_front_base": [], "lex_front_rt": [],
                               "reorder_diff": []})
    per = []

    for i, (q, cls) in enumerate(QS):
        res = retriever.retrieve(q, k=10, use_ked=True)
        chunks = list(getattr(res, "chunks", []) or getattr(res, "documents", []))
        if not chunks:
            print(f"  [{i}] cls={cls} 检索空，跳过")
            continue

        try:
            ob = org_base.execute(directive=directive, documents=chunks,
                                  question=q, task_type="general")
            ort = org_rt.execute(directive=directive, documents=chunks,
                                 question=q, task_type="general",
                                 rerank=True, truncate=True, signal="lex",
                                 max_chars=args.max_chars,
                                 max_chunks=args.max_chunks)
        except Exception as e:
            print(f"  [{i}] organizer 报错: {e}，跳过")
            continue

        base_docs = ob.get_all_documents()
        rt_docs = ort.get_all_documents()
        base_ctx = ob.get_context()
        rt_ctx = ort.get_context()

        drop_pct = (len(base_docs) - len(rt_docs)) / max(1, len(base_docs))
        lex_base = sum(1 for d in base_docs[:5] if _lex_hit(org_base, q, d)) / 5.0
        lex_rt = sum(1 for d in rt_docs[:5] if _lex_hit(org_rt, q, d)) / 5.0
        reorder_diff = len(_top5_set(base_docs) ^ _top5_set(rt_docs))

        rec = {
            "q": q, "cls": cls, "in_n": len(chunks),
            "base_n": len(base_docs), "rt_n": len(rt_docs),
            "drop_pct": round(drop_pct, 3),
            "ctx_base": len(base_ctx), "ctx_rt": len(rt_ctx),
            "lex_front_base": round(lex_base, 3), "lex_front_rt": round(lex_rt, 3),
            "reorder_diff": reorder_diff,
        }
        per.append(rec)
        a = agg[cls]
        a["n"] += 1
        for k in ("in_n", "base_n", "rt_n", "drop_pct", "ctx_base", "ctx_rt",
                  "lex_front_base", "lex_front_rt", "reorder_diff"):
            a[k].append(rec[k])

    print("\n===== 按 cls 分层（均值）=====")
    hdr = (f"{'cls':<4} {'n':>3} {'in':>4} {'base_n':>6} {'rt_n':>5} "
           f"{'drop%':>6} {'ctx_base':>9} {'ctx_rt':>8} {'lexB@5':>6} "
           f"{'lexR@5':>6} {'reorder':>7}")
    print(hdr)

    for cls in sorted(agg):
        a = agg[cls]
        def m(x):
            return sum(x) / len(x) if x else 0.0
        print(f"{cls:<4} {a['n']:>3} {m(a['in_n']):>4.0f} {m(a['base_n']):>6.1f} "
              f"{m(a['rt_n']):>5.1f} {100*m(a['drop_pct']):>5.1f}% "
              f"{m(a['ctx_base']):>9.0f} {m(a['ctx_rt']):>8.0f} "
              f"{m(a['lex_front_base']):>6.2f} {m(a['lex_front_rt']):>6.2f} "
              f"{m(a['reorder_diff']):>7.1f}")

    _all = "ALL"
    for k in list(agg.keys()):
        for field in ("in_n", "base_n", "rt_n", "drop_pct", "ctx_base", "ctx_rt",
                      "lex_front_base", "lex_front_rt", "reorder_diff"):
            agg[_all][field].extend(agg[k][field])
    a = agg[_all]
    def m(x):
        return sum(x) / len(x) if x else 0.0
    print(f"{_all:<4} {a['n']:>3} {m(a['in_n']):>4.0f} {m(a['base_n']):>6.1f} "
          f"{m(a['rt_n']):>5.1f} {100*m(a['drop_pct']):>5.1f}% "
          f"{m(a['ctx_base']):>9.0f} {m(a['ctx_rt']):>8.0f} "
          f"{m(a['lex_front_base']):>6.2f} {m(a['lex_front_rt']):>6.2f} "
          f"{m(a['reorder_diff']):>7.1f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"per_sample": per}, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] 逐题明细已写: {OUT}")


if __name__ == "__main__":
    main()
