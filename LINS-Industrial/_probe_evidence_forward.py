# -*- coding: utf-8 -*-
"""
evidence-forward 立项探针：base vs base+doc回捞（30 样本, seed=7, 句级覆盖率口径, 与 A/B 探针一致）。

思想（可行性快检已证实）：chunk 自带 document_id + chunk_index + total_chunks，
A 类超长 GT 的源文档有 52~235 块，而 top-10 只能覆盖其开头。故：
  1. retrieve(q, k=10) 拿 top-10 chunk 及各自的 document_id；
  2. 用 document_id 从 retriever.chunks 精确回捞该档全部块（按 chunk_index 还原顺序）；
  3. 与 top-10 合并去重，得到三种候选：
       base     = 仅 top-10（当前生产口径）
       ef_full  = top-10 + 全部 doc 回捞块（看天然覆盖上限）
       ef_n20   = top-10 + 每档限前 20 块（保守防爆候选池）
度量：句级覆盖率 bigram JT>=0.30；命中=至少覆盖 1 句；达标=cov>=50%。
重点看 A 类(超长GT>=50句)在 base+doc回捞下的增益。
"""
import os, csv, random, re
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7
IOU_SENT_TH = 0.30
MIN_COVERED = 1
COV_OK_TH = 0.50


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


def coverage(sents, chunks):
    covered = 0
    for sc in sents:
        if any(jt(sc, c) >= IOU_SENT_TH for c in chunks):
            covered += 1
    total = len(sents)
    return covered, total, (covered / total if total else 0.0)


def base_candidates(res):
    """仅 top-10 的 content 列表。"""
    return [_content(c) for c in getattr(res, "chunks", [])]


def doc_backfill(retriever, res, max_per_doc=None):
    """
    返回 (top10_content, 新增回捞 content) 。
    思路：取 top-10 每块 document_id -> 回捞该档全部块(按 chunk_index 排序)，
    再与 top-10 内容合并去重，得到候选。
    max_per_doc: 每档最多回捞前 max_per_doc 块（None=全部, 保守防爆）。
    """
    top = list(getattr(res, "chunks", []))
    top_contents = [_content(c) for c in top]
    got_docs = set()
    extra = []
    for c in top:
        did = getattr(c, "document_id", "") or ""
        if not did or did in got_docs:
            continue
        got_docs.add(did)
        # 收集该档全部块
        same = []
        for cid, data in retriever.chunks.items():
            dd = data.get("document_id") or data.get("doc_id") or ""
            if dd == did:
                same.append(data)
        same.sort(key=lambda d: d.get("chunk_index", 0))
        if max_per_doc is not None:
            same = same[:max_per_doc]
        for d in same:
            txt = d.get("content", "")
            if txt and txt not in top_contents and txt not in extra:
                extra.append(txt)
    return top_contents, extra


def main():
    import time
    sample = sample_questions()
    from retrieval.retriever import OpenDomainRetriever
    retriever = OpenDomainRetriever()
    retriever.load_from_manifest()

    cols = ["base", "ef_full", "ef_n20"]
    print(f"evidence-forward: base vs base+doc回捞  (N={len(sample)}, seed={SEED}, top-10, use_ked=True)")
    print(f"口径: 句级 bigram JT>={IOU_SENT_TH}; 命中=覆盖>=1句; 达标=cov>={COV_OK_TH:.0%}\n")

    rows = []
    t0 = time.time()
    for r in sample:
        q, kt = r.get("question", ""), r.get("knowledge_text", "")
        cap = r.get("capability", "")
        sents = split_sent(kt)
        nS = len(sents)
        cls = "A" if nS >= 50 else "B"
        qshort = q[:26]
        row = {"q": qshort, "cap": cap[:8], "cls": cls, "nS": nS}
        res = retriever.retrieve(q, k=10, use_ked=True)
        cands = {
            "base": base_candidates(res),
            "ef_full": None,
            "ef_n20": None,
        }
        topc, extra_full = doc_backfill(retriever, res, max_per_doc=None)
        _, extra_n20 = doc_backfill(retriever, res, max_per_doc=20)
        # 注意：为公平，均在"同一 top-10 + 回捞"基础上比较；base 单独用 topc
        cands["base"] = topc
        cands["ef_full"] = topc + extra_full
        cands["ef_n20"] = topc + extra_n20

        for name in cols:
            cov, total, ratio = coverage(sents, cands[name])
            row[f"{name}_cov"] = cov
            row[f"{name}_ratio"] = ratio
            row[f"{name}_ncand"] = len(cands[name])
        rows.append(row)

    # ---------- 汇总 ----------
    print(f"{'Q':26}{'cap':8}{'cl':2}{'#S':>4} | {'base':>7}{'ef_full':>9}{'ef_n20':>8} | {'b_n':>4}{'ef_n':>4}")
    print("-" * 80)
    for row in rows:
        print(f"{row['q']:26}{row['cap']:8}{row['cls']:2}{row['nS']:>4} | "
              f"{row['base_ratio']*100:6.1f}%{row['ef_full_ratio']*100:8.1f}%{row['ef_n20_ratio']*100:7.1f}% | "
              f"{row['base_ncand']:>4}{row['ef_full_ncand']:>4}")

    print("\n===== 聚合 =====")
    for cls in ["all", "A", "B"]:
        sub = [r for r in rows if cls == "all" or r["cls"] == cls]
        if not sub:
            continue
        print(f"[{cls}] n={len(sub)}")
        for name in cols:
            ratios = [r[f"{name}_ratio"] for r in sub]
            ok = sum(1 for r in sub if r[f"{name}_ratio"] >= COV_OK_TH)
            import statistics
            print(f"   {name:8} covMean={statistics.mean(ratios)*100:5.1f}%  "
                  f"covMed={statistics.median(ratios)*100:5.1f}%  达标({COV_OK_TH:.0%})={ok}/{len(sub)}  "
                  f"命中={sum(1 for r in sub if r[f'{name}_ratio']>0)}")
    print(f"\n耗时 {time.time()-t0:.1f}s")
    print("\n说明: ef_full=top10+整档回捞; ef_n20=top10+每档限前20块; base=仅top10")


if __name__ == "__main__":
    main()
