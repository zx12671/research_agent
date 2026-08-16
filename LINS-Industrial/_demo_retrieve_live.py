# -*- coding: utf-8 -*-
r"""
_demo_retrieve_live.py — 一次真实 hybrid_retrieve 的活体演示：

对一个真实 CSV 问题走完整的 OpenDomainRetriever.hybrid_retrieve，
打印三条链：
  1) query 的变换链条：原始 → KED 扩展(query_expanded) → multi-query 子查询(decompose)
  2) dense(+multi) 与 BM25 融合后的 top-k chunks
  3) 每个 chunk 的 chunk_id / score / capability / content 开头 → 看到 query 落到哪类知识块

⚠️ 注意（口径修正，2026-08-08）：
  本脚本最后“参考答案在 top-k 是否命中”用的是【整段字面】判定，会把
  “命中但措辞不同”的 chunk 误判成“未命中”（例：q1 答案要点“整流并联运行”
  其实在 rank1 chunk 里，只是字序/措辞不同）。可信的命中判定见
  `_demo_retrieve_verify.py`（按答案实体/要点短语判定）与
  `docs/demo_query_chunk_retrieval.md`。本脚本的前几步(query变换/top10/来源标注)
  是可靠的，仅末尾“命中与否”一行不具判定效力。

用法: cd LINS-Industrial && python _demo_retrieve_live.py
"""
import os, sys, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
sys.path.insert(0, os.path.abspath(os.path.join(_LINS, "..")))

from retrieval.retriever import OpenDomainRetriever

# ── 取一个真实 CSV 问题（流式读，不整载入内存）──
r = OpenDomainRetriever(project_root=_LINS)
r.load_from_manifest(
    manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json")
)
print(f"[就绪] chunks 数 = {len(r.chunks)}, embedding_model = {getattr(r,'embedding_model','?')}")

QUESTIONS = {
    "q1(型号+参数+场景)": "在冶金轧机这类高负载工艺中，若需超过单台SIMOREG DC Master 6RA70装置的输出能力，应采取何种运行方式来满足需求？",
    "q2(标准号安全)": "在升降式高杆照明装置中，若采用单根主钢丝绳作为升降系统的上绳，除需满足设计安全系数不小于8外，还应设置哪些安全装置以防止灯盘意外坠落？",
}

K = 10
for label, q in QUESTIONS.items():
    print("\n" + "=" * 78)
    print(f"[{label}] top_{K} 检索")
    print("=" * 78)

    # 1) query 变换链条
    print("\n◆ 1) query 变换链条")
    print(f"  原始 query   : {q}")
    expanded = r.ked.expand_query(q)
    print(f"  KED 扩展     : {expanded[:120]}{'...' if len(expanded)>120 else ''}")
    # 打印 KED 抽到的关键词
    kws = r.ked.extract(q)
    print(f"  KED 关键词   : {kws}")
    dec = r.ked.decompose(q)
    print(f"  multi-query  : {dec if len(dec)>1 else '(未拆分, 单查询)'}")

    # 2) 实际检索（hybrid = dense(multi) + BM25 → RRF）
    res = r.hybrid_retrieve(q, k=K, use_ked=True, use_multi_query=True,
                            use_sparse=True, dense_weight=1.0, sparse_weight=0.2)
    print("\n◆ 2) hybrid_retrieve 融合后 top-%d chunks" % K)
    print(f"  {'rank':<4}{'score':<8}{'capability':<12}{'source':<12} content 开头")
    print("  " + "-" * 70)
    for c in res.chunks:
        head = c.content.replace("\n", " ")[:48]
        print(f"  {c.rank:<4}{c.score:<8.4f}{c.capability:<12}{c.source:<12} {head}")
    # 每个 chunk 来自 dense 还是 sparse 只能通过是否出现在 dense 结果里判断 → 补一个区分
    dense_ids = {d.chunk_id for d in r.dense_search(q, k=K, use_ked=True).chunks}
    sparse_ids = {cid for cid, _ in r.bm25_search(q, k=K)}
    print("\n  标注：dense命中▲ / sparse命中◆ / 双路命中✚")
    for c in res.chunks:
        tag = ("✚" if c.chunk_id in dense_ids and c.chunk_id in sparse_ids
               else "▲" if c.chunk_id in dense_ids else "◆")
        print(f"    rank{c.rank} {tag}  {c.chunk_id}  {c.capability}")

    # 3) 对照参考答案是否在这些 chunk 里
    #   （用答案里的关键实体在 top10 content 里做字面匹配）
    ref = None
    with open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("question") or "").strip() == q:
                ref = row.get("answer") or ""
                break
    if ref:
        core = [t.strip() for t in ref.replace("。", "，").split("，") if len(t.strip()) >= 4]
        hit = []
        for c in res.chunks:
            if any(k in c.content for k in core):
                hit.append((c.rank, core))
        print("\n◆ 3) 参考答案在 top-%d 中的命中" % K)
        print(f"  参考答案(CSV): {ref}")
        if hit:
            for rk, core in hit:
                print(f"  ✓ rank{rk} 的 chunk 含答案片段: {core[:40]}")
        else:
            print("  ✗ top-%d 均未直接含答案片段（可能召回未命中 → 印证 S4 召回缺料）" % K)
