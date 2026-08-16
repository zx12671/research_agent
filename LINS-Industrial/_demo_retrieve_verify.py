# -*- coding: utf-8 -*-
r"""
_demo_retrieve_verify.py — 精细命中判定（对 _demo_retrieve_live.py 的补充）：

问题: q1 / q2 的参考答案到底在不在库里？若在，正确 chunk 的库内 rank / 内容 / capability 是什么？
  - 若"答案在库、且排序靠前" → 召回命中（组织/生成侧问题，不归 S4）
  - 若"答案在库、但 dense/BM25 rank 极靠后没进 top10" → 真·召回漏（归 S4，query 语义鸿沟）
  - 若"答案不在库" → 知识库缺料（归 S4 建库）

用法: cd LINS-Industrial && python _demo_retrieve_verify.py
"""
import os, sys, io, csv, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
sys.path.insert(0, os.path.abspath(os.path.join(_LINS, "..")))

from retrieval.retriever import OpenDomainRetriever

r = OpenDomainRetriever(project_root=_LINS)
r.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
print(f"[就绪] chunks = {len(r.chunks)}")

CASES = [
    ("q1 型号+参数(选型并联)",
     "在冶金轧机这类高负载工艺中，若需超过单台SIMOREG DC Master 6RA70装置的输出能力，应采取何种运行方式来满足需求？",
     "采用多台装置整流并联运行。",
     ["并联运行", "整流并联", "多台装置"]),
    ("q2 安全条款(高杆照明)",
     "在升降式高杆照明装置中，若采用单根主钢丝绳作为升降系统的上绳，除需满足设计安全系数不小于8外，还应设置哪些安全装置以防止灯盘意外坠落？",
     "应设置防止灯盘意外坠落制动装置或自动卸载装置。",
     ["制动装置", "自动卸载", "防止灯盘意外坠落"]),
]

# 答案片段 → 库内精确文档定位（全量扫描 55095 chunks content）
def locate_answer(needles):
    """返回 {needle: [chunk_id, capability, rank_in_loop]}；无 → None"""
    found = {}
    content_list = []
    cap_list = []
    for cid, data in r.chunks.items():
        content_list.append(data.get("content", ""))
        cap_list.append(data.get("capability", ""))
    for nd in needles:
        for i, c in enumerate(content_list):
            if nd in c:
                found[nd] = (list(r.chunks.keys())[i],
                             cap_list[i],
                             data.get("content", "")[:40])
                break
    return found

for label, q, ref, needles in CASES:
    print("\n" + "=" * 74)
    print(f"[{label}]")
    print("  参考答案: " + ref)
    res = r.hybrid_retrieve(q, k=10, use_ked=True, use_multi_query=True,
                            use_sparse=True)
    top_ids = [c.chunk_id for c in res.chunks]
    print("  top10 chunk 的 capability 分布:",
          {c.capability for c in res.chunks})
    # 1) 答案片段是否在 top10 内
    print("\n  (A) 答案片段在 top10 内命中：")
    any_hit = False
    for nd in needles:
        hits = [c.rank for c in res.chunks if nd in c.content]
        if hits:
            any_hit = True
            c0 = [c for c in res.chunks if nd in c.content][0]
            print(f"    ✓ 「{nd}」 → rank{hits}, chunk {c0.chunk_id}, cap={c0.capability}")
            print(f"       片段: …{c0.content[max(0,c0.content.find(nd)-12):c0.content.find(nd)+18]}…")
        else:
            print(f"    ✗ 「{nd}」 → top10 未见")
    if not any_hit:
        print("    → 答案要点未进 top10，判为召回/排序问题（需看 B/C）")
    # 2) 答案片段在 55095 全文库里的精确定位
    print("\n  (B) 答案片段在 55095 全文库精确字面定位：")
    for nd in needles:
        hit = None
        for cid, data in r.chunks.items():
            if nd in data.get("content", ""):
                hit = (cid, data.get("capability", ""))
                snip = data.get("content", "")
                i = snip.find(nd)
                print(f"    ✓ 「{nd}」 → chunk {hit[0]} cap={hit[1]}")
                print(f"       原文片段: …{snip[max(0,i-16):i+20].rstrip()}…")
                break
        if hit is None:
            print(f"    ✗ 「{nd}」 → 全库 55095 chunks 均无该字面 → 属知识库缺料(建库)")
    print("\n  (C) 该题在 agentic 里是否真的进组织层：top-id ⊂ evidence_cache（由 pipeline 决定）")
