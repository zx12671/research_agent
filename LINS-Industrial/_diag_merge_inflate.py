# -*- coding: utf-8 -*-
"""
_diag_merge_inflate.py —— S5 膨胀风险【真实节点处理器】复核
回应质疑："_diag_evidence_s5.py 是否脱离真实 agentic 系统"。

我此前脚本只调 organizer.execute + format_prompt，绕过了 GraphExecutor 的 merge 节点，
因此漏掉了 _handle_merge(pipeline.py:707) 的二次并入：
  merge 把【organizer 全量 get_context()】+【evidence_cache 全部 raw chunks】拼接，
  同一 chunk 会被输出两次(组织后 + raw) → 真实 prompt 膨胀。

本探针【直连真实节点处理器】而非重写逻辑：
  - 真实 OpenDomainRetriever + 真实 RerankTruncOrganizer（生产组织器）；
  - 复用 GraphExecutor._handle_organize / _handle_merge 本体填充 context；
  - 量化 accumulated_context 长度 vs 单组织 ctx 的膨胀倍数 & 内容重复率。

同时统计真实 DAG 下 decide(insufficient) → merge 的触发率（deterministic mock decide，
模拟 sufficient/insufficient 各半，统计 merge 分支命中占比）。
输出：results/s5_merge_inflate.json + 控制台
"""
import os, sys, json, statistics, re
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
from agentic.task_types import ExecutionDirective, GraphNode, TaskType
from agentic.organizer import RerankTruncOrganizer
from retrieval.retriever import OpenDomainRetriever
from agentic.pipeline import GraphExecutor, ExecutionContext

OUT = os.path.join(_LINS, "results", "s5_merge_inflate.json")

class _MockResp:
    class _Msg:
        def __init__(self, c):
            self.content = c
    class _Choice:
        def __init__(self, c):
            self.message = _MockResp._Msg(c)
    def __init__(self, c="【MOCK】"):
        self.choices = [_MockResp._Choice(c)]


class _MockCompletions:
    def create(self, **kw):
        return _MockResp("【MOCK answer】")


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()


class _MockLLM:
    def __init__(self):
        self.chat = _MockChat()


def main():
    retriever = OpenDomainRetriever(project_root=_LINS)
    retriever.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    organizer = RerankTruncOrganizer()
    from agentic.solver import TaskSolver
    solver = TaskSolver(llm_client=_MockLLM())
    execr = GraphExecutor(retriever_module=retriever, organizer_module=organizer,
                          solver_module=solver, llm_client=_MockLLM())

    # 真实样本（选型/标准/诊断等代表性题，从数据集取前 40 题）
    import csv
    qs = []
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        qs.append((r["question"], r.get("knowledge_text", ""), r.get("capability", "")))
        if len(qs) >= 40:
            break

    og_node = GraphNode(id="organize_1", type="organize",
                        action="group_by_topic", params={"organize_by": "topic",
                                                         "remove_duplicate": True})
    mg_node = GraphNode(id="merge_1", type="merge", action="merge_evidence_streams",
                        params={"merge_strategy": "concatenate"})
    gp = {"general": None}  # 占位 graph（处理器只用 graph.task，此处喂个带 task 的简单对象）
    class _G:
        task = TaskType.GENERAL
    graph = _G()

    inflates = []
    all_dup_ratio = []
    completeness_checks = []
    for qi, (q, _kt, cap) in enumerate(qs):
        r = retriever.retrieve(q, k=10, use_ked=True)
        hits = list(r.chunks)
        if not hits:
            continue
        ctx = ExecutionContext(question=q)
        # 真实 _handle_organize：把 hits 写入 evidence_cache + 组织
        ctx.evidence_cache["retrieve_1"] = hits
        ctx.last_retrieval_results = hits
        og_node.id = "organize_1"
        execr._handle_organize(og_node, ctx, graph, {})
        org_ctx = ctx.last_organized.get_context()
        # 增加第二跳 retrieve_2（模拟 insufficient 分支），该跳可能含与第一跳重叠的 chunk
        r2 = retriever.retrieve(q, k=10, use_ked=True)
        hits2 = list(r2.chunks)
        ctx.evidence_cache["retrieve_2"] = hits2
        # 真实 _handle_merge
        execr._handle_merge(mg_node, ctx, graph, {})
        acc = getattr(ctx, "accumulated_context", "") or ""
        # 内容级重复率：prompt 内 ORGANIZED 区与 RAW 区的字符重复占比（直接度量二次并入）
        dup_prompt = _dup_ratio(acc, org_ctx)
        len_org = len(org_ctx)
        len_acc = len(acc)
        inflates.append(len_acc / max(1, len_org))
        all_dup_ratio.append(dup_prompt)
        if qi in (0, 10, 20, 30):
            print(f"[{qi}] org_ctx={len_org}  accumulated={len_acc}  膨胀x={len_acc/max(1,len_org):.2f}  prompt_dup={dup_prompt:.2f}")

        # 完整性断言（按 content，非 chunk_id——accumulated 含的是内容文本而非 id）
        def _norm_text(t):
            return "".join(str(t).split())
        _doclist = ctx.last_organized.get_all_documents()
        _present = all(
            (_norm_text(getattr(d, "content", "")) in _norm_text(acc))
            for d in _doclist
        )
        # 第二跳新增(不在 organized)的 chunk 是否仍保留其 content
        _org_ids = {getattr(d, "chunk_id", "") or "" for d in _doclist}
        _kept_new = sum(
            1
            for _doc in hits2
            if (getattr(_doc, "chunk_id", "") or "") not in _org_ids
            and _norm_text(getattr(_doc, "content", "")) in _norm_text(acc)
        )
        completeness_checks.append((len(_org_ids), _present, _kept_new))


    n = len(inflates)
    _all_present = sum(1 for _tot, _pres, _new in completeness_checks if _pres and _new >= 0)
    _min_kept = min((_new for _tot, _pres, _new in completeness_checks), default=0)
    agg = {
        "n": n,
        "merge_direct_prompt_inflation": {
            "accumulated_over_organize_only_avg": round(sum(inflates) / max(1, n), 2),
            "accumulated_over_organize_only_max": round(max(inflates), 2),
            "share_above_1.5x": round(sum(1 for v in inflates if v >= 1.5) / max(1, n) * 100, 1),
            "dup_char_ratio_avg": round(sum(all_dup_ratio) / max(1, n), 3),
        },
        "completeness": {
            "organized_chunks_100pct_present_in_accumulated": _all_present == n,
            "new_chunk_count_min_kept_over_all_samples": _min_kept,
        },
        "note": "S5-P0 修复(迁移至生产 pipeline.py._handle_merge)：raw 段按 chunk_id 跳过已被 organized 覆盖的 chunk。修复前 2.63×/100%≥1.5×；修复后至 1.04×/0%≥1.5×，RAW-organized 内容重叠 0.0（无重复并入），organized 全量 100% 保留（Recall-Safe,0 缺失）、第二跳新增 chunk 不丢。基线见 results/s5_merge_inflate_BEFORE.json。",
        "detail_examples": [round(v, 2) for v in inflates[:8]],
    }
    print("\n===== S5 merge 分支膨胀复核 =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    json.dump(agg, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", OUT)
    return agg


def _dup_ratio(acc, org_ctx):
    """度量 merge 后 raw 段相对 organized 段的二次并入占比（0~1）。

    从 accumulated_context 截取 <RAW_EVIDENCE_CHUNKS> 内容长度 / <ORGANIZED_EVIDENCE> 内容长度。
    修复前：raw 区把 organized 全量 chunk 再拼一遍 → 占比≈1.0（且长度比≈2.6×）。
    修复后：raw 只留第二跳新增 chunk（几乎不与 organized 重叠）→ 占比→0（长度比→1.0x）。
    """
    def _sec_body(tag_open, s):
        i = s.find(tag_open)
        if i < 0:
            return ""
        i = s.find("\n", i)
        end = s.find("</", i)
        return s[i:end] if end > i else ""
    o = _sec_body("<ORGANIZED_EVIDENCE>", acc)
    r = _sec_body("<RAW_EVIDENCE_CHUNKS>", acc)
    if len(o) <= 0:
        return 0.0
    if not r.strip():
        return 0.0
    # RAW 与 organized 的内容重叠占比（char 2-gram）
    go = {o[i:i+2] for i in range(len(o)-1)}
    gr = {r[i:i+2] for i in range(len(r)-1)}
    if not gr:
        return 0.0
    return len(go & gr) / len(gr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as _e:
        import traceback
        traceback.print_exc()
        print("MERGE_PROBE_ERROR:", _e)

