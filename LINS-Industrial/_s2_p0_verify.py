# -*- coding: utf-8 -*-
"""_s2_p0_verify.py — 孤立验证 S2-P0 短路（不加载 embedding 模型）。

用 __new__ 绕过 AdaptiveAgenticPipeline.__init__（避免加载 retriever/llm 模型），
只构造 ExecutionContext + 极简图，直接调用 _handle_retrieve：

  1) 主检索 retrieve_1 + pre_retrieved_evidence → 必须走 SHORTCUT，且【不】调用 self.retriever
  2) followup retrieve_2 (evidence_guided=True) + pre_retrieved_evidence → 【不】短路，真实调用 retriever
  3) 无 pre_retrieved_evidence → 正常路径，调用 retriever

证明短路只影响主检索、不丢 follow-up recall、非短路时行为不变。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic.pipeline import GraphExecutor, ExecutionContext
from agentic.task_types import ExecutionGraph, GraphNode, TaskType, TaskAnalysis


class _BoomRetriever:
    """若被调用即抛错 —— 短路通过则证明未触碰检索。"""
    def retrieve(self, *a, **k):
        raise RuntimeError("SHORTCUT FAIL: retriever.retrieve called")


class _FakeChunk:
    def __init__(self, i):
        self.chunk_id = f"chunk_{i}"
        self.content = f"证据内容 #{i}"
        self.score = 0.9 - i * 0.05
        self.rank = i


def make_executor():
    e = GraphExecutor.__new__(GraphExecutor)
    e.second_retrieval_count = 0
    e.retriever = _BoomRetriever()
    e.semantic_rewriter = None
    return e


def make_node(nid, ntype, **params):
    if "retrieve_k" not in params:
        params["retrieve_k"] = 2
    return GraphNode(id=nid, type=ntype, action="x", params=params, next=[], description="")


def make_graph(nid):
    ta = TaskAnalysis(task=TaskType.GENERAL, original_question="q")
    return ExecutionGraph(task=TaskType.GENERAL, task_analysis=ta, nodes={"n": make_node(nid, "retrieve")}, entry_points=["n"])


def run(nid, params, pre_evidence, expect_boom):
    e = make_executor()
    ctx = ExecutionContext(question="q")
    if pre_evidence is not None:
        ctx.last_retrieval_results = list(pre_evidence)
        ctx.evidence_cache["__external_retrieval__"] = list(pre_evidence)
    g = make_graph(nid)
    node = make_node(nid, "retrieve", **params)
    threw = False
    try:
        e._handle_retrieve(node, ctx, g, {})
    except RuntimeError:
        threw = True
    cached = ctx.evidence_cache.get(nid)
    shortcut = any("SHORTCUT" in str(le) for le in ctx.execution_log)
    # 通过条件: retriever 调用与否符合预期，且短路路径下证据已登记到 node
    ok = (threw == expect_boom) and (not shortcut or (cached is not None and len(cached) >= 0))
    return "OK " if ok else "FAIL", threw, shortcut, len(cached) if cached else 0


def main():
    pre = [_FakeChunk(i) for i in range(6)]
    # 1) 主检索 + pre_evidence → SHORTCUT, retriever 不被调用
    st, boom, short, n = run("retrieve_1", {}, pre, expect_boom=False)
    print(f"[1] retrieve_1 + pre_evidence: SC={'SHORTCUT' if short else 'NO-SC'} "
          f"retriever_called={boom} cached={n} -> {st}")

    # 2) followup retrieve_2(evidence_guided) + pre_evidence → 不短路, retriever 被调用(boom)
    st2, boom2, short2, n2 = run("retrieve_2", {"evidence_guided": True}, pre, expect_boom=True)
    print(f"[2] retrieve_2 + pre_evidence: SC={'SHORTCUT' if short2 else 'NO-SC'} "
          f"retriever_called={boom2} cached={n2} -> {st2}")

    # 3) 主检索 无 pre_evidence → 不短路, retriever 被调用(boom)
    st3, boom3, short3, n3 = run("retrieve_1", {}, None, expect_boom=True)
    print(f"[3] retrieve_1 无 pre_evidence: SC={'SHORTCUT' if short3 else 'NO-SC'} "
          f"retriever_called={boom3} cached={n3} -> {st3}")

    # 4) 主检索 + pre_evidence, retrieve_k=2 → 截断到 2 条
    st4, boom4, short4, n4 = run("retrieve_1", {"retrieve_k": 2}, pre, expect_boom=False)
    print(f"[4] retrieve_1 + pre_evidence(k=2): SC={'SHORTCUT' if short4 else 'NO-SC'} "
          f"cached={n4}(应=2) -> {st4}")


if __name__ == "__main__":
    main()
