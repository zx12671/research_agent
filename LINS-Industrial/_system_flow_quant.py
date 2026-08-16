# -*- coding: utf-8 -*-
"""
_system_flow_quant.py — Agentic 系统【执行流水线全链路量化探针】

把 AdaptiveAgenticPipeline v2 拆成 5 个可独立量化的环节，对同一批 35 题逐环节采集指标：

    Q -> [1 TaskAnalyzer] -> [2 StrategyPlanner] -> [3 OpenDomainRetriever]
       -> [4 EvidenceOrganizer] -> [5 GraphExecutor(走完整 DAG)] -> 答案

两种 LLM 模式（`build_llm()` 控制）：
  - 默认 MockLLM（零成本、确定性）：planner/decide/verify/reason 返回固定内容，
    只量化"结构/调用点/证据流"，不访问 DeepSeek API。
  - `--real`（或环境变量 SYSTEM_FLOW_REAL=1）：真实 DeepSeek 客户端(经 CountingLLM 计数)，
    量化"真实端到端准确率 + 真实规划图/分支行为"。需可用 API key
    （优先环境变量 SYSTEM_FLOW_REAL_API_KEY，回退 experiments/config.DEEPSEEK_KEY）。

- Retriever 始终用真实 faiss 索引(真实证据流)。
- 输出: results/_system_flow_quant_{mode}.json   (rows + aggregated)

用法(在 LINS-Industrial 下)：
    python _system_flow_quant.py                          # 零成本回归(两模式,35题)
    python _system_flow_quant.py --mode fallback --max-q 5   # 只跑 fallback 前 5 题
    python _system_flow_quant.py --real --max-q 5            # 真实 LLM 前 5 题(建议限题)
"""

import os
import sys
import json
import time
import statistics
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

try:
    from experiments.config import register_paths, KNOWLEDGE_CORPUS_DIR
    register_paths()
except Exception as e:
    print("[WARN] register_paths 失败:", e)
    KNOWLEDGE_CORPUS_DIR = os.path.join(_LINS, "knowledge_corpus")

POOL = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")
OUT = os.path.join(_LINS, "results", "_system_flow_quant")


# ============================================================
# Mock LLM —— 记录所有调用点, 不访问真实 API
# ============================================================
class _MockChoice:
    def __init__(self, content):
        class _Msg:
            def __init__(self, c):
                self.content = c
        self.message = _Msg(content)


class _MockResponse:
    def __init__(self, content):
        self.choices = [_MockChoice(content)]
        self.content = content


class MockLLM:
    """OpenAI 兼容 mock: 记录调用点+prompt长度, 返回可控内容。
    planner_mode: "fallback"(坏JSON->fallback图) / "llm_graph"(合法JSON)
    reason_mode/decide_mode/verify_mode: 对应节点的返回值。
    """

    def __init__(self, planner_mode="fallback", reason_mode="answer",
                 decide_mode="sufficient", verify_mode="pass"):
        self.planner_mode = planner_mode
        self.reason_mode = reason_mode
        self.decide_mode = decide_mode
        self.verify_mode = verify_mode
        self.calls = []
        self.chat = self._ChatMethod(self)
        self.completions = self._Completions(self)
        self._completions = self._Completions(self)

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            return self.outer._handle(kwargs)

    class _ChatMethod:
        def __init__(self, outer):
            self.outer = outer

        def __call__(self, **kwargs):
            return self.outer._handle(kwargs)

    def _handle(self, kwargs):
        kind = self._detect_kind(kwargs)
        content = self._content_by_kind(kind)
        prompt = self._prompt_text(kwargs.get("messages", []))
        self.calls.append({
            "kind": kind,
            "prompt_len": len(prompt),
            "max_tokens": kwargs.get("max_tokens", 0),
        })
        return _MockResponse(content)

    @staticmethod
    def _prompt_text(messages):
        if not isinstance(messages, list):
            return str(messages)
        return " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in messages
        )

    def _detect_kind(self, kwargs):
        prompt = self._prompt_text(kwargs.get("messages", []))
        max_tokens = kwargs.get("max_tokens", 0)
        if ("OUTPUT FORMAT (JSON" in prompt or
                "designing an adaptive retrieval" in prompt or
                "ExecutionGraph" in prompt or "Now design the graph" in prompt):
            return "planner"
        if "Evaluate the following condition" in prompt:
            return "decide"
        if ("verify" in prompt.lower() and
                ("pass" in prompt.lower() or "fail" in prompt.lower()) and
                max_tokens and max_tokens <= 64):
            return "verify"
        return "reason"

    def _content_by_kind(self, kind):
        if kind == "planner":
            if self.planner_mode == "llm_graph":
                return json.dumps({
                    "entry_point": "retrieve_1",
                    "nodes": [
                        {"id": "retrieve_1", "type": "retrieve",
                         "action": "semantic_search",
                         "params": {"retrieve_k": 10, "multi_query": False,
                                    "use_ked": True},
                         "next": ["organize_1"], "description": "initial"},
                        {"id": "organize_1", "type": "organize",
                         "action": "group_by_topic",
                         "params": {"organize_by": "topic"},
                         "next": ["reason_1"], "description": "organize"},
                        {"id": "reason_1", "type": "reason",
                         "action": "execute_reasoning_workflow",
                         "params": {"reasoning_type": "general",
                                    "workflow_steps": ["synthesize"]},
                         "next": ["end"], "description": "reason"},
                        {"id": "end", "type": "end", "action": "complete",
                         "description": "terminal"},
                    ],
                })
            return "not valid json at all"  # -> planner 捕获异常走 fallback
        if kind == "reason":
            return "【MOCK】占位推理答案：综合给定证据后给出的结论。"
        if kind == "decide":
            return self.decide_mode
        if kind == "verify":
            return self.verify_mode
        return ""


# ============================================================
# 各环节量化测量函数


# ============================================================
# 真实 LLM / Mock LLM 可切换
# ============================================================
class CountingLLM:
    """包装真实 OpenAI 客户端：在透传调用的同时，记录每次调用的 kind/prompt_len/max_tokens。

    暴露与 OpenAIClient 一致的接口:
      llm.chat.completions.create(model=..., messages=[...], max_tokens=...)
      → pipeline `hasattr(self.llm.chat, "completions")` 为 True，走真实路径。
    """

    def __init__(self, client, coder=None):
        self._client = client          # 真实 OpenAI 客户端 (有 .chat.completions.create)
        self.calls = []                # 记录 [{kind, prompt_len, max_tokens}]
        self._coder = coder or MockLLM()  # 复用 _detect_kind/_prompt_text（纯静态，不触发网络）
        self.chat = self._Chat(self, client.chat)

    class _Completions:
        def __init__(self, owner, real_completions):
            self._owner = owner
            self._real = real_completions

        def create(self, **kwargs):
            owner = self._owner
            kind = owner._coder._detect_kind(kwargs)
            prompt = owner._coder._prompt_text(kwargs.get("messages", []))
            owner.calls.append({
                "kind": kind,
                "prompt_len": len(prompt),
                "max_tokens": kwargs.get("max_tokens", 0),
            })
            return self._real.create(**kwargs)

    class _Chat:
        def __init__(self, owner, real_chat):
            self._owner = owner
            self.completions = CountingLLM._Completions(owner, real_chat.completions)

    def _handle(self, kwargs):
        # 兼容 pipeline 的 else 分支（llm.chat 直接调用）——真实 openai 客户端不走此路径，保留兜底
        kind = self._coder._detect_kind(kwargs)
        prompt = self._coder._prompt_text(kwargs.get("messages", []))
        self.calls.append({"kind": kind, "prompt_len": len(prompt),
                           "max_tokens": kwargs.get("max_tokens", 0)})
        return self.chat.completions.create(**kwargs)

    def chat_method(self, **kwargs):
        return self._handle(kwargs)


def _real_key():
    """读取真实 key：优先环境变量 SYSTEM_FLOW_REAL_API_KEY，回退 config.DEEPSEEK_KEY。"""
    k = os.environ.get("SYSTEM_FLOW_REAL_API_KEY", "").strip()
    if k:
        return k
    try:
        from experiments.config import DEEPSEEK_KEY
        k = (DEEPSEEK_KEY or "").strip()
        return k
    except Exception:
        return ""

# ============================================================

def build_llm(planner_mode="fallback", real=False):
    """Build an LLM client: real OpenAI if real=True and a key exists, else MockLLM."""
    key = _real_key()
    if real and key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
            llm = CountingLLM(client)
            llm.mode = "real"
            return llm
        except Exception as e:
            print(f"[WARN] real LLM init failed, fallback to MockLLM: {e}")
    llm = MockLLM(planner_mode=planner_mode)
    llm.mode = "mock"
    return llm

def measure_retriever(retriever, question, k=10, use_ked=True):
    """环节3 OpenDomainRetriever —— 真实 faiss 检索。"""
    t0 = time.time()
    r = retriever.retrieve(question, k=k, use_ked=use_ked)
    dt_ms = (time.time() - t0) * 1000

    docs = list(getattr(r, "chunks", []) or [])
    if not docs:
        docs = list(getattr(r, "documents", []) or [])
    scores = [float(getattr(d, "score", 0.0)) for d in docs]
    inds = Counter(getattr(d, "industry", "?") for d in docs)
    caps = Counter(getattr(d, "capability", "?") for d in docs)
    src = Counter(getattr(d, "source", "?") for d in docs)

    return {
        "n_chunks": len(docs),
        "scores": [round(s, 4) for s in scores],
        "score_mean": round(statistics.mean(scores), 4) if scores else None,
        "score_min": round(min(scores), 4) if scores else None,
        "time_ms": round(dt_ms, 1),
        "industry_dist": dict(inds),
        "capability_dist": dict(caps),
        "source_dist": dict(src),
        "query_expanded": getattr(r, "query_expanded", None),
        "retrieval_failed": getattr(r, "retrieval_failed", None),
    }


def measure_organizer(organizer, evidence, question, task_type="general"):
    """环节4 EvidenceOrganizer —— 去重 + 分组 + 上下文规模。"""
    from agentic.task_types import ExecutionDirective

    directive = ExecutionDirective(
        module="organization", action="group_by_topic",
        params={"organize_by": "topic", "remove_duplicate": True},
    )
    t0 = time.time()
    org = organizer.execute(directive=directive, documents=list(evidence),
                            question=question, task_type=task_type)
    dt_ms = (time.time() - t0) * 1000

    try:
        ctx = org.get_context()
    except Exception as e:
        ctx = f"<get_context error: {e}>"
    groups = getattr(org, "groups", {}) or {}
    n_in = getattr(org, "original_count", len(evidence))
    n_out = getattr(org, "organized_count", len(getattr(org, "documents", [])))

    return {
        "n_in": n_in,
        "n_after_dedup": n_out,
        "n_dropped_dup": max(0, n_in - n_out),
        "dup_rate": round(max(0, n_in - n_out) / n_in, 4) if n_in else None,
        "n_groups": len(groups),
        "group_names": list(groups.keys()),
        "ctx_chars": len(ctx),
        "time_ms": round(dt_ms, 2),
    }



def measure_analyzer(analyzer, question, fmt=""):
    """环节1 TaskAnalyzer —— 验证中性化 + 各字段。"""
    ta = analyzer.analyze(question, format=fmt)
    bcount = sum(bool(b) for b in [
        ta.requires_multi_source, ta.requires_formula, ta.requires_step_reasoning])
    return {
        "task": ta.task.value if ta.task else None,
        "format": ta.format,
        "confidence": ta.confidence,
        "reasoning_complexity": ta.reasoning_complexity,
        "requires_multi_source": ta.requires_multi_source,
        "requires_formula": ta.requires_formula,
        "requires_step_reasoning": ta.requires_step_reasoning,
        "preferred_source": ta.preferred_source,
        "bool_true": bcount,
    }


def measure_planner(planner, task_analysis, mock):
    """环节2 StrategyPlanner —— 统计图拓扑(LLM 或 fallback)。"""
    n_before = len(mock.calls)
    g = planner.plan(task_analysis)
    n_calls = len(mock.calls) - n_before

    ntypes = Counter()
    ids = []
    for nid, node in g.nodes.items():
        ntypes[node.type] += 1
        ids.append(node.id)
    return {
        "planner_llm_calls": n_calls,
        "n_nodes": len(g.nodes),
        "node_types": dict(ntypes),
        "node_ids": ids,
        "entry_points": list(g.entry_points),
        "has_retrieve_2": "retrieve_2" in g.nodes,
        "has_merge": "merge_1" in g.nodes,
        "has_decide": any(n.type == "decide" for n in g.nodes.values()),
        "has_verify": any(n.type == "verify" for n in g.nodes.values()),
        "task": g.task.value if g.task else None,
    }


def measure_executor(executor, graph, question, mock, evidence):
    """环节5 GraphExecutor —— 走完整 DAG(mock LLM), 量化执行路径与调用点。"""
    n_before = len(mock.calls)
    t0 = time.time()
    result = executor.execute(graph=graph, question=question,
                              pre_retrieved_evidence=evidence)
    dt_ms = (time.time() - t0) * 1000

    new_calls = mock.calls[n_before:]
    call_positions = [c.get("kind") for c in new_calls]

    return {
        "exec_node_order": result.get("execution_nodes", []),
        "node_count": result.get("node_count"),
        "second_retrieval_triggers": result.get("second_retrieval_triggers"),
        "used_external_retrieval": result.get("used_external_retrieval"),
        "llm_calls_this_run": len(new_calls),
        "llm_call_positions": call_positions,
        "time_ms": round(dt_ms, 2),
        "answer_is_mock": "【MOCK】" in str(result.get("answer", "")),
        "answer_head": str(result.get("answer", ""))[:40],
    }



# ============================================================
# 汇总 + 主流程
# ============================================================
def load_pool():
    with open(POOL, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", data) if isinstance(data, dict) else data
    items = []
    for r in rows:
        if isinstance(r, dict) and r.get("q"):
            items.append({"q": r["q"], "cap": r.get("cap", "")})
    return items


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def aggregate(rows):
    N = len(rows)
    a = [r["analyzer"] for r in rows]
    p = [r["planner"] for r in rows]
    g = [r["retriever"] for r in rows]
    o = [r["organizer"] for r in rows]
    e = [r["executor"] for r in rows]
    return {
        "n_questions": N,
        "analyzer": {
            "task_general_rate": round(sum(1 for r in a if r["task"] == "general") / N, 4),
            "format_dist": dict(Counter(r["format"] for r in a)),
            "avg_bool_true": _avg([r["bool_true"] for r in a]),
            "confidence_all_equal": len({r["confidence"] for r in a}) == 1,
        },
        "planner": {
            "avg_nodes": _avg([r["n_nodes"] for r in p]),
            "retrieve2_rate": round(sum(1 for r in p if r["has_retrieve_2"]) / N, 4),
            "decide_rate": round(sum(1 for r in p if r["has_decide"]) / N, 4),
            "verify_rate": round(sum(1 for r in p if r["has_verify"]) / N, 4),
            "all_same_topology": len({
                tuple(sorted(r["node_ids"])) for r in p}) == 1,
        },
        "retriever": {
            "avg_n_chunks": _avg([len(r["scores"]) for r in g]),
            "score_mean": _avg([r["score_mean"] for r in g]),
            "score_min_avg": _avg([r["score_min"] for r in g]),
            "avg_time_ms": _avg([r["time_ms"] for r in g]),
            "failures": sum(1 for r in g if r["retrieval_failed"]),
        },
        "organizer": {
            "avg_n_in": _avg([r["n_in"] for r in o]),
            "avg_n_after": _avg([r["n_after_dedup"] for r in o]),
            "avg_dropped": _avg([r["n_dropped_dup"] for r in o]),
            "avg_dup_rate": _avg([r["dup_rate"] for r in o]),
            "avg_groups": _avg([r["n_groups"] for r in o]),
            "avg_ctx_chars": _avg([r["ctx_chars"] for r in o]),
        },
        "executor": {
            "avg_llm_calls": _avg([r["llm_calls_this_run"] for r in e]),
            "call_kinds_union": dict(Counter(
                k for r in e for k in r["llm_call_positions"])),
            "second_retry_triggers_total": sum(
                r["second_retrieval_triggers"] for r in e),
            "avg_time_ms": _avg([r["time_ms"] for r in e]),
            "all_answer_mock": all(r["answer_is_mock"] for r in e),
        },
    }



def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="使用真实 DeepSeek LLM（需 API key），否则用 MockLLM（默认，零成本）")
    ap.add_argument("--max-q", type=int, default=None,
                    help="只跑前 N 题（真实模式建议限制，避免 token 开销）")
    ap.add_argument("--mode", default=None,
                    help="指定单个 planner_mode（fallback / llm_graph），默认两个都跑")
    args = ap.parse_args()

    # 全局开关（兼容环境变量）
    real = args.real or os.environ.get("SYSTEM_FLOW_REAL", "0") == "1"

    from agentic import (TaskAnalyzer, StrategyPlanner, EvidenceOrganizer,
                         TaskSolver, GraphExecutor)
    from retrieval.retriever import OpenDomainRetriever

    items = load_pool()
    if args.max_q:
        items = items[: args.max_q]
    print(f"加载 35 题池 -> {len(items)} 题 (real={real})")
    if not items:
        print("!! 无题目, 退出")
        return

    print("[load] 加载真实 OpenDomainRetriever (faiss) ...")
    t0 = time.time()
    retriever = OpenDomainRetriever(project_root=_LINS)
    retriever.load_from_manifest(
        manifest_path=os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json"))
    print(f"      就绪: {len(getattr(retriever, 'chunks', {}))} chunks, "
          f"{(time.time() - t0):.1f}s")

    modes = [args.mode] if args.mode else ["fallback", "llm_graph"]
    for mode in modes:
        print(f"\n================ planner_mode = {mode} ================")
        mock = build_llm(planner_mode=mode, real=real)
        print(f"  LLM client mode = {getattr(mock, 'mode', 'mock')}")
        analyzer = TaskAnalyzer()
        planner = StrategyPlanner(llm_client=mock)
        organizer = EvidenceOrganizer()
        solver = TaskSolver(llm_client=mock)
        executor = GraphExecutor(retriever_module=retriever,
                                 organizer_module=organizer,
                                 solver_module=solver, llm_client=mock)

        rows = []
        for idx, item in enumerate(items):
            q, cap = item["q"], item.get("cap", "")
            an = measure_analyzer(analyzer, q)
            ta = analyzer.analyze(q, format="")
            pl = measure_planner(planner, ta, mock)
            graph = planner.plan(ta)  # 复用同一图给 executor
            rr = retriever.retrieve(q, k=10, use_ked=True)
            evidence = list(rr.chunks)
            re_m = measure_retriever(retriever, q, k=10, use_ked=True)
            or_m = measure_organizer(organizer, evidence, q)
            ex = measure_executor(executor, graph, q, mock, evidence)

            rows.append({
                "q": q, "cap": cap,
                "analyzer": an, "planner": pl,
                "retriever": re_m, "organizer": or_m, "executor": ex,
            })
            if idx % 10 == 0:
                print(f"  ... {idx}/{len(items)}")

        agg = aggregate(rows)
        with open(f"{OUT}_{mode}.json", "w", encoding="utf-8") as f:
            json.dump({"planner_mode": mode, "rows": rows, "aggregated": agg},
                      f, ensure_ascii=False, indent=2)

        print("\n-- 每题一行 --")
        for i, r in enumerate(rows):
            a, p, g2, o, e = (r["analyzer"], r["planner"], r["retriever"],
                              r["organizer"], r["executor"])
            print(f"[{i:02d}] task={a['task']} fmt={a['format']} | "
                  f"types={','.join(sorted(p['node_types']))} | "
                  f"n_chunk={g2['n_chunks']} avg={g2['score_mean']} | "
                  f"dup={o['n_in']}->{o['n_after_dedup']}({o['n_dropped_dup']}) "
                  f"grp={o['n_groups']} ctx={o['ctx_chars']} | "
                  f"llm={e['llm_call_positions']}")
        print("\n===== aggregated =====")
        print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

