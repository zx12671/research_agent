# -*- coding: utf-8 -*-
"""_s2_p1_real50.py — S2-P1 数据补全：真实 LLM 下 50 样本统计图分布/回退率/单环节耗时。

目标（对应 stagewise S2 指标）：图类型分布（简单/高级）、每样本 LLM 生成调用次数、
LLM 出图失败/回退模板率、plan 耗时。

数据来源：data/industrybench/huggingface_dataset.csv（2049 题池），按 seed 抽 50 题。
LLM：真实 DeepSeek（CountingLLM 透视计数；key 优先 SYSTEM_FLOW_REAL_API_KEY，回退 config.DEEPSEEK_KEY）。
Retriever：真实 faiss（需 TRANSFORMERS_OFFLINE=1 避免联网校验 embedding）。
Executor：完整 AdaptiveAgenticPipeline.run，注入 pre_retrieved_evidence —— 同时验证 S2-P0 短路。

回退率判定：真实 analyzer 中性化恒 general → fallback 恒为 4 节点
  {retrieve_1, organize_1, reason_1, end}。凡 LLM 出图失败回退模板（或恰好输出该形状）取此 pattern
  → fallback=True。

输出：results/s2_p1_real50.json（rows + aggregated）+ 终端报表。
用法：cd LINS-Industrial && python _s2_p1_real50.py [--max-q 50] [--seed 7]
"""
import os, sys, json, time, csv, random, argparse, statistics
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from _system_flow_quant import _real_key, CountingLLM  # noqa: E402

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
OUT = os.path.join(_LINS, "results", "s2_p1_real50.json")

_FALLBACK_SIMPLE = {"retrieve_1", "organize_1", "reason_1", "end"}


def load_pool(max_q, seed):
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    random.shuffle(rows)
    sel = rows[:max_q] if max_q else rows
    out = []
    for r in sel:
        q = r.get("question") or r.get("Question") or r.get("query")
        if q:
            out.append({"q": q, "cap": r.get("capability", ""),
                        "fmt": r.get("_format", "")})
    return out


def node_pattern(graph):
    return tuple(sorted(n.id for n in graph.nodes.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    key = _real_key()
    if not key:
        print("!! no real key (SYSTEM_FLOW_REAL_API_KEY / experiments.config.DEEPSEEK_KEY)")
        return

    from openai import OpenAI
    from experiments.config import register_paths, KNOWLEDGE_CORPUS_DIR
    register_paths()
    llm = CountingLLM(OpenAI(api_key=key, base_url="https://api.deepseek.com"))
    llm.mode = "real"

    from retrieval.retriever import OpenDomainRetriever
    print("[load] retriever (offline) ...")
    retriever = OpenDomainRetriever(project_root=_LINS)
    retriever.load_from_manifest(os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json"))
    print("[load] chunks:", len(getattr(retriever, "chunks", {})))

    from agentic import TaskAnalyzer, StrategyPlanner, EvidenceOrganizer, TaskSolver
    from agentic.pipeline import AdaptiveAgenticPipeline

    analyzer = TaskAnalyzer()
    planner = StrategyPlanner(llm_client=llm)
    organizer = EvidenceOrganizer()
    solver = TaskSolver(llm_client=llm)
    pipe = AdaptiveAgenticPipeline(analyzer=analyzer, planner=planner,
                                   retriever=retriever, organizer=organizer,
                                   solver=solver, llm_client=llm)

    items = load_pool(args.max_q, args.seed)
    print(f"[pool] {len(items)} 题 (real)")
    rows = []
    adv_cnt = simple_cnt = fb_cnt = 0

    for i, it in enumerate(items):
        q = it["q"]
        try:
            ta = analyzer.analyze(q, format=it["fmt"])
            t0 = time.time()
            graph = planner.plan(ta)
            plan_ms = (time.time() - t0) * 1000
            pat = node_pattern(graph)
            is_fallback = set(pat) == _FALLBACK_SIMPLE
            graph_types = Counter(n.type for n in graph.nodes.values())
            has_adv = any(t in ("decide", "verify") for t in graph_types) or graph_types.get("retrieve", 0) > 1

            rr = retriever.retrieve(q, k=10, use_ked=True)
            chunks = list(rr.chunks)

            n_before = len(llm.calls)
            t1 = time.time()
            result = pipe.run(question=q, pre_retrieved_evidence=chunks, format=it["fmt"])
            run_ms = (time.time() - t1) * 1000
            n_llm = len(llm.calls) - n_before
            pos = [c["kind"] for c in llm.calls[n_before:]]
            exec_nodes = result.get("execution_nodes", [])

            rows.append({
                "q": q[:60], "cap": it["cap"],
                "graph": {"pattern": list(pat), "types": dict(graph_types),
                          "n_nodes": len(graph.nodes), "advanced": has_adv,
                          "fallback": is_fallback, "plan_ms": round(plan_ms, 1)},
                "llm_calls_this_run": n_llm, "llm_call_positions": pos,
                "run_ms": round(run_ms, 1), "exec_nodes": len(exec_nodes),
            })
            adv_cnt += has_adv
            simple_cnt += (not has_adv)
            fb_cnt += is_fallback
            print(f"[{i:02d}] adv={has_adv} fb={is_fallback} n={len(graph.nodes)} "
                  f"plan={plan_ms:.0f}ms run={run_ms:.0f}ms llm={n_llm} {dict(Counter(pos))}")
        except Exception as e:
            print(f"[{i:02d}] ERROR: {e!r}")
            rows.append({"q": q[:60], "cap": it["cap"], "error": str(e)})

    N = len([r for r in rows if "error" not in r])
    def _avg(v):
        v = [x for x in v if x is not None]
        return round(statistics.mean(v), 2) if v else None
    agg = {
        "n": N, "max_q": args.max_q, "seed": args.seed,
        "graph_dist": {"advanced": adv_cnt, "simple": simple_cnt,
                       "advanced_ratio": round(adv_cnt / N, 4) if N else None},
        "fallback_rate": round(fb_cnt / N, 4) if N else None,
        "plan_ms_avg": _avg([r["graph"]["plan_ms"] for r in rows if "error" not in r]),
        "run_ms_avg": _avg([r["run_ms"] for r in rows if "error" not in r]),
        "llm_calls_per_q_avg": _avg([r["llm_calls_this_run"] for r in rows if "error" not in r]),
        "exec_node_count_avg": _avg([r["exec_nodes"] for r in rows if "error" not in r]),
        "llm_kind_hist": dict(Counter(k for r in rows if "error" not in r
                                      for k in r["llm_call_positions"])),
        "error_count": len([r for r in rows if "error" in r]),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "aggregated": agg}, f, ensure_ascii=False, indent=2)
    print("\n===== S2-P1 aggregated (真实 LLM) =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\n[OK] 已写入 {OUT}")


if __name__ == "__main__":
    main()
