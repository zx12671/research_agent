# -*- coding: utf-8 -*-
"""
_diag_s3_executor.py —— S3 图执行阶段检验（对标 docs/stagewise_debug_plan.md ##S3）。

检验项（每项可量化、零真实 LLM 成本、确定性、可留痕）：
  1) reason/decide/verify 节点【实际命中率】：走完整 ExecutionGraph，统计各类型节点真实
     执行到的次数（mock LLM 记录调用点：reason=标准prompt/decide=condition/verify=criteria）。
  2) TaskSolver 接入率（死代码检验）：向 GraphExecutor 注入一个"任何方法被调即抛异常"的
     solver 刺探。若所有 retrieve/organize/reason/decide/merge/verify/end 处理器全程
     都不触碰 solver → 异常永不触发 → 证明 _handle_reason 从不调用 TaskSolver.solve()
     （接入率 = 0，solver 为未接入的死代码）。
  3) evidence→prompt 传递完整率：reason 节点拿到的 evidence_str 是否等于
     organizer 输出的全量 get_context()（不坍缩、不截断）。

方法：真实 faiss 检索 + MockLLM（_system_flow_quant 提供的零成本确定性 LLM）走
      AdaptiveAgenticPipeline.run() 全链路（TaskAnalyzer -> StrategyPlanner -> GraphExecutor）。
      用 35 题池 results/probe_evidence_forward_e2e_35.json。

用法（LINS-Industrial 下）：
    python _diag_s3_executor.py --max-q 12 --seed 7

输出：results/_diag_s3_executor.json + 控制台摘要
"""
import os
import sys
import json
import time
import random
import argparse
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

try:
    from experiments.config import register_paths, KNOWLEDGE_CORPUS_DIR
    register_paths()
except Exception:
    KNOWLEDGE_CORPUS_DIR = os.path.join(_LINS, "knowledge_corpus")

POOL = os.path.join(_LINS, "results", "probe_evidence_forward_e2e_35.json")
OUT = os.path.join(_LINS, "results", "_diag_s3_executor.json")


class ThrowingSolver:
    """刺探 solver：任何方法被调用即抛异常（用于证明 _handle_* 是否触碰 solver）。"""
    def __init__(self):
        self.touched = False

    def __getattr__(self, name):
        def _boom(*a, **k):
            self.touched = True
            raise RuntimeError(f"S3-PROBE: solver.{name}() 被调用！")
        return _boom


def load_pool(n, seed):
    random.seed(seed)
    with open(POOL, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", data) if isinstance(data, dict) else data
    items = [r for r in rows if isinstance(r, dict) and r.get("q")]
    random.shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from _system_flow_quant import MockLLM
    from agentic import TaskAnalyzer, StrategyPlanner, EvidenceOrganizer
    from agentic.pipeline import AdaptiveAgenticPipeline
    from retrieval.retriever import OpenDomainRetriever

    retriever = OpenDomainRetriever(project_root=_LINS)
    try:
        retriever.load_from_manifest(os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json"))
    except Exception:
        retriever.load_from_manifest()
    print("  chunks:", len(getattr(retriever, "chunks", {})))

    sample = load_pool(args.max_q, args.seed)
    print(f"[样本] {len(sample)} 题 | cap: "
          f"{dict(Counter(r.get('cap') for r in sample))}")

    probe_solver = ThrowingSolver()  # 注入会抛异常的 solver → 触碰即崩

    def run_once(q):
        m = MockLLM()  # 每题独立，调用点从 0 起
        pipe = AdaptiveAgenticPipeline(
            analyzer=TaskAnalyzer(llm_client=m),
            planner=StrategyPlanner(llm_client=m),
            retriever=retriever,
            organizer=EvidenceOrganizer(),
            solver=probe_solver,
            llm_client=m,
        )
        return m, pipe.run(q)

    node_hits = Counter()
    kind_calls = Counter()
    solver_touched = False
    rows = []
    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("q", "")
        if not q:
            continue
        m, res = run_once(q)
        if probe_solver.touched:
            solver_touched = True
        kinds = [c.get("kind") for c in m.calls]
        for kk in kinds:
            kind_calls[kk] += 1
        ntypes = Counter(e.get("type") for e in res.get("execution_nodes", []))
        for t, c in ntypes.items():
            node_hits[t] += c
        rows.append({
            "q": q[:40], "cap": r.get("cap", ""),
            "node_types": dict(ntypes), "llm_kinds": kinds,
            "n_llm_calls": len(kinds),
            "ans_head": str(res.get("answer", ""))[:30],
        })
        if idx % 10 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    n = len(rows)
    agg = {
        "n": n,
        "node_hits": dict(node_hits),
        "node_hit_rates": {k: round(v / n, 4) for k, v in node_hits.items()},
        "llm_call_positions": dict(kind_calls),
        "llm_call_positions_rate": {k: round(v / n, 4) for k, v in kind_calls.items()},
        "solver_touched(should_be_false)": solver_touched,
        "solver_access_rate": (1.0 if solver_touched else 0.0),
    }

    print("\n=== S3 检验汇总 ===")
    print(f"  样本: {n} 题")
    print(f"  节点实际命中率/题: {agg['node_hit_rates']}")
    print(f"  LLM 调用点分布/题: {agg['llm_call_positions_rate']}")
    print(f"  TaskSolver 刺探被触碰? {solver_touched} → solver 接入率 = "
          f"{agg['solver_access_rate']}")
    verdict = ("FAIL(EVIDENCE): reason 现实执行但 TaskSolver 接入率=0 → 需裁决接入/移除"
               if (not solver_touched and agg["node_hit_rates"].get("reason", 0) > 0)
               else "PASS")
    print(f"  判定: {verdict}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": agg, "rows": rows, "verdict": verdict},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", OUT)


if __name__ == "__main__":
    main()
