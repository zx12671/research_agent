# -*- coding: utf-8 -*-
"""_perf_fullwalk.py — Agentic 全链路题逐环节量化【任务分析→最终输出】.
默认 MockLLM(零成本)，--real 走 DeepSeek。产物 results/_perf_fullwalk.json。
用法: python _perf_fullwalk.py --max-q 5 ; --cap 能力 ; --real
"""
import os, sys, json, time, random, csv, statistics
from collections import Counter

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
try:
    from experiments.config import register_paths, KNOWLEDGE_CORPUS_DIR
    register_paths()
except Exception:
    KNOWLEDGE_CORPUS_DIR = os.path.join(_LINS, "knowledge_corpus")

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
OUT = os.path.join(_LINS, "results", "_perf_fullwalk.json")


class MockLLM:
    def __init__(self):
        self.calls = []
        self.chat = type("_Chat", (), {"completions": self})()
        self.completions = self

    def create(self, **kwargs):
        prompt = ""
        for m in kwargs.get("messages", []) or []:
            prompt += (m.get("content") or "") if isinstance(m, dict) else ""
        kind = self._detect(prompt)
        self.calls.append({"kind": kind, "prompt_len": len(prompt),
                           "max_tokens": kwargs.get("max_tokens", 0)})
        content = self._content(kind, prompt)
        msg = type("_M", (), {"content": content})()
        ch = type("_C", (), {"message": msg})()
        return type("_R", (), {"choices": [ch], "content": content})()

    @staticmethod
    def _detect(p):
        p = p.lower()
        if "node types available" in p:
            return "planner"
        if "decision" in p and "sufficient" in p:
            return "decide"
        if "pass" in p and "fail" in p and "verify" in p:
            return "verify"
        return "reason"

    @staticmethod
    def _content(kind, prompt):
        if kind == "planner":
            return "not json"
        if kind == "decide":
            return "sufficient"
        if kind == "verify":
            return "pass"
        return "[MOCK-ANSWER] " + " ".join((prompt or "").split())[:60]


def load_questions(max_q=None, cap=None, seed=7):
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    random.seed(seed)
    if cap:
        rows = [r for r in rows if r.get("capability") == cap]
    random.shuffle(rows)
    return rows[:max_q] if max_q else rows


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q", type=int, default=5)
    ap.add_argument("--cap", type=str, default=None)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from retrieval.retriever import OpenDomainRetriever
    print("[load] OpenDomainRetriever ...")
    retriever = OpenDomainRetriever(project_root=_LINS)
    retriever.load_from_manifest(os.path.join(KNOWLEDGE_CORPUS_DIR, "manifest.json"))
    print("  chunks:", len(getattr(retriever, "chunks", {})))

    from agentic import (TaskAnalyzer, StrategyPlanner, EvidenceOrganizer, TaskSolver)
    from agentic.pipeline import AdaptiveAgenticPipeline
    from metrics.industrybench_scorer import RuleBasedScorer

    llm = MockLLM()
    analyzer = TaskAnalyzer(llm_client=llm)
    planner = StrategyPlanner(llm_client=llm)
    organizer = EvidenceOrganizer()
    solver = TaskSolver(llm_client=llm)
    pipeline = AdaptiveAgenticPipeline(analyzer=analyzer, planner=planner,
                                       retriever=retriever, organizer=organizer,
                                       solver=solver, llm_client=llm)

    items = load_questions(args.max_q, args.cap, args.seed)
    rows, all_calls = [], []
    for it in items:
        q = it["question"]
        ref = it.get("answer", "")

        # 1 TaskAnalyzer
        t0 = time.time()
        ta = analyzer.analyze(q)
        an = {"ms": (time.time()-t0)*1000, **ta.to_dict()}

        # 2 StrategyPlanner
        t0 = time.time()
        g = planner.plan(ta)
        pl = {"ms": (time.time()-t0)*1000, "n_nodes": len(g.nodes),
              "node_types": dict(Counter(n.type for n in g.nodes.values()))}

        # 3 Retriever
        t0 = time.time()
        rr = retriever.retrieve(q, k=10, use_ked=True)
        chunks = list(rr.chunks)
        scores = [c.score for c in chunks if getattr(c, "score", None) is not None]
        rv = {"ms": (time.time()-t0)*1000, "n_chunks": len(chunks),
              "score_mean": (_avg(scores) if scores else None)}

        # 6 端到端 run（注入外部证据）
        t0 = time.time()
        result = pipeline.run(question=q, pre_retrieved_evidence=chunks)
        run_ms = (time.time()-t0)*1000
        answer = result.get("answer", "")

        # 7 评分
        score = cov = None
        if ref:
            score = RuleBasedScorer.rule_based_score(q, ref, answer)
            cov = RuleBasedScorer.compute_coverage(ref, answer)

        all_calls += llm.calls
        rows.append({"q": q, "cap": it.get("capability"), "fmt": it.get("_format"),
                     "analyzer": {"task": an.get("task"), "format": an.get("format"),
                                  "ms": an["ms"]},
                     "planner": {"n_nodes": pl["n_nodes"], "node_types": pl["node_types"],
                                 "ms": pl["ms"]},
                     "retriever": rv,
                     "run_ms": run_ms, "answer_chars": len(answer),
                     "score": score, "coverage": cov,
                     "result_keys": list(result.keys())})

    agg = {"n": len(rows),
           "task_dist": dict(Counter(r["analyzer"]["task"] for r in rows)),
           "format_dist": dict(Counter(r["analyzer"]["format"] for r in rows)),
           "planner_nodes": _avg([r["planner"]["n_nodes"] for r in rows]),
           "planner_node_types": dict(Counter(t for r in rows
                                              for t in r["planner"]["node_types"])),
           "planner_ms": _avg([r["planner"]["ms"] for r in rows]),
           "retriever_n": _avg([r["retriever"]["n_chunks"] for r in rows]),
           "retriever_score": _avg([r["retriever"]["score_mean"] for r in rows]),
           "retriever_ms": _avg([r["retriever"]["ms"] for r in rows]),
           "run_ms_avg": _avg([r["run_ms"] for r in rows]),
           "answer_chars_avg": _avg([r["answer_chars"] for r in rows]),
           "score_avg": _avg([r["score"] for r in rows]),
           "score_dist": dict(Counter(r["score"] for r in rows if r["score"] is not None)),
           "coverage_avg": _avg([r["coverage"] for r in rows]),
           "llm_call_kinds": dict(Counter(c["kind"] for c in all_calls)),
           "llm_calls_per_q": (round(len(all_calls)/len(rows), 1) if rows else 0)}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "aggregated": agg}, f, ensure_ascii=False, indent=2)
    print("[已写]", OUT)

    print("\n===== 每题一行 =====")
    for i, r in enumerate(rows):
        print(f"[{i:02d}] task={r['analyzer']['task']} fmt={r['analyzer']['format']} | "
              f"图{r['planner']['n_nodes']}节点/{','.join(r['planner']['node_types'])} | "
              f"retr={r['retriever']['n_chunks']}块 score={r['retriever']['score_mean']} | "
              f"run={r['run_ms']:.0f}ms 答案={r['answer_chars']}字 "
              f"score={r['score']} cov={r['coverage']}")

    print("\n===== aggregated =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    if rows:
        print("\n===== PipelineResult 字段(第1样本) =====")
        for k in rows[0]["result_keys"]:
            print("  ", k)


if __name__ == "__main__":
    main()
