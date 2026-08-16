# -*- coding: utf-8 -*-
"""
_e2e_semantic_mv_prod.py —— 把 P1 语义级多视角改写(SemanticQueryRewriter)接进生产 agentic
路径后的【真实端到端验证】。

背景
----
生产 retrieve 分派（agentic/pipeline.py::_handle_retrieve）新增 P1 生产分支 `semantic_mv`
（默认关）：当 executor 注入 SemanticQueryRewriter 且节点 params["semantic_mv"]=True 时，
用 LLM 依 task/expected_evidence 生成语义互补子视角，各视角走 hybrid_retrieve(dense+BM25)，
再 _fusion_merge 融合（原 query 首保底 + RRF 高权重）。planner 侧新增
StrategyPlanner(retrieve_semantic_mv=...) 使模板图高级任务 retrieve_1 在开启时改走
semantic_mv 分支并关闭机械切逗号 multi_query（避免叠加稀释）。

本脚本验证：在真实 agentic 路径上（TaskAnalyzer->StrategyPlanner->GraphExecutor 的
retrieve 分派），语义多视角是否真的能捞回"机械切逗号 multi_query / 单查询"救不了的
语义鸿沟 hard-miss，并在覆盖(Cov@10 / hit@k)上不劣化。

两臂（同一题库）：
    base    = StrategyPlanner(retrieve_semantic_mv=False) + 不注入 rewriter（=生产现状）
    sem_mv  = StrategyPlanner(retrieve_semantic_mv=True)  + 注入 SemanticQueryRewriter
              （LLM 语义多视角；--mock 下走确定性视角）

分派忠实复用 _handle_retrieve 的 params 驱动逻辑，证据池即进入后续 agent 组织/推理的
真实证据，直接量化"检索路径是否解决 hard-miss"。

LLM 模式：
    --real : analyzer/planner/solver 用真实 DeepSeek，rewriter 走真实 LLM 语义视角
    --mock（默认）: MockLLM + rewriter mock 确定性视角，零成本跑通链路

用法（LINS-Industrial 下）：
    python _e2e_semantic_mv_prod.py --max-q 12 --seed 7             # mock 链路回归
    python _e2e_semantic_mv_prod.py --real --max-q 6 --seed 7       # 真实语义多视角

输出：results/_e2e_semantic_mv_prod_{mode}.json + 控制台逐题摘要
"""
import os
import sys
import csv
import json
import time
import random
import argparse
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)

try:
    from experiments.config import register_paths, DEEPSEEK_KEY, LLM_NAME
    register_paths()
except Exception:
    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    LLM_NAME = "deepseek-chat"

from retrieval.recall_metrics import sent_coverage
from retrieval.retriever import OpenDomainRetriever

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
IOU_TH = 0.20  # 整段 IoU 命中阈值（与 _sim_multi_groundfix 同口径）


def grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def load_samples(n, seed):
    random.seed(seed)
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig", newline="")))
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    s = []
    for cap, lst in by_cap.items():
        if lst:
            s += random.sample(lst, min((n // len(by_cap)) + 1, len(lst)))
    return s[:n]


def _content(c):
    return getattr(c, "content", getattr(c, "text", ""))


_SEMANTIC_MV_VIEWS = {"_rewriter": None, "_retriever": None}


def _semantic_mv_retrieve(q, k, use_ked, mv_max_views, mv_sparse_weight, rewriter):
    """语义级多视角检索：rewriter.rewrite -> 各视角 hybrid -> _fusion_merge。"""
    retr = _SEMANTIC_MV_VIEWS["_retriever"]
    views = rewriter.rewrite(q) or [q]
    views = [v for v in views if v] or [q]
    per_view = []
    for v in views[:mv_max_views]:
        try:
            vr = retr.hybrid_retrieve(
                v, k=k, use_ked=use_ked, use_multi_query=False, use_sparse=True,
                sparse_pool=50, dense_weight=1.0, sparse_weight=mv_sparse_weight,
            )
            per_view.append(vr)
        except Exception:
            continue
    if not per_view:
        return list(retr.retrieve(q, k=k, use_ked=use_ked).chunks)
    merged = retr._fusion_merge(per_view, k)
    return list(merged.chunks)


def _dispatch_by_params(retr, q, params):
    """忠实镜像 agentic/pipeline.py::_handle_retrieve 的 retrieve 分派，返回证据列表。"""
    k = int(params.get("retrieve_k", 10))
    use_ked = bool(params.get("use_ked", True))
    use_multi_query = bool(params.get("multi_query", False))
    use_fusion = bool(params.get("use_fusion", False))
    use_hybrid = bool(params.get("hybrid", False))
    if use_hybrid:
        rr = retr.hybrid_retrieve(
            q, k=k, use_ked=use_ked, use_multi_query=False, use_sparse=True,
            sparse_pool=int(params.get("hybrid_sparse_pool", 50)),
            dense_weight=float(params.get("hybrid_dense_weight", 1.0)),
            sparse_weight=float(params.get("hybrid_sparse_weight", 0.2)),
        )
        return list(rr.chunks)
    if use_multi_query:
        rr = retr.multi_query_retrieve(
            q, k=k, use_ked=use_ked,
            strategy="fusion" if use_fusion else "union",
        )
        return list(rr.chunks)
    rr = retr.retrieve(q, k=k, use_ked=use_ked)
    return list(rr.chunks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-q", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()

    sample = load_samples(args.max_q, args.seed)
    print(f"[样本] {len(sample)} 题 | cap: "
          f"{dict(Counter(r.get('capability') for r in sample))} | real={args.real}")

    from _system_flow_quant import MockLLM
    from agentic import TaskAnalyzer, StrategyPlanner, EvidenceOrganizer, TaskSolver
    from openai import OpenAI

    if args.real and DEEPSEEK_KEY:
        llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        rewriter_llm = llm
    elif args.real:
        print("[WARN] 无 DEEPSEEK_KEY，回退 mock 模式")
        llm = MockLLM()
        rewriter_llm = MockLLM()
    else:
        llm = MockLLM()
        rewriter_llm = MockLLM()

    retriever = OpenDomainRetriever(project_root=_LINS)
    try:
        retriever.load_from_manifest(os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retriever.load_from_manifest()
    _SEMANTIC_MV_VIEWS["_retriever"] = retriever
    print("  chunks:", len(getattr(retriever, "chunks", {})))

    from agentic.query_rewriter import SemanticQueryRewriter
    # 是否 mock：基于 --real 标志（OpenAI 客户端无 .mode）；真实模式必须走 LLM 语义视角。
    is_mock = not args.real
    rewriter = SemanticQueryRewriter(
        llm_client=(None if is_mock else rewriter_llm),
        model_name=LLM_NAME, num_views=3, mock=is_mock,
    )
    _SEMANTIC_MV_VIEWS["_rewriter"] = rewriter
    print(f"[rewriter] mock={is_mock} num_views=3")

    from agentic.pipeline import GraphExecutor
    organizer = EvidenceOrganizer()

    def _build(retrieve_semantic_mv):
        planner = StrategyPlanner(llm_client=llm, retrieve_semantic_mv=retrieve_semantic_mv)
        executor = GraphExecutor(
            retriever_module=retriever, organizer_module=organizer,
            solver_module=TaskSolver(llm_client=llm), llm_client=llm,
            semantic_rewriter=(rewriter if retrieve_semantic_mv else None),
        )
        return {"analyzer": TaskAnalyzer(llm_client=llm), "planner": planner,
                "executor": executor}

    base = _build(False)
    sem = _build(True)

    def run_arm(arm, q):
        ta = arm["analyzer"].analyze(q)
        graph = arm["planner"].plan(ta)
        node = graph.nodes.get("retrieve_1") or graph.nodes.get("retrieve")
        params = dict(node.params) if node else {"retrieve_k": 10, "multi_query": False}
        if params.get("semantic_mv"):
            evidence = _semantic_mv_retrieve(
                q, int(params.get("retrieve_k", 10)),
                bool(params.get("use_ked", True)),
                int(params.get("mv_max_views", 4)),
                float(params.get("mv_sparse_weight", 0.2)),
                rewriter,
            )
        else:
            evidence = _dispatch_by_params(retriever, q, params)
        return {
            "task": ta.task.value if ta else "?",
            "params_for_retrieve_1": params,
            "evidence": evidence,
            "graph_nodes": list(graph.nodes.keys()),
        }



    cov = {"base": [], "sem": []}
    topk = {"base": {1: 0, 3: 0, 5: 0}, "sem": {1: 0, 3: 0, 5: 0}}
    rows, hard_rows = [], []
    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", ""); cap = r.get("capability") or ""
        if not q or not kt:
            continue
        rb = run_arm(base, q)
        rs = run_arm(sem, q)
        evb, evs = rb["evidence"], rs["evidence"]

        cb = sent_coverage(kt, evb)[0]
        cs = sent_coverage(kt, evs)[0]
        cov["base"].append(cb); cov["sem"].append(cs)

        hitb = next((i + 1 for i, c in enumerate(evb) if iou(kt, _content(c)) >= IOU_TH), None)
        hits = next((i + 1 for i, c in enumerate(evs) if iou(kt, _content(c)) >= IOU_TH), None)
        # 相关块 min-rank（宽口径：整段 IoU 最高分块的 rank）——衡量"相关证据是否更靠前"
        def _rel_rank(ev):
            best = (None, 0.0)
            for i, c in enumerate(ev, 1):
                sc = iou(kt, _content(c))
                if sc > best[1]:
                    best = (i, sc)
            return best
        rkb, rks = _rel_rank(evb), _rel_rank(evs)
        for kkk in (1, 3, 5):
            if hitb is not None and hitb <= kkk:
                topk["base"][kkk] += 1
            if hits is not None and hits <= kkk:
                topk["sem"][kkk] += 1

        base_miss = hitb is None
        saved = base_miss and hits is not None
        if base_miss or (cs - cb > 0.02):
            hard_rows.append({
                "id": r.get("id"), "cap": cap,
                "q": q[:60], "cov_base": round(cb, 3), "cov_sem": round(cs, 3),
                "hit_base": hitb, "hit_sem": hits, "saved": saved,
                "params_base": {k: rb["params_for_retrieve_1"].get(k)
                                for k in ("multi_query", "hybrid", "semantic_mv", "retrieve_k")},
            })
        rows.append({
            "id": r.get("id"), "cap": cap, "q": q[:60],
            "cov_base": round(cb, 3), "cov_sem": round(cs, 3),
            "hit_base": hitb, "hit_sem": hits, "saved": saved,
            "rel_rank_base": rkb[0], "rel_iou_base": round(rkb[1], 3),
            "rel_rank_sem": rks[0], "rel_iou_sem": round(rks[1], 3),
            "task_base": rb["task"],
            "params_base": {k: rb["params_for_retrieve_1"].get(k)
                            for k in ("multi_query", "hybrid", "semantic_mv", "retrieve_k")},
            "params_sem": {k: rs["params_for_retrieve_1"].get(k)
                           for k in ("multi_query", "hybrid", "semantic_mv", "retrieve_k")},
        })
        flag = "SAVED" if saved else ("worse" if cs < cb - 0.01 else "even")
        print(f"[{idx:02d}] cov {cb*100:4.1f}->{cs*100:4.1f} | hit b={hitb}/s={hits} "
              f"| {flag} | task={rb['task']} | {cap}")

    n = len(cov["base"])
    avgb = sum(cov["base"]) / n
    avgs = sum(cov["sem"]) / n
    n_saved = sum(1 for x in rows if x["saved"])
    n_better = sum(1 for a, b in zip(cov["base"], cov["sem"]) if b > a + 1e-9)
    n_worse = sum(1 for a, b in zip(cov["base"], cov["sem"]) if b < a - 1e-9)
    base_miss_total = sum(1 for x in rows if x["hit_base"] is None)
    # 相关块 rank 前移统计（宽口径，能反映"语义改写把 GT 相关证据推前"的价值）
    rank_front = sum(1 for x in rows
                     if (x["rel_rank_base"] and x["rel_rank_sem"]
                         and (x["rel_rank_sem"] or 11) < (x["rel_rank_base"] or 11)))
    rank_back = sum(1 for x in rows
                    if (x["rel_rank_base"] and x["rel_rank_sem"] is not None
                        and (x["rel_rank_sem"] or 11) > (x["rel_rank_base"] or 11)))

    summary = {
        "n": n, "mode": "real" if args.real else "mock",
        "avg_cov_base": round(avgb, 4), "avg_cov_sem": round(avgs, 4),
        "delta_cov": round(avgs - avgb, 4),
        "cov_better_count": n_better, "cov_equal_count": n - n_better - n_worse,
        "cov_worse_count": n_worse,
        "base_miss_total": base_miss_total, "base_miss_saved_count": n_saved,
        "rel_rank_front_count": rank_front, "rel_rank_back_count": rank_back,
        "hit@1_3_5": {"base": topk["base"], "sem": topk["sem"]},
    }

    print("\n=== 汇总 ===")
    print(f"  avg_cov@10  base={avgb*100:.2f}%  sem_mv={avgs*100:.2f}%  "
          f"delta={100*(avgs-avgb):+.2f}pp")
    print(f"  coV 升降: better={n_better} even={n - n_better - n_worse} worse={n_worse}")
    print(f"  hit@1/3/5: base={topk['base']} sem={topk['sem']} (总 {n})")
    print(f"  base 在 top-10 未命中(IoU>={IOU_TH}) {base_miss_total} 题，其中 sem_mv 捞回 "
          f"{n_saved} 题")
    print(f"  相关块 rank 前移/后退: 前移={rank_front} 后退={rank_back} (宽口径, 体现语义改写价值)")
    print("\n  hard_miss 样本明细：")
    for h in hard_rows:
        print(f"    [{h['cap']}] cov {h['cov_base']}->{h['cov_sem']} "
              f"hit b={h['hit_base']}/s={h['hit_sem']} saved={h['saved']} | {h['q']}")

    out = os.path.join(_LINS, "results", f"_e2e_semantic_mv_prod_{summary['mode']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
