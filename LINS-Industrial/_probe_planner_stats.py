# -*- coding: utf-8 -*-
"""
_probe_planner_stats.py — StrategyPlanner 专项量化探针（v2：修正结构解析）

目标：统计 n 题中：
  (1) LLM 建图 vs fallback 模板占比
  (2) 平均节点数（规划图 num_nodes vs 执行日志 n）
  (3) verify / decide / 多检索占比（"adaptive 声称"）
  (4) 二次检索 retrieve 节点数（planned）与真实执行（execution log）
并将这些图结构指标与 Accuracy / SV 交叉，判断高级分支是否真提升质量。

v2 修正（诚实标注）：
  - execution_graph 真实结构已由 _diag_planner_graph_struct.py 实测确认：
      { "task", "num_nodes", "entry_points", "node_types": [type...],
        "node_summary": {node_id: type, ...}, "has_conditional_branching",
        "has_multi_hop_retrieval", "has_verification", "task_specific_hints" }
      => 用 node_summary/node_types/num_nodes/has_* 解析，不再虚构 nodes{}。
  - result 对象【没有】second_retrieval_triggers 字段（返回 None），
    故二次检索改由 (a) 规划图 retrieve 节点数 / (b) 执行日志实际 retrieve 数衡量。
  - Stable Knowledge Interface 会短路 GraphExecutor 内的 retrieve_1，
    但实测 retrieve_2 仍会真实执行（运行日志可见 secondary retrieve），
    故不再断言"二次检索必然为 0"，以实测合计为准。

运行:  cd LINS-Industrial
       python _probe_planner_stats.py --n 20 --scorer rule
输出:  results/planner_stats/planner_stats_report.md, per_sample.json, stats.json
"""
import sys
import os
import json
import time
import logging
import argparse
from collections import Counter, defaultdict

for _lg in ("httpx", "httpcore", "openai", "openai.http_client",
            "AgenticRAGEngine", "IndustrialRetriever", "GraphExecutor"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (_PROJECT_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _as_number(v):
    """把数值或数字字符串归一为 float / None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(str(v).strip())
        except (TypeError, ValueError):
            return None
    return None


def parse_graph(graph: dict) -> dict:
    """解析 execution_graph（实际结构见模块 docstring）。"""
    if not isinstance(graph, dict):
        graph = {}
    node_summary = graph.get("node_summary") or {}
    node_types = graph.get("node_types") or []
    num_nodes = graph.get("num_nodes")

    # 统一为 {node_id: type}
    id2type = {}
    if isinstance(node_summary, dict):
        for nid, t in node_summary.items():
            id2type[str(nid)] = str(t) if t else ""
    elif isinstance(node_summary, list):  # 兜底
        for it in node_summary:
            if isinstance(it, dict):
                id2type[str(it.get("id"))] = str(it.get("type") or "")
    if not id2type and isinstance(node_types, list):
        for i, t in enumerate(node_types):
            id2type[f"node_{i}"] = str(t)

    type_counter = Counter(t for t in id2type.values() if t)
    n_retrieve = type_counter.get("retrieve", 0)
    n_organize = type_counter.get("organize", 0)
    n_reason = type_counter.get("reason", 0)
    n_decide = type_counter.get("decide", 0)
    n_verify = type_counter.get("verify", 0)
    n_end = type_counter.get("end", 0)

    # 显式 meta 布尔
    mcb = _safe_get(graph, "has_conditional_branching")
    mhr = _safe_get(graph, "has_multi_hop_retrieval")
    vrf = _safe_get(graph, "has_verification")

    has_advanced = bool(
        (n_decide > 0) or (n_verify > 0) or (n_retrieve > 1)
        or mcb or mhr or vrf
    )
    is_fallback_shape = bool(
        n_retrieve == 1 and n_organize == 1 and n_reason == 1
        and n_end == 1 and n_decide == 0 and n_verify == 0
        and len(id2type) == 4 and not has_advanced
    )
    return {
        "node_count_planned": num_nodes if isinstance(num_nodes, int) else len(id2type),
        "n_ids": len(id2type),
        "type_counter": dict(type_counter),
        "id2type": id2type,
        "node_types_raw": node_types if isinstance(node_types, list) else [],
        "n_retrieve": n_retrieve, "n_organize": n_organize,
        "n_reason": n_reason, "n_decide": n_decide,
        "n_verify": n_verify, "n_end": n_end,
        "has_advanced": has_advanced,
        "has_conditional_branching": mcb,
        "has_multi_hop_retrieval": mhr,
        "has_verification": vrf,
        "task": _safe_get(graph, "task"),
        "entry_points": graph.get("entry_points") or [],
        "is_fallback_shape": is_fallback_shape,
        "graph_keys": list(graph.keys()),
    }


def parse_execution(exec_nodes) -> dict:
    """解析执行日志。字段名未知，用多候选键容错取类型计数。"""
    if exec_nodes is None:
        return {"n_log": 0, "reached_types": {}, "n_retrieve_exec": 0, "types": []}
    reached = Counter()
    types = []
    for n in exec_nodes:
        if isinstance(n, dict):
            t = (n.get("type") or n.get("node_type") or n.get("kind")
                 or n.get("name") or "")
            node_id = n.get("node_id") or n.get("id") or ""
            types.append({"type": t, "id": node_id})
            reached[t] += 1
    return {
        "n_log": len(exec_nodes),
        "reached_types": dict(reached),
        "n_retrieve_exec": sum(v for k, v in reached.items()
                               if "retrieve" in str(k)),
        "types": types,
    }


def main():
    from experiments.config import DEEPSEEK_KEY, LLM_NAME
    from experiments.exp1_agentic_rag import (
        AgenticRAGEngine, EnhancedScorer, load_question_dataset,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--scorer", default="rule", choices=["rule", "llm"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "planner_stats"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csv_path = None
    for cand in (os.path.join(_PROJECT_ROOT, "data", "industrybench", "huggingface_dataset.csv"),
                 os.path.join(_THIS_DIR, "data", "industrybench", "huggingface_dataset.csv")):
        if os.path.exists(cand):
            csv_path = cand
            break
    samples = load_question_dataset(csv_path, num_samples=args.start + args.n)
    samples = samples[args.start: args.start + args.n]
    if not samples:
        print("[ERROR] no samples loaded")
        return
    print(f"[INFO] 评估 {len(samples)} 题 (start={args.start})  scorer={args.scorer}")

    print("[INFO] 初始化 AgenticRAGEngine ...")
    engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
    scorer = EnhancedScorer(mode=args.scorer, api_key=DEEPSEEK_KEY)

    per = []
    agg = defaultdict(list)
    agg["fallback_adv"] = {"fallback_shape": [], "advanced": []}

    t0 = time.time()
    for i, s in enumerate(samples):
        print(f"\n>>> [{i+1}/{len(samples)}] {getattr(s,'id','?')} starting answer... "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
        try:
            answer, _ = engine.answer(s.question)
        except Exception as e:
            print(f"[ERR] {getattr(s,'id','?')} answer failed: {e}", flush=True)
            per.append({"sample_id": getattr(s, "id", "?"), "error": str(e)})
            continue
        pr = engine.last_pipeline_result
        prd = vars(pr) if hasattr(pr, "__dict__") else {}

        graph = prd.get("execution_graph", {}) or {}
        enodes = prd.get("execution_nodes") or []
        task_analysis = prd.get("task_analysis", {})
        task_val = task_analysis.get("task", "") if isinstance(task_analysis, dict) else ""
        task_val2 = prd.get("task", task_val)

        pg = parse_graph(graph)
        ex = parse_execution(enodes)

        sw = {}
        try:
            kt = getattr(s, "knowledge_text", "") or ""
            sw = scorer.score(s.question, answer, s.ref_answer,
                              knowledge_text=kt or None)
        except Exception as e:
            sw = {"adjusted_score": None, "has_violation": None,
                  "score_explanation": f"scorer err {e}"}

        adv = bool(pg["has_advanced"])
        bucket = "advanced" if adv else "fallback_shape"
        second_retrieve_planned = pg["n_retrieve"]  # 规划图中 retrieve 节点数
        adj_num = _as_number(sw.get("adjusted_score"))
        # 执行层二次检索：规划 retrieve>1 && 执行日志中出现 retrieve（含被短路但记录的）
        rec = {
            "sample_id": getattr(s, "id", "?"), "question": s.question[:40],
            "difficulty": getattr(s, "difficulty", ""),
            "capability": getattr(s, "capability", ""),
            "task_analysis.task": task_val or task_val2,
            "graph": pg, "execution": ex,
            "answer_len": len(answer),
            "scores": {
                "raw": sw.get("raw_score"),
                "adjusted": sw.get("adjusted_score"),
                "adjusted_num": adj_num,
                "has_violation": sw.get("has_violation"),
            },
            "planned_advanced_flag": adv,
            "planned_fallback_shape": pg["is_fallback_shape"],
            "second_retrieve_planned": second_retrieve_planned,
            "node_count_planned": pg["node_count_planned"],
        }
        per.append(rec)

        agg["bucket"].append(bucket)
        agg["n_nodes_planned"].append(pg["node_count_planned"])
        agg["n_nodes_exec"].append(ex["n_log"])
        agg["has_advanced"].append(1 if adv else 0)
        agg["second_ret_planned"].append(1 if second_retrieve_planned > 1 else 0)
        agg["n_verify"].append(pg["n_verify"])
        agg["adj_score"].append(adj_num)
        agg["violation"].append(1 if rec["scores"]["has_violation"] else 0)
        agg["fallback_adv"][bucket].append(adj_num)
        print(f"  [{i+1}/{len(samples)}] {getattr(s,'id','?')} nodes={pg['node_count_planned']}"
              f" retrieve={pg['n_retrieve']} verify={pg['n_verify']}"
              f" adv={int(adv)} score={rec['scores']['adjusted']}")

    # ---- 统计 ----
    n = len(per)
    out = {"samples": n}
    if n == 0:
        out["error"] = "no successful samples"
        per_sample_json = per
    else:
        from statistics import mean

        def avg(lst):
            vals = [v for v in lst if v is not None]
            return round(mean(vals), 4) if vals else None

        adv_cnt = sum(agg["has_advanced"])
        fb_cnt = n - adv_cnt
        sec_planned = sum(agg["second_ret_planned"])
        n_verify_tot = sum(agg["n_verify"])

        out.update({
            "llm_advanced_planned_arg": {
                "advanced_count": adv_cnt,
                "advanced_ratio": round(adv_cnt / n, 4),
                "fallback_shape_count": fb_cnt,
                "fallback_shape_ratio": round(fb_cnt / n, 4),
                "graph_meta_ratio": {
                    "branching_true": round(
                        sum(1 for r in per if r["graph"].get("has_conditional_branching")) / n, 4),
                    "multi_hop_true": round(
                        sum(1 for r in per if r["graph"].get("has_multi_hop_retrieval")) / n, 4),
                    "verify_true": round(
                        sum(1 for r in per if r["graph"].get("has_verification")) / n, 4),
                },
            },
            "avg_n_nodes": {
                "planned": round(mean(agg["n_nodes_planned"]), 3),
                "executed_log": round(mean(agg["n_nodes_exec"]), 3),
            },
            "second_retrieval_planned": {
                "retrieve_nodes_gt1_count": sec_planned,
                "ratio": round(sec_planned / n, 4),
                "avg_retrieve_nodes": round(mean(
                    [r["graph"]["n_retrieve"] for r in per]), 4),
                "note": (
                    "result 无 second_retrieval_triggers 字段；此处按规划图 retrieve 节点数>1 "
                    "统计 planned 二次检索；执行层是否真正触发以 execution log / 运行日志为准"
                    "（实测 retrieve_2 会真实二次检索，未像 retrieve_1 那样被 pre_evidence 短路）。"
                ),
            },
            "verify_nodes": {"count": n_verify_tot, "ratio": round(n_verify_tot / n, 4)},
            "score_cross_by_bucket": {
                "advanced": {
                    "count": len(agg["fallback_adv"]["advanced"]),
                    "avg_adj": avg(agg["fallback_adv"]["advanced"]),
                },
                "fallback_shape": {
                    "count": len(agg["fallback_adv"]["fallback_shape"]),
                    "avg_adj": avg(agg["fallback_adv"]["fallback_shape"]),
                },
            },
            "overall": {
                "avg_adjusted": avg(agg["adj_score"]),
                "violation_count": sum(agg["violation"]),
            },
            "capability_bucket_counts": dict(Counter(
                r["capability"] for r in per if "error" not in r)),
        })

    with open(os.path.join(args.outdir, "per_sample.json"), "w",
              encoding="utf-8") as f:
        json.dump(per, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(args.outdir, "stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    meta = out.get("llm_advanced_planned_arg", {})
    md = [
        "# StrategyPlanner 专项量化探针报告 (v2)\n",
        f"- 样本数: {n} | scorer: {args.scorer}\n",
        "## 1. LLM高级图 vs fallback 形状",
        f"- advanced(含 verify/decide/多检索或 meta 声明): {adv_cnt} ({meta.get('advanced_ratio')})",
        f"- fallback 形状(4节点线性无高级): {fb_cnt} ({meta.get('fallback_shape_ratio')})",
        f"- 规划图 meta: branching_true={meta.get('graph_meta_ratio',{}).get('branching_true')}, "
        f"multi_hop_true={meta.get('graph_meta_ratio',{}).get('multi_hop_true')}, "
        f"verify_true={meta.get('graph_meta_ratio',{}).get('verify_true')}",
        "## 2. 节点数",
        f"- 规划图平均节点数: {out.get('avg_n_nodes',{}).get('planned')}",
        f"- 执行日志平均节点数: {out.get('avg_n_nodes',{}).get('executed_log')}",
        "## 3. 二次检索 (planned)",
        f"- 规划图 retrieve 节点>1 占比: {out.get('second_retrieval_planned',{}).get('ratio')}",
        f"- 平均 retrieve 节点数: {out.get('second_retrieval_planned',{}).get('avg_retrieve_nodes')}",
        f"- 说明: {out.get('second_retrieval_planned',{}).get('note','')}",
        "## 4. verify 节点",
        f"- 平均 verify 节点数: {out.get('verify_nodes',{}).get('ratio')}",
        "## 5. 按规划图分组的 Accuracy/SV 交叉",
        f"- advanced 组: count={len(agg['fallback_adv']['advanced'])}, "
        f"avg_adj={out.get('score_cross_by_bucket',{}).get('advanced',{}).get('avg_adj')}",
        f"- fallback 组: count={len(agg['fallback_adv']['fallback_shape'])}, "
        f"avg_adj={out.get('score_cross_by_bucket',{}).get('fallback_shape',{}).get('avg_adj')}",
        "## 6. 总览",
        f"- overall avg_adjusted: {out.get('overall',{}).get('avg_adjusted')}",
        f"- SV 违规数: {out.get('overall',{}).get('violation_count')}",
        "\n> 注：advanced/fallback 按 execution_graph 真实结构（node_summary/node_types/has_*）判定，"
        "见模块 docstring。",
    ]
    with open(os.path.join(args.outdir, "planner_stats_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n[OK] 报表: {os.path.join(args.outdir, 'planner_stats_report.md')}")
    print(f"[OK] 明细: {os.path.join(args.outdir, 'per_sample.json')}")


if __name__ == "__main__":
    main()
