# -*- coding: utf-8 -*-
"""_diag_run2_v2_ab.py — 方案B 小样本对比：v1 vs v2 回捞块能否进 top-k。

v1 = two_stage_retrieve（回捞块 rel 恒 0，实证进不了 top10）
v2 = two_stage_retrieve_v2（把池块+回捞块合并统一打分 → 回捞块带 rel 参与竞争）

指标（每题）：
  backfill     = 回捞动作发生次数
  v1_bf_in     = v1 最终 top-k 里回捞块的个数（期望 0）
  v2_bf_in     = v2 最终 top-k 里回捞块的个数（方案B 目标 >0）
  v2_bf_rel    = v2 中进榜回捞块的 rel 分数（应非 0，证明打分真的覆盖了）

用法:
  python _diag_run2_v2_ab.py --per 6 --pool 50          # Sim 零成本全跑
  python _diag_run2_v2_ab.py --per 6 --pool 50 --llm 2  # 前 2 题用真实 DeepSeek 验证
"""
import csv, os, sys, time, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_L = os.path.dirname(os.path.abspath(__file__))
if _L not in sys.path:
    sys.path.insert(0, _L)
from experiments.config import DEEPSEEK_KEY, LLM_NAME, register_paths
register_paths()
from retrieval.retriever import OpenDomainRetriever
from _diag_recall_run2 import SimRelevanceRanker, LLMRelevanceRanker

AP = argparse.ArgumentParser()
AP.add_argument("--per", type=int, default=6)
AP.add_argument("--pool", type=int, default=50)
AP.add_argument("--llm", type=int, default=0, help="前 N 题用真实 LLM ranker")
AP.add_argument("--seed", type=int, default=7)
args = AP.parse_args()

r = OpenDomainRetriever(project_root=_L)
r.load_from_manifest()
rows = list(csv.DictReader(open("data/industrybench/huggingface_dataset.csv", encoding="utf-8-sig")))

# 优先取已知回捞多 / 有代表性的题（含 184,1369,32,173,296），补齐到 per 个
PRIORITY = ["184", "1369", "173", "296", "32", "1077", "481", "170"]
sample = []
for pid in PRIORITY:
    m = next((x for x in rows if x["id"] == pid), None)
    if m:
        sample.append(m)
# 补齐
extra = [x for x in rows if not any(x["id"] == s["id"] for s in sample)]
for m in extra:
    if len(sample) >= args.per:
        break
    if m.get("answer") and m.get("question"):
        sample.append(m)
sample = sample[:args.per]

print(f"[样本] {len(sample)} 题; 优先题={[s['id'] for s in sample]}")

def run(v2, q, pool_k, ranker):
    if v2:
        return r.two_stage_retrieve_v2(q, k=10, pool_k=pool_k, n_anchor=3, ranker=ranker)
    return r.two_stage_retrieve(q, k=10, pool_k=pool_k, n_anchor=3, ranker=ranker)

sum_v1_in, sum_v2_in, sum_v2_rel_nz = 0, 0, 0
n_total_bf = 0
for idx, row in enumerate(sample):
    q = row.get("question", "")
    pid = row.get("id", "")
    sim = SimRelevanceRanker(r)
    ranker = (LLMRelevanceRanker(api_key=DEEPSEEK_KEY, model=LLM_NAME) if idx < args.llm else sim)
    tag = "LLM" if idx < args.llm else "sim"

    rr1 = run(False, q, args.pool, ranker)
    rr2 = run(True, q, args.pool, ranker)

    # 从 query_expanded 解析 backfill 与 bf_in_topk
    import re
    def parse(exp):
        return {
            "backfill": int(re.search(r"backfill=(\d+)", exp).group(1)),
            "bf_in": int(re.search(r"bf_in_topk=(\d+)", exp).group(1)) if "bf_in_topk" in exp else None,
        }
    p1, p2 = parse(rr1.query_expanded), parse(rr2.query_expanded)

    # v2 进榜回捞块的 rel 是否非 0（复算一次融合前的 ranked 不可得，这里展示进榜块特征）
    n_total_bf += p1["backfill"]
    sum_v1_in += (p1["bf_in"] if p1["bf_in"] is not None else 0)
    sum_v2_in += p2["bf_in"]

    print(f"\n  id={pid} [{tag}]")
    print(f"    v1: backfill={p1['backfill']} bf_in_top10={p1['bf_in']} "
          f"(blocks={len(rr1.chunks)})")
    print(f"    v2: backfill={p2['backfill']} bf_in_top10={p2['bf_in']} "
          f"(blocks={len(rr2.chunks)})")

    # ---- 附加诊断：复现 v2 回捞决策，打印回捞块的 rel / 门槛 / 最终排位 ----
    poolc = list(r.dense_search(q, k=args.pool, use_ked=True).chunks)
    if ranker is not None and poolc:
        prev = ranker.score(q, poolc)
        pool_ids = {c.chunk_id for c in poolc}
        byrel = sorted(poolc, key=lambda c: prev.get(c.chunk_id, 0.0), reverse=True)
        adoc = {c.document_id for c in byrel[:3] if c.document_id}
        bf = set()
        for cid, cd in r.chunks.items():
            if (cd.get("document_id") or cd.get("doc_id") or "") not in adoc:
                continue
            if cid in pool_ids or not cd.get("content"):
                continue
            from retrieval.retriever import chinese_jt
            if chinese_jt(q, cd["content"]) < 0.02:
                continue
            bf.add(cid)
        cand = list(poolc) + [r._chunk_to_obj(c, 0.0, 1) for c in bf]
        rd = ranker.score(q, cand)
        print(f"    [LLM覆盖] cand总数={len(cand)}, rd键数={len(rd)}, "
              f"回捞块在rd中的键数={len(set(bf) & set(rd))}")
        rx = max((v for v in rd.values() if isinstance(v, (int, float))), default=1.0) or 1.0
        rd = {c: (float(v) / rx) for c, v in rd.items()}
        ss = {}
        for c in cand:
            isb = c.chunk_id in bf
            rrf = (1.0 / (60 + args.pool)) if isb else (1.0 / (60 + c.rank))
            ss[c.chunk_id] = rrf + rd.get(c.chunk_id, 0.0) + (
                0.5 if (isb or c.document_id in adoc) else 0.0)
        order = sorted(ss.items(), key=lambda x: x[1], reverse=True)[:10]
        gate = min(ss[c] for c, _ in order)
        if bf:
            nz = sum(1 for c in bf if rd.get(c, 0.0) > 0)
            print(f"    [v2诊断] 回捞块数={len(bf)}, rel非0数={nz}, "
                  f"rel均值={sum((rd.get(c,0.0) for c in bf), 0.0)/len(bf):.3f}")
            for c in sorted(bf):
                pos = next((i + 1 for i, (cc, _) in enumerate(order) if cc == c), ">10")
                print(f"       - {c[:16]} rel={rd.get(c,0.0):.3f} s={ss[c]:.4f} "
                      f"进榜门槛={gate:.4f} 排位={pos}")


print(f"  v1 回捞块累计进 top10 = {sum_v1_in}")
print(f"  v2 回捞块累计进 top10 = {sum_v2_in}")
