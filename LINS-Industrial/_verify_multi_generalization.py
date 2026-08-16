# -*- coding: utf-8 -*-
"""
_verify_multi_generalization.py —— 路径2「原 query 保底」的通用性验证（回答：
改善是"真实通用"还是"局部夸大优化"？）。

方法（真实落地后 retriever 复测，多 seed 交叉验证）：
  对多个 seed / 每 capability 多样本，逐题计算对 single / multi / hybrid 的：
    cov@10（主口径）、整段IoU hit@1 位次
  统计：
    · multi≥single 的题占比（越高越通用；若只有少数题达标而多数题退步→局部夸大）
    · 改善样本的 Δcov 分布（是否集中在个别"极端糟糕"样本拉均值）
    · 按 capability 分组：改善是否某些能力域独享（若集中在1-2个能力→可能局部）
    · 与 seed 无关性：跨 seed 结论一致才算稳健

输出：results/s4_recall/multi_generalize_check.json  +  控制台摘要
用法： python _verify_multi_generalization.py [--seeds 7,11,42] [--per_cap 6]
"""
import os, sys, json, random, argparse, csv
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
from retrieval.recall_metrics import sent_coverage, get_content
from retrieval.retriever import OpenDomainRetriever

CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]
IOU_TH = 0.15


def grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def hit1_rank(chunks, kt):
    for i, c in enumerate(chunks, 1):
        if iou(kt, get_content(c)) >= IOU_TH:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="7,11,42")
    ap.add_argument("--per_cap", type=int, default=6)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()

    by_cap = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        by_cap[r.get("capability") or ""].append(r)

    all_rows = []      # 每 seed 的逐题结果
    seed_summaries = []
    for seed in seeds:
        random.seed(seed)
        sample = []
        for cap in CAPS:
            picked = by_cap.get(cap, [])
            if picked:
                sample.extend(random.sample(picked, min(args.per_cap, len(picked))))
        per = []
        for r in sample:
            q = r.get("question", ""); kt = r.get("knowledge_text", "")
            if not q or not kt:
                continue
            sa = retr.retrieve(q, k=10, use_ked=True).chunks
            sb = retr.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion").chunks
            sc = retr.hybrid_retrieve(q, k=10, use_ked=True).chunks
            row = {
                "id": r.get("id"), "cap": r.get("capability"),
                "cA": sent_coverage(kt, sa)[0], "cB": sent_coverage(kt, sb)[0],
                "cC": sent_coverage(kt, sc)[0],
                "hA": hit1_rank(sa, kt), "hB": hit1_rank(sb, kt), "hC": hit1_rank(sc, kt),
            }
            per.append(row)
        n = len(per)
        avg = lambda key: sum(x[key] for x in per) / n
        def rate(cond):
            return sum(1 for x in per if cond(x)) / n
        g = {
            "seed": seed, "n": n,
            "covA": avg("cA"), "covB": avg("cB"), "covC": avg("cC"),
            "multi_ge_single": rate(lambda x: x["cB"] >= x["cA"] - 1e-9),
            "hybrid_ge_single": rate(lambda x: x["cC"] >= x["cA"] - 1e-9),
            "multi_gain_beyond_single": avg("cB") - avg("cA"),
            "hybrid_gain_beyond_single": avg("cC") - avg("cA"),
        }
        seed_summaries.append(g)
        all_rows.extend({"seed": seed, **x} for x in per)
        print(f"[seed={seed}] n={n} | cov single={avg('cA')*100:.1f}% multi={avg('cB')*100:.1f}% "
              f"hybrid={avg('cC')*100:.1f}% | multi≥single题占比={g['multi_ge_single']*100:.0f}% "
              f"hybrid≥single题占比={g['hybrid_ge_single']*100:.0f}%")

    print("\n=== 通用性判读 ===")
    print("· 若 multi≥single 题占比显著 >50%（如>60%）且多 seed 一致 → 通用；"
          "≈50% 或波动大 → 中性/局部")
    print("· 若增益只集中在少数样本、多数样本退步 → 局部夸大")
    # 全样本分布：Δmulti-single 的直方分布（看集中性）
    dsA = [x["cB"] - x["cA"] for x in all_rows]
    pos = sum(1 for d in dsA if d > 0); neg = sum(1 for d in dsA if d < 0); zero = sum(1 for d in dsA if d == 0)
    d_un0 = [d for d in dsA if d != 0]
    # 改善集中于少数极端样本? 看 top 占比
    srt = sorted([d for d in dsA], reverse=True)
    top10_share = sum(srt[: max(1, len(srt) // 5)]) / sum(srt) if srt and sum(srt) > 0 else 0
    print(f"全样本 Δcov(multi-single): 正={pos} 负={neg} 零={zero} / 总={len(dsA)}")
    print(f"  正负比={pos/(neg+1):.2f} | >0 中位Δ={ (sorted([d for d in d_un0 if d>0])[len([d for d in d_un0 if d>0])//2] if any(x>0 for x in d_un0) else 0):+.3f}")
    out = os.path.join(_LINS, "results", "s4_recall", "multi_generalize_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"seed_summaries": seed_summaries, "n_rows": len(all_rows),
               "pos": pos, "neg": neg, "zero": zero,
               "rows": all_rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
