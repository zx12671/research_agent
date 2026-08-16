# -*- coding: utf-8 -*-
"""
_calibrate_multi_gate.py —— 路径2 门控阈值标定（方法驱动，不拍脑袋）。

目标：为 multi_query_retrieve / hybrid 的 RRF 融合设计"子查询质量门控"，
     用数据标定一个合理的门控阈值（或决定默认关/开）。

方法：
  对每个样本，用 KED decompose 出子查询，逐个子查询做 dense top-10，
  记录每个子查询的 top-1 余弦相似度（quality 信号）。
  判定一个子查询是否"有价值"：其 top-10 内是否出现了与 GT 相关的 chunk
  （整段 IoU>=0.15，与 _diag_recall_s4 同口径）。
  对比 [有价值子查询] vs [无价值子查询] 的 top-1 相似度分布 →
  若二者可分（有价值子查询 top-1 明显更高），则门控阈值可从分布间距选出。

输出：results/s4_recall/multi_gate_calib.json  +  控制台分布摘要
用法： python _calibrate_multi_gate.py --per_cap 8 [--seed 7]
"""
import os, sys, json, random, time, argparse, csv
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", type=int, default=999)  # 不限制则每cap=per_cap
    args = ap.parse_args()
    random.seed(args.seed)
    from retrieval.retriever import OpenDomainRetriever
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()

    # 抽样
    by_cap = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap in CAPS:
        picked = by_cap.get(cap, [])
        if picked:
            sample.extend(random.sample(picked, min(args.per_cap, len(picked))))

    rows = []
    useful_top1, useless_top1 = [], []
    n_q = 0
    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        if not q or not kt:
            continue
        try:
            subs = retr.ked.decompose(q)
        except Exception as e:
            print(f"  decompose err {idx}: {e}")
            continue
        if len(subs) <= 1:
            continue
        for si, sq in enumerate(subs):
            n_q += 1
            dr = retr.dense_search(sq, k=10, use_ked=True)
            top1 = dr.chunks[0].score if dr.chunks else 0.0
            useful = any(iou(kt, c.content) >= IOU_TH for c in dr.chunks)
            (useful_top1 if useful else useless_top1).append(top1)
            rows.append({"id": r.get("id"), "cap": r.get("capability"),
                         "q": q[:30], "sub": sq[:25], "sub_idx": si,
                         "top1_score": round(top1, 4), "useful": bool(useful)})
        if idx % 10 == 0:
            print(f"  .. {idx}/{len(sample)} ({time.time()-t0:.0f}s)")

    def agg(xs):
        xs = sorted(xs)
        n = len(xs)
        return {"n": n, "min": round(xs[0], 3) if n else None,
                "p25": round(xs[n // 4], 3) if n else None,
                "median": round(xs[n // 2], 3) if n else None,
                "p75": round(xs[3 * n // 4], 3) if n else None,
                "max": round(xs[-1], 3) if n else None}

    print("\n=== 子查询质量门控标定 ===")
    print(f"  有效子查询(bearing GT 相关块): n={len(useful_top1)} top1-score={agg(useful_top1)}")
    print(f"  无效子查询(无 GT 相关块):     n={len(useless_top1)} top1-score={agg(useless_top1)}")
    # 提出一个候选阈值：有效子查询的中位数与 p25（保守点选 p25）
    if useful_top1:
        srt = sorted(useful_top1)
        cand = srt[len(srt) // 4]  # p25
        # 检查该阈值下误杀率/保留率
        keep = sum(1 for x in useful_top1 if x >= cand) / len(useful_top1)
        print(f"  候选阈值(取有效子查询 top1 的 p25) = {cand:.3f} → 有效子查询保留率 {keep*100:.0f}%")
    out = os.path.join(_LINS, "results", "s4_recall", "multi_gate_calib.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"useful_top1": agg(useful_top1), "useless_top1": agg(useless_top1),
               "n_subqueries": n_q, "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
