# -*- coding: utf-8 -*-
"""
_analyze_gate_predictors_35.py —— 逆推式协议 第2步（离线，零 LLM）：
  基于 ab_ef_pairs_35.json 的真值标签（delta=回捞实际升/降分），
  对多个 零-LLM 检索层信号做**阈值扫描**，评估每个信号对"该回捞(升分)"的判别力，
  选出最优门控配方（signal + threshold），并给出预估的成本/收益权衡。

输入：results/ab_ef_pairs_35.json（由 _e2e_agentic_ef_35_pairs.py 生成）
输出：控制台表格 + results/gate_predictor_analysis_35.json + 简短 markdown 结论段

运行：
    cd LINS-Industrial
    python _analyze_gate_predictors_35.py

判别口径（跟用户决策一致：省成本且科学）：
  - 真值标签 label：delta = score_ef - score_base
        "可回捞"(worth) = delta > 0   （回捞提升了分数）
        "不必回捞"(harm/neutral) = delta <= 0
  - 信号：全部在 base-top10 检索后、回捞前可得（在线零额外 LLM），另有 ef 侧实际回捞量作 oracle：
        mu_top, headroom, mu_tail, max_s, n_docs_top10, (n_extra_ef 仅 oracle)
  - 方向：值得回捞的题，其 base 检索应偏"弱/散" → 信号通常**低于**阈值（低 mu_top / 低 headroom）。
        但扫描同时评估"大于"与"小于"两种情况，取更优。
"""
import os, sys, json

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

IN = os.path.join(_LINS, "results", "ab_ef_pairs_35.json")
OUT = os.path.join(_LINS, "results", "gate_predictor_analysis_35.json")


def load_rows():
    with open(IN, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    sigs = ["mu_top", "headroom", "mu_tail", "max_s", "n_docs_top10"]
    for r in rows:
        s = r["sig_base"]
        for k in sigs:
            r.setdefault("_v_" + k, s.get(k))
        r.setdefault("_v_n_extra_ef", r.get("n_extra_ef"))
    return rows


def evaluate(gate_fn, rows):
    """gate_fn(r)->bool 返回"是否回捞"。以"回捞恰发生在升分题上"为目标打 2x2。"""
    tp = fp = tn = fn = 0
    for r in rows:
        worth = r["delta"] > 0
        trig = gate_fn(r)
        if worth and trig:
            tp += 1
        elif not worth and trig:
            fp += 1
        elif worth and not trig:
            fn += 1
        else:
            tn += 1
    tot = max(1, tp + fp + tn + fn)
    acc = (tp + tn) / tot
    prec = (tp / (tp + fp)) if (tp + fp) else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # 触发率=回捞比例（成本侧；越低越好，但不能漏掉升分）
    trig_rate = (tp + fp) / tot
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": round(acc, 3),
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "trig_rate": round(trig_rate, 3)}


def scan(signal, rows, n_steps=40):
    vals = sorted({r["_v_" + signal] for r in rows if r.get("_v_" + signal) is not None})
    if not vals:
        return None
    lo, hi = vals[0], vals[-1]
    best, best_dir, best_th = None, None, None
    for i in range(n_steps + 1):
        th = lo + (hi - lo) * i / n_steps
        for direction in ("lt", "gt"):
            gate = (lambda r, t=th, d=direction: r["_v_" + signal] < t) if direction == "lt" \
                else (lambda r, t=th, d=direction: r["_v_" + signal] > t)
            ev = evaluate(gate, rows)
            if best is None or ev["f1"] > best["f1"]:
                best, best_dir, best_th = ev, direction, round(th, 4)
    return {"signal": signal, "best_th": best_th, "direction": best_dir, **best}


def main():
    rows = load_rows()
    if not rows:
        print("无数据，请先运行 Step1 _e2e_agentic_ef_35_pairs.py 生成 ab_ef_pairs_35.json")
        return
    n = len(rows)
    up = sum(1 for r in rows if r["delta"] > 0)
    down = sum(1 for r in rows if r["delta"] < 0)
    same = sum(1 for r in rows if r["delta"] == 0)
    base_worth_rate = up / n  # 随机回捞的基准命中率

    print("=" * 74)
    print(f"Gate 判别力分析 | N={n} | 升分(worth)={up} 降分={down} 持平={same} | 基准命中率(w/o信号)={base_worth_rate:.2f}")
    print("=" * 74)

    sigs = ["mu_top", "headroom", "mu_tail", "max_s", "n_docs_top10", "n_extra_ef(o)"]  # (o)=oracle
    col_keys = {"mu_top": "mu_top", "headroom": "headroom", "mu_tail": "mu_tail",
                "max_s": "max_s", "n_docs_top10": "n_docs_top10", "n_extra_ef(o)": "n_extra_ef"}
    results = {}
    print(f"{'信号':<16}{'方向':<5}{'阈值':<8}{'f1':<6}{'acc':<6}{'prec':<6}{'rec':<6}{'触发率':<7}{'tp/fp/fn':<14}")
    for sig in sigs:
        k = col_keys[sig]
        if sig == "n_extra_ef(o)":
            # oracle：直接用 ef 实际回捞量 n>0 作为判别
            gate = lambda r: r["_v_n_extra_ef"] > 0
            ev = evaluate(gate, rows)
            res = {"signal": "n_extra_ef(o)", "threshold_note": "ef回捞量>0", **ev}
        else:
            res = scan(k, rows)
            if res is None:
                continue
            tag = res["direction"]
        print(f"{res['signal']:<16}{tag if 'threshold_note' not in res else '>0':<5}"
              f"{str(res.get('best_th', res.get('threshold_note',''))):<8}"
              f"{res['f1']:<6}{res['acc']:<6}{res['precision']:<6}{res['recall']:<6}"
              f"{res['trig_rate']:<7}[{res['tp']}/{res['fp']}/{res['fn']}]")
        results[res["signal"]] = res

    # 选择最优在线信号（排除 oracle）：优先 f1 高，其次触发率低（省成本）
    online = [v for k, v in results.items() if not k.endswith("(o)")]
    winner = max(online, key=lambda v: v["f1"]) if online else None
    print("\n" + "-" * 74)
    if winner:
        print(f"推荐门控: {winner['signal']} {winner['direction']} {winner['best_th']} | "
              f"f1={winner['f1']} acc={winner['acc']} 触发率={winner['trig_rate']} "
              f"(基准命中率 {base_worth_rate:.2f})")

    out = {
        "N": n, "up_down_same": [up, down, same], "base_worth_rate": round(base_worth_rate, 3),
        "per_signal": {k: v for k, v in results.items()},
        "winner_online": winner,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写:", OUT)
    print("\n[markdown 结论段]")
    if winner:
        print(f"> Gate 判定：{winner['signal']} {'<' if winner['direction']=='lt' else '>'} {winner['best_th']} → 回捞。")
    print(f"> 在 N={n} 上，回捞实际升分题占 {up} 条（命中率基准 {base_worth_rate:.2f}）；"
          f"最优门控 f1={winner['f1'] if winner else '?'}，触发率 {winner['trig_rate'] if winner else '?'}。")
    print("> 下一动作：据此配方实现 CtrEF 条件式回捞并回测（或先 6 题小样本复核）。")


if __name__ == "__main__":
    main()
