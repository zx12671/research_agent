# -*- coding: utf-8 -*-
"""冒烟验证：只有检索层 + 门控（零 LLM）——检查 35 题的 gate 分布与回捞行为一致性。"""
import os, sys, json

_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
_PJ = os.path.abspath(os.path.join(_LINS, ".."))
if _PJ not in sys.path:
    sys.path.insert(0, _PJ)

from experiments.config import DEEPSEEK_KEY, register_paths
register_paths()

from experiments.exp1_agentic_rag import IndustrialRetriever
from _ab_conditional_ef_35 import (
    EFExternalRetriever,
    CtrEFExternalRetriever,
    self_sufficiency_signal,
    is_gap_saturated,
    load_35_pool,
    TH_SAT,
)

base = IndustrialRetriever(corpus_dir=None)
base._ensure_initialized()
efu = EFExternalRetriever(corpus_dir=None)
ctr = CtrEFExternalRetriever(corpus_dir=None)

pool = load_35_pool()
print(f"池大小:{len(pool)} gate阈值={TH_SAT}\n")

gap_n = sat_n = 0
mismatch = []
for i, item in enumerate(pool, 1):
    q = item["q"]
    rb = base.retrieve(q, k=10)
    sig = self_sufficiency_signal(rb.scores)
    gap, label = is_gap_saturated(sig)
    if gap:
        gap_n += 1
    else:
        sat_n += 1
    # 回捞行为核验
    rb2 = efu.retrieve(q, k=10)      # ef 无条件
    rc = ctr.retrieve(q, k=10)       # ef 条件式
    n_efu = len(rb2.documents)
    n_ctr = len(rc.documents)
    n_base = len(rb.documents)
    gate_gap = rc.__gate__["gap"] if hasattr(rc, "__gate__") else None
    # 一致性: gap -> ctr 应 >= base； sat -> ctr 应 == base
    if gate_gap is not None and gate_gap:
        if not (n_ctr >= n_base):
            mismatch.append(("gap_no_backfill", q[:20], n_base, n_ctr))
    if gate_gap is not None and not gate_gap:
        if n_ctr != n_base:
            mismatch.append(("sat_extra", q[:20], n_base, n_ctr))
    if gate_gap is not None and gate_gap != gap:
        mismatch.append(("gate_mismatch", q[:20], gap, gate_gap))
    print(f"[{i:02d}] mu_top={sig['mu_top']:.3f} head={sig['headroom']:.3f} "
          f"label={label} n_base={n_base} n_efu={n_efu} n_ctr={n_ctr}")

print(f"\nGAP={gap_n} SAT={sat_n}")
print("门控↔回捞不一致:", mismatch if mismatch else "无")

# 保存门控信号分布供报告
dist = {"TH_SAT": TH_SAT, "GAP": gap_n, "SAT": sat_n,
        "rows": [{"q": it["q"], **self_sufficiency_signal(base.retrieve(it["q"], k=10).scores),
                  "label": is_gap_saturated(self_sufficiency_signal(base.retrieve(it["q"], k=10).scores))[1]} for it in pool]}
with open(os.path.join(_LINS, "results", "smoke_gate_35.json"), "w", encoding="utf-8") as f:
    json.dump(dist, f, ensure_ascii=False, indent=2)
print("已写 results/smoke_gate_35.json")
