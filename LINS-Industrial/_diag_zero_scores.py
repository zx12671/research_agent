# -*- coding: utf-8 -*-
"""诊断 ab_ef_pairs_35 全 0 得分：是 ref 没对上，还是 scorer 本身判 0？"""
import os, sys, json, csv
_LINS = os.path.dirname(os.path.abspath(__file__))
for p in (_LINS, os.path.abspath(os.path.join(_LINS, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
from experiments.config import INDUSTRYBENCH_CSV, register_paths
register_paths()

from metrics.industrybench_scorer import RuleBasedScorer

# 1) refs dict 构建（与 _load_refs 一致）
rows = {}
with open(INDUSTRYBENCH_CSV, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        q = (r.get("question") or "").strip()
        if q:
            rows[q] = r.get("answer", "")
print("CSV total refs:", len(rows))
print("CSV 列名:", list(next(csv.DictReader(open(INDUSTRYBENCH_CSV, encoding='utf-8-sig', newline=''))).keys()))

# 2) 35 题池 loaded via load_35_pool
from _ab_conditional_ef_35 import load_35_pool
pool = load_35_pool()
print("pool n:", len(pool))

# 3) 命中率
hit = 0
missing = []
for it in pool:
    q = it["q"]
    if q in rows:
        hit += 1
    else:
        missing.append(q)
print("refs 命中数:", hit, "/", len(pool))
print("未命中样例(<=5):")
for q in missing[:5]:
    print("   MISS:", repr(q[:40]))
    # 找 CSV 里最接近的 question
    cand = [(qq, len(qq)) for qq in rows.keys() if q[:12] in qq or qq[:12] in q]
    print("       类似候选:", [c[:40] for c, _ in cand[:3]])

# 4) 用真实 ref vs 空 ref 对一题打分，验证 scorer 行为
scorer = RuleBasedScorer()
probe_q = pool[0]["q"]
probe_ref = rows.get(probe_q, "")
dummy_ans = "答：应加强固定与防护。"
print("\n[scorer行为核验] q=", probe_q[:26])
print("  ref命中?", bool(probe_ref), " len(ref)=", len(probe_ref))
if probe_ref:
    print("  score(q, ref, ans) =", scorer.rule_based_score(probe_q, probe_ref, dummy_ans))
print("  score(q, '', ans)   =", scorer.rule_based_score(probe_q, "", dummy_ans))

# 5) 截断 q -> 完整 CSV question 模糊回填，确认是否有题能对上任一 ref
q_full = list(rows.keys())
def backfill(q):
    cands = [x for x in q_full if len(q) >= 10 and x.startswith(q)]
    if len(cands) == 1:
        return cands[0], "unique-prefix"
    sub = [x for x in q_full if len(x) >= 10 and q in x]
    return (sub[0], "substr") if len(sub) == 1 else (None, "none/ambig")

hitf = ambig = none_ = 0
matched = []
for it in pool:
    full, kind = backfill(it["q"])
    if full and kind == "unique-prefix":
        hitf += 1
        matched.append((it["q"], full))
    elif full and kind == "substr":
        ambig += 1
    else:
        none_ += 1
print("\n[回到填核验] 唯一前缀命中=%d  子串回填=%d  无匹配=%d / %d" % (hitf, ambig, none_, len(pool)))

# 6) 用回填到的真实 ref 造一个合格答案，验证打分器非 0（证明全0纯因 ref miss）
nonzero = 0
for q, full in matched[:25]:
    ref = rows[full]
    g = str(ref).replace("\n", "，")
    ents = [t.strip() for t in g.replace("；", "，").split("，") if 2 <= len(t.strip()) <= 24][:4]
    good = "答：关键在于" + "、".join(ents) + "，并说明相应措施。" if ents else "答：见上文。"
    s = scorer.rule_based_score(q, ref, good)
    if s != 0:
        nonzero += 1
print("回填题中，合格答案得分非零数 = %d/%d  (全0根源=ref未对上)" % (nonzero, len(matched)))
