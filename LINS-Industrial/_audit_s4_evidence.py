# -*- coding: utf-8 -*-
"""
_audit_s4_evidence.py —— S4 三条优化路径的证据审计（可溯源，仅读不改）。

动机：在决定"做不做"之前，先核查三条路径的核心假设是否成立，避免基于
      错误前提投入优化。输出结果用于 docs/stagewise_debug_plan.md S4 留痕。

核查项：
  [路径1] "工程计算/故障诊断 chunk≈0%" 是否真实、还是标签/口径失真。
  [路径3] 语料是否来自单一来源（判断语料质量 vs 排序失配）。
  结果留档：results/s4_recall/s4_evidence_audit.json  +  控制台输出。

用法： python _audit_s4_evidence.py
"""
import json, collections, os, sys
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
OUT = os.path.join(_LINS, "results", "s4_recall", "s4_evidence_audit.json")

report = {}

print("=" * 72)
print("[路径1] chunk 按 capability 的标签分布（核查 '≈0%' 是否真实）")
print("=" * 72)
c_cap = collections.Counter(); c_src = collections.Counter(); c_type = collections.Counter()
tot = 0; no_cap = 0
for line in open(CH, encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line); tot += 1
    if "capability" not in d:
        no_cap += 1
    c_cap[d.get("capability", "") or "(空)"] += 1
    c_src[d.get("source", "") or "(无)"] += 1
    c_type[d.get("knowledge_type", "") or "(无)"] += 1

rows = []
for k, v in c_cap.most_common():
    pct = v / tot * 100
    rows.append([k, v, round(pct, 1)])
    print(f"  capability[{k!r}] = {v}  ({pct:.1f}%)")
print("  total:", tot, "| 缺 capability 字段:", no_cap)
print("  source 分布:", dict(c_src.most_common(5)))
print("  knowledge_type 分布:", dict(c_type.most_common(5)))

# 关键判定：工程计算/故障诊断不是 0，而是"占比极小 + 检索不到"
for probe_cap in ("工程计算与估算", "故障诊断与排查"):
    n = c_cap.get(probe_cap, 0)
    verdict = "存在有限相关块但占比极小" if n > 0 else "真·零块"
    print(f"\n  判定[{probe_cap}]: {n} 块 → {verdict}（'补语料'与'修排序'责任需区分）")

# [路径1-补] 交叉验证：工程计算/故障诊断样本在 top10 内实际捞到多少相关块(来自 s4_rows)
print("\n" + "=" * 72)
print("[路径1-补] top10 内实际相关块入池量（交叉验证'是缺料'还是'检不到'）")
print("=" * 72)
_rows_json = os.path.join(_LINS, "results", "s4_recall", "s4_rows_20260807_221619.json")
if os.path.exists(_rows_json):
    rows = json.load(open(_rows_json, encoding="utf-8"))["rows"]
    for cap in ("工程计算与估算", "故障诊断与排查", "标准规范与术语"):
        subset = [r for r in rows if r.get("cap") == cap and "single" in r.get("methods", {})]
        if not subset:
            print(f"  {cap}: (无样本)"); continue
        rel = [r["methods"]["single"]["n_rel_in_pool"] for r in subset]
        cov = [r["methods"]["single"]["cov10"] for r in subset]
        print(f"  {cap}: n={len(subset)} avg_相关块入top10={sum(rel)/len(rel):.2f} "
              f"avg_cov10={sum(cov)/len(cov)*100:.1f}%")
    report["pool_note"] = (
        "若工程计算/故障诊断 avg_相关块入top10≈0 而库中又有 528/213 块 → 判定为"
        "'排序/信号失配'（与路径3同源），而非单纯'语料缺失'。"
    )
else:
    print("  未找到 s4_rows JSON，跳过。")
    report["pool_note"] = "s4_rows 缺失，未做入池交叉验证。"


report["capability_dist"] = rows
report["n_chunks"] = tot
report["no_cap_field"] = no_cap
report["source_dist"] = dict(c_src.most_common(5))
report["knowledge_type_dist"] = dict(c_type.most_common(5))
report["note"] = (
    "路径1 修正: 工程计算/故障诊断并非'零块'而是'占比极小(1%/0.4%)+检索不到(good%=0%)'。"
    "若这些有限块在 dense 排序中排位极低，则与路径3'排序失配'同源，非单纯语料缺失。"
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n[SAVED]", OUT)

