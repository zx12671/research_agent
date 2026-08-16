# -*- coding: utf-8 -*-
"""S2: 扫描 _recheck_err.txt 真实运行留痕，统计每张执行图的节点与调用。"""
import re, collections

PATH = "_recheck_err.txt"
lines = open(PATH, encoding="utf-8", errors="ignore").read().splitlines()

mark = [i for i, l in enumerate(lines) if "Executing graph with" in l]
print("Executing graph 条数:", len(mark))
if not mark:
    raise SystemExit(0)


def node_list_from(line):
    m = re.search(r"nodes: \[(.*?)\]", line)
    if not m:
        return []
    return [n.strip().strip("'") for n in m.group(1).split(",") if n.strip()]


patterns = collections.Counter()
r1cnt = r2cnt = vcnt = dcnt = r2_after_verify = 0
simple_cnt = 0
for idx, mi in enumerate(mark):
    end = mark[idx + 1] if idx + 1 < len(mark) else min(mi + 60, len(lines))
    seg = lines[mi:end]
    nodes = node_list_from(seg[0])
    has_r1 = any("retrieve_1] Retrieving" in l for l in seg)
    has_r2 = any("retrieve_2] Retrieving" in l for l in seg)
    has_v = any("Verifying" in l and "verify" in l.lower() for l in seg)
    has_d = any("decide" in l.lower() and "Evaluating" in l for l in seg)
    pat = tuple(sorted(set(nodes)))
    patterns[pat] += 1
    r1cnt += bool(has_r1)
    r2cnt += bool(has_r2)
    vcnt += bool(has_v)
    dcnt += bool(has_d)
    if has_v and has_r2:
        r2_after_verify += 1
    adv = ("verify_1" in pat) or ("retrieve_2" in pat) or ("decide_1" in pat) or ("merge_1" in pat)
    if not adv:
        simple_cnt += 1

print("\n[按去重节点集] 模式数:", len(patterns))
for p, c in patterns.most_common():
    adv = ("verify_1" in p) or ("retrieve_2" in p) or ("decide_1" in p) or ("merge_1" in p)
    print(("  [ADV]  " if adv else "  [simple]"), c, "x", sorted(p))

print("\n统计（按每次执行）:")
print("  含 retrieve_1 真实执行:", r1cnt, "/", len(mark))
print("  含 retrieve_2 真实执行:", r2cnt)
print("  含 verify 执行:", vcnt)
print("  含 decide 执行:", dcnt)
print("  verify 之后触发 retrieve_2 (fail retry):", r2_after_verify)
print("  纯简单图(无verify/retrieve2/decide/merge):", simple_cnt)

# 统计 real deepseek calls 与节点关系松散提示
http = sum(1 for l in lines if "api.deepseek.com/chat/completions" in l and "HTTP/1.1 200 OK" in l)
print("\n  真实 DeepSeek HTTP 200 调用:", http, "次（全文件）")
