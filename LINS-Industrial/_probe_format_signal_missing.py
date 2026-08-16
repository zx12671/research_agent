# -*- coding: utf-8 -*-
"""
探针：验证用户反馈 —— "原始题面无论题型都是疑问句，真人也难以仅凭question判断题型"
即：format 标签在 question 文本中的显式信号是否缺失？
量化四类题型在"题面显式题型字符"上的统计，支持归因修正。
"""
import os, re, json

def find_csv():
    import os
    hits = []
    base = os.getcwd()
    roots = [base]
    vend = os.path.join(base, "LINS-Industrial")
    if os.path.isdir(vend): roots.append(vend)
    for r in roots:
        for dp, _, files in os.walk(r):
            if any(x in dp for x in ["node_modules", ".git", "__pycache__"]):
                continue
            for fn in files:
                if fn.lower().endswith(".csv") and ("industry" in fn.lower() or "bench" in fn.lower() or "dataset" in fn.lower()):
                    hits.append(os.path.join(dp, fn))
    return sorted(hits)


def find_question_col(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    for k in ["question", "Question", "query", "Q", "_question"]:
        if k in header:
            return k
    return None

def main():
    import csv
    hits = find_csv()
    print("候选CSV:", hits if hits else "未找到/未列全，需指定路径")
    path = None
    for h in list(hits)[:3] if hits else []:
        if "industrybench" in h or "industry_bench" in h:
            path = h; break
    if path is None and hits:
        path = hits[0]
    if path is None:
        print("请确认数据集CSV路径后重试"); return

    qcol = find_question_col(path)
    fcol = "_format"
    print(f"CSV: {path}   question列={qcol}  format列={fcol}")
    if qcol is None:
        print("未找到 question 列"); return

    types = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if fcol not in reader.fieldnames:
            print(f"csv 无 {fcol} 列，实际列: {reader.fieldnames}"); return
        for row in reader:
            q = (row.get(qcol) or "").strip()
            fmt = (row.get(fcol) or "").strip()
            types.setdefault(fmt, {"n":0, "ends_qmark":0, "has_underscore":0, "has_option_letter":0, "has_calc_mark":0, "len":[]})
            t = types[fmt]
            t["n"] += 1
            t["len"].append(len(q))
            if q.endswith("？") or q.endswith("?"):
                t["ends_qmark"] += 1
            if "___" in q or "__" in q.replace(" ", "") or "（  ）" in q or "( )" in q or "填空" in q:
                t["has_underscore"] += 1
            # 选项字母：形如 A. abc 或 A、abc 或 A) abc
            if re.search(r"(?:^|\s)[A-D][.、)．]\s*\S", q):   # 仅匹配整题首选项，仍存疑
                t["has_option_letter"] += 1
            if re.search(r"[＝=<>+\-*/×÷^]|(\d+\s*(?:mm|kg|m|℃|度|L|g|Pa))", q) and any(k in q for k in ["多少","计算","为多少","约","何值","数值"]):
                t["has_calc_mark"] += 1

    print("\n=== format 题型在题面显式信号上的统计 ===")
    for fmt, t in types.items():
        n = t["n"]
        if n == 0: continue
        avg_len = sum(t["len"])/len(t["len"])
        print(f"[{fmt:14}] n={n:5d}  疑问句结尾={t['ends_qmark']/n:5.1%}  含填空符={t['has_underscore']/n:5.1%}  "
              f"含选项字母={t['has_option_letter']/n:5.1%}  含计算词/数字={t['has_calc_mark']/n:5.1%}  avg_len={avg_len:.0f}")

    # dump
    out = os.path.join("results", "probe_format_signal_missing", "format_signal_missing.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {"n":v["n"], "ends_qmark":v["ends_qmark"], "has_underscore":v["has_underscore"],
                       "has_option_letter":v["has_option_letter"], "has_calc_mark":v["has_calc_mark"],
                       "avg_len":round(sum(v["len"])/len(v["len"]),1)} for k,v in types.items()},
                   f, ensure_ascii=False, indent=2)
    print("\n保存:", out)

if __name__ == "__main__":
    main()
