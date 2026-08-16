# -*- coding: utf-8 -*-
"""
_probe_std_in_corpus.py —— 路径3 最后一关判定：含标准号且 B 类(池外) 的标准规范题，
其标准号在【整个知识库】中是否存在对应 chunk？据此裁决：
  · 库里有该编号 chunk 但检索捞不到 → 召回失配可修(应修 query/精确通道)
  · 库里根本没有该编号 chunk      → 真·语料缺失，改判为补料,路径3检索修法无效

方法：取 probe.bclass_bm25_probe 里的 187 题(含标准号 B 类)，抽标准号编号，
     在 chunks.jsonl 全部 content 中直接在库面搜该编号(全库全量字串匹配)。
输出：results/s4_recall/std_in_corpus.json + 摘要
"""
import os, sys, json, re
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
IOU_TH = 0.15

# 标准号模式：尽量贴合 query 中出现的带年份编号，如 GB/T 19862、GB 20074 等
P = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG|DL|QB|YB|CJ|SN)\s*[-—]?\s*(\d{3,6})[-—]?(\d{0,4})", re.I)


def norm(s):
    return re.sub(r"[\s\-\—/（()）]", "", str(s)).upper()


def main():
    probe = json.load(open(os.path.join(_LINS, "results", "s4_recall", "bclass_bm25_probe.json"),
                           encoding="utf-8"))
    # 建全库 content 汇总（转小写粗匹配）+ 抽号
    print("加载全库 chunk content 索引……")
    contents = []
    for line in open(CH, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        contents.append(d.get("content", "") or "")
    corp = " \u0001 ".join(contents)
    corp_norm = norm(corp)
    print("索引完毕, content 总字符:", len(corp))

    found = 0; hit_gt_link = 0; missing = 0
    rows = []
    for r in probe["rows"]:
        m = P.search(r["q"])
        if not m:
            continue
        num = m.group(2)
        # 标准号归一化形态：<前缀><数字>（如 GB/T19862 / GBT19862）与纯数字
        noj = m.group(0)
        forms = set()
        forms.add(num)                      # 19862
        forms.add(norm(noj))                # GBT19862
        forms.add(norm(noj).replace("GBT", "G B T"))  # 宽松
        in_corpus = any(f and f in corp_norm for f in forms if len(f) >= 3)
        rows.append({"id": r["id"], "q": r["q"][:35], "std_raw": m.group(0), "num": num,
                     "in_corpus": bool(in_corpus)})
        if in_corpus:
            found += 1
        else:
            missing += 1

    print(f"含标准号 B 类题(有编号): {len(rows)}")
    print(f"  标准号在【全库】存在: {found}  ({found/len(rows)*100:.0f}%)")
    print(f"  标准号全库不存在:     {missing}  ({missing/len(rows)*100:.0f}%)")
    print("→ 库里有号却检索捞不到 = 召回失配(可修); 库里无号 = 真缺料(改补料)")
    out = os.path.join(_LINS, "results", "s4_recall", "std_in_corpus.json")
    json.dump({"n_total": len(rows), "n_in_corpus": found, "n_missing": missing,
               "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
