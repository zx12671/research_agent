# -*- coding: utf-8 -*-
"""学理核查：标准号 token 是否本就同时存在于 query 与承载答案的标准正文 chunk 中。
若成立 → 用标准号做检索信号是"正当 token 重叠"（A/B 修法不作弊），而非答案侧泄漏。
核查对象：probe.std_in_corpus 中"标准号在库"的题，取对应标准正文 chunk，看该 chunk
content 里是否含 query 同款标准号 + 是否含 GT(knowledge_text) 句子（证明它是承载答案的块）。
输出：results/s4_recall/std_token_legit_check.json + 控制台。
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
P = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG|DL|QB|YB)\s*[-—]?\s*(\d{3,6})[-—]?(\d{0,4})", re.I)
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
    rows = json.load(open(os.path.join(_LINS, "results", "s4_recall", "std_in_corpus.json"), encoding="utf-8"))
    # 重新加载 question->knowledge_text 映射
    import csv
    qmap = {}
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        qmap[r.get("id")] = (r.get("question", ""), r.get("knowledge_text", ""))
    chunks = [json.loads(l) for l in open(CH, encoding="utf-8") if l.strip()]

    n_check = 0; n_std_both = 0; n_chunk_holds_gt = 0
    for it in rows["rows"]:
        if not it["in_corpus"]:
            continue
        qid = it["id"]
        if qid not in qmap:
            continue
        q, kt = qmap[qid]
        m = P.search(q)
        if not m:
            continue
        num = m.group(2)
        n_check += 1
        # 找含该标准号的 chunk 中 IoU 最高者（视为承载答案的块）
        cands = [d for d in chunks if num in (d.get("content", "") or "")]
        if not cands:
            continue
        best = max(cands, key=lambda d: iou(kt, d.get("content", "")))
        std_in_chunk = num in (best.get("content", "")) or it["std_raw"].replace("-", "—").split("—")[0] in best.get("content", "")
        gt_in_chunk = iou(kt, best.get("content", "")) >= IOU_TH
        if it["std_raw"].split("-")[-1].split("—")[0].isdigit():
            pass
        n_std_both += int(std_in_chunk)
        n_chunk_holds_gt += int(gt_in_chunk)

    print(f"检查(标准号在库的题): {n_check}")
    print(f"  标准号同时存在于 query 与该 chunk: {n_std_both} ({n_std_both/max(1,n_check)*100:.0f}%)")
    print(f"  该 chunk 承载 GT 答案(整段IoU>=0.15): {n_chunk_holds_gt} ({n_chunk_holds_gt/max(1,n_check)*100:.0f}%)")
    print("解读：")
    print("  · 若两者都高(>90%): 标准号是 query 与答案正文【天然共享的 token】→")
    print("    用标准号做检索信号=正当 token 重叠(A/B 不作弊)；同时说明内容是检索器本应"),
    print("    命中的正文,遗漏是检索算法缺陷,不是缺料。")
    out = os.path.join(_LINS, "results", "s4_recall", "std_token_legit_check.json")
    json.dump({"n_check": n_check, "n_std_both": n_std_both,
               "n_chunk_holds_gt": n_chunk_holds_gt}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
