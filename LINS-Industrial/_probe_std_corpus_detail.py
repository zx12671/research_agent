# -*- coding: utf-8 -*-
"""
_probe_std_corpus_detail.py —— 抽查标准号在【原库 chunk】里的实际书写形态，
对比 query 中的形态，判断"标准号在库但检索捞不到"的根因是归一化缺失还是索引错位。
输出：results/s4_recall/std_corpus_detail.txt（3~8 个 case 的原文摘录）
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")

P = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG|DL|QB|YB)\s*[-—]?\s*(\d{3,6})", re.I)
NUM = re.compile(r"\d{3,6}")


def main():
    probes = json.load(open(os.path.join(_LINS, "results", "s4_recall", "bclass_bm25_probe.json"),
                            encoding="utf-8"))
    cases = [r for r in probes["rows"] if not r.get("recover")][:8]
    # 全库按 content 缓存 index of 标准号数字出现
    chunks = []
    for line in open(CH, encoding="utf-8"):
        if line.strip():
            chunks.append(json.loads(line))
    print("全库 chunks:", len(chunks))
    out_lines = []
    for c in cases:
        m = P.search(c["q"])
        num = m.group(2) if m else None
        out_lines.append("=" * 60)
        out_lines.append(f"qid={c['id']}  std={c.get('std')}|num={num}")
        out_lines.append(f"  Q: {c['q'][:55]}...")
        # 找全库里含该数字前缀的 chunk
        hits = [d for d in chunks if num and num in (d.get("content", ""))]
        out_lines.append(f"  全库含'{num}'的 chunk 数: {len(hits)}")
        if hits:
            d0 = hits[0]; cnt = d0.get("content", "")
            # 找到该数字出现处附近原文
            i = cnt.find(num)
            out_lines.append(f"  首个命中 chunk(id={d0.get('chunk_id','')[:16]}…):")
            out_lines.append(f"    …{cnt[max(0,i-40):i+30]}…")
        else:
            out_lines.append("  （数字不在任何 chunk 中——需核对编号年分差）")
    print("\n".join(out_lines))
    with open(os.path.join(_LINS, "results", "s4_recall", "std_corpus_detail.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("[SAVED] std_corpus_detail.txt")


if __name__ == "__main__":
    main()
