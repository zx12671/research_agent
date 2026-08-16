# -*- coding: utf-8 -*-
"""
evidence-forward 可行性快检（纯静态，不检索、不加载 embedder/模型）。

回答三个决定性问题（为 evidence-forward 第0级"同名文档顺藤摸瓜"立项把关）：
  1. 同一 document_id 是否有多个 chunk？（有 → 同源"顺藤"才有意义；无 → 方案作废）
  2. chunk 记录里是否带可用 document_id / chunk_index 字段？（决定能否精确回捞该 doc 全部同源块）
  3. A 类超长 GT 对应的文档块数分布 —— 验证"top-10 只覆盖开头、doc 中后段可被同 doc_id 补全"
     这一 evidence-forward 的核心前提是否成立。

只依赖 industrybench_chunks.jsonl，秒级出结果。
"""
import os, json, csv, random, re
from collections import Counter, defaultdict
from functools import lru_cache

_LINS = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
PER_CAP, N, SEED = 5, 30, 7


# ---------------- chunk 元数据字段探查 ----------------
def load_chunks():
    chunks = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            chunks.append(c)
    return chunks


def key_field(c, cands):
    for k in cands:
        if k in c and c[k]:
            return c[k]
    return None


def main():
    chunks = load_chunks()
    print(f"chunks 总数: {len(chunks)}")

    # ---- 2) 字段 schema 探查（不假设名字）----
    probe = chunks[0] if chunks else {}
    print("\n== 示例 chunk 字段 ==")
    for k, v in probe.items():
        s = str(v)
        print(f"  {k:24} = {s[:70]}")

    doc_field = None
    for cand in ["document_id", "doc_id", "document"]:
        if any(cand in c for c in chunks[:2000]):
            doc_field = cand
            break
    cid_field = None
    for cand in ["chunk_id", "id"]:
        if any(cand in c for c in chunks[:2000]):
            cid_field = cand
            break

    print(f"\n→ document_id 字段 = {doc_field!r}; chunk_id 字段 = {cid_field!r}")

    dids = [key_field(c, ["document_id", "doc_id", "document"]) or "?" for c in chunks]
    uniq_docs = set(dids)
    doc_count = Counter(dids)
    print(f"\n== Q1: 同一 document_id 是否有多个 chunk ==")
    print(f"  独立 document_id 数: {len(uniq_docs)}")
    mult = {d: n for d, n in doc_count.items() if n > 1}
    print(f"  块数>1 的 doc 数: {len(mult)} ({len(mult)/len(uniq_docs)*100:.0f}%)")
    print(f"  块数>1 的 doc 中，块数分布(前15): "
          f"{sorted(mult.values(), reverse=True)[:15]}")

    # ---- 3) A 类超长 GT 对应的 doc 块数 ----
    print("\n== Q3: 30 样本 GT 首段应归属的源文档块数（模拟 evidence-forward 靶区）==")
    random.seed(SEED)
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r.get("capability") or "", []).append(r)
    sample = []
    for cap, lst in sorted(by_cap.items()):
        sample += random.sample(lst, min(PER_CAP, len(lst)))
    sample = sample[:N]

    # 用 chunk content 与 GT 开头 bigram 匹配定位"首个承载该 GT 的 chunk 的 doc"
    @lru_cache(maxsize=None)
    def _grams(s, n=2):
        s = str(s).replace(" ", "")
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}

    def jt(a, b):
        ga, gb = _grams(a), _grams(b)
        return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0

    sents = []
    for r in sample:
        kt = r.get("knowledge_text", "") or ""
        sents.append([p for p in re.split(r"(?<=[。；;！？])\s*", kt) if len(p.strip()) >= 6])

    rows_stat = []
    for i, r in enumerate(sample):
        cap = r.get("capability", "")
        q = (r.get("question", "") or "")[:26]
        kt = r.get("knowledge_text", "") or ""
        # 找一个与该 GT 开头句子最像的 chunk → 得到它的 doc
        best_doc, best_jt, best_n = None, 0.0, 0
        head = sents[i][0] if sents[i] else kt
        for c in chunks:
            content = c.get("content", "") or ""
            dd = key_field(c, ["document_id", "doc_id"])
            j = jt(head[:60], content[:120])
            if j > best_jt:
                best_jt, best_doc = j, dd
        if best_doc:
            best_n = doc_count.get(best_doc, 0)
            rows_stat.append((q, cap, best_jt, best_n))
        else:
            rows_stat.append((q, cap, best_jt, 0))

    # 分组：A类(超长GT, 句数>=50) vs 其他
    a_class = 0
    for q, cap, j, n in rows_stat:
        nS = len(sents[rows_stat.index((q, cap, j, n))]) if False else None
    # 直接按 sents 长度给类
    print(f"{'Q':28}{'cap':9}{'#S':>4}{'首段JT':>8}{'源doc块数':>9}   类")
    print("-" * 72)
    for idx, (q, cap, j, n) in enumerate(rows_stat):
        nS = len(sents[idx])
        cls = "A" if nS >= 50 else ("B" if n == 0 else "C")
        if cls == "A":
            a_class += 1
        print(f"{q:28}{cap[:8]:9}{nS:>4}{j:>8.2f}{n:>9}   {cls}")

    nA = a_class
    # A 类中"源doc块数>=2"的比例
    a_mult = [n for i, (q, cap, j, n) in enumerate(rows_stat) if len(sents[i]) >= 50]
    print(f"\nA 类(超长GT≥50句) 题数: {nA}")
    print(f"A 类中 源doc块数>=2（可顺藤）: {sum(1 for n in a_mult if n >= 2)}/{len(a_mult)}")
    print(f"A 类源doc块数分布: {sorted(a_mult, reverse=True)}")
    overall_mult = [n for n in [_n for _, _, _, _n in rows_stat] if n >= 2]
    print(f"全部 30 题中 源doc块数>=2（可顺藤）: {len(overall_mult)}/30")


if __name__ == "__main__":
    main()
