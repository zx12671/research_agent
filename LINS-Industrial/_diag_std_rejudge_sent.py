# -*- coding: utf-8 -*-
"""
_diag_std_rejudge_sent.py —— 方向修正验证：对标准规范"B类(池外失配)"题，改用
【句子覆盖率】口径重判——判断"低召回"是真实检索失败，还是整段IoU口径对长GT/标准正文
天然低估造成的误判(承前 agentic_recall_completeness_correction)。

方法：
  对每 B 类题，取含该标准号/或检索 top-10 的 chunks，算：
    sent_coverage(GT, top10_dense) ——检索结果句覆盖
    sent_coverage(GT, [含标准号的chunk]) ——"标准正文块"句覆盖
  若后者明显高于前者 → 标准正文块确实承载答案句,只是 dense 没把它顶上 top10(检索失配)；
  若两者都接近0 → GT 与库内含号 chunk 字形/语句都无关(评测口径判不到,非检索问题)。

输出：results/s4_recall/std_rejudge_sent.json + 摘要
"""
import os, sys, json, re, csv
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LINS)
from retrieval.recall_metrics import sent_coverage, get_content
from retrieval.retriever import OpenDomainRetriever
CHLIN = os.path.join(_LINS, "knowledge_corpus", "chunks", "industrybench_chunks.jsonl")
P = re.compile(r"(GB(?:\s*/?\s*T)?|JB|HG|DL|QB|YB)\s*[-—]?\s*(\d{3,6})", re.I)


def main():
    retr = OpenDomainRetriever(project_root=_LINS)
    try:
        retr.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    except Exception:
        retr.load_from_manifest()
    chunks_all = [json.loads(l) for l in open(CHLIN, encoding="utf-8") if l.strip()]
    qmap = {}
    for r in csv.DictReader(open(os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv"),
                                 encoding="utf-8-sig", newline="")):
        qmap[r.get("id")] = (r.get("question", ""), r.get("knowledge_text", ""))
    # 复用 std_token_legit 里 in_corpus 的题(B 类含标准号且库里有号)
    rows = json.load(open(os.path.join(_LINS, "results", "s4_recall", "std_in_corpus.json"), encoding="utf-8"))

    n = 0; cov_dense = []; cov_stdchunk = []; n_std_ge_dense = 0
    for it in rows["rows"]:
        if not it["in_corpus"] or it["id"] not in qmap:
            continue
        q, kt = qmap[it["id"]]
        m = P.search(q)
        if not m:
            continue
        num = m.group(2)
        n += 1
        dc = retr.retrieve(q, k=10, use_ked=True).chunks
        c_d = sent_coverage(kt, dc)[0]
        # 含该标准号的 chunk(s)
        cands = [chunks_all[i] for i in range(len(chunks_all))
                 if num in (chunks_all[i].get("content", "") or "")][:20]
        c_s = sent_coverage(kt, cands)[0]
        cov_dense.append(c_d); cov_stdchunk.append(c_s)
        if c_s > c_d:
            n_std_ge_dense += 1
    print(f"B类含标准号且库中有号: {n}")
    print(f"  top10_dense 句覆盖率均值: {sum(cov_dense)/max(1,n)*100:.1f}%")
    print(f"  含标准号chunk 句覆盖率均值: {sum(cov_stdchunk)/max(1,n)*100:.1f}%")
    print(f"  标准正文块覆盖>dense检索的题: {n_std_ge_dense} ({n_std_ge_dense/max(1,n)*100:.0f}%)")
    out = os.path.join(_LINS, "results", "s4_recall", "std_rejudge_sent.json")
    json.dump({"n": n, "cov_dense": round(sum(cov_dense)/max(1,n),4),
               "cov_stdchunk": round(sum(cov_stdchunk)/max(1,n),4),
               "n_std_ge_dense": n_std_ge_dense}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[SAVED]", out)


if __name__ == "__main__":
    main()
