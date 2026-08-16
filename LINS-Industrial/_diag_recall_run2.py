# -*- coding: utf-8 -*-
"""_diag_recall_run2.py — run2 两步式检索 vs 现状三方法（single/multi/hybrid）对比。

run1（baseline）= 当前 agentic 系统的检索实现（S4）：
  - single = retriever.retrieve(q,k=10,use_ked=True)
  - multi  = retriever.multi_query_retrieve(q,k=10,strategy='fusion')
  - hybrid = retriever.hybrid_retrieve(q,k=10,use_ked=True)
run2 = 两步式检索（EF 改进版，两步法）：
  第一步  现有 dense 检索取宽候选池 pool_k；
  第二步  AI（真实 DeepSeek）仅凭【问题+候选块】给相关性/适配分（**绝不碰 GT/答案**），
          取高相关锚块的 document_id 反向回捞同源文档其余块（jt 过滤防噪），再融合排序。

指标口径（与 _diag_recall_s4.py / recall_metrics.py 完全一致）：
  主口径   cov@10（GT 句子覆盖率）+ good%（>=0.5 拼齐率）；
  交叉参照 hit@1/3/10（整段 IoU）、MRR、NDCG（官方 RetrievalEvaluator）。
  相关集合 = chunk 与 knowledge_text 的 bigram IoU >= IOU_TH（仅用于评估，run2 打分器不可见）。

用法:
  python _diag_recall_run2.py --per_cap 4 [--pool 60] [--sim] [--seed 7]
    --sim  额外跑一个"零成本词法/嵌入打分"对照（不花 LLM），观察 LLM 打分是否带来增益。
输出: results/run2_vs_s4/run2_vs_s4_{ts}.md + .json
"""
import os, sys, json, time, csv, random, argparse, re, statistics
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from retrieval.recall_metrics import sent_coverage
from retrieval.retriever import OpenDomainRetriever, RelevanceRanker, chinese_jt

CSV = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
MANIFEST = os.path.join(_LINS, "knowledge_corpus", "manifest.json")
OUT_DIR = os.path.join(_LINS, "results", "run2_vs_s4")

IOU_TH = 0.15
GOOD_COV = 0.5
CAPS = ["选型与替代", "标准规范与术语", "工艺原理与参数影响", "安全合规与风险控制",
        "质量计量与检测", "故障诊断与排查", "工程计算与估算"]


# ---------------- run2 二阶段 相关性评估器（真实 DeepSeek，仅用问题+候选块） ----------------
class LLMRelevanceRanker(RelevanceRanker):
    """真实 DeepSeek 打分器。

    关键约束：prompt 只给【问题】与【候选块】，让模型逐块输出 0-1 适配分。
    **绝不注入 answers / knowledge_text / GT** —— 规避"直接读答案"的嫌疑。
    打分对象是第一步初检出的候选块（宽池），与评估用的 GT 完全解耦。
    """

    def __init__(self, api_key: str, model="deepseek-chat", max_tokens=900, temperature=0.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls = []
        self.prompt_total_chars = 0

    def _parse(self, text, cids):
        """解析模型输出中的 {chunk_id: 分数}。支持 JSON 或 'id: score' 行文本。"""
        scores = {}
        text = text or ""
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            scores[str(k)] = float(v)
            except Exception:
                pass
        if not scores:
            for cid in cids:
                pat = re.compile(re.escape(cid) + r"\s*[:：]\s*([0-1](?:\.\d+)?|\d)")
                mm = pat.search(text)
                if mm:
                    scores[cid] = min(1.0, float(mm.group(1)))
        return scores

    def score(self, query, chunks):
        cids = [c.chunk_id for c in chunks]
        body = []
        for c in chunks:
            content = (getattr(c, "content", "") or "").strip().replace("\n", " ")
            body.append(f"[{c.chunk_id}] {content[:200]}")
        user = (
            "你是工业知识检索的相关性评估器。请判断下面每个检索块与【问题】的适配/相关程度。\n"
            "规则：只需依据『问题』与『块内容』判断是否能辅助回答该问题（块里是否有答案所需的"
            "标准号/参数/工艺/合规要点等）。分数为 0~1 的实数：0 完全不相关，1 高度相关/几乎命中。\n"
            "严格按如下 JSON 返回（不要输出其它内容）：\n"
            '{"<chunk_id>": 0~1 分数, ...}\n\n'
            f"【问题】\n{query}\n\n【候选块】\n" + "\n".join(body)
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "你是严谨客观的检索相关性评估器，只打分不回答案。"},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content
        self.calls.append({"n_chunks": len(chunks), "q_len": len(query)})
        self.prompt_total_chars += len(user)
        return self._parse(text, cids)


class SimRelevanceRanker(RelevanceRanker):
    """零成本对照打分器（不花 LLM）：嵌入余弦 + 词法 jt 融合。
    仅用于观察『真实 LLM 打分』相对『纯检索层信号』是否带来额外增益，不参与主结论。"""

    def __init__(self, retriever):
        self.retriever = retriever

    def score(self, query, chunks):
        import numpy as np
        qe = self.retriever._encode_query(query)
        out = {}
        for c in chunks:
            ce = self.retriever._encode_query((getattr(c, "content", "") or "")[:400])
            cos = float(qe @ ce) / (float(np.linalg.norm(qe)) * float(np.linalg.norm(ce)) + 1e-9)
            jt = chinese_jt(query, getattr(c, "content", "") or "")
            out[c.chunk_id] = 0.7 * max(0.0, cos) + 0.3 * min(1.0, jt * 5)
        return out


# ---------------- 评估（与 _diag_recall_s4.py 同口径） ----------------
def _grams(s, n=2):
    s = str(s).replace(" ", "").replace("\u202e", "")
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def iou(a, b):
    ga, gb = _grams(a), _grams(b)
    return (len(ga & gb) / len(ga | gb)) if (ga and gb) else 0.0


def first_hit_rank(chunks, kt):
    for i, c in enumerate(chunks, 1):
        if iou(kt, getattr(c, "content", "") or "") >= IOU_TH:
            return i
    return None


def cover_at_k(chunks, kt):
    cov, _, _ = sent_coverage(kt, chunks)
    return cov, (cov >= GOOD_COV)


class RelEvaluator:
    def __init__(self, k_values=(1, 3, 10)):
        from eval_scripts.industrial_linkeval.retrieval_evaluator import RetrievalEvaluator
        self.ev = RetrievalEvaluator(k_values=list(k_values))

    def per_query(self, chunks, relevant_ids):
        rids = [c.chunk_id for c in chunks]
        return self.ev.evaluate(rids, set(relevant_ids))


def load_questions():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stratify_sample(rows, per_cap):
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r.get("capability") or ""].append(r)
    sample = []
    for cap in CAPS:
        picked = by_cap.get(cap, [])
        if picked:
            sample += random.sample(picked, min(per_cap, len(picked)))
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cap", type=int, default=4, help="每 capability 抽样数")
    ap.add_argument("--pool", type=int, default=60, help="run2 第一步候选池宽")
    ap.add_argument("--n_anchor", type=int, default=3)
    ap.add_argument("--sim", action="store_true", help="额外跑零成本 SimRelevanceRanker 对照")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    rows_full = load_questions()
    sample = stratify_sample(rows_full, args.per_cap)
    print(f"[样本] {len(sample)} 题; cap 分布={dict(Counter(r['capability'] for r in sample))}")

    from experiments.config import LLM_NAME, DEEPSEEK_KEY, register_paths
    register_paths()

    from retrieval.retriever import OpenDomainRetriever
    retr = OpenDomainRetriever(project_root=_LINS)
    retr.load_from_manifest(manifest_path=MANIFEST)
    retr.retrieve("预热", k=1)
    print(f"[检索器] {len(retr.chunks)} chunks")

    llm_ranker = LLMRelevanceRanker(api_key=DEEPSEEK_KEY, model=LLM_NAME)
    sim_ranker = SimRelevanceRanker(retr) if args.sim else None

    def meth_run1(m, q):
        if m == "single":
            return retr.retrieve(q, k=10, use_ked=True)
        if m == "multi":
            return retr.multi_query_retrieve(q, k=10, use_ked=True, strategy="fusion")
        return retr.hybrid_retrieve(q, k=10, use_ked=True)

    methods_run1 = ["single", "multi", "hybrid"]
    methods = methods_run1 + ["run2_llm"]
    if sim_ranker:
        methods += ["run2_sim"]

    ev = RelEvaluator()
    agg = {}
    for m in methods:
        agg[m] = {cap: {"n": 0, "hit1": 0, "hit3": 0, "hit10": 0, "mrr": 0.0,
                        "ndcg": 0.0, "cov10": 0.0, "good10": 0} for cap in CAPS}

    t0 = time.time()
    for idx, r in enumerate(sample, 1):
        q = r.get("question", ""); kt = r.get("knowledge_text", "")
        cap = r.get("capability", "")
        if not q or not kt:
            continue
        detail = {"id": r.get("id", ""), "cap": cap, "q": q[:60], "methods": {}}
        for m in methods_run1:
            try:
                res = meth_run1(m, q)
            except Exception as e:
                print(f"  [{m}] err: {e}"); continue
            _accumulate(agg, m, cap, res, kt, ev, detail)
        _run2(agg, "run2_llm", cap, q, kt, retr, llm_ranker, ev, detail, args)
        if sim_ranker:
            _run2(agg, "run2_sim", cap, q, kt, retr, sim_ranker, ev, detail, args)
        if idx % max(1, len(sample) // 5) == 0:
            print(f"  .. {idx}/{len(sample)}  ({time.time()-t0:.0f}s)")

    _report(methods, agg, sample, args, ts, llm_ranker)


def _run2(agg, m, cap, q, kt, retr, ranker, ev, detail, args):
    try:
        res = retr.two_stage_retrieve(q, k=10, pool_k=args.pool, n_anchor=args.n_anchor,
                                      ranker=ranker)
    except Exception as e:
        print(f"  [run2:{m}] err: {e}"); return
    _accumulate(agg, m, cap, res, kt, ev, detail)


def _accumulate(agg, m, cap, res, kt, ev, detail):
    chunks = list(getattr(res, "chunks", [])) or []
    rel_ids = [c.chunk_id for c in chunks if iou(kt, getattr(c, "content", "") or "") >= IOU_TH]
    rank = first_hit_rank(chunks, kt)
    cov10, good10 = cover_at_k(chunks, kt)
    a = agg[m][cap]
    a["n"] += 1
    if rank and rank <= 1: a["hit1"] += 1
    if rank and rank <= 3: a["hit3"] += 1
    if rank and rank <= 10: a["hit10"] += 1
    if rank: a["mrr"] += 1.0 / rank
    rq = ev.per_query(chunks, rel_ids)
    a["ndcg"] += rq.ndcg
    a["cov10"] += cov10
    a["good10"] += int(good10)
    detail["methods"][m] = {"rank": rank, "n_rel": len(rel_ids),
                            "cov10": round(cov10, 3), "good10": bool(good10)}


def _report(methods, agg, sample, args, ts, llm_ranker):
    rep = []
    R = rep.append
    R("=" * 84)
    R(f"run2 两步式检索 vs run1 三方法 | 样本 {len(sample)} 题(每cap≈{args.per_cap}) "
      f"| pool={args.pool} anchor={args.n_anchor} | IOU={IOU_TH} | {ts}")
    R("=" * 84)
    R("\n【A】方法对比 (k=10) —— 主口径 cov@10 + good%；交叉参照 hit/MRR/NDCG")
    R(f"  {'方法':<9}{'n':>4}{'hit@1':>7}{'hit@3':>7}{'hit@10':>8}{'cov@10':>8}{'good%':>7}{'MRR':>7}{'NDCG':>8}")
    for m in methods:
        n = sum(agg[m][c]["n"] for c in CAPS)
        if n == 0:
            continue
        h1 = sum(agg[m][c]["hit1"] for c in CAPS) / n
        h3 = sum(agg[m][c]["hit3"] for c in CAPS) / n
        h10 = sum(agg[m][c]["hit10"] for c in CAPS) / n
        cov = sum(agg[m][c]["cov10"] for c in CAPS) / n
        good = sum(agg[m][c]["good10"] for c in CAPS) / n
        mrr = sum(agg[m][c]["mrr"] for c in CAPS) / n
        ndcg = sum(agg[m][c]["ndcg"] for c in CAPS) / n
        R(f"  {m:<9}{n:>4}{h1*100:>6.0f}%{h3*100:>6.0f}%{h10*100:>7.0f}%"
          f"{cov*100:>7.1f}%{good*100:>6.0f}%{mrr*100:>6.1f}%{ndcg*100:>7.1f}%")

    R("\n【B】按 capability 分层 (cov@10 / hit@1)")
    R(f"  {'能力':<12}{'n':>4} " + " ".join(f"{m[:7]:>9}" for m in methods if any(agg[m][c]["n"] for c in CAPS)))
    for cap in CAPS:
        if sum(agg[m][cap]["n"] for m in methods) == 0:
            continue
        line = f"  {cap:<12}"
        for m in methods:
            a = agg[m][cap]
            if a["n"] == 0:
                line += " " * 22
                continue
            line += f"  {a['cov10']/a['n']*100:>5.0f}/{a['hit1']/a['n']*100:>3.0f}"
        R(line)

    n = sum(agg["run2_llm"][c]["n"] for c in CAPS)
    base_best_cov = max(sum(agg["single"][c]["cov10"] for c in CAPS),
                        sum(agg["hybrid"][c]["cov10"] for c in CAPS)) / n
    r2_cov = sum(agg["run2_llm"][c]["cov10"] for c in CAPS) / n
    R("\n【C】run2 vs baseline（主口径 cov@10）")
    R(f"  baseline取best(single,hybrid) cov@10 = {base_best_cov*100:.1f}%")
    R(f"  run2(LLM打分)              cov@10 = {r2_cov*100:.1f}%   Δ={(r2_cov-base_best_cov)*100:+.1f}pp")
    try:
        nc = statistics.mean([c["n_chunks"] for c in llm_ranker.calls] or [0])
    except Exception:
        nc = 0
    R(f"  run2 LLM 调用点: {len(llm_ranker.calls)} 题级 | 平均每题候选 {nc:.0f} | "
      f"累计 prompt chars ≈ {llm_ranker.prompt_total_chars}")

    text = "\n".join(rep)
    print(text)
    md = os.path.join(args.out, f"run2_vs_s4_{ts}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    js = os.path.join(args.out, f"run2_vs_s4_{ts}.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "methods": methods,
                   "agg_by_cap": {m: {c: dict(agg[m][c]) for c in CAPS} for m in methods}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {md}")
    print(f"[SAVED] {js}")


if __name__ == "__main__":
    main()
