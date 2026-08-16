# -*- coding: utf-8 -*-
"""
_diag_evidence_s5.py —— S5【证据组织】专项量化检查（完整性 + 不膨胀 + 超长截断审计）
参照 S4 留痕式检查：口径明确 → 大规模实测 → 留痕 → 归因。
覆盖 4 判据：A完整性 / B去重保留率 / C长度与比值+超长定位 / D merge膨胀(重复块)。
生产对标：RerankTruncOrganizer（生产落地，truncate=True,max_chars=6000）；
对照：EvidenceOrganizer（纯保真无截断）。
"""
import os, sys, json, statistics, csv, hashlib, re
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")
_LINS = os.path.dirname(os.path.abspath(__file__))
if _LINS not in sys.path:
    sys.path.insert(0, _LINS)
from agentic.task_types import ExecutionDirective, TaskType
from agentic.organizer import EvidenceOrganizer, RerankTruncOrganizer
from agentic.prompts import format_prompt
from retrieval.retriever import OpenDomainRetriever
from retrieval.recall_metrics import sent_coverage

OUT = os.path.join(_LINS, "results", "evidence_s5_audit.json")
CSV_PATH = os.path.join(_LINS, "data", "industrybench", "huggingface_dataset.csv")
CAP_HARDPOOL = {
    "selection": 20, "工艺原理": 20, "标准规范": 20, "质量计量": 18,
    "故障诊断": 15, "安全合规": 15, "工程计算": 12,
}
CAP_ALIAS = {"selection": "选型替代", "comparison": "比较释义"}


def normcap(cap):
    return CAP_ALIAS.get(cap, cap)



def main():
    rows_by_cap = defaultdict(list)
    for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig", newline="")):
        rows_by_cap[normcap(r.get("capability", ""))].append(r)
    order = []
    used = set()
    for c in list(CAP_HARDPOOL):
        bucket = [r for r in rows_by_cap.get(c, []) if id(r) not in used]
        cur = bucket[: CAP_HARDPOOL[c]]
        order.extend(cur)
        used |= {id(r) for r in cur}
    target = sum(CAP_HARDPOOL.values())
    n_have = len(order)
    if n_have < target:
        allr = []
        for rr in rows_by_cap.values():
            allr += rr
        for r in allr:
            if id(r) in used:
                continue
            order.append(r); used.add(id(r))
            if len(order) >= target:
                break
    sample = order
    print(f"S5 证据组织审计样本 N={len(sample)}")

    retriever = OpenDomainRetriever(project_root=_LINS)
    retriever.load_from_manifest(manifest_path=os.path.join(_LINS, "knowledge_corpus", "manifest.json"))
    org_prod = RerankTruncOrganizer()
    org_pure = EvidenceOrganizer()
    directive = ExecutionDirective(module="organization", action="group_by_topic",
                                   params={"organize_by": "topic", "remove_duplicate": True})
    task_obj = TaskType.GENERAL

    stats = {"n": 0, "complete_ok": 0, "miss_total": 0,
             "retention_prod": [], "retention_pure": [],
             "ctx_prod": [], "ctx_pure": [], "prompt_len": [], "ratio_ev_prompt": [],
             "truncated_cases": [], "miss_id_cases": [], "dup_block_cases": []}

    for idx, row in enumerate(sample):
        q, kt = row.get("question", ""), row.get("knowledge_text", "")
        cap = normcap(row.get("capability", ""))
        try:
            rr = retriever.retrieve(q, k=10, use_ked=True)
        except Exception as e:
            print(f"  [{idx}] retrieve err:", e)
            continue
        hits = list(rr.chunks)
        if not hits:
            continue
        orp = org_prod.execute(directive=directive, documents=list(hits), question=q, task_type=cap)
        ctx_prod = orp.get_context()
        n_in = getattr(orp, "original_count", len(hits))
        n_prod_after = getattr(orp, "organized_count", len(getattr(orp, "documents", [])))
        n_doc_prod = len(getattr(orp, "documents", []))
        # 完整性：get_context 输出 self.documents（与 organized_count 一致），且每个 doc 的
        # 内容前段(归一空白)都在 ctx 中逐字出现。不在正文里数 "(score=" 以避免正文误计。
        block_hits_prod = n_doc_prod
        missing_ids = []
        for d_ in getattr(orp, "documents", []):
            cpre = re.sub(r"\s+", "", str(getattr(d_, "content", ""))[:50])
            if cpre and cpre not in re.sub(r"\s+", "", ctx_prod):
                missing_ids.append(getattr(d_, "chunk_id", "?"))
        complete_ok = int((n_doc_prod == n_prod_after) and not missing_ids)
        miss = len(missing_ids)


        opu = org_pure.execute(directive=directive, documents=list(hits), question=q, task_type=cap)
        ctx_pure = opu.get_context()
        n_pure_after = getattr(opu, "organized_count", 0)

        evidence_str = ctx_prod if orp else ""
        full_prompt = ""
        try:
            full_prompt = format_prompt(task_key="general", question=q, evidence=evidence_str, prompt_extra="")
        except Exception:
            full_prompt = ""
        prompt_len = len(full_prompt)

        # merge/重复块检测：证据串内同 content 指纹 >1
        blocks = re.findall(r"\[?\d+\]? \(score=\d+\.\d+\)[^\n]*\n(.*?)(?=\n\[?\d+\]? \(score=|\Z)", ctx_prod, re.S)
        sig = Counter()
        for b in blocks:
            sig[hashlib.md5(re.sub(r"\s+", "", b).encode()).hexdigest()] += 1
        dup_seen = sum(1 for v in sig.values() if v > 1)

        # 超长/截断定位：生产 vs 纯保真后缺失的 chunk 里是否含 GT 相关块
        if n_prod_after < n_pure_after:
            prod_ids = {id(d) for d in getattr(orp, "documents", [])}
            dropped = [d for d in getattr(opu, "documents", []) if id(d) not in prod_ids]
            if dropped:
                rc = sent_coverage(kt, list(dropped))[0]
                if rc >= 0.30:
                    stats["truncated_cases"].append(dict(qid=row.get("id"), cap=cap,
                                n_trunc=len(dropped), sent_cov_dropped=round(rc, 2),
                                ctx_ev_len=len(ctx_prod)))
        stats["n"] += 1
        stats["complete_ok"] += complete_ok
        stats["miss_total"] += miss
        stats["retention_prod"].append(n_prod_after / n_in if n_in else 0)
        stats["retention_pure"].append(n_pure_after / n_in if n_in else 0)
        stats["ctx_prod"].append(len(ctx_prod))
        stats["ctx_pure"].append(len(ctx_pure))
        stats["prompt_len"].append(prompt_len)
        stats["ratio_ev_prompt"].append(len(evidence_str) / prompt_len if prompt_len else 0)
        if miss > 0:
            stats["miss_id_cases"].append(dict(qid=row.get("id"), cap=cap, miss=miss,
                                               block_hits=block_hits_prod, n_doc=n_doc_prod))
        if dup_seen > 0:
            stats["dup_block_cases"].append(dict(qid=row.get("id"), cap=cap, dup_blocks=dup_seen))
        if idx % 20 == 0:
            print(f"  ... {idx}/{len(sample)}")
    def avg(x):
        x = [v for v in x if v is not None]
        return statistics.mean(x) if x else None

    def pct(n, d):
        return round(n / d * 100, 1) if d else 0.0

    agg = {
        "n": stats["n"],
        "completeness": {
            "get_context_contains_all_after_dedup_pct": pct(stats["complete_ok"], stats["n"]),
            "total_missing_chunks": stats["miss_total"],
            "missing_case_count": len(stats["miss_id_cases"]),
            "missing_detail": stats["miss_id_cases"][:8],
        },
        "retention": {
            "prod_dedup_retention_pct": round(avg(stats["retention_prod"]) * 100, 1),
            "pure_dedup_retention_pct": round(avg(stats["retention_pure"]) * 100, 1),
        },
        "lengths": {
            "ctx_prod_avg": round(avg(stats["ctx_prod"]), 1),
            "ctx_prod_p95": round(_pct(stats["ctx_prod"], 95), 1),
            "ctx_pure_avg": round(avg(stats["ctx_pure"]), 1),
            "prompt_avg": round(avg(stats["prompt_len"]), 1),
            "prompt_p95": round(_pct(stats["prompt_len"], 95), 1),
            "evidence_over_prompt_pct": round(avg(stats["ratio_ev_prompt"]) * 100, 1),
            "overlong_gt6000": sum(1 for v in stats["ctx_prod"] if v > 6000),
            "overlong_gt8000": sum(1 for v in stats["ctx_prod"] if v > 8000),
        },
        "truncation": {
            "cases_losing_gt_core": len(stats["truncated_cases"]),
            "detail": stats["truncated_cases"][:8],
        },
        "merge_bloat": {
            "dup_block_case_count": len(stats["dup_block_cases"]),
            "note": "reason 用 organized.get_context() 二选一，非叠加；仅在证据串内同内容指纹>1 记为可疑",
            "detail": stats["dup_block_cases"][:5],
        },
    }
    print("\n========== S5 证据组织量化 ==========")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    json.dump({"agg": agg, "stats": {k: (v if isinstance(v, int) else v[:200]) for k, v in stats.items()}},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[SAVED]", OUT)


def _pct(xs, q):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(round(len(xs) * q / 100)))]


if __name__ == "__main__":
    main()

