# -*- coding: utf-8 -*-
"""
e2e_agentic_ef_backfill_count.py: 仅统计 ef 整档回捞的真实回捞块数（不跑 agentic/LLM）。
供 e2e_agentic_ef 汇总使用：修正运行脚本中 n_extra 因 citation 标记误判为 0 的问题。
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _e2e_agentic_ef import EFExternalRetriever, pick_rows, FOCUS
from experiments.config import register_paths
register_paths()

def main():
    selected = pick_rows()
    retriever = EFExternalRetriever()
    out = {}
    for pre, (flag, note, row_m) in selected.items():
        q = row_m.get("question", "")
        r = retriever.retrieve(q, k=10)
        n_extra = sum(1 for d in r.documents if "*ef" in (d.citation or ""))
        out[pre] = {"flag": flag, "note": note, "n_total": len(r.documents), "n_extra": n_extra}
        print(f"[{flag}] {note:<10} total={len(r.documents)} backfill={n_extra}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2e_agentic_ef_backfill.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("written.")


if __name__ == "__main__":
    main()
