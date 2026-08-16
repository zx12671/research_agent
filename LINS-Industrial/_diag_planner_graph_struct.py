# -*- coding: utf-8 -*-
"""诊断 execution_graph / execution_nodes 真实结构（探针解析失真的根因）"""
import sys, os, json
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ABOVE = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (_THIS_DIR, _ABOVE):
    if p not in sys.path:
        sys.path.insert(0, p)
from experiments.config import DEEPSEEK_KEY, LLM_NAME
from experiments.exp1_agentic_rag import AgenticRAGEngine, load_question_dataset

print("[INFO] init engine ...")
engine = AgenticRAGEngine(deepseek_key=DEEPSEEK_KEY, model_name=LLM_NAME, verbose=False)
samples = load_question_dataset(os.path.join(_THIS_DIR, "data", "industrybench", "huggingface_dataset.csv"), num_samples=2)
q = samples[0].question
_, _ = engine.answer(q)
pr = engine.last_pipeline_result
prd = vars(pr) if hasattr(pr, "__dict__") else {}
print("\n=== pr full keys ===")
print(sorted(prd.keys()))
for key in ("execution_graph", "graph", "planned_graph", "task_analysis", "second_retrieval_triggers", "node_count"):
    print(f"\n--- prd[{key!r}] ---")
    v = prd.get(key)
    if isinstance(v, (dict, list)):
        print(json.dumps(v, ensure_ascii=False, indent=1, default=str)[:2500])
    else:
        print(repr(v))

print("\n=== execution_nodes (first 3 raw items) ===")
en = prd.get("execution_nodes") or []
print(f"len={len(en)}")
for it in en[:3]:
    print(json.dumps(it, ensure_ascii=False, indent=1, default=str)[:1200])
