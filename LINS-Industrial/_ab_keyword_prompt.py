# -*- coding: utf-8 -*-
"""
_ab_keyword_prompt.py  —— 小样本 A/B 消融：验证「KEYWORD_MAP 得到的 TaskType」对回答是否有正向帮助。

核心问题：
    KEYWORD_MAP 把 question 判成 8 类 task（compare/diagnosis/calculation/selection/
    procedure/interpret/explain/general）。该 task 会经 Planner 决定 reasoning_type 与
    workflow_steps，进而让 Solver 选用【专门的推理 system prompt】（solver._get_reasoning_system_prompt）。

    本实验纯化隔离这一步：对同一批样本，用【同一份证据 (knowledge_text)】，
    仅切换「推理 prompt」：
        Arm A = 按 KEYWORD_MAP 的 task 走专门的 reasoning prompt
        Arm B = 一律走 general 推理 prompt（基线）
    再用 IndustryBench 的 JudgeLLM 对两版回答打分，对比净影响。

    全程复用 solver.py 的真实 prompt 资产 + planner 的真实 task_configs，
    不修改任何 agentic 源码，仅用少量样本（默认 16）快速验证。

"""
import os, sys, json, csv, collections

CUR = os.path.dirname(os.path.abspath(__file__))
if CUR not in sys.path:
    sys.path.insert(0, CUR)

API_KEY = "<DEEPSEEK_API_KEY_FROM_ENV>"
CSV_PATH = os.path.join(CUR, "data", "industrybench", "huggingface_dataset.csv")
OUT_JSON = os.path.join(CUR, "results", "ab_keyword_prompt_result.json")

BASE_URL = "https://api.deepseek.com"   # DeepSeek 端点，否则 openai 默认端点会 401

# ---------- LLM client（DeepSeek 官方端点，OpenAI 兼容形态） ----------
def make_llm():
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return client


# ---------- 加载数据 ----------
def load_all():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    return [r for r in rows if r.get("question", "") and r.get("answer", "")]


# ---------- Arm B：general 基线（复用 solver._solve_general 的 prompt） ----------
def build_b_prompt(question, evidence):
    sysb = ("You are a knowledgeable industrial domain expert. Answer the question "
            "using the provided evidence context. Cite evidence using citation IDs.")
    ctx = evidence
    usrb = f"Question: {question}\n\nEvidence Context:\n{ctx}"
    return sysb, usrb

# ---------- Arm A：按 KEYWORD_MAP 的 task 走专门推理 prompt ----------
def build_a_prompts(question, evidence, task_value, reasoner, task_steps):
    """返回 (system_prompt, user_prompt)。reasoner: TaskSolver 实例（提供 _get_reasoning_system_prompt）"""
    system_prompt = reasoner._get_reasoning_system_prompt(reasoning_type_for(task_value))

    workflow_desc = "\n".join(f"  Step {i+1}. {s}" for i, s in enumerate(task_steps))
    citation = ("IMPORTANT: You MUST cite specific evidence chunks using their citation IDs (e.g., [1], [2]). "
                "Each claim must be supported by at least one citation.\n")
    user_prompt = f"""Question: {question}

Evidence Context:
{evidence}

Reasoning Workflow:
Please follow these steps to reason through the answer:
{workflow_desc}

{citation}

Provide your answer following the reasoning workflow above."""
    return system_prompt, user_prompt

_REASONING_TYPE_MAP = {
    "comparison": "compare",
    "diagnosis": "diagnosis",
    "calculation": "calculation",
    "selection": "selection",
    "procedure": "procedure",
    "standard_interpretation": "interpret",
    "explanation": "explain",
    "general": "general",
}

def reasoning_type_for(task_value):
    return _REASONING_TYPE_MAP.get(task_value, "general")


# planner._fallback_graph 的真实 workflow steps（task_configs[task]["steps"]），内联复用源码
_TASK_STEPS = {
    "comparison": [
        "Identify the specific items being compared",
        "Extract evidence for each item from the documents",
        "Analyze similarities between the items",
        "Analyze differences between the items",
        "Draw a balanced conclusion with recommendation",
    ],
    "diagnosis": [
        "Identify symptoms from the question",
        "Extract possible root causes from evidence",
        "Match evidence to each possible cause",
        "Determine the most likely cause",
        "Generate remediation recommendation",
    ],
    "calculation": [
        "Extract variables and parameters from the question",
        "Identify applicable formulas from evidence",
        "Perform step-by-step calculation",
        "Verify units and reasonableness of the result",
        "State final answer with interpretation",
    ],
    "selection": [
        "Identify requirements and constraints from the question",
        "List candidate options from evidence",
        "Evaluate each candidate against the requirements",
        "Perform trade-off analysis",
        "Make recommendation with justification",
    ],
    "procedure": [
        "Identify the procedure or process being described",
        "Identify prerequisites and safety measures",
        "Describe each step in the correct sequence",
        "Highlight critical parameters and quality checks",
        "Summarize the complete procedure",
    ],
    "standard_interpretation": [
        "Identify the applicable standards",
        "Extract key requirements from evidence",
        "Interpret each requirement in practical terms",
        "Note compliance guidance",
        "Provide practical examples",
    ],
    "explanation": [
        "Define the concept or principle",
        "Explain the underlying mechanism",
        "Provide industrial context and applications",
        "Give concrete examples",
        "Summarize key takeaways",
    ],
    "general": [
        "Understand the question",
        "Extract relevant evidence",
        "Synthesize an answer from evidence",
        "Verify completeness and accuracy",
        "Provide the final answer with citations",
    ],
}

def _chat(client, system, user, max_tokens=2048):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def main(N=6):
    # 1) 数据
    rows = load_all()
    print(f"[INFO] 数据总样本 {len(rows)}")

    # 2) KEYWORD_MAP 分类
    from agentic.analyzer import TaskAnalyzer
    analyzer = TaskAnalyzer(llm_client=None)  # 离线关键词分类，无需 LLM

    # 3) TaskSolver（提供真实的专门 system prompt 字典）
    from agentic.solver import TaskSolver
    llm = make_llm()
    solver = TaskSolver(llm_client=llm, model_name="deepseek-chat", temperature=0.1)

    # 4) 抽小样本：优先 KEYWORD_MAP 判到非 general 类的（才有 A/B 差异体现）
    tagged = []
    for r in rows:
        try:
            ta = analyzer.analyze(question=r["question"], format=r.get("_format", ""))
            tv = ta.task.value
        except Exception as e:
            tv = "general"
        tagged.append((r, tv))

    # 聚一下非 general 各类，均匀抽：按 order 循环取每类的下一个样本，直到填满 N
    by_task = collections.defaultdict(list)
    for r, tv in tagged:
        by_task[tv].append(r)
    order = ["comparison", "diagnosis", "selection", "explanation", "procedure",
             "standard_interpretation", "general"]
    picked = []
    idx = {t: 0 for t in order}
    progressed = True
    while len(picked) < N and progressed:
        progressed = False
        for t in order:
            pool = by_task.get(t, [])
            if idx[t] < len(pool):
                picked.append(pool[idx[t]])
                idx[t] += 1
                progressed = True
            if len(picked) >= N:
                break


    print(f"[INFO] 抽取 {len(picked)} 个样本")

    print(f"{'ID':<8}{'task(KM)':<22}{'fmt':<6}")
    for r in picked:
        ta = analyzer.analyze(question=r["question"], format=r.get("_format", ""))
        tv = ta.task.value
        print(f"{str(r.get('id',''))[:8]:<8}{tv:<22}{r.get('_format',''):<6}")

    # 5) Judge 判分：复用 scorer 官方 RUBRIC prompt 走 DeepSeek
    sys.path.insert(0, os.path.join(CUR, "metrics"))
    from metrics.industrybench_scorer import ScoringRubric

    def judge_score(q, ref, ans):
        prompt = ScoringRubric.format_prompt(q, ref, ans)
        out = _chat(llm, "You are a strict evaluator. Reply only with the JSON block.", prompt, max_tokens=400)
        score, expl = ScoringRubric.parse_response(out)
        return int(score)

    results = []
    wins = 0; ties = 0; loses = 0
    diffs = []
    for r in picked:
        q = r["question"]; ref = r["answer"]; ev = r["knowledge_text"] or ""
        ta = analyzer.analyze(question=q, format=r.get("_format", ""))
        tv = ta.task.value
        rid = r.get("id", "")

        # Arm B
        sb, ub = build_b_prompt(q, ev)
        ans_b = _chat(llm, sb, ub)
        # Arm A
        sa, ua = build_a_prompts(q, ev, tv, solver, _TASK_STEPS.get(tv, _TASK_STEPS["general"]))
        ans_a = _chat(llm, sa, ua)

        # 打 A、B 分（0-3 rubric）
        ra = judge_score(q, ref, ans_a)
        rb = judge_score(q, ref, ans_b)

        diffs.append(ra - rb)

        if ra > rb: wins += 1
        elif ra == rb: ties += 1
        else: loses += 1

        results.append({
            "id": rid, "task_km": tv, "format": r.get("_format", ""),
            "score_A_taskaware": ra, "score_B_general": rb, "delta": ra - rb,
            "ans_A": ans_a[:2000], "ans_B": ans_b[:2000],
            "question": q[:150], "ref": ref[:150],
        })
        print(f"[{rid}] task={tv:<8} A={ra} vs B={rb} delta={ra-rb:+.1f}")

    # 6) 汇总
    n = len(results)
    avg_a = sum(x["score_A_taskaware"] for x in results) / n
    avg_b = sum(x["score_B_general"] for x in results) / n
    avg_delta = sum(diffs) / n
    summary = {
        "N": n,
        "avg_A_taskaware": round(avg_a, 2),
        "avg_B_general": round(avg_b, 2),
        "avg_delta": round(avg_delta, 2),
        "wins_A": wins, "ties": ties, "loses_A": loses,
        "majority": "A(task-aware)更好" if wins > loses else ("B(general)更好" if loses > wins else "平手/需更大样本"),
    }
    print("\n===== A/B 汇总 (KEYWORD_MAP task 对回答净效果) =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] 结果已保存: {OUT_JSON}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": results}, f, ensure_ascii=False, indent=2, default=str)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)

    args = ap.parse_args()
    main(N=args.N)
