import json

# 详细分析错误案例2（GT=yes, 预测=B(no)）和案例4（GT=no, 预测=A(yes)）
with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f.readlines()]

for w in lines:
    if w.get('is_correct') is not False:
        continue
    
    print(f"\n{'='*80}")
    print(f"ID: {w['id']}")
    print(f"问题: {w['question']}")
    print(f"GT: {w['ground_truth']}, 预测: {w['options_response']}")
    
    # 打印mAirag_passages全文
    passages = w.get('mAirag_passages', [])
    for j, p in enumerate(passages):
        print(f"\n--- 段落[{j}] (全文) ---")
        print(p)
    
    # 打印完整LLM回答
    resp = w.get('mAirag_response', '')
    if resp:
        print(f"\n\n--- mAirag_response 完整内容 ---")
        print(resp)
    
    print(f"\n{'='*80}")
