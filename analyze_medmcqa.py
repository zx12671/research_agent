import json

with open('eval_results_lins_full/medmcqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    data = json.loads(line)
    print(f"--- [{i+1}] ID={data['id']} ---")
    q = data['question']
    print(f"问题(前150字): {q[:150]}")
    print(f"GT: {data.get('ground_truth', '?')}, 预测: {data.get('options_response', '?')}, 正确: {data.get('is_correct')}")
    print(f"检索段落数: {data.get('retrieved_count', 0)}, KED检索: {data.get('ked_retrieved_count', 0)}")
    print(f"URL数: {len(data.get('mAirag_urls', []))}, Passage数: {len(data.get('mAirag_passages', []))}")
    print()
