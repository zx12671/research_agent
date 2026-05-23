import json

with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

total = len(lines)
correct = 0
wrong = 0
no_gt = 0

for i, line in enumerate(lines):
    data = json.loads(line)
    status = data.get('is_correct', None)
    
    ked = data.get('ked_retrieved_count', 0)
    ret = data.get('retrieved_count', 0)
    gt = data.get('ground_truth', '?')
    pred = data.get('options_response', '?')
    
    if status is True:
        correct += 1
        tag = "CORRECT"
    elif status is False:
        wrong += 1
        tag = "WRONG"
    else:
        no_gt += 1
        tag = "NO_GT"
    
    print(f"[{i+1:2d}] ID={data['id']} | GT={gt:8s} | pred={pred} | {tag} | ked={ked:2d} | passages={ret:2d}")

print(f'\n总计: {total} 条')
print(f'正确: {correct}, 错误: {wrong}, 无GT: {no_gt}')
if correct + wrong > 0:
    print(f'准确率: {correct/(correct+wrong)*100:.1f}%')
