import json

with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    if data.get('is_correct') is not False:
        continue
    
    print(f"\n{'='*80}")
    print(f"问题: {data['question']}")
    print(f"Ground Truth: {data['ground_truth']}")
    print(f"模型预测: {data['options_response']}")
    
    # LLM的完整响应
    ma_resp = data.get('mAirag_response', '')
    if ma_resp:
        print(f"\n--- LLM完整响应 (mAirag_response) ---")
        print(ma_resp[:1000])
    else:
        print("\n--- 无mAirag_response字段 ---")
    
    # 分析检索到的文献
    urls = data.get('mAirag_urls', [])
    print(f"\n--- 检索到的 {len(urls)} 篇文献 ---")
    for url in urls:
        pmid = url.split('/')[-1]
        print(f"  PMID: {pmid}")
    
    # 分析检索到的段落（前200字符）
    passages = data.get('mAirag_passages', [])
    print(f"\n--- 检索到的 {len(passages)} 个段落 ---")
    for j, p in enumerate(passages):
        print(f"  段落[{j}]: {p[:200]}...")
        print()
