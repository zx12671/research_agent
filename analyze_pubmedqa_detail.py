import json

with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 分析检索段落内容对判断的影响
for line in lines:
    data = json.loads(line)
    is_correct = data.get('is_correct')
    if is_correct is False:  # 只分析错误样本
        print(f"\n{'='*80}")
        print(f"问题: {data['question']}")
        print(f"Ground Truth: {data['ground_truth']}")
        print(f"模型预测: {data['options_response']}")
        print(f"LLM选项回答: {data.get('options_response', 'N/A')}")
        
        # 显示 LLM 的生成响应(如果有)
        llm_response = data.get('llm_response', '')
        if llm_response:
            print(f"\n--- LLM完整响应 ---")
            print(llm_response[:500])
        
        # 检查所有检索到的段落
        passages = data.get('mAirag_passages', [])
        print(f"\n--- 检索到的 {len(passages)} 个段落 ---")
        for j, p in enumerate(passages):
            content = p.get('content', p.get('text', str(p)))[:200]
            url = p.get('url', p.get('source', 'N/A'))
            print(f"  段落[{j}]: {url}")
            print(f"  内容: {content}...")
            print()
        
        # 是否包含PubMedQA原始答案证据的线索
        gt = data['ground_truth']
        print(f"\n--- 段落中是否包含与GT '{gt}' 相关的证据 ---")
        evidence_found = False
        for j, p in enumerate(passages):
            content = p.get('content', p.get('text', str(p))).lower()
            if gt.lower() in content:
                evidence_found = True
                print(f"  段落[{j}] 中包含 GT 关键词!")
        if not evidence_found:
            print(f"  未在段落中找到 GT 关键词")
