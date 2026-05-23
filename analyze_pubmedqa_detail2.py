import json

with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 先查看第一个错误案例的数据结构
for line in lines:
    data = json.loads(line)
    if data.get('is_correct') is False:
        print(f"问题: {data['question'][:60]}...")
        print(f"GT: {data['ground_truth']}, 预测: {data['options_response']}")
        
        # 检查mAirag_passages的类型
        passages = data.get('mAirag_passages', [])
        print(f"passages类型: {type(passages)}, 长度: {len(passages)}")
        if len(passages) > 0:
            print(f"passages[0]类型: {type(passages[0])}")
            print(f"passages[0]内容: {str(passages[0])[:300]}")
        
        # 检查mAirag_urls
        urls = data.get('mAirag_urls', [])
        print(f"urls: {urls}")
        
        # 检查llm_response
        llm_resp = data.get('llm_response', 'N/A')
        print(f"llm_response (前300字): {str(llm_resp)[:300]}")
        
        # 检查所有可用的键
        print(f"所有keys: {list(data.keys())}")
        break

    # 只分析第一个错误样本的结构
    if data.get('is_correct') is False:
        break
