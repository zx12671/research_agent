import json

# ========== 分析 LINS-Full 的 PubMedQA 结果 ==========
with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f.readlines()]

total = len(lines)
correct = sum(1 for l in lines if l.get('is_correct') is True)
wrong = sum(1 for l in lines if l.get('is_correct') is False)
no_gt = sum(1 for l in lines if 'is_correct' not in l)

print(f"PubMedQA 评估结果 (LINS-Full)")
print(f"总样本: {total}")
print(f"正确: {correct} ({correct/total*100:.1f}%)")
print(f"错误: {wrong} ({wrong/total*100:.1f}%)")
print(f"无GT: {no_gt}")

# 错误案例分析
wrong_cases = [l for l in lines if l.get('is_correct') is False]
print(f"\n{'='*80}")
print(f"错误案例详细分析 ({len(wrong_cases)} 个)")
print(f"{'='*80}")

for i, w in enumerate(wrong_cases):
    print(f"\n--- 错误案例 [{i+1}] ---")
    print(f"问题: {w['question'][:80]}...")
    print(f"GT: {w['ground_truth']}, 预测: {w['options_response']}")
    
    # 分析检索质量
    passages = w.get('mAirag_passages', [])
    urls = w.get('mAirag_urls', [])
    
    print(f"\n检索段落数: {len(passages)}")
    print(f"引用文献数: {len(urls)}")
    
    # 分析 LLM 的完整回答
    ma_resp = w.get('mAirag_response', '')
    if ma_resp:
        # 判断LLM最终倾向
        resp_lower = ma_resp.lower()
        yes_indicators = ['yes', 'is', 'do play', 'does play', 'are associated', 'suggests that', 'support', 'confirm', 'indeed']
        no_indicators = ['no evidence', 'no direct', 'does not', 'do not', 'not suggest', 'not support', 'no significant']
        
        yes_score = sum(1 for ind in yes_indicators if ind in resp_lower)
        no_score = sum(1 for ind in no_indicators if ind in resp_lower)
        
        print(f"回答倾向: yes_score={yes_score}, no_score={no_score}")
        
        # 看LLM是否在回答中表达了不确定
        uncertainty_words = ['however', 'but', 'unclear', 'uncertain', 'limited', 'further research', 'more studies']
        has_uncertainty = any(wd in resp_lower for wd in uncertainty_words)
        print(f"回答含不确定性词: {has_uncertainty}")
        
        # 打印回答的结尾部分（通常包含最终结论）
        last_300 = ma_resp[-300:]
        print(f"回答结尾(最后300字): {last_300}")

# ========== 与 LLM Baseline 对比 ==========
print(f"\n{'='*80}")
print(f"与 LLM Baseline 对比")
print(f"{'='*80}")

try:
    with open('eval_results_llm_baseline/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
        baseline_lines = [json.loads(line) for line in f.readlines()]
    
    baseline_correct = sum(1 for l in baseline_lines if l.get('is_correct') is True)
    baseline_wrong = sum(1 for l in baseline_lines if l.get('is_correct') is False)
    baseline_total = len(baseline_lines)
    print(f"LLM Baseline (纯DeepSeek无RAG):")
    print(f"总样本: {baseline_total}, 正确: {baseline_correct} ({baseline_correct/baseline_total*100:.1f}%), 错误: {baseline_wrong}")
    
    # 检查哪些案例在baseline中正确但在LINS中错误
    for w in wrong_cases:
        wid = w['id']
        for b in baseline_lines:
            if b.get('id') == wid and b.get('is_correct') is True:
                print(f"\n  [退化案例] {wid}: LINS错误(预测={w['options_response']}), 但纯LLM正确")
                # 进一步分析可能是检索引入了噪音
                print(f"  问题: {w['question'][:60]}...")
                if w.get('mAirag_passages'):
                    # 检查检索内容是否相关
                    pass_text = ' '.join(w['mAirag_passages'][:3])
                    if w['ground_truth'].lower() == 'no' and ('yes' in pass_text.lower() or 'significant' in pass_text.lower()):
                        print(f"  推测: 检索到的文献可能与问题相关，但方向偏向\"Yes\"")
                    if w['ground_truth'].lower() == 'yes':
                        # 检查检索到的文献是否充分
                        if len(w.get('mAirag_urls', [])) < 2:
                            print(f"  推测: 检索到的文献数量不足")
                        if w.get('mAirag_response', '') and 'no direct' in w.get('mAirag_response', '').lower():
                            print(f"  推测: 检索内容误导了LLM (检索到了不充分的证据)")

except FileNotFoundError:
    print("无 LLM baseline 对比数据")
