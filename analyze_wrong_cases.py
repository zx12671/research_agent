import json

with open('eval_results_lins_full/pubmedqa_details.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

wrong_cases = []
for line in lines:
    data = json.loads(line)
    if data.get('is_correct') == False:
        wrong_cases.append(data)

print(f"共 {len(wrong_cases)} 个错误案例\n")

for data in wrong_cases:
    print("=" * 80)
    print(f"ID: {data['id']}")
    print(f"问题: {data['question']}")
    print(f"Ground Truth: {data['ground_truth']}")
    print(f"模型预测: {data['options_response']}")
    print(f"retrieved_count: {data['retrieved_count']}")
    print(f"ked_retrieved_count: {data['ked_retrieved_count']}")
    
    # 分析检索到的文章的PubMed IDs
    print(f"检索到的URL数: {len(data.get('mAirag_urls', []))}")
    print(f"检索到的段落数: {len(data.get('mAirag_passages', []))}")
    
    # 检查是否包含关键文献（该问题的原始文献）
    orig_pmid = data['id']
    ked_urls = data.get('ked_urls', [])
    all_urls = data.get('mAirag_urls', [])
    
    # 关键文献在KED中是否出现
    origin_in_ked = any(orig_pmid in url for url in ked_urls)
    origin_in_all = any(orig_pmid in url for url in all_urls)
    
    print(f"关键文献({orig_pmid})在KED结果中: {'是' if origin_in_ked else '否'}")
    print(f"关键文献在最终检索结果中: {'是' if origin_in_all else '否'}")
    
    # 统计 ked_urls 中非原文的额外文献
    extra_in_ked = [u for u in ked_urls if orig_pmid not in u]
    extra_in_all = [u for u in all_urls if orig_pmid not in u]
    print(f"KED额外检索到的文献数: {len(extra_in_ked)}")
    print(f"最终检索额外文献数: {len(extra_in_all)}")
    
    print()

analysis = """
错误分析:
1. ID=16418930 (GT=no, pred=A): 检索了14篇文献，KED检索了19篇！大量噪声文献导致模型误判。
   关键文献(16418930)不在任何检索结果中？检查...原始研究中Landolt C和Snellen E在斜视弱视中没有差异。

2. ID=9488747 (GT=yes, pred=B): 只检索到1篇文献且就是原文，但模型选了B而不是A。
   PubMedQA中答案是"yes"，模型判错了，但答案格式有A/B选项问题。

3. ID=26037986 (GT=maybe, pred=A): 检索到3篇文献，均与主题相关。模型得出肯定结论但GT是"maybe"。
   模型过于肯定。

4. ID=26852225 (GT=no, pred=A): 检索到2篇文献。第二篇明确说"correction...is not a necessary tool"，但模型判A。
   模型没有正确理解原文结论。
"""

print(analysis)
