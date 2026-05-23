"""验证PubMed连接和搜索功能"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'LINS-main'))

# 1. 基本Bio.Entrez连接测试
print("="*50)
print("1️⃣ Bio.Entrez 直接搜索测试")
from Bio import Entrez, Medline
Entrez.email = '869347360@qq.com'

# 搜索cancer
try:
    handle = Entrez.esearch(db='pubmed', term='cancer', retmax=3, sort='relevance')
    record = Entrez.read(handle)
    print(f'  ✅ cancer -> PMIDs: {record["IdList"]}')
except Exception as e:
    print(f'  ❌ cancer 搜索失败: {e}')

# 搜索长句子
try:
    handle = Entrez.esearch(db='pubmed', term='treatment for lung cancer', retmax=3, sort='relevance')
    record = Entrez.read(handle)
    print(f'  ✅ "treatment for lung cancer" -> PMIDs: {record["IdList"]}')
except Exception as e:
    print(f'  ❌ 搜索失败: {e}')

# 2. 测试database.Pubmed
print("\n" + "="*50)
print("2️⃣ Pubmed.get_data_list 测试")
from model.database import Pubmed
db = Pubmed()

# 短词
result = db.get_data_list('cancer', retmax=3)
print(f'  ✅ "cancer" -> {len(result["texts"])} texts, first AB: {result["texts"][0][:80] if result["texts"] else "EMPTY"}...')

# 长词
result2 = db.get_data_list('treatment for lung cancer', retmax=3)
print(f'  ✅ "treatment for lung cancer" -> {len(result2["texts"])} texts')

# 3. 完整LINS检索流程测试
print("\n" + "="*50)
print("3️⃣ LINS.get_passages 测试")
sys.path.append(os.path.join(os.path.dirname(__file__), 'LINS-main', 'model'))
from model.model_LINS import LINS
lins = LINS(LLM_name='gpt-4o')
lins.database_name = 'pubmed'
result3 = lins.get_passages('cancer treatment', topk=3, recall_top_k=30)
if result3 and result3.get('texts'):
    print(f'  ✅ 检索到 {len(result3["texts"])} 段, URLs: {result3["urls"]}')
else:
    print(f'  ❌ 检索失败或结果为空')

print("\n" + "="*50)
print("🎉 测试完成")
