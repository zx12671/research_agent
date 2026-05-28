import os
# os.environ['OPEN_API_KEY'] = 'sk-your-openai-key-here'  # 填入有效的 OpenAI API Key 用于检索
# os.environ['QIANWEN_KEY'] = '18cdc8814e87c3ddac0f458451b1fdf7:ZGFiMDEwMTc3ZTFlOTE3NTlkZTM2ZGY4'  # 填入你的千问 API Key

from model.model_LINS import LINS

# 初始化 LINS（默认使用 GPT-4o、text-embedding-3-large、PubMed 数据库）
# lins = LINS()
lins = LINS(
    LLM_name='qwen-turbo',
    assistant_LLM_name='qwen-turbo',
    retriever_name='text-embedding-3-large',  # 需要有效的 OpenAI Key
    LLM_keys=os.environ.get('OPENAI_API_KEY'),
    QianWen_keys=os.environ.get('QIANWEN_KEY'),
    database_name='pubmed'
)

# 1. 多轮对话
response, history = lins.chat(question="What is BCR-ABL1?", history=None)
print("Chat response:", response)

# 2. 生成带引用的回答（MAIRAG）
response, urls, passages, history, sub_qs = lins.MAIRAG(
    question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression?"
)
print("MAIRAG response:", response)
print("URLs:", urls)

# 3. 循证推荐（AEBMP）
response, urls, passages, history, pico = lins.AEBMP(
    PICO_question="For Parkinson's disease, whether prasinezumab showed greater benefits on motor signs progression?",
    if_guidelines=False,
    patient_information="A 76-year-old female..."   # 见 test.ipynb 中的示例
)
print("EBM response:", response)

# 4. 给患者解释医嘱（MOEQA）
medical_explanations, qa_answer = lins.MOEQA(
    if_QA=True,
    if_explanation=True,
    question="Why do I have ischemic bowel disease?",
    explain_text="Preliminary Diagnosis: Ischemic Bowel Disease...",
    patient_information="Gender: Female, Age: 53 years..."
)
print("Explanations:", medical_explanations)
print("QA Answer:", qa_answer)