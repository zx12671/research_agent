"""
快速诊断：运行单个 PubmedQA 样本的 MAIRAG，打印完整堆栈
"""
import os
os.environ['DEEPSEEK_API_KEY'] = 'sk-ed0d07bcdbfe4d9e99f8653760954022'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
import json

LINS_MAIN_PATH = os.path.join(os.getcwd(), "LINS-main")
if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)

from model.model_LINS import LINS

# 1. 初始化 LINS
print(">>> 初始化 LINS...")
lins = LINS(
    LLM_name='deepseek-chat',
    assistant_LLM_name='deepseek-chat',
    retriever_name='BGE',
    DeepSeek_keys=os.environ.get('DEEPSEEK_API_KEY'),
    database_name='pubmed'
)
print("  LINS 初始化成功 ✓")

# 2. 检查 self.retriever 的类型和属性
print(f"\n>>> 诊断 self.retriever:")
print(f"  type(lins.retriever) = {type(lins.retriever)}")
print(f"  type(lins.retriever).__name__ = {type(lins.retriever).__name__}")
print(f"  dir(lins.retriever)[:20] = {dir(lins.retriever)[:20]}")

# 3. 测试 encode
print(f"\n>>> 测试 encode:")
try:
    emb = lins.retriever.encode("test question")
    print(f"  encode 成功! len={len(emb)}")
except Exception as e:
    import traceback
    print(f"  encode 失败: {e}")
    traceback.print_exc()

# 4. 测试 MAIRAG
print(f"\n>>> 测试 MAIRAG:")
try:
    response, urls, passages, history, sub_qs = lins.MAIRAG(
        question="Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
        if_PRA=True,
        if_SKA=True,
        if_QDA=True,
    )
    print(f"  MAIRAG 成功! response[:100] = {str(response)[:100]}")
except Exception as e:
    import traceback
    print(f"  MAIRAG 失败: {type(e).__name__}: {e}")
    traceback.print_exc()

# 5. 测试 get_passages 
print(f"\n>>> 测试 get_passages:")
try:
    ret = lins.get_passages(
        question="Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
        topk=5,
        if_split_n=False,
        recall_top_k=20
    )
    if ret:
        print(f"  get_passages 成功! 获取 {len(ret['texts'])} 条结果")
        print(f"  URLs: {ret['urls']}")
    else:
        print(f"  get_passages 返回 None")
except Exception as e:
    import traceback
    print(f"  get_passages 失败: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n>>> 诊断完成")
