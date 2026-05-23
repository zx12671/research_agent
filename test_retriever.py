import os
os.environ['DEEPSEEK_API_KEY'] = 'sk-ed0d07bcdbfe4d9e99f8653760954022'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'LINS-main'))

# 只测试 LINS_Retriever 初始化
from model.retriever_model import LINS_Retriever

try:
    ret = LINS_Retriever('BGE', max_thread=100, BGE_encoder_path='./model/retriever/bge/bge-m3')
    print('LINS_Retriever 创建成功')
    print('  type(ret) =', type(ret))
    print('  hasattr retriever:', hasattr(ret, 'retriever'))
    if hasattr(ret, 'retriever'):
        print('  ret.retriever =', type(ret.retriever))
    # 尝试 encode
    emb = ret.encode('test question')
    print('  encode 成功! len=', len(emb))
except Exception as e:
    print('错误:', type(e).__name__, ':', e)
    import traceback
    traceback.print_exc()
