"""
config.py: 实验管线全局配置

所有实验共享的路径、模型、API Key 等配置。
"""

import os

# ============================================================
# 1. 项目根路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))

# ============================================================
# 2. 数据路径
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'industrybench')
KNOWLEDGE_CORPUS_DIR = os.path.join(PROJECT_ROOT, 'knowledge_corpus')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiments')

# ============================================================
# 3. 模型配置
# ============================================================
LLM_NAME = "deepseek-chat"
EMBEDDING_MODEL = "BGE"
RETRIEVER_NAME = "BGE"
TOP_K = 10
RECALL_TOP_K = 50

# ============================================================
# 4. API Key
# ============================================================
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '<DEEPSEEK_API_KEY_FROM_ENV>')
os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_KEY

# ============================================================
# 5. CSV 文件名
# ============================================================
INDUSTRYBENCH_CSV = os.path.join(DATA_DIR, 'huggingface_dataset.csv')

# ============================================================
# 6. 路径注册（确保 import 可用）
# ============================================================
def register_paths():
    """注册项目路径到 sys.path"""
    import sys
    for p in [PROJECT_ROOT, LINS_MAIN_PATH]:
        if p not in sys.path:
            sys.path.insert(0, p)
