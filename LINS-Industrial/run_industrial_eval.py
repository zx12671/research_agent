"""
LINS-Industrial: 工业场景评估主入口脚本
基于 LINS-main 核心框架，用于工业知识问答评估
"""
import os
import sys
import json
import time
from datetime import datetime

# ========== 1. 配置原项目路径 ==========
# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 计算 LINS-main 的路径（与 LINS-Industrial 同级）
LINS_MAIN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'LINS-main'))

# 将 LINS-main 添加到 Python 搜索路径
if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)
    print(f"[INFO] 已添加 LINS-main 路径: {LINS_MAIN_PATH}")

# ========== 2. 配置环境变量 ==========
DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '<DEEPSEEK_API_KEY_FROM_ENV>')
os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_KEY
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ========== 3. 导入 LINS 核心类 ==========
try:
    from model.model_LINS import LINS
    print("[INFO] LINS 核心模块导入成功")
except ImportError as e:
    print(f"[ERROR] 导入 LINS 失败: {e}")
    print(f"[ERROR] 请确认 LINS-main 路径正确: {LINS_MAIN_PATH}")
    sys.exit(1)

# ========== 4. 初始化 LINS 工业版 ==========
print("\n" + "="*70)
print("LINS-Industrial 初始化")
print("="*70)

# 配置工业场景参数
INDUSTRIAL_CONFIG = {
    "LLM_name": "deepseek-chat",
    "assistant_LLM_name": "deepseek-chat",
    "retriever_name": "BGE",  # 或 'text-embedding-3-large'
    "database_name": "pubmed",  # 后续可替换为工业知识库
    "DeepSeek_keys": DEEPSEEK_KEY,
}

try:
    lins = LINS(**INDUSTRIAL_CONFIG)
    print("[SUCCESS] LINS 工业版初始化成功")
    print(f"  - LLM: {INDUSTRIAL_CONFIG['LLM_name']}")
    print(f"  - Retriever: {INDUSTRIAL_CONFIG['retriever_name']}")
    print(f"  - Database: {INDUSTRIAL_CONFIG['database_name']}")
except Exception as e:
    print(f"[ERROR] LINS 初始化失败: {e}")
    sys.exit(1)

# ========== 5. 工业测试查询 ==========
print("\n" + "="*70)
print("执行工业测试查询")
print("="*70)

TEST_QUESTIONS = [
    "What is the recommended torque for an M8 bolt in a structural steel connection?",
    "How to identify a faulty bearing from vibration analysis data?",
    "What are the safety requirements for operating a CNC milling machine?",
]

results = []

for idx, question in enumerate(TEST_QUESTIONS, 1):
    print(f"\n[测试 {idx}/{len(TEST_QUESTIONS)}]")
    print(f"问题: {question}")

    try:
        start_time = time.time()
        response, urls, passages, history, sub_qs = lins.MAIRAG(
            question=question + "\nPlease include citation numbers like [1], [2] in your answer.",
            topk=5,
            if_PRA=True,
            if_SKA=False,
            if_QDA=False,
            if_PCA=False,
            recall_top_k=50
        )
        elapsed = time.time() - start_time

        results.append({
            "question": question,
            "answer": response,
            "sources": urls[:5] if urls else [],
            "retrieved_count": len(passages) if passages else 0,
            "time_elapsed": round(elapsed, 2)
        })

        print(f"[SUCCESS] 回答完成 (耗时: {elapsed:.1f}s)")
        print(f"  引用数: {len(urls) if urls else 0}")
        print(f"  回答预览: {response[:150]}..." if len(response) > 150 else response)

    except Exception as e:
        print(f"[ERROR] 回答失败: {type(e).__name__}: {str(e)}")
        results.append({
            "question": question,
            "error": str(e)[:200]
        })

# ========== 6. 保存结果 ==========
RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(RESULTS_DIR, f"industrial_test_{timestamp}.json")

output_data = {
    "timestamp": timestamp,
    "config": INDUSTRIAL_CONFIG,
    "total_questions": len(TEST_QUESTIONS),
    "success_count": sum(1 for r in results if "error" not in r),
    "results": results
}

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print("测试完成！")
print(f"结果已保存至: {result_file}")
print("="*70)
