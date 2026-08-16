"""
eval_rag.py: RAG (检索增强生成) 评测脚本

评测在 IndustryBench 数据集上，使用 LangChain + 本地 / 云端 LLM 的 RAG 方案。

流程:
1. 加载 IndustryBench 测试题
2. 从知识库检索相关文档 (IndustrialRetriever)
3. 构建 Prompt（上下文 + 问题）
4. LLM 生成答案
5. 评测指标计算 (EM, F1, BLEU, ROUGE-L, MEDQA Acc)
"""
import os
import sys
import json
import logging
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 1. 配置
# ============================================================

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 2. RAG Evaluator
# ============================================================

class RAGEvaluator:
    """
    RAG 评测器。
    
    支持多种 LLM 后端:
    - "deepseek": DeepSeek Chat API
    - "openai": OpenAI API
    - "local": 本地模型 (通过 Ollama)
    """
    
    def __init__(self, retriever=None, llm_type: str = "deepseek",
                 k: int = 5, chunk_size: int = 512,
                 model_name: str = "deepseek-chat"):
        """
        Args:
            retriever: 已有的 IndustrialRetriever 实例 (None 则自动创建)
            llm_type: LLM 类型
            k: 检索返回的文档数
            chunk_size: 检索块大小
            model_name: 模型名称
        """
        self.k = k
        self.chunk_size = chunk_size
        self.llm_type = llm_type
        self.model_name = model_name
        self.llm = None
        
        # 初始化检索器
        if retriever is not None:
            self.retriever = retriever
        else:
            from retrieval.retriever import IndustrialRetriever
            self.retriever = IndustrialRetriever()
            # 尝试加载已有索引
            self._try_load_index()
    
    def _try_load_index(self):
        """尝试加载已有索引"""
        index_dir = os.path.join(self.retriever.project_root, 'knowledge_corpus', 'index')
        if os.path.exists(index_dir):
            faiss_files = [f for f in os.listdir(index_dir) if f.endswith('.faiss')]
            if faiss_files:
                index_path = os.path.join(index_dir, sorted(faiss_files)[-1])
                logger.info(f"加载索引: {index_path}")
                self.retriever.load_index(index_path)
                return
        logger.warning("未找到已存在的索引。请先运行 build_pipeline()")
    
    def _init_llm(self):
        """初始化 LLM"""
        if self.llm is not None:
            return
        
        if self.llm_type == "deepseek":
            try:
                from langchain_deepseek import ChatDeepSeek
                self.llm = ChatDeepSeek(
                    model=self.model_name,
                    temperature=0.1,
                    max_tokens=512,
                )
                logger.info(f"LLM initialized: DeepSeek {self.model_name}")
            except ImportError:
                logger.warning("langchain_deepseek not installed. Using basic API call.")
                self.llm_type = "deepseek_api"
        elif self.llm_type == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model_name or "gpt-4o-mini",
                temperature=0.1,
                max_tokens=512,
            )
        elif self.llm_type == "local":
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=self.model_name or "qwen2.5:7b",
                    temperature=0.1,
                    num_predict=512,
                )
            except ImportError:
                logger.error("langchain-ollama not installed")
                raise
        else:
            raise ValueError(f"Unknown llm_type: {self.llm_type}")
    
    def _build_rag_prompt(self, question: str, options: List[str],
                          context_docs: List[Dict]) -> str:
        """
        构建 RAG Prompt。
        
        Args:
            question: 问题
            options: 选项列表
            context_docs: 检索到的文档列表
        
        Returns:
            prompt: 完整提示
        """
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get('content', '')
            source = doc.get('title', doc.get('source', f'文档{i+1}'))
            context_parts.append(f"[{i+1}] 来源: {source}\n{content}")
        context = "\n\n".join(context_parts)
        
        # 构建选项文本
        option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        
        # 构建完整 Prompt
        prompt = f"""你是一个专业的工业知识问答助手。请根据提供的参考文档回答以下问题。

参考文档:
{context}

问题: {question}
选项:
{option_text}

请直接输出对应的选项字母 (A, B, C, D 等)，不需要输出其他内容。
如果无法从参考文档中找到答案，请选择最合理的选项。

答案:"""
        
        return prompt
    
    def _extract_answer(self, response: str, num_options: int) -> str:
        """
        从 LLM 回复中提取答案。
        
        Returns:
            选项字母 (A, B, C, D...)
        """
        response = response.strip()
        
        # 匹配单个字母
        letters = [chr(65+i) for i in range(num_options)]
        for letter in letters:
            pattern = rf'^\s*{letter}\s*$'
            if re.match(pattern, response, re.IGNORECASE):
                return letter.upper()
            if response.upper().startswith(letter) and len(response.strip()) <= 3:
                return letter.upper()
        
        # 匹配 "答案是 X"
        match = re.search(r'答案[是为]:?\s*([A-D])', response)
        if match:
            return match.group(1)
        
        # 匹配选项内容
        for i in range(num_options):
            if response.find(chr(65+i)) >= 0:
                return chr(65+i)
        
        return "Unknown"
    
    def answer_single(self, question: str, options: List[str]) -> Tuple[str, str, float]:
        """
        回答单个问题。
        
        Returns:
            (prediction, explanation, latency)
        """
        start_time = time.time()
        
        # Step 1: 检索
        docs = self.retriever.retrieve(question, k=self.k)
        
        # Step 2: 构建 Prompt + LLM 推理
        self._init_llm()
        prompt = self._build_rag_prompt(question, options, docs)
        
        if self.llm_type in ("deepseek", "openai", "local"):
            response = self.llm.invoke(prompt)
            answer_text = response.content if hasattr(response, 'content') else str(response)
        else:
            # raw API fallback
            import requests
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
                timeout=60,
            )
            answer_text = resp.json()["choices"][0]["message"]["content"]
        
        # Step 3: 提取答案
        prediction = self._extract_answer(answer_text, len(options))
        latency = time.time() - start_time
        
        return prediction, answer_text, latency
    
    def evaluate(self, test_data: List[Dict], output_path: str = None,
                 max_samples: int = None) -> Dict:
        """
        在测试数据上评测 RAG 性能。
        
        Args:
            test_data: 测试数据列表 (每项含 question, options, answer_idx)
            output_path: 结果保存路径
            max_samples: 最大评测样本数 (None = 全部)
        
        Returns:
            metrics: 评测指标
        """
        if max_samples:
            test_data = test_data[:max_samples]
        
        logger.info(f"开始 RAG 评测: {len(test_data)} 条")
        
        results = []
        correct = 0
        total_latency = 0
        
        for idx, item in enumerate(test_data):
            question = item.get("question", "")
            options = item.get("options", [])
            gold_idx = item.get("answer_idx", item.get("gold", 0))
            gold = chr(65 + gold_idx)
            
            logger.info(f"[{idx+1}/{len(test_data)}] 问题: {question[:50]}...")
            
            try:
                pred, raw_answer, latency = self.answer_single(question, options)
                is_correct = pred == gold
                if is_correct:
                    correct += 1
                total_latency += latency
                
                results.append({
                    "question": question,
                    "options": options,
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_correct,
                    "raw_answer": raw_answer,
                    "latency": latency,
                })
                
                logger.info(f"   Gold={gold}, Pred={pred}, Correct={is_correct}, Lat={latency:.1f}s")
                
            except Exception as e:
                logger.error(f"   Error: {e}")
                results.append({
                    "question": question,
                    "options": options,
                    "gold": gold,
                    "prediction": "Error",
                    "correct": False,
                    "raw_answer": str(e),
                    "latency": 0,
                })
        
        # 计算指标
        n = len(results)
        accuracy = correct / n if n > 0 else 0
        avg_latency = total_latency / n if n > 0 else 0
        
        metrics = {
            "total": n,
            "correct": correct,
            "accuracy": accuracy,
            "avg_latency_s": avg_latency,
            "llm_type": self.llm_type,
            "model_name": self.model_name,
            "k": self.k,
            "chunk_size": self.chunk_size,
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"RAG 评测完成!")
        logger.info(f"  总样本: {n}")
        logger.info(f"  正确: {correct}")
        logger.info(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  平均延迟: {avg_latency:.1f}s")
        logger.info(f"{'='*50}")
        
        # 保存结果
        if output_path:
            output = {
                "metrics": metrics,
                "results": results,
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存: {output_path}")
        
        return metrics


# ============================================================
# 3. 加载 IndustryBench 数据
# ============================================================

def load_industrybench_data(data_path: str = None) -> List[Dict]:
    """加载 IndustryBench 测试数据"""
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, 'data', 'industry_eval_zh.json')
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"已加载 {len(data)} 条 IndustryBench 测试题")
    return data


# ============================================================
# 4. Main
# ============================================================

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 工业知识评测")
    parser.add_argument("--llm", type=str, default="deepseek", choices=["deepseek", "openai", "local"],
                        help="LLM 类型")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="模型名称")
    parser.add_argument("--k", type=int, default=5,
                        help="检索文档数")
    parser.add_argument("--max", type=int, default=None,
                        help="最大评测样本数")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出路径")
    parser.add_argument("--data", type=str, default=None,
                        help="IndustryBench 数据路径")
    
    args = parser.parse_args()
    
    # 加载数据
    test_data = load_industrybench_data(args.data)
    if not test_data:
        return
    
    # 输出路径
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(PROJECT_ROOT, 'results', 
                                    f'rag_eval_{args.llm}_{args.model.replace("/", "_")}_{timestamp}.json')
    
    # 评测
    evaluator = RAGEvaluator(llm_type=args.llm, k=args.k, model_name=args.model)
    metrics = evaluator.evaluate(test_data, output_path=args.output, max_samples=args.max)
    
    # 打印摘要
    print(f"\n{'='*50}")
    print(f"📊 RAG 评测摘要")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
