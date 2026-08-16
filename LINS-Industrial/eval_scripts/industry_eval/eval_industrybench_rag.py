"""
eval_industrybench_rag.py: IndustryBench RAG 模式评估
绕过 LINS-main 的 get_passages 限制（不识别 industry_kb），
通过直接调用 General_Local_Database 实现手动检索 + LLM 调用

对比基线：
  - quick: 直接注入 knowledge_text (upper bound)
  - closed_book: 无外部知识 (lower bound)
  - rag: 从 industry_kb 数据库检索 (target)
"""
import sys, os, json, time, csv
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
LINS_MAIN_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'LINS-main'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'industrybench')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
QUICK_RESULT_FILE = os.path.join(RESULTS_DIR, 'industrybench_eval_quick_20260715_152317.json')
CLOSED_RESULT_FILE = os.path.join(RESULTS_DIR, 'industrybench_eval_closed_book_20260715_153310.json')

# 用户指定的 DeepSeek API Key
DEEPSEEK_API_KEY = "<DEEPSEEK_API_KEY_FROM_ENV>"
os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if LINS_MAIN_PATH not in sys.path:
    sys.path.insert(0, LINS_MAIN_PATH)

import torch
from model.model_LINS import LINS
from model.retriever_model import LINS_Retriever

# ========== 数据 Schema ==========
@dataclass
class EvalSample:
    id: str
    question: str
    ref_answer: str
    knowledge_text: str = ''
    difficulty: str = 'medium'
    capability: str = ''
    industry_primary: str = ''
    domain: str = ''

# ========== 1. 加载数据 ==========
def load_industrybench_data(csv_path: str, num_samples: Optional[int] = None) -> List[EvalSample]:
    """加载 IndustryBench 数据集"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    samples = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = EvalSample(
                id=row.get('id', '').strip(),
                question=row.get('question', '').strip(),
                ref_answer=row.get('answer', '').strip(),
                knowledge_text=row.get('knowledge_text', '').strip(),
                difficulty=row.get('difficulty', '').strip().lower(),
                capability=row.get('capability', '').strip(),
                industry_primary=row.get('industry_primary', '').strip(),
                domain=row.get('domain', '').strip(),
            )
            if sample.question and sample.ref_answer:
                samples.append(sample)

    if num_samples:
        samples = samples[:num_samples]

    print(f"[INFO] 加载了 {len(samples)} 个有效样本")
    return samples

# ========== 2. 工业知识库检索器 ==========
class IndustryKBRetriever:
    """手动从 industry_kb 嵌入文件检索，绕过 LINS.main 的 database_name 限制"""

    def __init__(self, embedding_path: str):
        print(f"  [KBRetriever] 加载嵌入文件: {embedding_path}")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"嵌入文件不存在: {embedding_path}")

        # 读取所有 embedding
        self.texts = []
        self.embeddings = []
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj['text'])
                self.embeddings.append(obj['embedding'])

        self.embeddings_tensor = torch.tensor(self.embeddings)
        print(f"  [KBRetriever] 加载了 {len(self.texts)} 条知识, embedding dim={self.embeddings_tensor.shape[1]}")

    def retrieve(self, query_embedding, topk=5):
        """余弦相似度检索"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query = torch.tensor(query_embedding).to(device)
        embeds = self.embeddings_tensor.to(device)

        scores = torch.matmul(query, embeds.T)
        topk = min(topk, len(scores))
        topk_indices = torch.topk(scores, topk, dim=0).indices

        results = {"texts": [], "urls": [], "scores": []}
        for i in range(len(topk_indices)):
            idx = topk_indices[i]
            results["texts"].append(self.texts[idx])
            results["urls"].append(f"local/industry_kb/entry/{idx}")
            results["scores"].append(scores[idx].item())

        return results

# ========== 3. 评分模块 ==========
class RAGScorer:
    """RAG 模式评分：规则匹配 + 引用检查"""

    def evaluate(self, question: str, answer: str, ref_answer: str) -> dict:
        """综合评分"""
        raw = self._keyword_match_score(ref_answer, answer)
        has_violation = self._detect_refusal_violation(answer, question)
        adjusted = raw * (0 if has_violation else 1)

        return {
            'raw_score': raw,
            'has_violation': has_violation,
            'adjusted_score': adjusted,
        }

    def _keyword_match_score(self, ref_answer: str, answer: str, verbose: bool = False) -> int:
        """技术词级别关键词匹配评分 0-3
        将中英文文本分割成独立的技术词汇进行匹配
        """
        import re

        if not ref_answer or not answer:
            return 0

        def extract_word_tokens(text: str) -> set:
            """提取独立技术词汇，非整句"""
            # 1. 标准化：去除标点符号
            text = re.sub(r'[，。、；：！？""''（）【】\[\]\(\)\.\,\!\?\:\;「」\*\-_]', ' ', text.lower())
            # 2. 按空格拆分
            tokens = set()
            for t in text.split():
                t = t.strip().strip('[](){}【】\'"')
                if not t:
                    continue
                # 数字+单位组合保留（如 50次, 14毫米, 8倍）
                if re.match(r'^\d+[a-zμnmk°%#x]', t, re.IGNORECASE):
                    tokens.add(t)
                # 纯中文词长度>=2保留
                elif re.match(r'^[\u4e00-\u9fff]+$', t) and len(t) >= 2:
                    tokens.add(t)
                # 英文词长度>=3保留
                elif re.match(r'^[a-z]+$', t) and len(t) >= 3:
                    tokens.add(t)
                # 混合词（字母数字中文混合）保留
                elif len(t) >= 2 and not re.match(r'^\d+$', t):
                    tokens.add(t)
            return tokens

        ref_keywords = extract_word_tokens(ref_answer)
        ans_keywords = extract_word_tokens(answer)

        if not ref_keywords:
            return 0

        matched = ref_keywords & ans_keywords
        match_ratio = len(matched) / len(ref_keywords)

        if verbose:
            print(f"  参考词({len(ref_keywords)}): {sorted(ref_keywords)[:20]}")
            print(f"  匹配词({len(matched)}): {sorted(matched)[:20]}")
            print(f"  比率: {match_ratio:.2f}")

        if match_ratio >= 0.80:
            return 3
        elif match_ratio >= 0.50:
            return 2
        elif match_ratio >= 0.20:
            return 1
        else:
            return 0

    def _detect_refusal_violation(self, answer: str, question: str) -> bool:
        """检测拒绝/回避回答"""
        if not answer:
            return True
        ans_lower = answer.lower()
        refusal_patterns = [
            'i cannot', "i can't", "i'm unable", 'i am unable',
            'i do not have', "i don't have", 'not able to',
            'cannot provide', 'cannot answer', 'cannot generate',
            'sorry', '作为ai', '作为人工智能', '我不能',
            '我无法', '没有相关信息', '无法回答', '无法提供',
            'not specified', 'no information', 'insufficient',
            'my knowledge cutoff', 'training data',
        ]
        return any(p in ans_lower for p in refusal_patterns)

# ========== 4. RAG 评估（手动检索 + LLM） ==========
def evaluate_rag_manual(lins, retriever, sample: EvalSample, scorer: RAGScorer) -> dict:
    """手动 RAG：用 IndustryKBRetriever 检索 + LINS.chat 推理"""
    question = sample.question
    ref_answer = sample.ref_answer

    print(f"  问题: {question[:80]}")

    result = {
        'sample_id': sample.id,
        'question': question,
        'ref_answer': ref_answer,
        'capability': sample.capability,
        'industry': sample.industry_primary,
        'difficulty': sample.difficulty,
        'model_answer_preview': '',
    }

    try:
        # 1. 用 retriever 编码问题
        start_time = time.time()
        query_embedding = lins.retriever.encode(text=question)

        # 2. 从 industry_kb 检索
        retrieved = retriever.retrieve(query_embedding, topk=5)
        passages = retrieved['texts']
        urls = retrieved['urls']

        has_retrieved = len(passages) > 0

        # 3. 构建 RAG prompt
        if has_retrieved:
            references_str = ''.join(
                f"[{ix+1}] {passages[ix]}\n" for ix in range(len(passages))
            )
            prompt = (
                "你是一个工业领域知识问答助手。请根据以下检索到的参考资料回答问题。\n"
                "在回答中标注引用编号如 [1], [2] 以标明信息来源。\n\n"
                f"**检索到的参考资料**:\n{references_str}\n"
                f"**问题**: {question}\n"
                "**回答**:"
            )
        else:
            prompt = (
                f"**问题**: {question}\n"
                "**回答**:"
            )

        # 4. LLM 生成回答
        response, history = lins.chat(question=prompt, history=None)
        elapsed = time.time() - start_time

        result['answer'] = response or ''
        result['model_answer_preview'] = (response or '')[:100]
        result['sources'] = urls[:5]
        result['retrieved_passage_count'] = len(passages)
        result['time_elapsed'] = round(elapsed, 2)
        result['success'] = True

        # 5. 评分
        scores = scorer.evaluate(question, response or '', ref_answer)
        result.update(scores)

        print(f"  检索段落: {len(passages)}")
        print(f"  回答: {(response or '')[:80]}...")
        print(f"  得分: raw={scores['raw_score']}, adj={scores['adjusted_score']}")
        print(f"  耗时: {elapsed:.1f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['error'] = str(e)[:200]
        result['answer'] = ''
        result['success'] = False
        result['raw_score'] = 0
        result['has_violation'] = True
        result['adjusted_score'] = 0
        print(f"  [ERROR] {e}")

    return result

# ========== 5. 主流程 ==========
def run_rag_evaluation(num_samples=5):
    print("=" * 70)
    print("IndustryBench RAG模式评估")
    print("数据库: industry_kb (嵌入检索, 手动 RAG 流程)")
    print("=" * 70)

    # 初始化 retriever (用于编码问题)
    print("\n[1/3] 初始化检索器和 LLM...")
    lins = LINS(
        LLM_name='deepseek-chat',
        assistant_LLM_name='deepseek-chat',
        retriever_name='BGE',
        database_name='none',  # 绕过 LINS 的数据库检查
        DeepSeek_keys=DEEPSEEK_API_KEY,
    )

    # 初始化 industry_kb 检索器
    kb_embedding_path = os.path.abspath(
        os.path.join(PROJECT_ROOT, '..', 'LINS-main', 'add_dataset', 'industry_kb', 'industry_kb_embedding.json')
    )
    kb_retriever = IndustryKBRetriever(kb_embedding_path)

    # 加载数据
    csv_path = os.path.join(DATA_PATH, 'huggingface_dataset.csv')
    print(f"\n[2/3] 加载数据: {csv_path}...")
    samples = load_industrybench_data(csv_path, num_samples)

    # 执行 RAG 评估
    print(f"\n[3/3] 执行 RAG 评估 ({len(samples)} 条)...")
    scorer = RAGScorer()
    results = []

    for idx, sample in enumerate(samples):
        print(f"\n  [{idx+1}/{len(samples)}] id={sample.id}")
        result = evaluate_rag_manual(lins, kb_retriever, sample, scorer)
        results.append(result)

    # 汇总统计
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)

    total = len(results)
    success = sum(1 for r in results if r.get('success'))
    avg_raw = sum(r.get('raw_score', 0) for r in results) / total if total > 0 else 0
    avg_adj = sum(r.get('adjusted_score', 0) for r in results) / total if total > 0 else 0
    violations = sum(1 for r in results if r.get('has_violation'))

    print(f"  总样本: {total}")
    print(f"  成功: {success}")
    print(f"  平均原始分: {avg_raw:.2f}/3")
    print(f"  平均调整分: {avg_adj:.2f}/3")
    print(f"  违规数: {violations}")
    print(f"  平均违规率: {violations/total*100:.1f}%")

    # 分难度统计
    by_diff = {}
    for r in results:
        d = r.get('difficulty', 'unknown')
        if d not in by_diff:
            by_diff[d] = []
        by_diff[d].append(r)

    for diff in ['easy', 'medium', 'hard']:
        if diff in by_diff:
            items = by_diff[diff]
            avg = sum(r.get('adjusted_score', 0) for r in items) / len(items)
            print(f"  [{diff}] avg_adj={avg:.2f} (n={len(items)})")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_file = os.path.join(RESULTS_DIR, f'industrybench_eval_rag_{timestamp}.json')
    output = {
        'timestamp': timestamp,
        'mode': 'rag',
        'database': 'industry_kb',
        'num_samples': len(results),
        'summary': {
            'total': total,
            'success': success,
            'avg_raw_score': round(avg_raw, 3),
            'avg_adjusted_score': round(avg_adj, 3),
            'violation_rate': round(violations/total*100, 1) if total > 0 else 0,
        },
        'results': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

if __name__ == '__main__':
    run_rag_evaluation(num_samples=5)
