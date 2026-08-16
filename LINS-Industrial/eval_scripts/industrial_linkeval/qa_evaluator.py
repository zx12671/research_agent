"""
qa_evaluator.py: 开放式问答 (QA) 评估器

支持的评估方法:
1. Semantic Similarity — 基于嵌入向量的语义相似度
2. LLM Judge — 调用大模型评分 (0-3)
3. Evidence Consistency — 答案与检索证据的一致性
"""

import re
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class QAScoreResult:
    """QA 评估结果"""
    semantic_similarity: float = 0.0       # 语义相似度 (0-1)
    llm_judge_score: float = 0.0           # LLM 评分 (0-3)
    evidence_consistency: float = 0.0      # 证据一致性 (0-1)
    raw_scores: Dict[str, float] = field(default_factory=dict)


class SemanticSimilarityScorer:
    """基于嵌入的语义相似度评估"""
    
    def __init__(self, model_name: str = "BGE", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._encoder = None
    
    def _get_encoder(self):
        """懒加载嵌入模型"""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_map = {
                    "BGE": "BAAI/bge-large-zh-v1.5",
                    "text2vec": "shibing624/text2vec-base-chinese",
                    "all-MiniLM": "all-MiniLM-L6-v2",
                }
                model_path = model_map.get(self.model_name, model_map["BGE"])
                self._encoder = SentenceTransformer(model_path, device=self.device)
            except ImportError:
                # fallback: 字符级 Jaccard 相似度
                self._encoder = None
        return self._encoder
    
    def compute_similarity(self, answer: str, reference: str) -> float:
        """计算语义相似度"""
        encoder = self._get_encoder()
        if encoder:
            emb1 = encoder.encode(answer, normalize_embeddings=True)
            emb2 = encoder.encode(reference, normalize_embeddings=True)
            return float(np.dot(emb1, emb2))
        else:
            # fallback: Jaccard 字符相似度
            return self._jaccard_similarity(answer, reference)
    
    def _jaccard_similarity(self, a: str, b: str) -> float:
        """字符级 Jaccard 相似度（fallback）"""
        chars_a = set(a.lower().replace(' ', ''))
        chars_b = set(b.lower().replace(' ', ''))
        if not chars_a or not chars_b:
            return 0.0
        intersection = chars_a & chars_b
        union = chars_a | chars_b
        return len(intersection) / len(union)
    
    def compute_batch(self, answers: List[str], references: List[str]) -> List[float]:
        """批量计算相似度"""
        return [self.compute_similarity(a, r) for a, r in zip(answers, references)]


class LLMJudgeScorer:
    """调用 LLM 作为 Judge 评分"""
    
    RUBRIC = """You are an expert evaluator for industrial knowledge QA.

Score the model answer against the reference answer on a 0-3 scale:
- Score 3 (Correct): Answer aligns closely with reference, covering all key info
- Score 2 (Acceptable): Directionally correct but has notable omissions
- Score 1 (Partial): Somewhat relevant but significant inconsistencies
- Score 0 (Incorrect): Irrelevant, contains factual errors, or contradicts reference

Return JSON: {{"score": <0-3>, "explanation": "<reason>"}}"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
    
    def score_one(self, question: str, answer: str, reference: str) -> Dict[str, Any]:
        """单样本评分"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            
            user_prompt = f"""## Question:
{question}

## Reference Answer (correct):
{reference}

## Model Answer (to evaluate):
{answer}

## Output (JSON):"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.RUBRIC},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = response.choices[0].message.content
            
            # 解析 JSON
            json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d[^{}]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                score = max(0, min(3, int(data.get("score", 0))))
                explanation = data.get("explanation", "")
                return {"score": score, "explanation": explanation}
            
            # fallback: 正则提取
            score_match = re.search(r'score["\']?\s*[:=]\s*(\d)', text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 0
            return {"score": max(0, min(3, score)), "explanation": text[:200]}
            
        except Exception as e:
            return {"score": 0, "explanation": f"Judge call failed: {str(e)}", "error": str(e)}
    
    def score_batch(self, questions: List[str], answers: List[str],
                    references: List[str]) -> List[Dict[str, Any]]:
        """批量评分"""
        return [
            self.score_one(q, a, r)
            for q, a, r in zip(questions, answers, references)
        ]


class EvidenceConsistencyScorer:
    """评估答案与检索证据的一致性"""
    
    def __init__(self, method: str = "keyword"):
        """
        Args:
            method: keyword / llm / hybrid
        """
        self.method = method
    
    def score_one(self, answer: str, evidence_texts: List[str]) -> float:
        """
        评估答案与证据的一致性
        
        Returns:
            0.0 ~ 1.0 的一致性分数
        """
        if not evidence_texts or not answer:
            return 0.0
        
        if self.method == "keyword":
            return self._keyword_consistency(answer, evidence_texts)
        elif self.method == "llm":
            return self._llm_consistency(answer, evidence_texts)
        else:  # hybrid
            kw_score = self._keyword_consistency(answer, evidence_texts)
            return kw_score
    
    def _keyword_consistency(self, answer: str, evidence_texts: List[str]) -> float:
        """关键词覆盖度评估一致性"""
        import re
        
        def extract_keywords(text: str) -> set:
            text = text.lower()
            # 提取中文词（>=2字）和英文词（>=3字母）
            words = set()
            for m in re.finditer(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text):
                words.add(m.group())
            return words
        
        answer_kw = extract_keywords(answer)
        if not answer_kw:
            return 0.0
        
        # 合并所有证据的关键词
        evidence_kw = set()
        for ev in evidence_texts:
            evidence_kw |= extract_keywords(ev)
        
        if not evidence_kw:
            return 0.0
        
        # 答案关键词被证据覆盖的比例
        covered = answer_kw & evidence_kw
        coverage = len(covered) / len(answer_kw)
        
        return coverage
    
    def score_batch(self, answers: List[str],
                    evidence_list: List[List[str]]) -> List[float]:
        """批量评分"""
        return [
            self.score_one(a, evs)
            for a, evs in zip(answers, evidence_list)
        ]


class QAEvaluator:
    """
    QA 评估器：整合三种评估方法
    
    评估指标:
    - Semantic Similarity: 基于嵌入的语义相似度
    - LLM Judge: 大模型评分 (0-3 scale)
    - Evidence Consistency: 答案与检索证据的一致性
    """
    
    def __init__(
        self,
        use_semantic: bool = True,
        use_llm_judge: bool = False,
        use_evidence: bool = True,
        embedding_model: str = "BGE",
        llm_model: str = "deepseek-chat",
        api_key: Optional[str] = None,
    ):
        self.use_semantic = use_semantic
        self.use_llm_judge = use_llm_judge
        self.use_evidence = use_evidence
        
        self.semantic_scorer = SemanticSimilarityScorer(model_name=embedding_model)
        self.llm_judge = LLMJudgeScorer(api_key=api_key, model=llm_model)
        self.evidence_scorer = EvidenceConsistencyScorer(method="keyword")
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
        evidence_list: Optional[List[List[str]]] = None,
    ) -> List[QAScoreResult]:
        """
        批量评估 QA 回答
        
        Args:
            questions: 问题列表
            answers: 模型回答列表
            references: 参考答案列表
            evidence_list: 检索到的证据文本列表（每个样本多条）
        
        Returns:
            List[QAScoreResult]
        """
        n = len(answers)
        results = []
        
        # 语义相似度
        sim_scores = []
        if self.use_semantic:
            sim_scores = self.semantic_scorer.compute_batch(answers, references)
        else:
            sim_scores = [0.0] * n
        
        # LLM Judge
        judge_results = []
        if self.use_llm_judge:
            judge_results = self.llm_judge.score_batch(questions, answers, references)
        else:
            judge_results = [{"score": 0.0}] * n
        
        # 证据一致性
        evi_scores = []
        if self.use_evidence and evidence_list:
            evi_scores = self.evidence_scorer.score_batch(answers, evidence_list)
        else:
            evi_scores = [0.0] * n
        
        for i in range(n):
            results.append(QAScoreResult(
                semantic_similarity=sim_scores[i],
                llm_judge_score=judge_results[i].get("score", 0.0),
                evidence_consistency=evi_scores[i],
                raw_scores={
                    "semantic_sim": sim_scores[i],
                    "llm_judge": judge_results[i].get("score", 0.0),
                    "evidence_consistency": evi_scores[i],
                }
            ))
        
        return results
    
    def evaluate_one(
        self,
        question: str,
        answer: str,
        reference: str,
        evidence: Optional[List[str]] = None,
    ) -> QAScoreResult:
        """单样本评估"""
        return self.evaluate(
            [question], [answer], [reference],
            [evidence] if evidence else None
        )[0]
