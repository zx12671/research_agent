"""
blank_evaluator.py: 填空题 (Fill-in-the-Blank) 评估器

支持的评估方法:
1. Normalization — 标准化处理（去空格、标点、大小写）
2. Exact Match — 精确匹配
3. Synonym Matching — 同义词匹配
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class BlankScoreResult:
    """填空题评估结果"""
    exact_match: bool = False               # 精确匹配
    normalized_match: bool = False          # 标准化后匹配
    synonym_match: bool = False             # 同义词匹配
    match_score: float = 0.0               # 综合匹配得分 (0-1)
    raw_scores: Dict[str, float] = field(default_factory=dict)


class TextNormalizer:
    """文本标准化工具"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """标准化：去空格、标点、统一大小写"""
        text = text.strip()
        # 去除标点符号
        text = re.sub(r'[，。、；：！？""''（）【】\[\]\(\)\.\,\!\?\:\;「」\*\-_\s+]', '', text)
        # 统一大小写
        text = text.lower()
        # 统一全角半角
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('Ａ', 'A').replace('Ｂ', 'B').replace('Ｃ', 'C').replace('Ｄ', 'D')
        return text.strip()
    
    @staticmethod
    def extract_answer_from_blank(question: str, model_answer: str) -> str:
        """
        从完整回答中提取填空答案
        例如问题 "螺栓的直径是___mm"，回答 "螺栓的直径是14mm" → 提取 "14"
        """
        # 替换问题中的占位符为正则捕获组
        blank_patterns = [r'_{3,}', r'（\s*）', r'\(\s*\)', r'__+', r'\[BLANK\]']
        pattern_str = question
        for bp in blank_patterns:
            pattern_str = re.sub(bp, '(.+)', pattern_str)
        
        # 转义并构建匹配模式
        pattern_str = re.escape(pattern_str).replace(r'\(\.\+\)', '(.+)')
        
        try:
            match = re.search(pattern_str, model_answer)
            if match:
                return match.group(1).strip()
        except:
            pass
        
        # fallback: 返回完整回答
        return model_answer.strip()


class SynonymMatcher:
    """同义词匹配工具"""
    
    # 工业领域常见同义词表
    SYNONYM_DICT = {
        # 中文同义词
        '螺栓': {'螺丝', '螺钉', '螺柱', '紧固件'},
        '螺母': {'螺帽', '螺丝帽'},
        '垫圈': {'垫片', '密封垫'},
        '轴承': {'bearing', '軸承'},
        '齿轮': {'齿', 'gear'},
        '电机': {'电动机', '马达', 'motor', '马达'},
        '电压': {'伏特', '伏', 'V', 'v'},
        '电流': {'安培', '安', 'A', 'a'},
        '功率': {'瓦特', '瓦', 'W', 'watt'},
        '频率': {'赫兹', 'Hz', 'hz'},
        '扭矩': {'转矩', '力矩', 'torque'},
        '转速': {'速度', '转/分', 'rpm', 'RPM'},
        '温度': {'温', '摄氏', '℃', '°C'},
        '压力': {'压强', '气压', '液压', 'MPa'},
        '焊接': {'焊', '熔接'},
        '热处理': {'热加工', 'heat treatment'},
        '淬火': {'淬', 'quenching'},
        '回火': {'tempering'},
        '退火': {'annealing'},
        '硬度': {'hardness', 'HRC', 'HB'},
        '抗拉强度': {'拉伸强度', 'tensile strength'},
        '屈服强度': {'yield strength'},
        # 英文同义词
        'bolt': {'screw', 'fastener'},
        'nut': {'locknut'},
        'washer': {'gasket', 'seal'},
        'bearing': {'轴承'},
        'voltage': {'V', 'volt'},
        'current': {'A', 'amp', 'ampere'},
        'power': {'W', 'watt'},
        'torque': {'扭矩', '力矩'},
        'temperature': {'temp', 'T', '温度'},
        'pressure': {'P', '压力'},
    }
    
    @classmethod
    def get_synonyms(cls, word: str) -> Set[str]:
        """获取单词的同义词集合"""
        word = word.lower().strip()
        synonyms = set()
        
        # 直接查询
        if word in cls.SYNONYM_DICT:
            synonyms |= cls.SYNONYM_DICT[word]
        
        # 反向查询（如果一个同义词表中包含该词）
        for key, syn_set in cls.SYNONYM_DICT.items():
            if word in syn_set:
                synonyms.add(key)
                synonyms |= syn_set
        
        # 包含自身
        synonyms.add(word)
        
        return synonyms
    
    @classmethod
    def are_synonyms(cls, word1: str, word2: str) -> bool:
        """判断两个词是否为同义词"""
        if word1.lower().strip() == word2.lower().strip():
            return True
        syns1 = cls.get_synonyms(word1)
        syns2 = cls.get_synonyms(word2)
        return bool(syns1 & syns2)


class BlankEvaluator:
    """
    填空题评估器
    
    评估方法:
    - Exact Match: 精确匹配（区分大小写）
    - Normalized Match: 标准化后匹配
    - Synonym Match: 同义词匹配
    """
    
    def __init__(self, use_synonym: bool = True):
        self.use_synonym = use_synonym
        self.normalizer = TextNormalizer()
        self.synonym_matcher = SynonymMatcher()
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str],
    ) -> List[BlankScoreResult]:
        """
        批量评估填空题
        
        Args:
            questions: 问题列表（含 __ 占位符）
            answers: 模型回答列表（可能是完整句子或仅填空词）
            references: 参考答案列表（正确的填空词）
        
        Returns:
            List[BlankScoreResult]
        """
        results = []
        
        for q, a, r in zip(questions, answers, references):
            result = self._evaluate_one(q, a, r)
            results.append(result)
        
        return results
    
    def evaluate_one(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> BlankScoreResult:
        """单样本评估"""
        return self._evaluate_one(question, answer, reference)
    
    def _evaluate_one(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> BlankScoreResult:
        """内部单样本评估"""
        # 提取填空答案
        extracted = self.normalizer.extract_answer_from_blank(question, answer)
        
        # 标准化
        norm_answer = self.normalizer.normalize(extracted)
        norm_ref = self.normalizer.normalize(reference)
        
        # 1. Exact Match (精确匹配)
        exact_match = (answer.strip() == reference.strip())
        
        # 2. Normalized Match (标准化匹配)
        normalized_match = (norm_answer == norm_ref)
        
        # 3. Synonym Match (同义词匹配)
        synonym_match = False
        if self.use_synonym and not normalized_match and norm_answer and norm_ref:
            synonym_match = self.synonym_matcher.are_synonyms(norm_answer, norm_ref)
        
        # 综合得分
        scores = []
        if exact_match:
            scores.append(1.0)
        if normalized_match:
            scores.append(1.0)
        if synonym_match:
            scores.append(0.8)
        
        if not scores:
            # 部分字符匹配
            if norm_answer and norm_ref:
                chars_a = set(norm_answer)
                chars_r = set(norm_ref)
                intersection = chars_a & chars_r
                union = chars_a | chars_r
                partial = len(intersection) / len(union) if union else 0.0
                scores.append(partial * 0.3)
            else:
                scores.append(0.0)
        
        match_score = max(scores)
        
        return BlankScoreResult(
            exact_match=exact_match,
            normalized_match=normalized_match,
            synonym_match=synonym_match,
            match_score=match_score,
            raw_scores={
                "exact_match": 1.0 if exact_match else 0.0,
                "normalized_match": 1.0 if normalized_match else 0.0,
                "synonym_match": 0.8 if synonym_match else 0.0,
                "partial_match": match_score,
            }
        )
    
    def get_accuracy(self, results: List[BlankScoreResult]) -> float:
        """计算准确率（精确匹配或标准化匹配视为正确）"""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.exact_match or r.normalized_match)
        return correct / len(results)
    
    def get_synonym_accuracy(self, results: List[BlankScoreResult]) -> float:
        """计算含同义词匹配的准确率"""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.exact_match or r.normalized_match or r.synonym_match)
        return correct / len(results)
