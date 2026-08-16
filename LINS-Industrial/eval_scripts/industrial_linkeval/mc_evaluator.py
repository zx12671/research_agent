"""
mc_evaluator.py: 选择题 (Multiple Choice) 评估器

评估方法:
- Accuracy: 准确率
- 支持从模型回答中自动提取选项字母
- 支持标准化匹配 (A/a, 全角/半角)
"""

import re
from typing import List, Dict, Optional, Set, Union
from dataclasses import dataclass, field


@dataclass
class MCScoreResult:
    """选择题评估结果"""
    is_correct: bool = False         # 是否正确
    predicted: str = ""              # 模型预测的选项
    expected: str = ""               # 正确答案选项
    confidence: float = 0.0          # 置信度 (0-1)
    raw_scores: Dict[str, float] = field(default_factory=dict)


class MCOptionExtractor:
    """从模型回答中提取选项字母"""
    
    # 优先级最高的提取模式（越靠前越优先）
    EXTRACT_PATTERNS = [
        # "答案是A" / "选择B"
        r'(?:答案|选择|选|正确)[是为：:．\s]*([A-Ea-e])',
        # "A." (选项字母开头)
        r'^\s*([A-Ea-e])[.、）)\s]',
        # "A" (单独字母)
        r'(?:^|\s)([A-Ea-e])(?:\s|[.，。、！？]|$)',
        # 括号中的字母 "(A)"
        r'[（(]([A-Ea-e])[）)]',
    ]
    
    @classmethod
    def extract(cls, answer: str) -> Optional[str]:
        """从回答中提取选项字母"""
        if not answer:
            return None
        
        answer = answer.strip()
        
        for pattern in cls.EXTRACT_PATTERNS:
            match = re.search(pattern, answer)
            if match:
                letter = match.group(1).upper()
                if letter in 'ABCDE':
                    return letter
        
        return None


class MCEvaluator:
    """
    选择题评估器
    
    评估指标:
    - Accuracy: 正确率
    - Per-class accuracy: 每个选项的正确率
    """
    
    def __init__(self):
        self.extractor = MCOptionExtractor()
    
    def evaluate(
        self,
        answers: List[str],
        correct_answers: List[str],
        options_list: Optional[List[dict]] = None,
    ) -> List[MCScoreResult]:
        """
        批量评估选择题
        
        Args:
            answers: 模型回答列表（原始文本或提取后的选项字母）
            correct_answers: 正确答案列表（如 "A", "B", "C"）
            options_list: 选项字典列表 [{A: "xxx", B: "yyy", ...}]
        
        Returns:
            List[MCScoreResult]
        """
        results = []
        
        for i, (ans, correct) in enumerate(zip(answers, correct_answers)):
            result = self._evaluate_one(ans, correct)
            results.append(result)
        
        return results
    
    def evaluate_one(
        self,
        answer: str,
        correct_answer: str,
        options: Optional[dict] = None,
    ) -> MCScoreResult:
        """单样本评估"""
        return self._evaluate_one(answer, correct_answer, options)
    
    def _evaluate_one(
        self,
        answer: str,
        correct_answer: str,
        options: Optional[dict] = None,
    ) -> MCScoreResult:
        """内部单样本评估"""
        correct = correct_answer.upper().strip()
        
        # 尝试从回答中提取选项字母
        predicted = self.extractor.extract(answer)
        
        is_correct = (predicted == correct) if predicted else False
        
        # 置信度：如果是直接提取且匹配
        confidence = 1.0 if is_correct else (0.5 if predicted else 0.0)
        
        return MCScoreResult(
            is_correct=is_correct,
            predicted=predicted or "",
            expected=correct,
            confidence=confidence,
            raw_scores={
                "accuracy": 1.0 if is_correct else 0.0,
                "has_prediction": 1.0 if predicted else 0.0,
            }
        )
    
    def get_accuracy(self, results: List[MCScoreResult]) -> float:
        """计算准确率"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.is_correct) / len(results)
    
    def get_confusion_stats(self, results: List[MCScoreResult]) -> dict:
        """获取混淆统计"""
        from collections import Counter
        correct = Counter()
        incorrect = Counter()
        
        for r in results:
            if r.is_correct:
                correct[r.expected] += 1
            else:
                incorrect[r.expected] += 1
        
        return {
            "correct_by_answer": dict(correct),
            "incorrect_by_answer": dict(incorrect),
        }
