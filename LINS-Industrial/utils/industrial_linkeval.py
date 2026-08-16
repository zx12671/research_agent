"""
industrial_linkeval.py: Link-Eval 的工业场景适配版
支持工业知识问答的引用准确性评估
"""
import re
import json
from itertools import combinations


class IndustrialLinkEval:
    """工业场景 Link-Eval 评估器"""

    def __init__(self, nli_model=None):
        """
        初始化工业 Link-Eval
        Args:
            nli_model: 可选的自然语言推理模型，用于判断陈述与引用的关系
        """
        self.nli_model = nli_model

    def extract_statements(self, text):
        """从带引用的文本中提取陈述和引用"""
        pattern = re.compile(r"\[(\d+)\]")
        sentences = re.split(r'(?<=\.\s)', text.strip())

        statements = []
        for sentence in sentences:
            refs = pattern.findall(sentence)
            ref_numbers = {int(ref) - 1 for ref in refs}
            clean_sentence = pattern.sub('', sentence).strip()
            if clean_sentence:
                statements.append((clean_sentence, ref_numbers))
        return statements

    def calculate_citation_metrics(self, statements, refs):
        """
        计算引用相关指标
        Args:
            statements: [(statement, {citation_indices}), ...]
            refs: 引用文本列表
        Returns:
            dict: {precision, recall, f1}
        """
        # 简化实现：检查引用是否有效
        total_citations = sum(len(s) for _, s in statements)
        valid_citations = 0

        for _, citation_set in statements:
            for citation in citation_set:
                if citation < len(refs):
                    valid_citations += 1

        precision = valid_citations / total_citations if total_citations > 0 else 0
        recall = valid_citations / len(refs) if refs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "citation_precision": precision,
            "citation_recall": recall,
            "f1_score": f1
        }

    def evaluate_statement_correctness(self, statements):
        """检查陈述之间是否存在矛盾"""
        texts = [s[0] for s in statements]
        for premise, hypothesis in combinations(texts, 2):
            # 简单的矛盾检测：检查是否包含相反的断言
            # 实际实现应使用 NLI 模型
            if self._check_contradiction(premise, hypothesis):
                return 0
        return 1

    def _check_contradiction(self, text1, text2):
        """简单的矛盾检测（占位实现）"""
        # 在实际实现中，这里应该调用 NLI 模型
        # 这里简化为检查是否包含相反词
        neg_words = ['not', 'no', 'never', 'incorrect', 'invalid']
        has_neg1 = any(w in text1.lower() for w in neg_words)
        has_neg2 = any(w in text2.lower() for w in neg_words)

        # 如果一个有否定词而另一个没有，可能构成矛盾
        if has_neg1 != has_neg2:
            # 进一步检查主题是否相似
            common_words = set(text1.lower().split()) & set(text2.lower().split())
            if len(common_words) > 3:  # 至少有3个共同词
                return True
        return False

    def evaluate_fluency(self, text):
        """评估陈述流畅度（占位实现）"""
        # 实际实现应使用 UniEval 等专用模型
        # 这里简化为基于句子长度和复杂度的评分
        sentences = text.split('.')
        avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)

        # 简单评分：长度适中为佳
        if 20 < avg_len < 80:
            return 0.9
        elif 10 < avg_len < 120:
            return 0.6
        else:
            return 0.3

    def evaluate(self, question, response, refs, correct_answer=None):
        """
        完整评估
        Returns:
            dict: 各项评估指标
        """
        statements = self.extract_statements(response)

        if not statements or not refs:
            return {
                "citation_precision": 0,
                "citation_recall": 0,
                "f1_score": 0,
                "statement_correctness": 0,
                "statement_fluency": 0,
                "overall_score": 0
            }

        # 计算引用指标
        citation_metrics = self.calculate_citation_metrics(statements, refs)

        # 计算陈述正确性
        correctness = self.evaluate_statement_correctness(statements)

        # 计算流畅度
        fluency = self.evaluate_fluency(response)

        # 计算总体评分
        overall = (
            citation_metrics['f1_score'] * 0.3 +
            correctness * 0.4 +
            fluency * 0.3
        )

        return {
            **citation_metrics,
            "statement_correctness": correctness,
            "statement_fluency": fluency,
            "overall_score": overall
        }


# 便捷函数
def create_industrial_linkeval():
    """创建工业 Link-Eval 实例"""
    return IndustrialLinkEval()


def format_eval_results(metrics):
    """格式化评估结果"""
    return json.dumps({
        "Citation Precision": f"{metrics['citation_precision']:.3f}",
        "Citation Recall": f"{metrics['citation_recall']:.3f}",
        "F1 Score": f"{metrics['f1_score']:.3f}",
        "Statement Correctness": metrics['statement_correctness'],
        "Statement Fluency": f"{metrics['statement_fluency']:.3f}",
        "Overall Score": f"{metrics['overall_score']:.3f}"
    }, indent=2)
