"""
format_detector.py: 自动检测问题格式

支持自动识别三种格式：
- QA: 开放式问答
- Fill-in-the-Blank: 填空题 (含 ___ 或 （）占位符)
- Multiple Choice: 选择题 (含 A/B/C/D 选项)

也能通过 override 手动指定格式。
"""

import re
from enum import Enum
from typing import List, Optional, Union


class QuestionFormat(str, Enum):
    """问题格式枚举"""
    QA = "qa"                     # 开放式问答
    FILL_IN_BLANK = "fill_blank"  # 填空题
    MULTIPLE_CHOICE = "mc"        # 选择题
    UNKNOWN = "unknown"           # 无法识别


class FormatDetector:
    """
    问题格式检测器
    
    自动检测逻辑优先级:
    1. 如果手动指定格式 → 使用指定格式
    2. 包含 ___ 或 （） 或 （   ） → FILL_IN_BLANK
    3. 包含 A. B. C. D. 或 A) B) C) D) → MULTIPLE_CHOICE
    4. 其他 → QA
    """
    
    # 填空题占位符模式
    BLANK_PATTERNS = [
        r'_{3,}',           # ___ 三个及以上下划线
        r'（\s*）',          # （） 或 （   ）
        r'\(\s*\)',         # () 或 (   )
        r'（\s*_\s*）',      # （_）
        r'\[BLANK\]',       # [BLANK]
        r'__+',             # __ 两个及以上下划线
    ]
    
    # 选择题选项模式
    MC_PATTERNS = [
        # A. B. C. D. 模式
        r'(?:^|\n)\s*A\.[\s\S]*?(?:B\.[\s\S]*?)?(?:C\.[\s\S]*?)?(?:D\.[\s\S]*?)(?:E\.[\s\S]*?)?(?=\n|$)',
        # A) B) C) D) 模式
        r'(?:^|\n)\s*A\)[\s\S]*?(?:B\)[\s\S]*?)?(?:C\)[\s\S]*?)?(?:D\)[\s\S]*?)(?:E\)[\s\S]*?)?(?=\n|$)',
        # (A) (B) (C) (D) 模式
        r'(?:^|\n)\s*\(A\)[\s\S]*?(?:\(B\)[\s\S]*?)?(?:\(C\)[\s\S]*?)?(?:\(D\)[\s\S]*?)(?:\(E\)[\s\S]*?)?(?=\n|$)',
    ]
    
    # 选项字母检测（用于确认多选题）
    OPTION_LETTERS = re.compile(
        r'(?:(?:^|\n)\s*(?:[A-E])[.)])|(?:^|\n)\s*\([A-E]\)',
        re.MULTILINE
    )
    
    @classmethod
    def detect(cls, question: str) -> QuestionFormat:
        """
        自动检测问题格式
        
        Args:
            question: 问题文本
        
        Returns:
            QuestionFormat 枚举
        """
        if not question:
            return QuestionFormat.UNKNOWN
        
        # 1. 检测填空题
        for pattern in cls.BLANK_PATTERNS:
            if re.search(pattern, question):
                return QuestionFormat.FILL_IN_BLANK
        
        # 2. 检测多选题
        # 检查是否至少有3个选项标记
        matches = cls.OPTION_LETTERS.findall(question)
        if len(matches) >= 3:
            return QuestionFormat.MULTIPLE_CHOICE
        
        # 3. 默认作为 QA
        return QuestionFormat.QA
    
    @classmethod
    def detect_batch(cls, questions: List[str]) -> List[QuestionFormat]:
        """批量检测"""
        return [cls.detect(q) for q in questions]
    
    @classmethod
    def format_name(cls, fmt: QuestionFormat) -> str:
        """获取格式的中文名称"""
        names = {
            QuestionFormat.QA: "开放式问答 (QA)",
            QuestionFormat.FILL_IN_BLANK: "填空题 (Fill-in-the-Blank)",
            QuestionFormat.MULTIPLE_CHOICE: "选择题 (Multiple Choice)",
            QuestionFormat.UNKNOWN: "未知格式",
        }
        return names.get(fmt, "未知")
    
    @staticmethod
    def extract_mc_options(question: str) -> dict:
        """提取选择题的选项"""
        options = {}
        
        # 尝试 A. xxx 格式
        pattern_a = re.findall(r'(?:^|\n)\s*([A-E])[.)]\s*(.*?)(?=(?:\n\s*[A-E][.)]\s*)|\n|$)', question, re.DOTALL)
        if pattern_a and len(pattern_a) >= 2:
            for letter, text in pattern_a:
                options[letter] = text.strip()
            return options
        
        # 尝试 (A) xxx 格式
        pattern_b = re.findall(r'(?:^|\n)\s*\(([A-E])\)\s*(.*?)(?=(?:\n\s*\([A-E]\))|\n|$)', question, re.DOTALL)
        if pattern_b and len(pattern_b) >= 2:
            for letter, text in pattern_b:
                options[letter] = text.strip()
            return options
        
        return options
    
    @staticmethod
    def extract_blank_position(question: str) -> int:
        """返回填空题第一个空白的位置（字符偏移）"""
        patterns = [r'_{3,}', r'（\s*）', r'\(\s*\)', r'__+']
        for pat in patterns:
            m = re.search(pat, question)
            if m:
                return m.start()
        return -1
