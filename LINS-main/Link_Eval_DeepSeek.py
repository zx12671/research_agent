"""
LinkEval-DeepSeek: 基于 DeepSeek API 的精简引用评估模块
完全对齐 LINS 论文 (Nature Communications 2025) 中 LinkEval 的核心评价指标

精简后评估指标（去除了冗余指标，保留核心评价能力）:
1. Citation Precision (引用精准率) - 正确引用数 / 总引用数（最细粒度的引用准确度）
2. Citation Recall (引用召回率) - 正确且有效引用数 / 有效 Ref 总数（引用覆盖度）
3. F1 Score - Precision 和 Recall 的调和平均（综合指标，替代冗余的 Overall Score）
4. Statement Correctness (陈述正确性) - 语句间无矛盾 + 答案正确则为 1
5. Statement Fluency (陈述流畅度) - DeepSeek 评估流畅度 (0-1)

使用方法：
    from Link_Eval_DeepSeek import LinkEvalDeepSeek, convert_to_statements
    
    # 初始化（需要 DEEPSEEK_API_KEY 环境变量）
    evaluator = LinkEvalDeepSeek(api_key="sk-94ccad564a7542228ad52f6b2654e11e")
    
    # 进行完整评估
    metrics = evaluator.evaluate(question, statements, refs, correct_answer=None)
    
    # 输出格式:
    # {
    #   "citation_precision": 0.7661,       # 引用精准率
    #   "citation_recall": 0.7280,          # 引用召回率
    #   "f1_score": 0.7466,                 # F1 综合
    #   "statement_correctness": 1.0,       # 陈述正确性
    #   "statement_fluency": 0.8876,        # 陈述流畅度
    # }
"""

import numpy as np
import re
import json
import os
from itertools import combinations
from openai import OpenAI
import httpx


# ============================================================
# 1. DeepSeek NLI Entailment 模块
# ============================================================
# 请通过环境变量 DEEPSEEK_API_KEY 设置你的 API Key
# 或在创建 LinkEvalDeepSeek 时传入 api_key 参数
# os.environ['DEEPSEEK_API_KEY'] = 'sk-your-deepseek-api-key'
class DeepSeekNLI:
    """使用 DeepSeek 进行 NLI 蕴涵判断，替代 T5-11B"""

    def __init__(self, api_key=None, model_name="deepseek-chat"):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", http_client=http_client)
        self.model_name = model_name

    def nli(self, premise, hypothesis):
        """
        判断 premise 是否蕴涵 hypothesis

        完全对齐论文中 NLI 系统的输出：
        "output labels {0, 1, 2} representing {entailment, contradiction, neutral} respectively"

        Args:
            premise: 前提文本
            hypothesis: 假设文本

        Returns:
            0: entailment (蕴涵)
            1: contradiction (矛盾)
            2: neutral (中立)
        """
        system_prompt = """You are an NLI (Natural Language Inference) expert. 
Your task is to determine the relationship between a PREMISE and a HYPOTHESIS.

Rules:
- Return ONLY "entailment" if the premise logically implies the hypothesis
- Return ONLY "contradiction" if the premise contradicts the hypothesis
- Return ONLY "neutral" if neither entailment nor contradiction can be determined
- Do NOT include any explanation or additional text
- Do NOT include punctuation

Examples:
Premise: A dog is playing in the park.
Hypothesis: An animal is outside.
Answer: entailment

Premise: The cat is sleeping on the couch.
Hypothesis: The cat is running outside.
Answer: contradiction

Premise: I bought a new phone yesterday.
Hypothesis: The phone is black.
Answer: neutral"""

        user_prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=20
            )

            output_text = response.choices[0].message.content.strip().lower()

            if "entailment" in output_text:
                return 0
            elif "contradiction" in output_text:
                return 1
            else:
                return 2  # neutral

        except Exception as e:
            print(f"[DeepSeekNLI] API error: {e}")
            return 2  # 出错时默认 neutral


# ============================================================
# 2. DeepSeek Fluency 模块
# ============================================================

class DeepSeekFluency:
    """使用 DeepSeek 评估文本流畅度，替代 UniEval"""

    def __init__(self, api_key=None, model_name="deepseek-chat"):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", http_client=http_client)
        self.model_name = model_name

    def score_fluency(self, text):
        """
        评估文本流畅度，返回 0-1 之间的分数

        Returns:
            float: 流畅度分数 (0.0 ~ 1.0)
        """
        system_prompt = """You are an expert evaluator of text fluency.
Rate the following text on a scale from 0.0 to 1.0 based on these criteria:
- 1.0: Perfect fluency, natural and grammatically correct
- 0.8: Good fluency, minor issues
- 0.6: Acceptable fluency, some grammatical or stylistic issues
- 0.4: Poor fluency, notable issues
- 0.2: Very poor fluency, hard to understand
- 0.0: Completely unreadable

Return ONLY a JSON object with a single key "fluency" and a float value between 0.0 and 1.0.
Example: {"fluency": 0.95}
Do NOT include any other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text to evaluate:\n{text}\n\nScore (JSON):"}
                ],
                temperature=0.0,
                max_tokens=50
            )

            output = response.choices[0].message.content.strip()

            # 尝试解析 JSON
            try:
                json_match = re.search(r'\{[^}]+\}', output)
                if json_match:
                    score_dict = json.loads(json_match.group())
                    return score_dict.get("fluency", 0.5)
            except (json.JSONDecodeError, KeyError):
                pass

            # 如果 JSON 解析失败，尝试提取数字
            numbers = re.findall(r'0\.\d+|1\.0', output)
            if numbers:
                return float(numbers[0])

            return 0.5  # 默认中等分数

        except Exception as e:
            print(f"[DeepSeekFluency] API error: {e}")
            return 0.5


# ============================================================
# 3. 主评估器 LinkEvalDeepSeek
# ============================================================

class LinkEvalDeepSeek:
    """
    基于 DeepSeek API 的 LinkEval 评估器
    完全对齐 LINS 论文 (Nature Communications 2025) 中的 LinkEval 评估系统

    评估指标（论文 Fig.5b-e）:
    - Citation Set Precision: 引用集精准率
    - Citation Precision: 引用精准率
    - Citation Recall: 引用召回率
    - Statement Correctness: 陈述正确性
    - Statement Fluency: 陈述流畅度
    """

    def __init__(self, api_key=None, model_name="deepseek-chat", verbose=False):
        """
        初始化评估器

        Args:
            api_key: DeepSeek API Key，默认从 DEEPSEEK_API_KEY 环境变量读取
            model_name: DeepSeek 模型名，默认 "deepseek-chat"
            verbose: 是否打印详细信息
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model_name = model_name
        self.verbose = verbose

        # 初始化评估组件
        self.nli = DeepSeekNLI(api_key=self.api_key, model_name=self.model_name)
        self.fluency_evaluator = DeepSeekFluency(api_key=self.api_key, model_name=self.model_name)

        # 初始化 DeepSeek 客户端（用于 _get_valid_refs 等直接调用）
        http_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", http_client=http_client)

        if self.verbose:
            print(f"[LinkEvalDeepSeek] 初始化完成，模型: {self.model_name}")

    # ----------------------------------------------------------
    # 兼容原 LinkEval 类的 API 接口
    # 完全对齐 LINS/Link_Eval.py 中的 compute_correct_citations 方法
    # ----------------------------------------------------------

    def compute_correct_citations(self, statements, refs):
        """
        计算正确引用集（兼容原 LinkEval 类 API）。

        完全对齐 LINS/Link_Eval.py 中 LinkEval.compute_correct_citations() 的逻辑：
        
        原方法：
        ```
        def compute_correct_citations(self, statements, refs):
            correct_citations = []
            for statement, citation_set in statements:
                if not citation_set:
                    continue
                if any(i >= len(refs) for i in citation_set):
                    continue
                premise = " ".join([refs[i] for i in citation_set])
                hypothesis = statement
                entailment_label = self.nli_entailment(premise, hypothesis)
                if entailment_label == 0:  # entailment
                    correct_citations.append((statement, citation_set))
            return correct_citations
        ```

        Args:
            statements: list of (text, citation_set)
            refs: list of reference texts

        Returns:
            list of (statement, citation_set): 被判定为正确引用集的 (陈述, 引用集) 列表
        """
        correct_citations = []
        for statement, citation_set in statements:
            if not citation_set:
                continue
            # 检查引用索引是否越界
            if any(i >= len(refs) for i in citation_set):
                continue
            premise = " ".join([refs[i] for i in citation_set])
            hypothesis = statement
            entailment_label = self.nli_entailment(premise, hypothesis)
            if entailment_label == 0:  # entailment
                correct_citations.append((statement, citation_set))
        return correct_citations

    def compute_correct_citation_count(self, correct_citations, refs):
        """
        计算正确引用数目（兼容原 LinkEval 类 API）。

        完全对齐 LINS/Link_Eval.py 中 LinkEval.compute_correct_citation_count() 的逻辑：

        原方法：
        ```
        def compute_correct_citation_count(self, correct_citations, refs):
            correct_count = 0
            correct_citation_details = []
            for statement, citation_set in correct_citations:
                correct_set = set()
                if len(citation_set) == 1:
                    citation = citation_set.pop()
                    correct_count += 1
                    correct_set.add(citation)
                else:
                    for citation in citation_set:
                        remaining_citations = citation_set - {citation}
                        if any(i >= len(refs) for i in remaining_citations):
                            continue
                        premise = " ".join([refs[i] for i in remaining_citations])
                        hypothesis = statement
                        entailment_label = self.nli_entailment(premise, hypothesis)
                        if entailment_label != 0:  # not entailment
                            correct_count += 1
                            correct_set.add(citation)
                correct_citation_details.append((statement, citation_set, correct_set))
            return correct_count, correct_citation_details
        ```

        Args:
            correct_citations: compute_correct_citations 的返回结果
            refs: list of reference texts

        Returns:
            tuple: (correct_count, correct_citation_details)
                - correct_count: 正确引用的总数
                - correct_citation_details: list of (statement, citation_set, correct_set)
        """
        correct_count = 0
        correct_citation_details = []  # List to store details of correct citations
        for statement, citation_set in correct_citations:
            correct_set = set()
            # 如果集合大小为1
            if len(citation_set) == 1:
                citation = list(citation_set)[0]  # 使用 list pop 会修改原集合，改用 list 转换
                correct_count += 1
                correct_set.add(citation)
            else:
                for citation in citation_set:
                    remaining_citations = citation_set - {citation}
                    if any(i >= len(refs) for i in remaining_citations):
                        continue
                    premise = " ".join([refs[i] for i in remaining_citations])
                    hypothesis = statement
                    entailment_label = self.nli_entailment(premise, hypothesis)
                    if entailment_label != 0:  # not entailment
                        correct_count += 1
                        correct_set.add(citation)
            correct_citation_details.append((statement, citation_set, correct_set))
        return correct_count, correct_citation_details

    def compute_precision_and_recall(self, question, statements, refs, p=0.60):
        """
        计算精准率和召回率（兼容原 LinkEval 类 API）。

        完全对齐 LINS/Link_Eval.py 中 LinkEval.compute_precision_and_recall() 的逻辑。

        Args:
            question: 问题文本
            statements: list of (text, citation_set)
            refs: list of reference texts
            p: 相关性分数阈值，默认 0.60（对齐论文）

        Returns:
            tuple: (set_precision, precision, recall)
        """
        filtered_statements = [statement for statement in statements if statement[1]]
        total_citations = sum(len(citation_set) for _, citation_set in statements)
        correct_citations = self.compute_correct_citations(filtered_statements, refs)
        correct_count, correct_citation_details = self.compute_correct_citation_count(correct_citations, refs)

        # 使用 DeepSeek 评估引用与问题的相关性（替代原版 medlinker_compute_score）
        valid_ref_indices = self._get_valid_refs(question, refs)
        valid_refs = list(valid_ref_indices)

        # 计算正确且有效的引用
        valid_correct_union = set()
        for _, _, correct_set in correct_citation_details:
            valid_correct_union.update(correct_set & set(valid_refs))

        valid_correct_count = len(valid_correct_union)

        precision = correct_count / total_citations if total_citations > 0 else 1
        recall = valid_correct_count / len(valid_refs) if len(valid_refs) > 0 else 1
        set_precision = len(correct_citations) / len(filtered_statements) if len(filtered_statements) > 0 else 1

        return set_precision, precision, recall

    # ----------------------------------------------------------
    # 核心 NLI 接口
    # ----------------------------------------------------------

    def nli_entailment(self, premise, hypothesis):
        """
        NLI 蕴涵判断

        Returns:
            0: entailment, 1: contradiction, 2: neutral
        """
        return self.nli.nli(premise, hypothesis)

    # ----------------------------------------------------------
    # 正确引用集判定 (论文 Fig.5c)
    # ----------------------------------------------------------

    def _is_correct_citation_set(self, statement, citation_indices, refs):
        """
        判断一个引用集对于某个陈述是否是"正确引用集"。

        论文定义:
        "We define a citation set C for a statement S as a correct citation set
        if the concatenated text of all citations in C contains the meaning of S."

        实现: 将所有引用的文本拼接作为 premise，陈述作为 hypothesis，
        如果 NLI 结果为 entailment (0)，则该引用集是正确的。

        Args:
            statement: 陈述文本
            citation_indices: 引用索引集合
            refs: 所有引用文本列表

        Returns:
            bool: 是否为正确的引用集
        """
        # 过滤越界索引
        valid_indices = [i for i in citation_indices if i < len(refs)]
        if not valid_indices:
            return False

        premise = " ".join([refs[i] for i in valid_indices])
        hypothesis = statement

        ent_label = self.nli_entailment(premise, hypothesis)
        return ent_label == 0  # entailment

    # ----------------------------------------------------------
    # 正确引用判定 (论文 Fig.5d)
    # ----------------------------------------------------------

    def _is_correct_citation(self, statement, citation_set, citation_idx, refs):
        """
        判断一个引用在引用集中是否是"正确引用"。

        论文定义:
        "Suppose C is the correct citation set for statement S. We enumerate all
        citations c in C, and by removing c from C to get C', if the concatenated
        text corresponding to C' can't entail S, then c is deemed irreplaceable
        in C, and thus considered a correct citation."

        简单说: 如果移除该引用后，剩余引用的拼接不再蕴涵陈述，则该引用是正确的。

        Args:
            statement: 陈述文本
            citation_set: 完整的引用集
            citation_idx: 要检查的引用在 refs 中的索引
            refs: 所有引用文本列表

        Returns:
            bool: 是否为正确引用
        """
        # 先检查原始引用集是否为正确引用集
        if not self._is_correct_citation_set(statement, citation_set, refs):
            return False

        # 移除该引用
        remaining = [i for i in citation_set if i != citation_idx]
        if not remaining:
            # 如果移除后没有引用了，说明该引用是不可替代的（因为移除后为空集不蕴涵陈述）
            # 但原论文规定：空集不能蕴涵任何陈述，所以该引用是正确的
            return True

        # 检查移除后剩余引用是否能蕴涵陈述
        premise = " ".join([refs[i] for i in remaining])
        hypothesis = statement
        ent_label = self.nli_entailment(premise, hypothesis)

        # 如果移除后不能蕴涵，则该引用是不可替代的 → 正确引用
        return ent_label != 0  # not entailment → correct citation

    # ----------------------------------------------------------
    # 有效 Ref 判定 (论文 Fig.5c)
    # ----------------------------------------------------------

    def _get_valid_refs(self, question, refs):
        """
        使用 DeepSeek 评估引用与问题的相关性，找出有效 Ref。

        原论文方法:
        "Through statistical analysis, we established a threshold value p=0.60.
        When the ranking score > p, R has a 99.9% probability of being a valid Ref."

        由于 DeepSeek 版本没有检索器的 ranking score，我们使用 DeepSeek
        直接判断每个 ref 是否与问题相关。

        Args:
            question: 问题文本
            refs: 引用文本列表

        Returns:
            set: 有效引用的索引集合
        """
        if not refs:
            return set()

        system_prompt = """You are a relevance evaluator. For each reference passage, determine if it is RELEVANT to the given question.

A passage is RELEVANT if it contains information that directly helps answer the question.
Return ONLY a JSON array of indices (0-based) of relevant passages.
Example: [0, 2, 3]
If none are relevant, return an empty array [].
Do NOT include any explanation."""

        refs_text = "\n".join(f"[{i}] {ref[:200]}" for i, ref in enumerate(refs))
        user_prompt = f"Question: {question}\n\nReferences:\n{refs_text}\n\nRelevant indices (JSON array):"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )

            output = response.choices[0].message.content.strip()

            # 解析 JSON 数组
            try:
                json_match = re.search(r'\[.*?\]', output, re.DOTALL)
                if json_match:
                    valid_indices = json.loads(json_match.group())
                    return set(valid_indices)
            except (json.JSONDecodeError, TypeError):
                pass

            # 如果解析失败，尝试提取数字
            indices = re.findall(r'\d+', output)
            return set(int(i) for i in indices if int(i) < len(refs))

        except Exception as e:
            print(f"  [DeepSeek Relevancy] API error: {e}")
            return set(range(len(refs)))

    # ----------------------------------------------------------
    # 指标计算 (论文 Fig.5c, 5d, 5e)
    # ----------------------------------------------------------

    def compute_citation_metrics(self, question, statements, refs):
        """
        计算所有引用相关的指标。

        论文 Fig.5c-d 定义:
        - Citation Set Precision = RC / NC
          正确引用集数 / 总引用集数

        - Citation Precision = N / M
          正确引用数 / 总引用数

        - Citation Recall = W / V
          正确且有效引用数 / 有效 Ref 总数

        Args:
            question: 问题文本
            statements: list of (text, citation_set)
            refs: list of reference texts

        Returns:
            dict: 包含 set_precision, precision, recall 的字典
        """
        # 过滤有引用的陈述
        filtered = [(s, c) for s, c in statements if c]
        total_statements = len(filtered)

        if total_statements == 0:
            if self.verbose:
                print("  [警告] 没有引用集")
            return {
                "citation_set_precision": 1.0,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "correct_citation_sets": 0,
                "total_citation_sets": 0,
                "correct_citations": 0,
                "total_citations": 0,
                "valid_correct_citations": 0,
                "valid_refs": 0,
            }

        # 统计总引用集数和总引用数
        total_citation_sets = total_statements
        total_citations = sum(len(c) for _, c in filtered)

        # Step 1: 判断每个引用集是否为正确引用集
        correct_set_count = 0
        correct_citation_sets = []  # 存储 (statement, citation_set)

        for statement, citation_set in filtered:
            if self._is_correct_citation_set(statement, citation_set, refs):
                correct_set_count += 1
                correct_citation_sets.append((statement, citation_set))
                if self.verbose:
                    print(f"  [正确引用集] 陈述: {statement[:60]}... → {citation_set}")

        if self.verbose:
            print(f"  正确引用集数: {correct_set_count} / {total_citation_sets}")

        # Step 2: 计算正确引用数（只在正确引用集中计算）
        correct_citation_count = 0

        for statement, citation_set in correct_citation_sets:
            for c in citation_set:
                if self._is_correct_citation(statement, citation_set, c, refs):
                    correct_citation_count += 1
                    if self.verbose:
                        print(f"  [正确引用] idx={c} 在引用集 {citation_set} 中")

        if self.verbose:
            print(f"  正确引用数: {correct_citation_count} / {total_citations}")

        # Step 3: 获取有效 Ref
        valid_ref_indices = self._get_valid_refs(question, refs)
        num_valid_refs = len(valid_ref_indices)

        if self.verbose:
            print(f"  有效 Ref: {valid_ref_indices} (共 {num_valid_refs})")

        # Step 4: 计算正确且有效的引用数 W
        # 论文定义: W = |R_i ∩ (∪_{i=1}^{e} (C*_i))|
        # 即正确引用集合中，同时也是有效 Ref 的个数
        all_correct_citations = set()
        for statement, citation_set in correct_citation_sets:
            for c in citation_set:
                if self._is_correct_citation(statement, citation_set, c, refs):
                    all_correct_citations.add(c)

        valid_correct_count = len(all_correct_citations & valid_ref_indices)

        if self.verbose:
            print(f"  正确且有效引用数: {valid_correct_count}")

        # Step 5: 计算最终指标
        citation_set_precision = (correct_set_count / total_citation_sets
                                  if total_citation_sets > 0 else 1.0)
        citation_precision = (correct_citation_count / total_citations
                              if total_citations > 0 else 1.0)
        citation_recall = (valid_correct_count / num_valid_refs
                           if num_valid_refs > 0 else 1.0)

        return {
            "citation_set_precision": round(citation_set_precision, 4),
            "citation_precision": round(citation_precision, 4),
            "citation_recall": round(citation_recall, 4),
            "correct_citation_sets": correct_set_count,
            "total_citation_sets": total_citation_sets,
            "correct_citations": correct_citation_count,
            "total_citations": total_citations,
            "valid_correct_citations": valid_correct_count,
            "valid_refs": num_valid_refs,
        }

    def compute_statement_correctness(self, statements, correct_answer=None):
        """
        计算陈述正确性 (论文 Fig.5e)。

        论文定义:
        "Statement correctness is used to evaluate the correctness of the model.
        For any pair of statements Si and Sj (i≠j) in {S}, we use an NLI model
        to determine if a conflict relationship exists. If a conflict is detected,
        then statement correctness = 0.

        When there is no conflict between any two statements, we check if {S}
        satisfies the correct answer. If it does, statement correctness = 1;
        otherwise it is 0."

        Args:
            statements: list of (text, citation_set)
            correct_answer: 正确答案文本（可选）。如果提供，会检查陈述是否与正确答案一致。

        Returns:
            int: 1 (正确) 或 0 (不正确)
        """
        texts = [s[0] for s in statements]

        # Step 1: 检查所有两两组合是否存在矛盾
        if len(texts) >= 2:
            for t1, t2 in combinations(texts, 2):
                ent_label = self.nli_entailment(t1, t2)
                if ent_label == 1:  # contradiction
                    if self.verbose:
                        print(f"  [陈述矛盾]")
                        print(f"    S1: {t1[:60]}...")
                        print(f"    S2: {t2[:60]}...")
                    return 0  # 存在矛盾 → 不正确

        # Step 2: 如果提供了正确答案，检查陈述是否与正确答案一致
        if correct_answer is not None and correct_answer:
            # 将正确答案与每个陈述分别做 NLI 检查
            # 只要有一个陈述与正确答案矛盾，就返回 0
            for t in texts:
                ent_label = self.nli_entailment(t, correct_answer)
                if ent_label == 1:  # contradiction
                    if self.verbose:
                        print(f"  [与正确答案矛盾] 陈述: {t[:60]}...")
                    return 0

            # 检查正确答案是否能蕴涵所有陈述的集合（可选）
            # 更严格：检查所有陈述拼接是否与正确答案蕴涵关系
            all_statements = " ".join(texts)
            # 检查正确答案 → 所有陈述 (entailment 或 neutral 都可以)
            # 但不能矛盾
            ent_correct_to_all = self.nli_entailment(correct_answer, all_statements)
            if ent_correct_to_all == 1:  # contradiction with correct answer
                if self.verbose:
                    print(f"  [陈述整体与正确答案矛盾]")
                return 0

        return 1  # 无矛盾且符合答案

    def compute_statement_fluency(self, text):
        """
        评估陈述流畅度 (论文 Fig.5e)。

        Args:
            text: 要评估的文本

        Returns:
            float: 流畅度分数 (0.0 ~ 1.0)
        """
        return self.fluency_evaluator.score_fluency(text)

    # ----------------------------------------------------------
    # 一站式评估接口
    # ----------------------------------------------------------

    def evaluate(self, question, statements, refs, correct_answer=None, return_details=False):
        """
        完整评估：计算所有 LinkEval 指标。

        论文 Fig.5b 定义评估维度:
        - Citation: set precision, precision, recall
        - Statement: correctness, fluency

        Args:
            question: 原始问题
            statements: list of (text, citation_set)
            refs: list of reference texts
            correct_answer: 正确答案（可选，用于 Statement Correctness）
            return_details: 是否返回详细信息

        Returns:
            metrics: dict 包含所有评估指标
            (optional) details: dict 包含详细信息
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LinkEval-DeepSeek 评估")
            print(f"问题: {question[:60]}...")
            print(f"陈述数: {len(statements)}")
            print(f"引用数: {len(refs)}")
            print(f"{'='*60}")

        # 1. 计算引用相关指标 (Citation Set Precision, Citation Precision, Citation Recall)
        citation_metrics = self.compute_citation_metrics(question, statements, refs)

        # 2. 计算 F1 Score
        p = citation_metrics["citation_precision"]
        r = citation_metrics["citation_recall"]
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # 3. 计算陈述正确性
        statement_correctness = self.compute_statement_correctness(statements, correct_answer)

        # 4. 计算流畅度
        all_text = " ".join([s[0] for s in statements])
        statement_fluency = self.compute_statement_fluency(all_text)

        # 5. 构建结果（精简版：去除冗余的 Citation Set Precision 和 Overall Score）
        metrics = {
            "citation_precision": citation_metrics["citation_precision"],
            "citation_recall": citation_metrics["citation_recall"],
            "f1_score": round(f1, 4),
            "statement_correctness": statement_correctness,
            "statement_fluency": round(statement_fluency, 4),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"评估结果:")
            print(f"  Citation Precision:     {metrics['citation_precision']:.4f}")
            print(f"  Citation Recall:        {metrics['citation_recall']:.4f}")
            print(f"  F1 Score:               {metrics['f1_score']:.4f}")
            print(f"  Statement Correctness:  {metrics['statement_correctness']}")
            print(f"  Statement Fluency:      {metrics['statement_fluency']:.4f}")
            print(f"{'='*60}")

        if return_details:
            details = {
                "correct_citation_sets": citation_metrics["correct_citation_sets"],
                "total_citation_sets": citation_metrics["total_citation_sets"],
                "correct_citations": citation_metrics["correct_citations"],
                "total_citations": citation_metrics["total_citations"],
                "valid_correct_citations": citation_metrics["valid_correct_citations"],
                "valid_refs": citation_metrics["valid_refs"],
            }
            return metrics, details

        return metrics


# ============================================================
# 便捷函数：文本格式转换
# ============================================================

def convert_to_statements(text):
    """
    将带引用的文本转换为 statements 列表。

    输入示例:
        "Prasinezumab targets alpha-synuclein [1]. This is supported by trials [2]."

    输出:
        [("Prasinezumab targets alpha-synuclein .", {0}),
         ("This is supported by trials .", {1})]

    Args:
        text: 带 [n] 引用标记的文本

    Returns:
        list of (clean_text, citation_set)
    """
    pattern = re.compile(r"\[(\d+)\]")

    # 以句号分割段落
    sentences = re.split(r'(?<=\.\s)', text.strip())

    statements = []

    for sentence in sentences:
        refs = pattern.findall(sentence)
        ref_numbers = {int(ref) - 1 for ref in refs}  # 1-based → 0-based
        clean_sentence = pattern.sub('', sentence).strip()
        if clean_sentence:
            statements.append((clean_sentence, ref_numbers))

    return statements


def format_metrics(metrics):
    """
    格式化指标输出为易读的表格形式。

    Args:
        metrics: evaluate() 返回的指标字典

    Returns:
        str: 格式化的指标文本
    """
    lines = [
        f"{'Metric':<30} {'Value':<10}",
        "-" * 42,
        f"{'Citation Precision':<30} {metrics.get('citation_precision', 'N/A'):<10}",
        f"{'Citation Recall':<30} {metrics.get('citation_recall', 'N/A'):<10}",
        f"{'F1 Score':<30} {metrics.get('f1_score', 'N/A'):<10}",
        f"{'Statement Correctness':<30} {metrics.get('statement_correctness', 'N/A'):<10}",
        f"{'Statement Fluency':<30} {metrics.get('statement_fluency', 'N/A'):<10}",
    ]
    return "\n".join(lines)



# ============================================================
# 自测代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LinkEval-DeepSeek 自测")
    print("=" * 60)

    # 检查 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请设置: set DEEPSEEK_API_KEY=sk-your-key")
        exit(1)

    print("DEEPSEEK_API_KEY 已找到\n")

    # 初始化评估器
    evaluator = LinkEvalDeepSeek(verbose=True)

    # 测试数据（模拟论文 Fig.5d 的示例）
    test_text = (
        "Prasinezumab targets alpha-synuclein and shows benefit in early Parkinson's [1]. "
        "This is supported by evidence from clinical trials [2]. "
        "Regular exercise may also help manage symptoms [3]."
    )
    test_refs = [
        "Prasinezumab is a monoclonal antibody that targets alpha-synuclein and is being investigated for Parkinson's disease.",
        "Clinical trials have shown that prasinezumab may slow motor progression in early Parkinson's disease patients.",
        "Exercise has been shown to have neuroprotective effects and may help manage motor symptoms in Parkinson's disease.",
    ]
    test_question = "For Parkinson's disease, what treatment options are available?"

    # 转换陈述
    statements = convert_to_statements(test_text)
    print(f"\n转换后的 statements: {statements}")

    # 完整评估
    metrics, details = evaluator.evaluate(
        test_question, statements, test_refs,
        correct_answer="Prasinezumab and exercise can help Parkinson's disease.",
        return_details=True
    )

    print("\n" + "=" * 60)
    print("格式化输出:")
    print(format_metrics(metrics))
    print("=" * 60)
    print("自测完成")
    print("=" * 60)
