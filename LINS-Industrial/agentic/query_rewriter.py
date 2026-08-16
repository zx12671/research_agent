# -*- coding: utf-8 -*-
"""
query_rewriter.py — 语义级多视角 Query 改写器（P1：补全度主杠杆）。

背景
----
前几轮验证（docs/agentic_query_vision_ab_result.md）定位：机械 multi_query（KED 按逗号/子句
切分）会稀释语义，是分解方向根因；若走分解方向，必须换成**语义级子视角**——
复用 TaskAnalysis 的 task/expected_evidence 信号，由 LLM 生成 2~4 个语义互补的子视角 query，
再分别检索后融合。本模块落地该"多视角改写器"。

设计
----
- SemanticQueryRewriter.rewrite(question, task_analysis) -> List[str]：
  - LLM（DeepSeek）根据 task/expected_evidence 生成语义互补的 2~4 个子视角 query（JSON 数组）。
  - 解析失败 / 超时 / 异常 → 降级返回 [question]（与纯 base 等价，不引入噪声）。
- 保底语义锚：返回列表第一个元素恒为原 question，确保基 recall 不丢。
- mock 模式：不触发网络，直接调用 _mock_views() 用确定性规则生成视角，供 A/B 链路先行验证。
- 视角数/长度自适应（见 _plan_view_counts）：comparison>=3 视角，其余按 task 动态取 2~4。

生产纪律
--------
本模块为**独立组件**，默认不进生产分派（pipeline 的 _handle_retrieve 不改）。
A/B 脚本通过 GraphExecutor 子类注入；有正向结论后再沉淀为 pipeline 可选注入。
"""

import json
import logging
import re
import traceback
from typing import Any, List

logger = logging.getLogger(__name__)

# 保留型号/标准号/数值参数的规整辅助（避免改写过程拆碎工业标识）
_ALNUM_STD = re.compile(
    r"\b(?:GB|GB/T|JB/T|JG/T|DL/T|NB/T|QB/T|YY/T|HJ|AQ|TSG)\s*/?\s*\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:℃|°C|K|MPa|kPa|Pa|bar|kV|V|kW|W|mm|cm|m|Hz|rpm|%)"
)


def _squeeze_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class SemanticQueryRewriter:
    """
    语义级多视角 Query 改写器。

    Args:
        llm_client: OpenAI 兼容 / MockLLM 兼容客户端（需 .chat.completions.create）。
        model_name: LLM 模型名（默认 deepseek-chat）。
        num_views: 期望视角数（2~4，实际由 task 与保底共同决定）。
        temperature: 改写温度（低=更稳定）。
        mock: 强制走确定性 mock 视角（不触发网络）。
    """

    def __init__(self, llm_client: Any = None, model_name: str = "deepseek-chat",
                 num_views: int = 3, temperature: float = 0.3,
                 mock: bool = False):
        self.llm = llm_client
        self.model_name = model_name
        self.num_views = max(1, min(4, int(num_views)))
        self.temperature = float(temperature)
        self.mock = bool(mock)
        self.last_views: List[str] = []
        self.last_degraded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rewrite(self, question: str, task_analysis: Any = None) -> List[str]:
        """返回 1+ 个语义子视角 query 列表（含原问题保底）。失败降级 [question]。"""
        q = _squeeze_ws(question)
        if not q:
            return [""]
        if self.mock or self.llm is None:
            views = self._mock_views(q, task_analysis)
        else:
            views = self._llm_views(q, task_analysis)
        return self._finalize(q, views)

    # ------------------------------------------------------------------
    # LLM 改写（真实模式）
    # ------------------------------------------------------------------
    def _llm_views(self, q: str, task_analysis: Any = None) -> List[str]:
        task = self._field(task_analysis, "task")
        expected = self._field(task_analysis, "expected_evidence", "general")
        n = self._plan_view_counts(task)
        prompt = (
            "你是工业知识库检索的查询改写器。给定一个工业问答及其任务类型/期望证据类型，\n"
            f"生成 {n} 个【语义互补、两两不重叠】的子视角检索 query。\n\n"
            f"原问题：{q}\n"
            f"任务类型：{task}\n"
            f"期望证据类型：{expected}\n\n"
            "要求：\n"
            "1. 每个视角聚焦独立语义面（如：标准号维度 / 数值参数维度 / 对象原理维度 / 工程实操维度）；\n"
            "2. 保留型号、标准号、数值+单位原文（如 GB/T 20476、380V、65℃），不得拆分改写；\n"
            "3. 视角之间尽量不重叠，合起来覆盖原问题全部语义；\n"
            "4. 只输出 JSON 数组（query 字符串列表），不要任何其他文字。\n"
            'OUTPUT FORMAT (JSON array only):\n["视角1", "视角2", ...]'
        )
        try:
            resp = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=self.temperature,
            )
            content = ""
            if resp is not None:
                content = getattr(getattr(resp, "choices", [{}])[0].message, "content", "") or ""
            views = self._parse_json_array(content)
            if not views:
                self.last_degraded = True
                return []
            return views
        except Exception:
            logger.warning("SemanticQueryRewriter LLM 改写失败，降级为原问题:\n%s",
                           traceback.format_exc(limit=2))
            self.last_degraded = True
            return []


    # ------------------------------------------------------------------
    # 确定性 mock 视角（零成本，验证链路）
    # ------------------------------------------------------------------
    def _mock_views(self, q: str, task_analysis: Any = None) -> List[str]:
        n = self._plan_view_counts(self._field(task_analysis, "task"))
        views = [q]

        stds = [m.group(0).strip() for m in _ALNUM_STD.finditer(q)]
        nums = [m.group(0).strip() for m in _NUM_UNIT.finditer(q)]

        # 视角2：标准/规范维度
        if stds:
            views.append("标准与规范：" + "；".join(stds))
        # 视角3：数值参数维度
        if nums:
            views.append("数值参数：" + "；".join(nums[:4]))
        # 视角4：对象/原理维度（取核心名词短语，启发式）
        core = self._core_phrase(q)
        if core and core not in (views + [q]):
            views.append("原理与机理：" + core)

        # 视角数保底：若不足 n，用原问题补足
        while len(views) < n:
            views.append(q)
        # 去重保序，截断到最多 4 个
        dedup, seen = [], set()
        for v in views:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        return dedup[:4]

    def _core_phrase(self, q: str) -> str:
        """启发式抽取核心对象短语（去掉疑问词/场景前缀），用于 mock 视角补足。"""
        s = re.sub(r"(请问|求|什么是|如何|为什么|怎样|多少|是否)", " ", q)
        s = re.sub(r"\?|？|。|；|，", " ", s)
        s = _squeeze_ws(s)
        return s[:60] if s else ""

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _plan_view_counts(self, task: Any) -> int:
        t = str(task).lower()
        if "comparison" in t or "compare" in t:
            return 3
        if "procedure" in t:
            return 4
        return max(2, self.num_views)

    def _finalize(self, q: str, views: List[str]) -> List[str]:
        """保底原问题 + 规范 + 去重 + 截断。"""
        out: List[str] = [q]
        for v in views:
            v = _squeeze_ws(v)
            if not v or v == q or v in out:
                continue
            out.append(v[:120])
        # 视角数上界 4，下界 1（原问题恒在）
        self.last_views = out[:4]
        return self.last_views

    @staticmethod
    def _field(obj: Any, name: str, default: str = "") -> Any:
        if obj is None:
            return default
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    @staticmethod
    def _parse_json_array(content: str) -> List[str]:
        content = (content or "").strip()
        # 去除可能的 markdown 代码围栏
        content = re.sub(r"```(?:json)?", "", content).strip()
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            content = m.group(0)
        data = None
        try:
            data = json.loads(content)
        except Exception:
            data = None
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
