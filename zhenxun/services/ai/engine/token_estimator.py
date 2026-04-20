"""
LLM Token 动态预估与上下文管理模块
"""

from abc import ABC, abstractmethod
import math
import re
from typing import Any

from zhenxun.services.ai.types.messages import (
    AudioPart,
    FilePart,
    ImagePart,
    LLMMessage,
    TextPart,
    ThoughtPart,
    ToolCallPart,
    ToolMessage,
    ToolReturnPart,
    UsageInfo,
    VideoPart,
)


class BaseTokenEstimator(ABC):
    """
    Token 预估器协议接口
    允许第三方开发者未来挂载 tiktoken 等硬核离线分词库
    """

    @abstractmethod
    def estimate_message(self, msg: LLMMessage, model_name: str) -> int:
        """估算单条消息的 Token 数量"""
        pass

    @abstractmethod
    def estimate_context(
        self, messages: list[LLMMessage], model_name: str, base_overhead: int = 0
    ) -> int:
        """估算整个对话上下文的 Token 数量"""
        pass

    @abstractmethod
    def calibrate(
        self,
        actual_prompt_tokens: int,
        estimated_messages: list[LLMMessage],
        model_name: str,
    ) -> None:
        """接收 API 的后验反馈 (静态引擎中此方法为空，动态锚定将由 Session 完成)"""
        pass


def estimate_tools_schema(obj: dict | list | str | Any) -> int:
    """
    启发式预估工具 JSON Schema 的 Token 开销 (The Hidden Token Tax)。
    根据研究，每个 Key 约占 12 Tokens，描述文本(description)按字符计算。
    """
    if isinstance(obj, dict):
        cost = len(obj.keys()) * 12
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                cost += int(len(v) * 0.3)
            else:
                cost += estimate_tools_schema(v)
        return cost
    elif isinstance(obj, list):
        return sum(estimate_tools_schema(item) for item in obj)
    return 0


class StaticTokenEngine(BaseTokenEstimator):
    """
    静态多模态计算引擎 (Static Rule Engine)
    基于第一性原理数学计算，摆脱外部库依赖，提供绝对稳定的预估基线。
    """

    def _estimate_text(self, text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
        ascii_chars = len(text) - cjk_chars
        return math.ceil(cjk_chars * 1.2 + ascii_chars * 0.3)

    def _estimate_image(self, resolution_hint: str | None, model_name: str) -> int:
        if "gemini" in model_name.lower():
            res = (resolution_hint or "").upper()
            if "ULTRA_HIGH" in res:
                return 6192
            if "HIGH" in res:
                return 3096
            if "LOW" in res:
                return 258
            return 1032
        return 765

    def estimate_message(self, msg: LLMMessage, model_name: str) -> int:
        if msg.token_cost is not None:
            return msg.token_cost

        total_tokens = 4

        if isinstance(msg, ToolMessage):
            total_tokens += 40

        if isinstance(msg.content, str):
            total_tokens += self._estimate_text(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, TextPart) and part.text:
                    total_tokens += self._estimate_text(part.text)
                elif isinstance(part, ImagePart):
                    total_tokens += self._estimate_image(
                        getattr(part, "media_resolution", None), model_name
                    )
                elif isinstance(
                    part,
                    (
                        VideoPart,
                        AudioPart,
                        FilePart,
                    ),
                ):
                    total_tokens += 1032
                elif isinstance(part, ThoughtPart) and part.thought_text:
                    total_tokens += self._estimate_text(part.thought_text)
                elif isinstance(part, ToolCallPart) and part.args:
                    total_tokens += self._estimate_text(str(part.args))
                elif isinstance(part, ToolReturnPart) and part.output:
                    total_tokens += self._estimate_text(str(part.output))

        return total_tokens

    def estimate_context(
        self, messages: list[LLMMessage], model_name: str, base_overhead: int = 0
    ) -> int:
        if not messages:
            return base_overhead
        return base_overhead + sum(
            self.estimate_message(msg, model_name) for msg in messages
        )

    def calibrate(
        self,
        actual_prompt_tokens: int,
        estimated_messages: list[LLMMessage],
        model_name: str,
    ) -> None:
        pass


global_estimator = StaticTokenEngine()


def parse_usage_info(usage_info: dict | None) -> UsageInfo:
    """
    全协议统一遥测解析器 (Universal Telemetry Parser)
    兼容 OpenAI Standard、OpenAI Responses (v1/responses) 以及 Gemini (usageMetadata)。
    """
    if not usage_info or not isinstance(usage_info, dict):
        return UsageInfo()

    prompt = 0
    completion = 0
    total = 0
    cache_hit = 0
    cache_miss = 0
    reasoning = 0

    if "promptTokenCount" in usage_info or "candidatesTokenCount" in usage_info:
        prompt = usage_info.get("promptTokenCount", 0)
        completion = usage_info.get("candidatesTokenCount", 0)
        total = usage_info.get("totalTokenCount", 0)
        reasoning = usage_info.get("thoughtsTokenCount", 0)
        cache_hit = usage_info.get("cachedContentTokenCount", 0)

    elif "input_tokens" in usage_info or "output_tokens" in usage_info:
        prompt = usage_info.get("input_tokens", 0)
        completion = usage_info.get("output_tokens", 0)
        total = usage_info.get("total_tokens", 0)
        cache_hit = (usage_info.get("input_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        reasoning = (usage_info.get("output_tokens_details") or {}).get(
            "reasoning_tokens", 0
        )

    else:
        prompt = usage_info.get("prompt_tokens", 0)
        completion = usage_info.get("completion_tokens", 0)
        total = usage_info.get("total_tokens", 0)

        cache_hit = usage_info.get("prompt_cache_hit_tokens") or (
            usage_info.get("prompt_tokens_details") or {}
        ).get("cached_tokens", 0)
        cache_miss = usage_info.get("prompt_cache_miss_tokens", 0)
        reasoning = (usage_info.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        )

    if cache_miss == 0 and prompt > 0:
        cache_miss = max(0, prompt - cache_hit)

    return UsageInfo(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_cache_hit_tokens=cache_hit,
        prompt_cache_miss_tokens=cache_miss,
        reasoning_tokens=reasoning,
    )
