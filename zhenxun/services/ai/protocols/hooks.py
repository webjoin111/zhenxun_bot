"""
生命周期拦截器域类型定义
"""

from typing import Any, Protocol, runtime_checkable

from zhenxun.services.ai.types.messages import LLMMessage, LLMResponse


@runtime_checkable
class BeforeLLMCallHook(Protocol):
    """LLM 调用前钩子协议"""

    async def __call__(
        self, messages: list[LLMMessage], kwargs: dict[str, Any]
    ) -> list[LLMMessage]: ...


@runtime_checkable
class AfterLLMCallHook(Protocol):
    """LLM 调用后钩子协议"""

    async def __call__(
        self, response: LLMResponse, kwargs: dict[str, Any]
    ) -> LLMResponse: ...


__all__ = [
    "AfterLLMCallHook",
    "BeforeLLMCallHook",
]
