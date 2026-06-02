"""
LLM 中间件协议定义
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import LLMMessage, LLMResponse
from zhenxun.services.ai.core.models import ToolChoice


class LLMContext(BaseModel):
    """LLM 执行上下文，用于在中间件管道中传递请求状态"""

    messages: list[LLMMessage]
    """当前请求的消息列表。"""
    config: Any
    """本次调用使用的模型生成配置。"""
    tools: list[Any] | None
    """可供模型调用的工具集合。"""
    tool_choice: str | dict[str, Any] | ToolChoice | None
    """工具调用策略或指定工具配置。"""
    timeout: float | None
    """本次调用的超时时间（秒）。"""
    extra: dict[str, Any] = Field(default_factory=dict)
    """透传给模型层的附加参数。"""
    request_type: Literal[
        "generation", "embedding", "rerank", "image_generation", "speech_generation"
    ] = "generation"
    """请求类型标识。"""
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    """中间件运行时的临时状态存储。"""
    cancellation_token: Any | None = Field(default=None)
    """全局取消令牌。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


NextCall = Callable[[LLMContext], Awaitable[LLMResponse]]
LLMMiddleware = Callable[[LLMContext, NextCall], Awaitable[LLMResponse]]


class BaseLLMMiddleware(ABC):
    """LLM 中间件抽象基类"""

    @abstractmethod
    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        pass
