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
from zhenxun.services.ai.core.options import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)
from zhenxun.services.ai.run.models import CancellationToken


class LLMContext(BaseModel):
    """LLM 执行上下文，用于在中间件管道中传递请求状态"""

    messages: list[LLMMessage]
    """当前请求的消息列表。"""
    config: GenerationConfig | LLMEmbeddingConfig | TTSConfig | None
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
    cancellation_token: CancellationToken | None = Field(default=None)
    """全局取消令牌。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


NextCall = Callable[[LLMContext], Awaitable[LLMResponse]]
"""中间件管道中下一个节点（或最终执行函数）的调用签名"""
LLMMiddleware = Callable[[LLMContext, NextCall], Awaitable[LLMResponse]]
"""LLM 中间件函数的调用签名，遵循洋葱模型嵌套包裹设计"""


class BaseLLMMiddleware(ABC):
    """LLM 中间件抽象基类"""

    @abstractmethod
    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        pass
