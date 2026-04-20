"""
LLM 中间件协议定义
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.types.messages import LLMMessage, LLMResponse
from zhenxun.services.ai.types.tools import ToolChoice


class LLMContext(BaseModel):
    """LLM 执行上下文，用于在中间件管道中传递请求状态"""

    messages: list[LLMMessage]
    config: Any
    tools: list[Any] | None
    tool_choice: str | dict[str, Any] | ToolChoice | None
    timeout: float | None
    extra: dict[str, Any] = Field(default_factory=dict)
    request_type: Literal["generation", "embedding", "rerank"] = "generation"
    runtime_state: dict[str, Any] = Field(
        default_factory=dict,
        description="中间件运行时的临时状态存储",
    )
    cancellation_token: Any | None = Field(default=None, description="全局取消令牌")

    model_config = ConfigDict(arbitrary_types_allowed=True)


NextCall = Callable[[LLMContext], Awaitable[LLMResponse]]
LLMMiddleware = Callable[[LLMContext, NextCall], Awaitable[LLMResponse]]


class BaseLLMMiddleware(ABC):
    """LLM 中间件抽象基类"""

    @abstractmethod
    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        pass
