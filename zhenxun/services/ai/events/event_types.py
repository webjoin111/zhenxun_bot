from typing import Any

from pydantic import Field

from .base import AIEvent


class AgentStartEvent(AIEvent):
    messages: list[Any]


class AgentEndEvent(AIEvent):
    final_history: list[Any]
    duration_ms: float


class ModelStartEvent(AIEvent):
    model_name: str
    messages: list[Any]


class ModelEndEvent(AIEvent):
    response: Any
    duration_ms: float


class ToolCallEvent(AIEvent):
    """工具准备执行前触发。监听器可直接修改 arguments，或抛出异常以拦截执行"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    context: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolErrorEvent(AIEvent):
    """工具执行抛出异常时触发。监听器可通过设置 recovered_result 来修复错误"""

    tool_call_id: str
    tool_name: str
    error: Exception
    recovered_result: Any | None = None


class ToolResultEvent(AIEvent):
    """工具执行结束（无论成功或被修复）时触发"""

    tool_call_id: str
    tool_name: str
    result: Any | None
    error: Exception | None
    duration_ms: float


class ToolStreamEvent(AIEvent):
    """工具执行过程中产生流式输出时触发"""

    tool_call_id: str
    tool_name: str
    chunk: Any


class SandboxExecutionStartedEvent(AIEvent):
    session_id: str | None = None
    code: str


class SandboxExecutionCompletedEvent(AIEvent):
    session_id: str | None = None
    exit_code: int
    duration_ms: float
