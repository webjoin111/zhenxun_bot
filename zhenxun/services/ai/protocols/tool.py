"""
工具执行与管理协议定义
"""

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from zhenxun.services.ai.types.tools import ToolDefinition, ToolResult


class ToolExecutable(Protocol):
    """
    一个协议，定义了所有可被LLM调用的工具必须实现的行为。
    它将工具的"定义"（给LLM看）和"执行"（由框架调用）封装在一起。
    """

    async def get_definition(self, context: Any | None = None) -> ToolDefinition | None:
        """
        异步地获取一个结构化的工具定义。如果返回 None，则该工具对大模型不可见。
        """
        ...

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        """
        异步执行工具并返回一个结构化的结果。
        """
        ...

    async def should_confirm(self, **kwargs: Any) -> str | None:
        """
        [可选] 异步判定工具执行前是否需要用户交互确认。
        如果需要确认，返回一段提示文本；否则返回 None。
        """
        ...

@runtime_checkable
class ToolResolvable(Protocol):
    """
    鸭子类型解析协议。任何实现了此协议的对象，
    都可以直接被 Agent 或 LLM 的 tools 参数接收。
    """

    async def __resolve_to_tools__(self) -> list[ToolExecutable]: ...


ToolNextCall = Callable[[dict[str, Any], Any], Awaitable[ToolResult]]


class ToolMiddleware(Protocol):
    """
    工具执行中间件协议。
    洋葱模型：拦截执行、修改参数、修改上下文、校验权限、金币扣除或篡改返回值。
    """

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: Any,
        next_call: ToolNextCall,
    ) -> ToolResult: ...


class ToolProvider(Protocol):
    """
    一个协议，定义了"工具提供者"的行为。
    工具提供者负责发现或实例化具体的 ToolExecutable 对象。
    """

    async def initialize(self) -> None:
        """
        异步初始化提供者。
        """
        ...

    async def discover_tools(
        self,
        allowed_servers: list[str] | None = None,
        excluded_servers: list[str] | None = None,
    ) -> dict[str, ToolExecutable]:
        """
        异步发现此提供者提供的所有工具。
        """
        ...

    async def get_tool_executable(
        self, name: str, config: dict[str, Any]
    ) -> ToolExecutable | None:
        """
        如果此提供者能处理名为 'name' 的工具，则返回一个可执行实例。
        """
        ...
