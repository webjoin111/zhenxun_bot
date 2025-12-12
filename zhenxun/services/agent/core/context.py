"""
LLM 执行上下文管理

提供一个机制，允许正在执行的工具安全地回调LLM服务，以实现递归思考或任务委托。
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from zhenxun.services.llm.types import (
        LLMContentPart,
        LLMMessage,
        LLMResponse,
        ModelName,
    )
    from zhenxun.services.llm.types.models import ToolChoice

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentContext:
    """
    Agent 执行上下文，封装会话必需数据用于参数透传与记忆。
    """

    session_id: str
    user_input: str
    message_history: list["LLMMessage"] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)


class ToolTrustPolicy(BaseModel):
    """定义工具执行的信任策略"""

    trust_all: bool = False
    trusted_servers: list[str] | None = None

    def trusts_server(self, server_name: str) -> bool:
        """检查此策略是否信任给定的服务器"""
        if self.trust_all:
            return True
        if self.trusted_servers and server_name in self.trusted_servers:
            return True
        return False


class LLMInterface(Protocol):
    """
    一个协议，定义了工具在执行期间可以安全调用的LLM能力。
    这是对完整LLM服务的一个受限、安全的子集。
    """

    async def chat(
        self,
        message: "str | LLMMessage | list[LLMContentPart]",
        *,
        model: "ModelName" = None,
        tools: list[dict[str, Any] | str] | None = None,
    ) -> "LLMResponse":
        """
        执行一次无状态的、一次性的聊天调用。
        它不会影响或使用调用者（工具）的外部会话历史。
        """
        ...

    async def generate_structured(
        self,
        message: "str | LLMMessage | list[LLMContentPart]",
        response_model: type[T],
        *,
        model: "ModelName" = None,
        tools: list[dict[str, Any] | str] | None = None,
        tool_choice: "str | dict[str, Any] | ToolChoice | None" = None,
        instruction: str | None = None,
    ) -> T:
        """
        执行一次无状态的、一次性的结构化内容生成。
        """
        ...


_llm_interface_context: ContextVar[LLMInterface | None] = ContextVar(
    "llm_interface_context", default=None
)
_tool_trust_policy_context: ContextVar[ToolTrustPolicy | None] = ContextVar(
    "tool_trust_policy_context", default=None
)


def get_llm_interface() -> LLMInterface | None:
    """
    供工具开发者使用的公共API。
    在工具的 execute 方法内部调用，以获取LLM能力的接口。
    如果在AgentExecutor的工具执行上下文之外调用，将返回None。

    返回:
        LLMInterface | None: 一个可用的LLM接口实例，或None。
    """
    return _llm_interface_context.get()


def get_tool_trust_policy() -> ToolTrustPolicy | None:
    """
    在工具执行期间获取当前的工具信任策略。
    如果在 with_tool_trust_policy 上下文之外调用，将返回None。

    返回:
        ToolTrustPolicy | None: 一个可用的信任策略实例，或None。
    """
    return _tool_trust_policy_context.get()


@asynccontextmanager
async def with_llm_interface(interface: LLMInterface):
    """
    一个异步上下文管理器，用于在执行代码块期间设置LLM接口上下文。
    """
    token = _llm_interface_context.set(interface)
    try:
        yield
    finally:
        _llm_interface_context.reset(token)


@asynccontextmanager
async def with_tool_trust_policy(policy: ToolTrustPolicy):
    """
    一个异步上下文管理器，用于在执行代码块期间设置工具信任策略。
    """
    token = _tool_trust_policy_context.set(policy)
    try:
        yield
    finally:
        _tool_trust_policy_context.reset(token)
