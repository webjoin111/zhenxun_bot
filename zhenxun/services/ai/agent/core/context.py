"""
LLM 执行上下文管理

提供一个机制，允许正在执行的工具安全地回调LLM服务，以实现递归思考或任务委托。
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TypeVar

from pydantic import BaseModel

from zhenxun.services.ai.protocols.llm import LLMInterface

T = TypeVar("T", bound=BaseModel)


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
