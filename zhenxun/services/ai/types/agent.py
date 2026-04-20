"""
Agent 相关核心类型定义
"""

import asyncio
from collections.abc import Callable
from typing import Any, Generic
from typing_extensions import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .configs import LLMGenerationConfig
from .messages import LLMMessage, UsageInfo


class CancellationToken:
    """全局取消令牌，用于在异步链路中传递中止信号"""

    def __init__(self):
        self._cancelled = False
        self._futures: list[asyncio.Future] = []

    def cancel(self) -> None:
        self._cancelled = True
        for f in self._futures:
            if not f.done():
                f.cancel()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError(
                "任务已被主动取消 (CancellationToken triggered)"
            )

    def link_future(self, future: asyncio.Future) -> None:
        if self._cancelled:
            future.cancel()
        else:
            self._futures.append(future)


AgentDepsT = TypeVar("AgentDepsT", default=Any)

OutputDataT = TypeVar("OutputDataT", default=str)


class AgentRunResult(BaseModel, Generic[OutputDataT]):
    """Agent 单次无状态运行的结果"""

    output: Any = Field(description="最终输出数据")
    messages: list[LLMMessage] = Field(
        default_factory=list, description="本次运行产生/更新的历史消息"
    )
    usage: UsageInfo = Field(
        default_factory=UsageInfo, description="本次运行的Token消耗总计"
    )
    handoff_target: str | None = Field(default=None, description="移交的目标Agent名称")
    handoff_args: dict[str, Any] | None = Field(
        default=None, description="移交附带的参数"
    )
    handoff_payload: dict[str, Any] | None = Field(
        default=None, description="移交携带的强类型Payload数据"
    )

    class Config:
        arbitrary_types_allowed = True


class ExecutionConfig(BaseModel):
    """Agent 执行引擎的配置"""

    max_cycles: int = Field(default=10, description="工具调用最大循环次数")
    enable_parallel_calls: bool = Field(default=True, description="允许并行工具调用")
    reflexion_retries: int = Field(default=1, description="反思重试次数")


class AgentConfig(BaseModel):
    """Agent 的核心配置模型"""

    instruction: str = Field(default="", description="Agent系统指令")
    model: str | Callable[[], str] | None = Field(
        default=None, description="模型名称或返回模型名的函数"
    )
    handoffs: list[Any] | None = Field(default=None, description="允许移交的目标")
    tools: list[str | Any] | None = Field(default=None, description="绑定工具列表")
    knowledge: list[Any] | Any | None = Field(
        default=None, description="挂载的知识库(BaseKnowledge)"
    )
    skills: list[str] | None = Field(default=None, description="静态预装技能列表")
    available_skills: list[str] | None = Field(
        default=None, description="动态发现技能列表(Catalog)"
    )
    namespace: str | None = Field(default=None, description="所属命名空间")
    resources: list[str] | None = Field(default=None, description="MCP 资源")
    prompts: list[str] | None = Field(default=None, description="MCP Prompt")
    generation_config: LLMGenerationConfig | None = Field(
        default=None, description="LLM 生成配置"
    )
    response_model: type[BaseModel] | None = Field(
        default=None, description="结构化响应模型"
    )
    memory_reducers: list[str | Any] | None = Field(
        default=None, description="记忆压缩策略"
    )
    context_threshold: float | None = Field(default=None, description="压缩触发阈值")
    max_history_turns: int | None = Field(
        default=None, description="触发压缩的对话条数上限"
    )
    callbacks: list[Any] | None = Field(default=None, description="生命周期回调")
    system_prompts: list[Any] | None = Field(default=None, description="动态系统提示词")
    result_validators: list[Any] | None = Field(default=None, description="结果校验")
    stateless: bool = Field(default=True, description="是否无状态执行")

    model_config = ConfigDict(arbitrary_types_allowed=True)
