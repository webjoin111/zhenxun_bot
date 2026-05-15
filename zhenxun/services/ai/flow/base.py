from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import contextlib
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from zhenxun.services.ai.run.models import StreamedRunResult

T_RunResult = TypeVar("T_RunResult")


class BaseRuntimeConfig(BaseModel):
    """所有可执行实体（Agent/Team/Workflow）的通用基础运行时配置"""

    stateless: bool = Field(default=True)
    """是否使用临时会话，不持久化历史记录"""
    ui_streamer: str | None = Field(default=None)
    """自动绑定的前端UI渲染器标识符（如 'markdown'）"""


class BaseRunnable(ABC, Generic[T_RunResult]):
    """
    所有可执行 AI 编排实体的统一基类 (Composite Pattern)。
    统一了 Agent, Team, Workflow 的核心契约，支持物理上的任意嵌套。
    """

    name: str
    """可执行实体的名称标识"""

    description: str
    """可执行实体的详细描述。用于外部路由(Router)或上层智能体(DelegateTool)决定是否调用它"""

    persona: Any | None = None
    """(可选) 实体的角色设定 (Persona)。包含 role 和 goal，在多智能体路由移交时优先级最高"""

    runtime_config: BaseRuntimeConfig
    """运行时配置，如是否无状态、UI输出模式等"""

    def bind(self, **kwargs: Any) -> Any:
        """DI 注入语法糖：返回 Depends，自动绑定当前上下文"""
        from nonebot.params import Depends

        from zhenxun.services.ai.flow.agent.bridge import AgentRunner

        async def _dependency() -> AgentRunner[Any]:
            return AgentRunner[Any](self, **kwargs)

        return Depends(_dependency)

    async def reply(
        self,
        prompt: Any = None,
        reply_to: bool = False,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> T_RunResult:
        """交互执行语法糖，自动渲染流式进度并最终将结果回复给终端用户"""
        from zhenxun.services.ai.flow.agent.bridge import AgentRunner

        runner = AgentRunner(self, context=context, **kwargs)
        return cast(T_RunResult, await runner.reply(prompt=prompt, reply_to=reply_to))

    async def run(
        self, prompt: Any = None, *, context: Any = None, **kwargs: Any
    ) -> T_RunResult:
        """阻塞式核心运行入口，默认通过 run_stream 消费提取结果"""
        async with self.run_stream(
            prompt=prompt, context=context, **kwargs
        ) as stream_result:
            return cast(T_RunResult, await stream_result.get_run_result())

    @abstractmethod
    @contextlib.asynccontextmanager
    async def run_stream(
        self, prompt: Any = None, *, context: Any = None, **kwargs: Any
    ) -> "AsyncIterator[StreamedRunResult[Any]]":
        """流式运行入口，返回上下文管理器，用于消费底层执行流事件 (StreamedRunResult)"""
        yield cast(Any, None)
