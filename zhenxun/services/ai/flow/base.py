from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import contextlib
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.run.context import RunContext

if TYPE_CHECKING:
    from zhenxun.services.ai.flow.agent.models import Persona
    from zhenxun.services.ai.run.models import StreamedRunResult

from zhenxun.services.ai.core.messages import PromptInput

T_RunResult = TypeVar("T_RunResult")


class ConcurrencyPolicy(str, Enum):
    """并发执行策略枚举"""

    ALLOW = "allow"
    """允许并发：不做任何限制（适用于无状态或绝对独立任务）"""
    REJECT = "reject"
    """拒绝新请求：当前有任务在执行时，直接丢弃新任务并提醒"""
    QUEUE = "queue"
    """排队等待：当前有任务在执行时，新任务排队等待（先进先出）"""
    INTERRUPT = "interrupt"
    """中断旧任务：新任务到达时，立即强制取消并覆盖正在执行的旧任务"""


class BaseRuntimeConfig(BaseModel):
    """所有可执行实体（Agent/Team/Workflow）的通用基础运行时配置"""

    stateless: bool = Field(default=True)
    """是否使用临时会话，不持久化历史记录"""
    concurrency_policy: ConcurrencyPolicy | None = Field(default=None)
    """并发执行策略。如果未显式指定，无状态(stateless=True)默认为ALLOW，有状态(stateless=False)默认为QUEUE。"""


class BaseRunnable(ABC, Generic[T_RunResult]):
    """
    所有可执行 AI 编排实体的统一基类 (Composite Pattern)。
    统一了 Agent, Team, Workflow 的核心契约，支持物理上的任意嵌套。
    """

    name: str
    """可执行实体的名称标识"""

    description: str
    """可执行实体的详细描述。用于外部路由(Router)或上层智能体(DelegateTool)决定是否调用它"""

    persona: "Persona | dict | None" = None
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
        prompt: PromptInput | None = None,
        reply_to: bool = False,
        *,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> T_RunResult:
        """交互执行语法糖，自动渲染流式进度并最终将结果回复给终端用户"""
        from zhenxun.services.ai.flow.agent.bridge import AgentRunner

        runner = AgentRunner(self, context=context, **kwargs)
        return cast(T_RunResult, await runner.reply(prompt=prompt, reply_to=reply_to))

    async def run(
        self,
        prompt: PromptInput | None = None,
        *,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> T_RunResult:
        """阻塞式核心运行入口，安全捕获内部抛出的静默退出信号"""
        from zhenxun.services.ai.core.exceptions import ControlFlowExit
        from zhenxun.services.log import logger

        try:
            async with self.run_stream(
                prompt=prompt, context=context, **kwargs
            ) as stream_result:
                return cast(T_RunResult, await stream_result.get_run_result())
        except ControlFlowExit as e:
            logger.info(f"[{self.name}] 触发底层控制流，已安全退出: {e}")

            display_msg = getattr(e, "display", None) or getattr(
                e, "display_content", None
            )
            if display_msg:
                try:
                    from nonebot_plugin_alconna import UniMessage

                    if isinstance(display_msg, UniMessage):
                        bot = context.get_bot() if context else None
                        event = context.get_event() if context else None
                        if bot and event:
                            await display_msg.send(event, bot=bot)
                    else:
                        from zhenxun.utils.message import MessageUtils

                        await MessageUtils.build_message(str(display_msg)).send()
                except Exception:
                    pass

            import asyncio

            raise asyncio.CancelledError()

    @abstractmethod
    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: PromptInput | None = None,
        *,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> "AsyncIterator[StreamedRunResult[Any]]":
        """流式运行入口，返回上下文管理器，用于消费底层执行流事件 (StreamedRunResult)"""
        yield cast(Any, None)
