from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel

from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.flow.team.models import TeamRuntimeConfig
from zhenxun.services.ai.flow.team.strategy import (
    BaseTeamStrategy,
)
from zhenxun.services.ai.run import AgentRunResult, RunContext, Task


class Team(BaseRunnable[AgentRunResult[Any]]):
    """
    多智能体动态编排与路由控制器 (Facade)。
    继承自 BaseRunnable，支持被嵌套在其他 Team 或 Workflow 中。
    """

    def __init__(
        self,
        name: str,
        members: list[BaseRunnable[Any]],
        strategy: BaseTeamStrategy,
        description: str | None = None,
        persona: Any | None = None,
        runtime_config: TeamRuntimeConfig | dict | None = None,
    ):
        """
        多智能体协作团队初始化。

        参数:
            name: 团队的名称标识。
            members: 团队成员列表，可以包含 Agent、Workflow 或其他 Team。
            strategy: 团队协作策略实例。
            description: 团队的职能描述，用于上层节点路由。
            persona: 团队的整体人设或宏观设定。
            runtime_config: 团队级别的运行时宏观配置。
        """
        self.name = name
        self.members = members
        self.strategy = strategy
        self.description = (
            description
            or f"一个名为 {self.name} 的协作团队，包含 {len(self.members)} 个处理节点。"
        )
        self.persona = persona

        if isinstance(runtime_config, dict):
            runtime_config = TeamRuntimeConfig(**runtime_config)
        self.runtime_config = runtime_config or TeamRuntimeConfig(stateless=True)

        self.selector_func = getattr(strategy, "selector_func", None)


    async def run(
        self,
        prompt: str | Task | None = None,
        *,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        团队级运行阻塞核心入口，内部静默分配任务给成员直至汇总结束。

        参数:
            prompt: 派发给多智能体团队的任务描述 or 契约对象 (Task)。
            context: 显式传入的会话与运行上下文。
            kwargs: 透传的其他附加参数。

        返回:
            AgentRunResult[Any]: 包含最终融合输出、消息历史和用量统计的运行结果对象。
        """
        return await super().run(prompt=prompt, context=context, **kwargs)

    import contextlib

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: str | Task | None = None,
        *,
        context: RunContext | None = None,
        **kwargs: Any,
    ):
        import asyncio

        if context is None:
            context = RunContext()


        from zhenxun.services.ai.core.stream_events import EventStreamer
        from zhenxun.services.ai.flow.team.runner import TeamRunner
        from zhenxun.services.ai.run import StreamedRunResult

        streamer = EventStreamer()
        context.run.streamer = streamer
        runner = TeamRunner(self, self.strategy)

        policy = getattr(self.runtime_config, "concurrency_policy", None)
        if policy is None:
            from zhenxun.services.ai.flow.base import ConcurrencyPolicy

            policy = (
                ConcurrencyPolicy.ALLOW
                if getattr(self.runtime_config, "stateless", True)
                else ConcurrencyPolicy.QUEUE
            )

        async def _execution_task():
            from zhenxun.services.ai.run.models import CancellationToken
            from zhenxun.services.ai.run.session_manager import session_manager

            cancel_token = context.run.cancellation_token or CancellationToken()
            context.run.cancellation_token = cancel_token

            try:
                async with session_manager.apply_concurrency_policy(
                    session_id=context.session_id or "default_session",
                    policy=policy,
                    cancel_token=cancel_token,
                ):
                    async for event in runner.run_stream(prompt, context, **kwargs):
                        await streamer.send(event)
            except BaseException as e:
                from zhenxun.services.ai.run.models import AgentRunError

                if isinstance(e, asyncio.CancelledError):
                    from zhenxun.services.ai.core.exceptions import (
                        ConcurrencyInterruptException,
                    )

                    e = ConcurrencyInterruptException("团队执行已被新请求打断并接管")
                await streamer.send(AgentRunError(error=e))
            finally:
                await streamer.end()


        task = asyncio.create_task(_execution_task())
        result_obj = StreamedRunResult[Any](streamer)

        try:
            yield result_obj
        finally:
            if not task.done():
                task.cancel()
