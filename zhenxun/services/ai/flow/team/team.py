from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.flow.agent.models import AgentRuntimeConfig
from zhenxun.services.ai.flow.team.models import TeamMode
from zhenxun.services.ai.flow.team.router import BaseRouter
from zhenxun.services.ai.flow.team.strategy import (
    BaseTeamStrategy,
)
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.run import AgentRunResult, RunContext, Task


class Team:
    """
    多智能体动态编排与路由控制器 (Facade)。
    """

    def __init__(
        self,
        name: str,
        members: list[Agent],
        mode: str | TeamMode | BaseTeamStrategy = TeamMode.COORDINATE,
        leader_model: str | None = None,
        leader_tools: list[Any] | None = None,
        blackboard_schema: type[BaseModel] | None = None,
        initial_blackboard_state: BaseModel | None = None,
        custom_prompt: str | None = None,
        state_flow: dict[str, list[str | Any]] | Callable | None = None,
        selector_func: Callable[..., str | None | Awaitable[str | None]] | None = None,
        router: "BaseRouter | None" = None,
    ):
        self.name = name
        self.members = members
        self.leader_model = leader_model or get_global_default_model_name()
        self.leader_tools = leader_tools or []

        self.runtime_config = AgentRuntimeConfig(stateless=True, enable_hitl=False)

        if isinstance(state_flow, dict):
            from zhenxun.services.ai.flow.team.models import Transition

            normalized_flow = {}
            for k, targets in state_flow.items():
                normalized_targets = []
                for t in targets:
                    if isinstance(t, str):
                        normalized_targets.append(Transition(target=t))
                    else:
                        normalized_targets.append(t)
                normalized_flow[k] = normalized_targets
            self.state_flow = normalized_flow
        else:
            self.state_flow = state_flow

        self.selector_func = selector_func

        if isinstance(mode, BaseTeamStrategy):
            self.strategy = mode
        else:
            mode_str = mode.value if isinstance(mode, TeamMode) else mode.lower()
            from zhenxun.services.ai.flow.team.registry import TeamStrategyRegistry

            strategy_cls = TeamStrategyRegistry.get(mode_str)
            if not strategy_cls:
                raise ValueError(f"未找到指定的 Team 模式: {mode_str}")

            self.strategy = strategy_cls(
                custom_prompt=custom_prompt,
                router=router,
            )

        self.blackboard = None
        if blackboard_schema is not None:
            from zhenxun.services.ai.run.blackboard import (
                BlackboardManager,
                create_blackboard_tools,
            )

            self.blackboard = BlackboardManager(
                schema=blackboard_schema, initial_state=initial_blackboard_state
            )
            bb_tools = create_blackboard_tools(self.blackboard)

            if not self.leader_tools:
                self.leader_tools = []
            self.leader_tools.extend(bb_tools)

            for member in self.members:
                if not member.tool_definitions:
                    member.tool_definitions = []
                member.tool_definitions.extend(bb_tools)

    def bind(self, **kwargs: Any) -> Any:
        """DI 注入语法糖"""
        from nonebot.params import Depends

        from zhenxun.services.ai.flow.agent.bridge import AgentRunner

        async def _dependency() -> AgentRunner[Any]:
            return AgentRunner[Any](self, **kwargs)

        return Depends(_dependency)

    async def reply(
        self, prompt: Any = None, reply_to: bool = False, **kwargs: Any
    ) -> AgentRunResult[Any]:
        """
        团队级交互执行语法糖，隐式推导上下文，自动渲染协作进度并最终将团队产出回复给终端用户。

        参数:
            prompt: 派发给多智能体团队的任务描述。
            reply_to: 是否将结果作为回复消息发送 (at用户或引用原消息)。
            kwargs: 传递给底层 AgentRunner 的附加执行参数。

        返回:
            AgentRunResult[Any]: 包含最终融合输出、消息历史和用量统计的运行结果对象。
        """
        from zhenxun.services.ai.flow.agent.bridge import AgentRunner

        runner = AgentRunner[Any](self, **kwargs)
        return await runner.reply(prompt=prompt, reply_to=reply_to)

    async def run(
        self,
        prompt: str | Task | None = None,
        context: RunContext | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        团队级运行阻塞核心入口，内部静默分配任务给成员直至汇总结束。

        参数:
            prompt: 派发给多智能体团队的任务描述或契约对象 (Task)。
            context: 显式传入的会话与运行上下文。
            kwargs: 透传的其他附加参数。

        返回:
            AgentRunResult[Any]: 包含最终融合输出、消息历史和用量统计的运行结果对象。
        """
        async with self.run_stream(prompt, context, **kwargs) as stream_result:
            return await stream_result.get_run_result()

    import contextlib

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: str | Task | None = None,
        context: RunContext | None = None,
        **kwargs: Any,
    ):
        if context is None:
            context = RunContext()

        if self.blackboard is not None:
            context.session.blackboard = self.blackboard

        from zhenxun.services.ai.core.stream_events import EventStreamer
        from zhenxun.services.ai.flow.team.runner import TeamRunner
        from zhenxun.services.ai.run import StreamedRunResult

        streamer = EventStreamer()
        context.run.streamer = streamer
        runner = TeamRunner(self, self.strategy)

        async def _execution_task():
            try:
                async for event in runner.run_stream(prompt, context, **kwargs):
                    await streamer.send(event)
            except BaseException as e:
                from zhenxun.services.ai.run.models import AgentRunError

                await streamer.send(AgentRunError(error=e))
            finally:
                await streamer.end()

        import asyncio

        task = asyncio.create_task(_execution_task())
        result_obj = StreamedRunResult[Any](streamer)

        try:
            yield result_obj
        finally:
            if not task.done():
                task.cancel()
