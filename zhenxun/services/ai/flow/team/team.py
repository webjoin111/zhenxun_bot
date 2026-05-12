from typing import Any

from pydantic import BaseModel

from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.flow.agent.models import AgentRuntimeConfig
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.run import AgentRunResult, RunContext, Task
from zhenxun.services.ai.flow.team.mode import TeamMode
from zhenxun.services.ai.flow.team.strategy import (
    BaseTeamStrategy,
    BroadcastStrategy,
    CoordinateStrategy,
    RouteStrategy,
)
from zhenxun.services.log import logger


class RouteDecision(BaseModel):
    """大模型动态路由决策的数据契约"""

    selected_member_name: str
    """选定的最合适的团队成员名称"""
    reason: str
    """选择该成员的详细理由"""


class Team:
    """
    多智能体动态编排与路由控制器 (Facade)。
    极简门面模式：用户仅需通过 mode 指定协作方式，底层自动装配 Tool 与状态转移流。
    也支持高阶开发者直接传入自定义的 BaseTeamStrategy 实例。
    """

    def __init__(
        self,
        name: str,
        members: list[Agent],
        mode: str | TeamMode | BaseTeamStrategy = TeamMode.COORDINATE,
        leader_model: str | None = None,
        custom_prompt: str | None = None,
    ):
        self.name = name
        self.members = members
        self.leader_model = leader_model or get_global_default_model_name()

        self.runtime_config = AgentRuntimeConfig(stateless=True, enable_hitl=False)

        if isinstance(mode, BaseTeamStrategy):
            self.strategy = mode
        else:
            mode_str = mode.value if isinstance(mode, TeamMode) else mode.lower()
            if mode_str == "route":
                self.strategy = RouteStrategy(custom_prompt)
            elif mode_str == "broadcast":
                self.strategy = BroadcastStrategy(custom_prompt)
            else:
                self.strategy = CoordinateStrategy(custom_prompt)

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

        from zhenxun.services.ai.run import StreamedRunResult
        from zhenxun.services.ai.core.stream_events import EventStreamer

        streamer = EventStreamer()

        async def _execution_task():
            try:
                async for event in self.strategy.run_stream(
                    self, prompt, context, **kwargs
                ):
                    await streamer.send(event)
            except BaseException as e:
                from zhenxun.services.ai.core.stream_events import AgentRunError

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


