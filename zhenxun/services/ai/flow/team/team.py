from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.core.messages import PromptInput
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.flow.team.models import TeamRuntimeConfig
from zhenxun.services.ai.flow.team.strategy import (
    BaseTeamStrategy,
)
from zhenxun.services.ai.run import AgentRunResult, RunContext, Task
from zhenxun.services.ai.tools.providers.skills.models import Skill, SkillSource
from zhenxun.utils.utils import infer_plugin_namespace

if TYPE_CHECKING:
    from zhenxun.services.ai.flow.agent.agent import CapabilitySource
    from zhenxun.services.ai.flow.agent.models import Persona


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
        persona: "Persona | dict | None" = None,
        runtime_config: TeamRuntimeConfig | dict | None = None,
        capabilities: list["CapabilitySource"] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
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

        self.namespace = infer_plugin_namespace() or "unknown"

        self.capabilities: list[Any] = []
        if capabilities:
            from zhenxun.services.ai.protocols.capabilities import (
                AbstractCapability,
                DynamicCapability,
            )

            for cap in capabilities:
                if isinstance(cap, AbstractCapability):
                    self.capabilities.append(cap)
                elif callable(cap):
                    self.capabilities.append(DynamicCapability(cap))

        if isinstance(runtime_config, dict):
            runtime_config = TeamRuntimeConfig(**runtime_config)
        self.runtime_config = runtime_config or TeamRuntimeConfig(stateless=True)

        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            self.capabilities.append(
                SkillCapability(skills=skills, namespace=self.namespace)
            )

        self.selector_func = getattr(strategy, "selector_func", None)

    async def run(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        context: "RunContext | None" = None,
        capabilities: list["CapabilitySource"] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        团队级运行阻塞核心入口，内部静默分配任务给成员直至汇总结束。

        参数:
            prompt: 派发给多智能体团队的任务描述 or 契约对象 (Task)。
            context: 显式传入的会话与运行上下文。
            capabilities: 仅针对本次团队执行动态注入的临时拦截器列表。
            kwargs: 透传的其他附加参数。

        返回:
            AgentRunResult[Any]: 包含最终融合输出、消息历史和用量统计的运行结果对象。
        """
        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            capabilities = list(capabilities) if capabilities else []
            capabilities.append(
                SkillCapability(skills=skills, namespace=self.namespace)
            )

        return await super().run(
            prompt=prompt, context=context, capabilities=capabilities, **kwargs
        )

    import contextlib

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        context: "RunContext | None" = None,
        capabilities: list["CapabilitySource"] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        **kwargs: Any,
    ):
        import asyncio

        if context is None:
            context = RunContext()

        if not hasattr(context, "capabilities"):
            context.capabilities = []

        if hasattr(self, "capabilities") and self.capabilities:
            context.capabilities.extend(self.capabilities)

        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            capabilities = list(capabilities) if capabilities else []
            capabilities.append(
                SkillCapability(skills=skills, namespace=self.namespace)
            )

        if capabilities:
            from zhenxun.services.ai.protocols.capabilities import (
                AbstractCapability,
                DynamicCapability,
            )

            for cap in capabilities:
                if isinstance(cap, AbstractCapability):
                    context.capabilities.append(cap)
                elif callable(cap):
                    context.capabilities.append(DynamicCapability(cap))

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
