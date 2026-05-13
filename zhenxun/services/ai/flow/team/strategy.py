from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    TeamMemberEndEvent,
    TeamMemberStartEvent,
    TeamRouteDecisionEvent,
    TeamRunEndEvent,
    TeamRunStartEvent,
    TeamSynthesizeStartEvent,
)
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.core.stream_events import ToolStreamChunk
from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.flow.team.capabilities import TeamRoutingCapability
from zhenxun.services.ai.flow.team.router import BaseRouter
from zhenxun.services.ai.run import RunContext, Task
from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunError
from zhenxun.services.ai.tools.bridges.delegate import DelegateTool
from zhenxun.services.log import logger


class BaseTeamStrategy(ABC):
    """多智能体团队协作策略基类"""

    default_system_prompt: str = ""

    def __init__(
        self,
        custom_prompt: str | None = None,
        state_flow: Any = None,
        selector_func: Callable[..., str | None | Awaitable[str | None]] | None = None,
    ):
        self.custom_prompt = custom_prompt
        self.state_flow = state_flow
        self.selector_func = selector_func

    def get_prompt(self, **kwargs) -> str:
        template = self.custom_prompt or self.default_system_prompt
        return PromptTemplate(template).render(**kwargs)

    @abstractmethod
    async def run_stream(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """核心执行流"""
        yield None


class RouteStrategy(BaseTeamStrategy):
    """路由策略：基于挂载的 Router 进行最合适的专家分发"""

    def __init__(self, router: BaseRouter):
        self.router = router
        self.custom_prompt = None

    async def run_stream(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ) -> AsyncGenerator:
        task_desc_str = (
            prompt.description if isinstance(prompt, Task) else (prompt or "")
        )
        session_id = context.session_id or "default_team_session"
        await EventCenter.publish(
            TeamRunStartEvent(
                session_id=session_id, team_name=team.name, task=task_desc_str
            )
        )

        routing_cap = TeamRoutingCapability(
            team_name=team.name,
            members=team.members,
            state_flow=getattr(team, "state_flow", None),
        )

        cycle_count = 0
        exec_config = kwargs.get("config")
        max_cycles = getattr(exec_config, "max_cycles", 15) if exec_config else 15

        logger.info(f"🛣️ [RouteStrategy] '{team.name}' 正在获取初始路由决策...")

        decision = await self.router.route(context, [], prompt)
        if not decision:
            from zhenxun.services.ai.core.exceptions import AbortException

            logger.warning(
                f"🚨 [RouteStrategy] Team '{team.name}' 的所有路由策略未能命中目标。"
            )
            raise AbortException(
                reason=f"Team '{team.name}' 无法找到合适的路由节点处理该任务",
                display="🚨 团队协作失败，无法分配任务。",
            )

        from zhenxun.services.ai.core.exceptions import HandoffException

        handoff_exception = HandoffException(
            target=decision.target_name,
            payload={"reason": decision.reason, "context_data": decision.context_data},
            display=f"⚡ 初始路由分配至 {decision.target_name}...",
        )

        current_agent = None
        final_result = None
        handoff_history_messages: list[LLMMessage] = []

        while True:
            cycle_count += 1
            if cycle_count > max_cycles:
                from zhenxun.services.ai.core.exceptions import AbortException

                logger.error(
                    f"🚨 [RouteStrategy] Team '{team.name}' 路由陷入死循环！"
                    f"已达到最大限制 {max_cycles} 次。"
                )
                raise AbortException(
                    reason=f"Team '{team.name}' 路由流转超过最大次数限制 ({max_cycles}次)，已强制熔断。",
                    display="🚨 团队协作陷入死循环，已被系统强制中断。",
                )

            if handoff_exception:
                target_name = handoff_exception.target
                handoff_reason = (
                    handoff_exception.payload.get("reason", "")
                    if handoff_exception.payload
                    else ""
                )
                context_data = (
                    handoff_exception.payload.get("context_data", "")
                    if handoff_exception.payload
                    else ""
                )

                upstream_info = []
                if handoff_reason:
                    upstream_info.append(f"【移交说明】\n{handoff_reason}")
                if context_data:
                    upstream_info.append(f"【核心上下文数据】\n{context_data}")

                combined_info = "\n\n".join(upstream_info)

                if combined_info:
                    handoff_msg = LLMMessage.system(
                        f"### 🔄 [来自上游节点的移交数据]\n{combined_info}"
                    )
                    handoff_history_messages.append(handoff_msg)

                if current_agent is not None:
                    await EventCenter.publish(
                        TeamMemberEndEvent(
                            session_id=session_id,
                            team_name=team.name,
                            member_name=current_agent.name,
                            result=f"移交至 {target_name}",
                        )
                    )

                await EventCenter.publish(
                    TeamRouteDecisionEvent(
                        session_id=session_id,
                        team_name=team.name,
                        selected_member=target_name,
                        reason=handoff_reason,
                    )
                )

                target_member = next(
                    (m for m in team.members if m.name == target_name), None
                )
                if not target_member:
                    logger.warning(
                        f"状态机断链：未找到名为 {target_name} 的成员，移交循环结束。"
                    )
                    from zhenxun.services.ai.core.messages import UsageInfo
                    from zhenxun.services.ai.run import AgentRunResult

                    final_result = AgentRunResult(
                        output=f"移交失败：未找到目标成员 {target_name}",
                        usage=UsageInfo(),
                    )
                    break

                current_agent = target_member
                handoff_exception = None

            if current_agent is None:
                raise RuntimeError("系统错误：current_agent 未能成功解析。")

            sub_context = context.clone_for_member(current_agent.name)
            sub_context.capabilities = list(sub_context.capabilities)
            sub_context.capabilities.append(routing_cap)

            await EventCenter.publish(
                TeamMemberStartEvent(
                    session_id=session_id,
                    team_name=team.name,
                    member_name=current_agent.name,
                    task=str(prompt.description if isinstance(prompt, Task) else prompt)
                    if prompt
                    else "",
                )
            )

            run_result = None

            try:
                async with current_agent.run_stream(
                    prompt=prompt,
                    context=sub_context,
                    message_history=handoff_history_messages,
                    **kwargs,
                ) as stream_result:
                    async for event in stream_result.stream_events():
                        if isinstance(event, AgentRunEnd):
                            run_result = event.result
                        elif isinstance(event, AgentRunError):
                            from zhenxun.services.ai.core.exceptions import (
                                HandoffException,
                            )

                            if isinstance(event.error, HandoffException):
                                handoff_exception = event.error
                            else:
                                raise event.error
                        yield event
            except BaseException as e:
                from zhenxun.services.ai.core.exceptions import HandoffException

                if isinstance(e, HandoffException):
                    handoff_exception = e
                else:
                    raise e

            if not handoff_exception:
                if not run_result:
                    raise RuntimeError(
                        f"Agent {current_agent.name} 未能成功返回运行结果。"
                    )

                final_result = run_result
                await EventCenter.publish(
                    TeamMemberEndEvent(
                        session_id=session_id,
                        team_name=team.name,
                        member_name=current_agent.name,
                        result=final_result.output,
                    )
                )
                break

        await EventCenter.publish(
            TeamRunEndEvent(
                session_id=session_id,
                team_name=team.name,
                result=final_result.output if final_result else None,
            )
        )

        if final_result:
            yield AgentRunEnd(result=final_result)


class CoordinateStrategy(BaseTeamStrategy):
    """协作策略：Leader 自主规划，委派任务给 Sub-Agents 并汇总结果"""

    default_system_prompt = (
        "## 角色与目标\n"
        "你是一个多智能体团队的协调者（Leader）。\n"
        "请分析用户的目标，将其拆解为逻辑连贯的子任务，并委派给合适的下属专员。\n"
        "当你收集齐所有需要的专员报告后，请汇总生成最终回复向用户汇报。如果你能自己解答，也可以不调用专员。"
    )

    async def run_stream(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ) -> AsyncGenerator:
        task_desc_str = (
            prompt.description if isinstance(prompt, Task) else (prompt or "")
        )
        session_id = context.session_id or "default_team_session"
        await EventCenter.publish(
            TeamRunStartEvent(
                session_id=session_id, team_name=team.name, task=task_desc_str
            )
        )

        delegation_tools = []
        for m in team.members:
            if getattr(m, "persona", None):
                desc = f"角色：{m.persona.role}，目标：{m.persona.goal}"
            else:
                instr = getattr(m, "instruction", "")
                desc = str(instr)[:100] + "..." if instr else "无特殊说明"

            delegation_tools.append(
                DelegateTool(
                    runnable=m,
                    name=f"delegate_to_{m.name}",
                    description=f"将子任务委派给专员 [{m.name}] 处理。专长：{desc}",
                )
            )

        leader_agent = Agent(
            name=f"{team.name}_Leader",
            instruction=self.get_prompt(),
            model=team.leader_model,
            tools=delegation_tools,
            runtime_config=team.runtime_config,
        )

        logger.info(f"👨‍💼 [CoordinateStrategy] '{team.name}' 正在启动协调推理循环...")
        await EventCenter.publish(
            TeamSynthesizeStartEvent(session_id=session_id, team_name=team.name)
        )

        final_result = None
        async with leader_agent.run_stream(
            prompt=prompt, context=context, **kwargs
        ) as stream_result:
            async for event in stream_result.stream_events():
                if isinstance(event, AgentRunEnd):
                    final_result = event.result.output
                    await EventCenter.publish(
                        TeamRunEndEvent(
                            session_id=session_id,
                            team_name=team.name,
                            result=final_result,
                        )
                    )
                yield event


class BroadcastStrategy(BaseTeamStrategy):
    """广播策略：并发让所有成员处理同一个任务，最后由 Leader 总结"""

    default_system_prompt = (
        "## 角色与目标\n"
        "你是一个多智能体团队的总结者（Leader）。\n"
        "以下是各位专家的独立处理结果，请融合各方观点，取长补短，给出一份最终的总结报告。"
    )

    async def run_stream(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ) -> AsyncGenerator:
        task_desc_str = (
            prompt.description if isinstance(prompt, Task) else (prompt or "")
        )
        session_id = context.session_id or "default_team_session"
        await EventCenter.publish(
            TeamRunStartEvent(
                session_id=session_id, team_name=team.name, task=task_desc_str
            )
        )

        event_streamer = kwargs.get("event_streamer")
        if event_streamer:
            await event_streamer.send(
                ToolStreamChunk(
                    tool_name="Team Broadcaster",
                    content=f"🚀 正在并发广播任务给 {len(team.members)} 位专家...",
                )
            )

        async def _run_member(m: Agent):
            sub_ctx = context.clone_for_member(f"bc_{m.name}")

            await EventCenter.publish(
                TeamMemberStartEvent(
                    session_id=session_id,
                    team_name=team.name,
                    member_name=m.name,
                    task=task_desc_str,
                )
            )

            try:
                res = await m.run(prompt=prompt, context=sub_ctx, **kwargs)
                await EventCenter.publish(
                    TeamMemberEndEvent(
                        session_id=session_id,
                        team_name=team.name,
                        member_name=m.name,
                        result="Success",
                    )
                )
                return m.name, res.output
            except Exception as e:
                await EventCenter.publish(
                    TeamMemberEndEvent(
                        session_id=session_id,
                        team_name=team.name,
                        member_name=m.name,
                        result=f"Error: {e}",
                    )
                )
                return m.name, f"执行失败: {e}"

        tasks = [_run_member(m) for m in team.members]
        results = await asyncio.gather(*tasks)

        if event_streamer:
            await event_streamer.send(
                ToolStreamChunk(
                    tool_name="Team Leader",
                    content="✨ 所有专家汇报完毕，Leader 正在融合各方观点...",
                )
            )

        summary_text = "\n\n".join(
            [f"### 【{name} 的意见】:\n{out}" for name, out in results]
        )
        synthesize_prompt = (
            f"**用户原始任务**: {task_desc_str}\n\n"
            f"以下是各位专家的独立处理结果，请融合各方观点，给出一份最终的总结报告：\n\n"
            f"{summary_text}"
        )

        await EventCenter.publish(
            TeamSynthesizeStartEvent(session_id=session_id, team_name=team.name)
        )

        leader_agent = Agent(
            name=f"{team.name}_Leader",
            instruction=self.get_prompt(),
            model=team.leader_model,
            runtime_config=team.runtime_config,
        )

        final_result = None
        async with leader_agent.run_stream(
            prompt=synthesize_prompt, context=context, **kwargs
        ) as stream_result:
            async for event in stream_result.stream_events():
                if isinstance(event, AgentRunEnd):
                    final_result = event.result.output
                    await EventCenter.publish(
                        TeamRunEndEvent(
                            session_id=session_id,
                            team_name=team.name,
                            result=final_result,
                        )
                    )
                yield event
