from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator
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
from zhenxun.services.ai.core.stream_events import ToolStreamChunk
from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.protocols.capabilities import AbstractCapability
from zhenxun.services.ai.run import RunContext, Task
from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunError
from zhenxun.services.ai.tools.bridges.delegate import DelegateTool
from zhenxun.services.ai.tools.bridges.handoff import HandoffTool
from zhenxun.services.log import logger


class TeamRoutingCapability(AbstractCapability):
    """团队路由能力组件：动态向所有团队成员（包括 Router 和 Expert）注入互相移交的工具及规则说明。"""

    def __init__(self, team_name: str, members: list[Any]):
        self.team_name = team_name
        self.members = members

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []
        for m in self.members:
            if context.run.agent_name != m.name:
                if getattr(m, "persona", None):
                    desc = f"角色：{m.persona.role}，目标：{m.persona.goal}"
                else:
                    instr = getattr(m, "instruction", "")
                    desc = str(instr)[:100] + "..." if instr else "领域专家"

                tools.append(
                    HandoffTool(
                        target_name=m.name,
                        target_description=desc,
                    )
                )
        return tools

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        if context.run.agent_name != f"{self.team_name}_Router":
            return [
                "### 🤝 [团队协作规范]\n"
                f"你是跨域协作团队 '{self.team_name}' 的一员。如果你认为当前任务超出了你的职责范畴，"
                "或你目前已经完成了前置处理但需要其他专家的处理结果进行下一步推进，"
                "请务必使用移交工具 (transfer_to_...) 将控制权移交给合适的队友。\n"
                "移交时必须在 `reason` 参数中详细说明你的移交原因，并附带你已经处理好的上下文关键数据！"
            ]
        return []


class BaseTeamStrategy(ABC):
    """多智能体团队协作策略基类"""

    default_system_prompt: str = ""

    def __init__(self, custom_prompt: str | None = None):
        self.custom_prompt = custom_prompt

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
    """路由策略：动态选择一个最合适的专家处理问题"""

    default_system_prompt = (
        "## 角色与目标\n"
        "你是一个高级任务路由器 (所在团队: {{ team_name }})。\n"
        "请根据用户的输入意图，立刻调用相应的移交工具 (transfer_to_...) 将对话物理转移给合适的专员处理。\n"
        "你必须且只能选择移交，不能自己作答。"
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

        route_prompt = self.get_prompt(team_name=team.name)

        routing_cap = TeamRoutingCapability(team_name=team.name, members=team.members)

        router_agent = Agent(
            name=f"{team.name}_Router",
            instruction=route_prompt,
            model=team.leader_model,
            runtime_config=team.runtime_config,
        )

        current_agent = router_agent
        current_task_desc = prompt
        final_result = None

        logger.info(f"🛣️ [RouteStrategy] '{team.name}' 正在判定意图...")
        while True:
            is_router = current_agent == router_agent
            sub_context = context.clone_for_member(current_agent.name)

            sub_context.capabilities = list(sub_context.capabilities)
            sub_context.capabilities.append(routing_cap)

            if not is_router:
                await EventCenter.publish(
                    TeamMemberStartEvent(
                        session_id=session_id,
                        team_name=team.name,
                        member_name=current_agent.name,
                        task=str(
                            current_task_desc.description
                            if isinstance(current_task_desc, Task)
                            else current_task_desc
                        )
                        if current_task_desc
                        else "",
                    )
                )

            run_result = None
            handoff_exception = None
            try:
                async with current_agent.run_stream(
                    prompt=current_task_desc, context=sub_context, **kwargs
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
                        if not is_router:
                            yield event
            except BaseException as e:
                from zhenxun.services.ai.core.exceptions import HandoffException

                if isinstance(e, HandoffException):
                    handoff_exception = e
                else:
                    raise e

            if not run_result and not handoff_exception:
                raise RuntimeError(f"Agent {current_agent.name} 未能成功返回运行结果。")

            if handoff_exception:
                target_name = handoff_exception.target
                handoff_reason = (
                    handoff_exception.payload.get("reason", "")
                    if handoff_exception.payload
                    else ""
                )
                if handoff_reason:
                    import copy

                    if isinstance(prompt, Task):
                        current_task_desc = copy.copy(prompt)
                        current_task_desc.context = f"[来自上游的移交说明]\n{handoff_reason}\n\n[前置背景]\n{current_task_desc.context or ''}"
                    else:
                        current_task_desc = f"[来自上一个处理节点的移交说明]\n{handoff_reason}\n\n[用户的原始诉求]\n{prompt}"
                else:
                    current_task_desc = prompt

                if not is_router:
                    await EventCenter.publish(
                        TeamMemberEndEvent(
                            session_id=session_id,
                            team_name=team.name,
                            member_name=current_agent.name,
                            result=f"Handoff to {target_name}",
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
                        output=f"Handoff failed: {target_name} not found",
                        usage=UsageInfo(),
                    )
                    break

                current_agent = target_member
            else:
                final_result = run_result
                if final_result and not is_router:
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
