from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any

from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.flow.agent.agent import Agent
from zhenxun.services.ai.flow.team.models import (
    CallAction,
    ConcurrentCallAction,
    FinishAction,
    TeamAction,
)
from zhenxun.services.ai.flow.team.registry import team_strategy
from zhenxun.services.ai.flow.team.router import BaseRouter
from zhenxun.services.ai.run import RunContext, Task
from zhenxun.services.ai.tools.bridges.delegate import DelegateTool
from zhenxun.services.log import logger


class BaseTeamStrategy(ABC):
    """多智能体团队协作策略基类"""

    default_system_prompt: str = ""

    def __init__(self, custom_prompt: str | None = None, **kwargs):
        self.custom_prompt = custom_prompt
        self.kwargs = kwargs

    def get_prompt(self, **kwargs) -> str:
        template = self.custom_prompt or self.default_system_prompt
        return PromptTemplate(template).render(**kwargs)

    async def generate_plan(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ) -> AsyncGenerator[TeamAction, Any]:
        """
        核心决策生成器 (Action Yielding Pattern)。

        第三方开发者只需重写此方法：
        1. 使用 `yield CallAction(...)` 派发任务，系统会自动拦截并执行，
           然后将 `AgentRunResult` 通过 .asend() 传回。
        2. 使用 `yield FinishAction(...)` 结束团队协作。

        不再需要手动处理 EventCenter、上下文隔离和异常兜底。
        """
        yield FinishAction(
            result="The Strategy has not implemented generate_plan() yet."
        )


@team_strategy("route", namespace="builtin")
class RouteStrategy(BaseTeamStrategy):
    """路由策略：基于挂载的 Router 进行最合适的专家分发"""

    def __init__(
        self,
        router: BaseRouter | None = None,
        custom_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(custom_prompt=custom_prompt, **kwargs)
        self.router = router

    async def generate_plan(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ):
        router = self.router
        if not router:
            from .router import ChainRouter, FunctionRouter, LLMRouter

            routers = []
            selector_func = getattr(team, "selector_func", None)
            if selector_func:
                routers.append(FunctionRouter(selector_func))
            routers.append(
                LLMRouter(
                    team_name=team.name,
                    members=team.members,
                    leader_model=getattr(team, "leader_model", None),
                    leader_tools=getattr(team, "leader_tools", []),
                    state_flow=getattr(team, "state_flow", None),
                    runtime_config=getattr(team, "runtime_config", None),
                    custom_prompt=self.custom_prompt,
                )
            )
            router = ChainRouter(routers)

        cycle_count = 0
        exec_config = kwargs.get("config")
        max_cycles = getattr(exec_config, "max_cycles", 15) if exec_config else 15

        logger.info(f"🛣️ [RouteStrategy] '{team.name}' 正在获取初始路由决策...")

        decision = await router.route(context, [], prompt)
        if not decision:
            from zhenxun.services.ai.core.exceptions import AbortException

            logger.warning(
                f"🚨 [RouteStrategy] Team '{team.name}' 的所有路由策略未能命中目标。"
            )
            raise AbortException(
                reason=f"Team '{team.name}' 无法找到合适的路由节点处理该任务",
                display="🚨 团队协作失败，无法分配任务。",
            )

        current_target = decision.target_name
        handoff_reason = decision.reason
        context_data = decision.context_data

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

            handoff_history_messages = []
            upstream_info = []
            if handoff_reason:
                upstream_info.append(f"【移交说明】\n{handoff_reason}")
            if context_data:
                if isinstance(context_data, dict):
                    import json

                    formatted_data = json.dumps(
                        context_data, ensure_ascii=False, indent=2
                    )
                    upstream_info.append(
                        f"【结构化上下文载荷】\n```json\n{formatted_data}\n```"
                    )
                else:
                    upstream_info.append(f"【核心上下文数据】\n{context_data}")
            combined_info = "\n\n".join(upstream_info)

            if combined_info:
                from zhenxun.services.ai.core.messages import LLMMessage

                handoff_msg = LLMMessage.system(
                    f"### 🔄 [来自上游节点的移交数据]\n{combined_info}"
                )
                handoff_history_messages.append(handoff_msg)

            run_result = yield CallAction(
                agent=current_target,
                task=prompt,
                history=handoff_history_messages,
                kwargs=kwargs,
            )

            output_str = str(run_result.output)

            if output_str.startswith("__HANDOFF__:"):
                parts = output_str[len("__HANDOFF__:") :].split("|", 2)
                current_target = parts[0]
                handoff_reason = parts[1] if len(parts) > 1 else ""
                context_data = parts[2] if len(parts) > 2 else ""
                continue

            fast_routed = False
            state_flow = getattr(team, "state_flow", None)
            if isinstance(state_flow, dict) and current_target in state_flow:
                for t in state_flow[current_target]:
                    if getattr(t, "trigger_regex", None):
                        import re

                        if re.search(t.trigger_regex, output_str):
                            current_target = t.target
                            handoff_reason = ""
                            context_data = output_str
                            fast_routed = True
                            break
                    if getattr(t, "trigger_func", None):
                        try:
                            if t.trigger_func(output_str):
                                current_target = t.target
                                handoff_reason = ""
                                context_data = output_str
                                fast_routed = True
                                break
                        except Exception:
                            pass

            if fast_routed:
                from zhenxun.services.ai.core.events import (
                    EventCenter,
                    TeamRouteDecisionEvent,
                )

                await EventCenter.publish(
                    TeamRouteDecisionEvent(
                        session_id=context.session_id or "default_team_session",
                        team_name=team.name,
                        selected_member=current_target,
                        reason="[系统拦截：正则/函数状态流发生转移]",
                    )
                )
                continue

            yield FinishAction(result=run_result.output)
            break


@team_strategy("coordinate", namespace="builtin")
class CoordinateStrategy(BaseTeamStrategy):
    """协作策略：Leader 自主规划，委派任务给 Sub-Agents 并汇总结果"""

    default_system_prompt = (
        "## 角色与目标\n"
        "你是一个多智能体团队的协调者（Leader）。\n"
        "你可以使用你自身携带的工具先查阅、收集资料；也可以分析用户的目标将其拆解为子任务，并委派给合适的下属专员。\n"
        "当你收集齐所有需要的信息或专员报告后，请汇总生成最终回复向用户汇报。"
    )

    async def generate_plan(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ):
        delegation_tools = []
        for m in team.members:
            if getattr(m, "persona", None):
                desc = f"角色：{m.persona.role}，目标：{m.persona.goal}"
            else:
                desc = getattr(m, "description", "") or "处理节点"

            delegation_tools.append(
                DelegateTool(
                    runnable=m,
                    name=f"delegate_to_{m.name}",
                    description=f"将子任务委派给专员 [{m.name}] 处理。专长：{desc}",
                )
            )

        leader_tools = getattr(team, "leader_tools", []).copy()
        leader_tools.extend(delegation_tools)

        leader_agent = Agent(
            name=f"{team.name}_Leader",
            instruction=self.get_prompt(),
            model=team.leader_model,
            tools=leader_tools,
            runtime_config=team.runtime_config,
        )

        session_id = context.session_id or "default_team_session"
        from zhenxun.services.ai.core.events import EventCenter
        from zhenxun.services.ai.core.events.event_types import TeamSynthesizeStartEvent

        await EventCenter.publish(
            TeamSynthesizeStartEvent(session_id=session_id, team_name=team.name)
        )

        if context.run.streamer:
            from zhenxun.services.ai.core.stream_events import ToolStreamChunk

            await context.run.streamer.send(
                ToolStreamChunk(
                    tool_name="Team Leader",
                    content="✨ 团队 Leader 正在汇总各方报告...",
                )
            )

        logger.info(f"👨💼 [CoordinateStrategy] '{team.name}' 正在启动协调推理循环...")

        leader_res = yield CallAction(agent=leader_agent, task=prompt)

        yield FinishAction(result=leader_res.output)


@team_strategy("broadcast", namespace="builtin")
class BroadcastStrategy(BaseTeamStrategy):
    """广播策略：并发让所有成员处理同一个任务，最后由 Leader 总结"""

    default_system_prompt = (
        "## 角色与目标\n"
        "你是一个多智能体团队的总结者（Leader）。\n"
        "以下是各位专家的独立处理结果，请融合各方观点，取长补短，给出一份最终的总结报告。"
    )

    async def generate_plan(
        self, team: Any, prompt: str | Task | None, context: RunContext, **kwargs
    ):
        task_desc_str = (
            prompt.description if isinstance(prompt, Task) else (prompt or "")
        )

        session_id = context.session_id or "default_team_session"

        if context.run.streamer:
            from zhenxun.services.ai.core.stream_events import ToolStreamChunk

            await context.run.streamer.send(
                ToolStreamChunk(
                    tool_name="Team Broadcaster",
                    content=f"🚀 正在并发广播任务给 {len(team.members)} 位专家...",
                )
            )

        actions = [CallAction(agent=m.name, task=task_desc_str) for m in team.members]
        results = yield ConcurrentCallAction(actions=actions)

        if context.run.streamer:
            from zhenxun.services.ai.core.stream_events import ToolStreamChunk

            await context.run.streamer.send(
                ToolStreamChunk(
                    tool_name="Team Leader",
                    content="✨ 所有专家汇报完毕，Leader 正在融合各方观点...",
                )
            )

        from zhenxun.services.ai.core.events import EventCenter
        from zhenxun.services.ai.core.events.event_types import TeamSynthesizeStartEvent

        await EventCenter.publish(
            TeamSynthesizeStartEvent(session_id=session_id, team_name=team.name)
        )

        summary_text = "\n\n".join(
            [f"### 【{name} 的意见】:\n{res.output}" for name, res in results]
        )

        synthesize_prompt = (
            f"**用户原始任务**: {task_desc_str}\n\n"
            f"以下是各位专家的独立处理结果，请融合各方观点，给出一份最终的总结报告：\n\n"
            f"{summary_text}"
        )

        leader_agent = Agent(
            name=f"{team.name}_Leader",
            instruction=self.get_prompt(),
            model=team.leader_model,
            tools=getattr(team, "leader_tools", []),
            runtime_config=team.runtime_config,
        )

        leader_res = yield CallAction(agent=leader_agent, task=synthesize_prompt)

        yield FinishAction(result=leader_res.output)
