from abc import ABC, abstractmethod

from zhenxun.services.ai.core.events.event_types import (
    ConditionExecutionCompletedEvent,
    ConditionExecutionStartedEvent,
    LoopExecutionCompletedEvent,
    LoopExecutionStartedEvent,
    LoopIterationCompletedEvent,
    LoopIterationStartedEvent,
    RouterExecutionCompletedEvent,
    RouterExecutionStartedEvent,
    StepCompletedEvent,
    StepFallbackEvent,
    StepHealingEvent,
    StepPausedEvent,
    StepRetryEvent,
    StepStartedEvent,
    TaskRunEndEvent,
    TaskRunErrorEvent,
    TaskRunStartEvent,
    TeamMemberEndEvent,
    TeamMemberStartEvent,
    TeamRouteDecisionEvent,
    TeamRunEndEvent,
    TeamRunStartEvent,
    TeamSynthesizeStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowStartedEvent,
)


class BaseUIStreamer(ABC):
    """UI 渲染策略抽象基类"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.lines: list[str] = []

    def on_tool_call(self, event: ToolCallEvent) -> None:
        pass

    def on_tool_result(self, event: ToolResultEvent) -> None:
        pass

    def on_tool_stream(self, event: ToolStreamEvent) -> None:
        pass

    def on_team_run_start(self, event: TeamRunStartEvent) -> None:
        pass

    def on_team_route_decision(self, event: TeamRouteDecisionEvent) -> None:
        pass

    def on_team_member_start(self, event: TeamMemberStartEvent) -> None:
        pass

    def on_team_member_end(self, event: TeamMemberEndEvent) -> None:
        pass

    def on_team_synthesize_start(self, event: TeamSynthesizeStartEvent) -> None:
        pass

    def on_team_run_end(self, event: TeamRunEndEvent) -> None:
        pass

    def on_task_run_start(self, event: TaskRunStartEvent) -> None:
        pass

    def on_task_run_end(self, event: TaskRunEndEvent) -> None:
        pass

    def on_task_run_error(self, event: TaskRunErrorEvent) -> None:
        pass

    def on_workflow_started(self, event: WorkflowStartedEvent) -> None:
        pass

    def on_workflow_completed(self, event: WorkflowCompletedEvent) -> None:
        pass

    def on_workflow_error(self, event: WorkflowErrorEvent) -> None:
        pass

    def on_step_started(self, event: StepStartedEvent) -> None:
        pass

    def on_step_completed(self, event: StepCompletedEvent) -> None:
        pass

    def on_step_paused(self, event: StepPausedEvent) -> None:
        pass

    def on_step_retry(self, event: StepRetryEvent) -> None:
        pass

    def on_step_healing(self, event: StepHealingEvent) -> None:
        pass

    def on_step_fallback(self, event: StepFallbackEvent) -> None:
        pass

    def on_condition_started(self, event: ConditionExecutionStartedEvent) -> None:
        pass

    def on_condition_completed(self, event: ConditionExecutionCompletedEvent) -> None:
        pass

    def on_router_started(self, event: RouterExecutionStartedEvent) -> None:
        pass

    def on_router_completed(self, event: RouterExecutionCompletedEvent) -> None:
        pass

    def on_loop_started(self, event: LoopExecutionStartedEvent) -> None:
        pass

    def on_loop_iteration_started(self, event: LoopIterationStartedEvent) -> None:
        pass

    def on_loop_iteration_completed(self, event: LoopIterationCompletedEvent) -> None:
        pass

    def on_loop_completed(self, event: LoopExecutionCompletedEvent) -> None:
        pass

    @abstractmethod
    def render(self, duration: float) -> str:
        """渲染最终的战报字符串"""
        pass


class MarkdownUIStreamer(BaseUIStreamer):
    """默认的 Markdown 战报渲染器实现"""

    def on_tool_call(self, event: ToolCallEvent) -> None:
        if "🗣️" in event.tool_name:
            self.lines.append(f"🔄 正在并发指派: {event.tool_name}")
        else:
            self.lines.append(f"🔄 正在调用: `{event.tool_name}`")

    def on_tool_result(self, event: ToolResultEvent) -> None:
        if event.error or (event.result and event.result.is_error):
            self.lines.append(f"❌ 调用失败: `{event.tool_name}`")
        else:
            if "🗣️" in event.tool_name:
                self.lines.append(
                    f"✅ 执行完毕: {event.tool_name} ({event.duration_ms:.0f}ms)"
                )
            else:
                self.lines.append(
                    f"✅ 调用成功: `{event.tool_name}` ({event.duration_ms:.0f}ms)"
                )

    def on_tool_stream(self, event: ToolStreamEvent) -> None:
        self.lines.append(f"  └ ⏳ {event.chunk.content}")

    def on_team_run_start(self, event: TeamRunStartEvent) -> None:
        self.lines.append(f"🤝 **团队 [{event.team_name}] 开始协作**: `{event.task}`")

    def on_team_route_decision(self, event: TeamRouteDecisionEvent) -> None:
        reason_str = f" (理由: {event.reason})" if event.reason else ""
        self.lines.append(
            f"🛣️ **路由决策**: 委派给专员 👨‍💼`{event.selected_member}`{reason_str}"
        )

    def on_team_member_start(self, event: TeamMemberStartEvent) -> None:
        self.lines.append(f"🚀 **专员 👨‍💼`{event.member_name}`** 开始执行子任务...")

    def on_team_member_end(self, event: TeamMemberEndEvent) -> None:
        self.lines.append(f"✅ **专员 👨‍💼`{event.member_name}`** 完成任务！")

    def on_team_synthesize_start(self, event: TeamSynthesizeStartEvent) -> None:
        self.lines.append(f"✨ **团队 [{event.team_name}] Leader** 正在汇总各方报告...")

    def on_team_run_end(self, event: TeamRunEndEvent) -> None:
        self.lines.append(f"🏁 **团队 [{event.team_name}]** 协作圆满结束！")

    def on_task_run_start(self, event: TaskRunStartEvent) -> None:
        self.lines.append(
            f"📋 **开始任务**: `{event.task_name}` (由 {event.agent_name} 执行)"
        )

    def on_task_run_end(self, event: TaskRunEndEvent) -> None:
        self.lines.append(f"✅ **任务完成**: `{event.task_name}`")

    def on_task_run_error(self, event: TaskRunErrorEvent) -> None:
        self.lines.append(f"❌ **任务失败**: `{event.task_name}` - {event.error}")

    def on_workflow_started(self, event: WorkflowStartedEvent) -> None:
        self.lines.append(f"🏭 **工作流 [{event.workflow_name}] 启动**")

    def on_step_started(self, event: StepStartedEvent) -> None:
        self.lines.append(f"  ┣ ⚙️ [节点] `{event.step_name}` 开始执行...")

    def on_step_completed(self, event: StepCompletedEvent) -> None:
        res = event.result
        if res and not res.success:
            self.lines.append(f"  ┣ ❌ [节点] `{event.step_name}` 执行异常/中断")
        elif res and res.stop:
            self.lines.append(f"  ┣ 🛑 [节点] `{event.step_name}` 主动终止了后续流程")
        else:
            self.lines.append(f"  ┣ ✅ [节点] `{event.step_name}` 成功完成")

    def on_step_paused(self, event: StepPausedEvent) -> None:
        self.lines.append(f"  ┣ ⏸️ **[节点挂起]** `{event.step_name}`: {event.reason}")

    def on_step_retry(self, event: StepRetryEvent) -> None:
        delay_str = f"，等待 {event.delay}s 后" if event.delay > 0 else ""
        self.lines.append(
            f"  ┣ 🔄 [节点重试] `{event.step_name}` 遇到异常{delay_str}将进行第 {event.attempt} 次重试..."
        )

    def on_step_healing(self, event: StepHealingEvent) -> None:
        healer = event.healer_agent_name or "智能体"
        self.lines.append(
            f"  ┣ 🩹 [自愈介入] {healer} 正在尝试修复 `{event.step_name}` 的参数输入错误..."
        )

    def on_step_fallback(self, event: StepFallbackEvent) -> None:
        self.lines.append(
            f"  ┣ 🔀 [降级路由] `{event.step_name}` 发生严重故障，正在安全切换至备用节点 `{event.fallback_node_name}`..."
        )

    def on_workflow_completed(self, event: WorkflowCompletedEvent) -> None:
        self.lines.append(f"🏭 **工作流 [{event.workflow_name}] 运行结束**")

    def on_workflow_error(self, event: WorkflowErrorEvent) -> None:
        self.lines.append(f"❌ **工作流执行异常**: {event.error}")

    def on_condition_started(self, event: ConditionExecutionStartedEvent) -> None:
        self.lines.append(f"  ┣ 🔀 [条件分流] 开始评估 `{event.step_name}`")

    def on_condition_completed(self, event: ConditionExecutionCompletedEvent) -> None:
        self.lines.append(f"  ┣ ⤵️ [条件分流] 评估完毕，进入 `{event.branch}` 分支")

    def on_router_started(self, event: RouterExecutionStartedEvent) -> None:
        self.lines.append("  ┣ 🧭 [智能路由] 分析意图中...")

    def on_router_completed(self, event: RouterExecutionCompletedEvent) -> None:
        self.lines.append(f"  ┣ 🎯 [智能路由] 决定分发至节点: `{event.selected_steps}`")

    def on_loop_started(self, event: LoopExecutionStartedEvent) -> None:
        self.lines.append(
            f"  ┣ 🔁 开始循环: [Loop] `{event.step_name}` (最大 {event.max_iterations} 次)"
        )

    def on_loop_iteration_started(self, event: LoopIterationStartedEvent) -> None:
        self.lines.append(f"  ┃  ┣ 🔄 第 {event.iteration} 次迭代...")

    def on_loop_completed(self, event: LoopExecutionCompletedEvent) -> None:
        self.lines.append(
            f"  ┣ ✅ 循环结束: [Loop] `{event.step_name}` (共执行 {event.total_iterations} 次)"
        )

    def render(self, duration: float) -> str:
        if not self.lines:
            return ""
        header = f"🚀 **Agent 思考与执行流** (耗时 {duration:.1f}s)\n" + "-" * 20 + "\n"
        body = "\n".join(self.lines)
        footer = "\n" + "-" * 20
        return header + body + footer
