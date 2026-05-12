from zhenxun.services.ai.core.events import (
    EventCenter,
    TeamMemberEndEvent,
    TeamMemberStartEvent,
    TeamRouteDecisionEvent,
    TeamRunEndEvent,
    TeamRunStartEvent,
    TeamSynthesizeStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
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
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowStartedEvent,
)
from zhenxun.services.log import logger

from .base import BaseUIStreamer, MarkdownUIStreamer


class UIStreamerRegistry:
    """UI 渲染器注册中心"""

    _registry: dict[str, type[BaseUIStreamer]] = {}

    @classmethod
    def register(cls, name: str, streamer_cls: type[BaseUIStreamer]) -> None:
        cls._registry[name] = streamer_cls
        logger.debug(f"已注册 UI Streamer: {name}")

    @classmethod
    def get(cls, name: str) -> type[BaseUIStreamer]:
        if name not in cls._registry:
            logger.warning(f"未找到 UI Streamer '{name}'，降级使用 markdown。")
            return cls._registry.get("markdown", MarkdownUIStreamer)
        return cls._registry[name]


UIStreamerRegistry.register("markdown", MarkdownUIStreamer)

active_streamers: dict[str, BaseUIStreamer] = {}


def _dispatch(event, method_name: str):
    """根据 session_id 分发事件给活跃的 Streamer 实例"""
    if getattr(event, "session_id", None) and event.session_id in active_streamers:
        streamer = active_streamers[event.session_id]
        handler = getattr(streamer, method_name, None)
        if handler:
            handler(event)


@EventCenter.subscribe(ToolCallEvent, priority=10)
async def _on_tool_call(event: ToolCallEvent):
    _dispatch(event, "on_tool_call")


@EventCenter.subscribe(ToolResultEvent, priority=10)
async def _on_tool_result(event: ToolResultEvent):
    _dispatch(event, "on_tool_result")


@EventCenter.subscribe(ToolStreamEvent, priority=10)
async def _on_tool_stream(event: ToolStreamEvent):
    _dispatch(event, "on_tool_stream")


@EventCenter.subscribe(TeamRunStartEvent, priority=10)
async def _on_team_run_start(event):
    _dispatch(event, "on_team_run_start")


@EventCenter.subscribe(TeamRouteDecisionEvent, priority=10)
async def _on_team_route_decision(event):
    _dispatch(event, "on_team_route_decision")


@EventCenter.subscribe(TeamMemberStartEvent, priority=10)
async def _on_team_member_start(event):
    _dispatch(event, "on_team_member_start")


@EventCenter.subscribe(TeamMemberEndEvent, priority=10)
async def _on_team_member_end(event):
    _dispatch(event, "on_team_member_end")


@EventCenter.subscribe(TeamSynthesizeStartEvent, priority=10)
async def _on_team_synthesize_start(event):
    _dispatch(event, "on_team_synthesize_start")


@EventCenter.subscribe(TeamRunEndEvent, priority=10)
async def _on_team_run_end(event):
    _dispatch(event, "on_team_run_end")


@EventCenter.subscribe(TaskRunStartEvent, priority=10)
async def _on_task_run_start(event):
    _dispatch(event, "on_task_run_start")


@EventCenter.subscribe(TaskRunEndEvent, priority=10)
async def _on_task_run_end(event):
    _dispatch(event, "on_task_run_end")


@EventCenter.subscribe(TaskRunErrorEvent, priority=10)
async def _on_task_run_error(event):
    _dispatch(event, "on_task_run_error")


@EventCenter.subscribe(WorkflowStartedEvent, priority=10)
async def _on_workflow_started(event):
    _dispatch(event, "on_workflow_started")


@EventCenter.subscribe(WorkflowCompletedEvent, priority=10)
async def _on_workflow_completed(event):
    _dispatch(event, "on_workflow_completed")


@EventCenter.subscribe(WorkflowErrorEvent, priority=10)
async def _on_workflow_error(event):
    _dispatch(event, "on_workflow_error")


@EventCenter.subscribe(StepStartedEvent, priority=10)
async def _on_step_started(event):
    _dispatch(event, "on_step_started")


@EventCenter.subscribe(StepCompletedEvent, priority=10)
async def _on_step_completed(event):
    _dispatch(event, "on_step_completed")


@EventCenter.subscribe(StepPausedEvent, priority=10)
async def _on_step_paused(event):
    _dispatch(event, "on_step_paused")


@EventCenter.subscribe(StepRetryEvent, priority=10)
async def _on_step_retry(event):
    _dispatch(event, "on_step_retry")


@EventCenter.subscribe(StepHealingEvent, priority=10)
async def _on_step_healing(event):
    _dispatch(event, "on_step_healing")


@EventCenter.subscribe(StepFallbackEvent, priority=10)
async def _on_step_fallback(event):
    _dispatch(event, "on_step_fallback")


@EventCenter.subscribe(ConditionExecutionStartedEvent, priority=10)
async def _on_condition_started(event):
    _dispatch(event, "on_condition_started")


@EventCenter.subscribe(ConditionExecutionCompletedEvent, priority=10)
async def _on_condition_completed(event):
    _dispatch(event, "on_condition_completed")


@EventCenter.subscribe(RouterExecutionStartedEvent, priority=10)
async def _on_router_started(event):
    _dispatch(event, "on_router_started")


@EventCenter.subscribe(RouterExecutionCompletedEvent, priority=10)
async def _on_router_completed(event):
    _dispatch(event, "on_router_completed")


@EventCenter.subscribe(LoopExecutionStartedEvent, priority=10)
async def _on_loop_started(event):
    _dispatch(event, "on_loop_started")


@EventCenter.subscribe(LoopIterationStartedEvent, priority=10)
async def _on_loop_iteration_started(event):
    _dispatch(event, "on_loop_iteration_started")


@EventCenter.subscribe(LoopIterationCompletedEvent, priority=10)
async def _on_loop_iteration_completed(event):
    _dispatch(event, "on_loop_iteration_completed")


@EventCenter.subscribe(LoopExecutionCompletedEvent, priority=10)
async def _on_loop_completed(event):
    _dispatch(event, "on_loop_completed")
