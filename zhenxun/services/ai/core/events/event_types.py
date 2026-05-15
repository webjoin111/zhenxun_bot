from typing import Any

from pydantic import Field

from .base import AIEvent


class AgentStartEvent(AIEvent):
    agent_name: str
    prompt: str | None = None


class AgentEndEvent(AIEvent):
    agent_name: str
    result: Any
    duration_ms: float


class ModelStartEvent(AIEvent):
    model_name: str
    messages: list[Any]


class ModelEndEvent(AIEvent):
    response: Any
    duration_ms: float


class ToolCallEvent(AIEvent):
    """工具准备执行前触发。监听器可直接修改 arguments，或抛出异常以拦截执行"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    context: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolErrorEvent(AIEvent):
    """工具执行抛出异常时触发。监听器可通过设置 recovered_result 来修复错误"""

    tool_call_id: str
    tool_name: str
    error: Exception


class ToolResultEvent(AIEvent):
    """工具执行结束（无论成功或被修复）时触发"""

    tool_call_id: str
    tool_name: str
    result: Any | None
    error: Exception | None
    duration_ms: float


class ToolStreamEvent(AIEvent):
    """工具执行过程中产生流式输出时触发"""

    tool_call_id: str
    tool_name: str
    chunk: Any


class SandboxExecutionStartedEvent(AIEvent):
    session_id: str | None = None
    code: str


class SandboxExecutionCompletedEvent(AIEvent):
    session_id: str | None = None
    exit_code: int
    duration_ms: float


class TeamRunStartEvent(AIEvent):
    """团队执行开始"""

    team_name: str
    task: str


class TeamRouteDecisionEvent(AIEvent):
    """团队路由分发决策"""

    team_name: str
    selected_member: str
    reason: str | None = None


class TeamMemberStartEvent(AIEvent):
    """子智能体开始执行"""

    team_name: str
    member_name: str
    task: str


class TeamMemberEndEvent(AIEvent):
    """子智能体执行结束"""

    team_name: str
    member_name: str
    result: Any


class TeamSynthesizeStartEvent(AIEvent):
    """团队Leader开始汇总生成"""

    team_name: str


class TeamRunEndEvent(AIEvent):
    """团队执行完全结束"""

    team_name: str
    result: Any


class TaskRunStartEvent(AIEvent):
    """数据契约任务执行开始"""
    task_id: str
    task_name: str
    agent_name: str


class TaskRunEndEvent(AIEvent):
    """数据契约任务执行结束"""
    task_id: str
    task_name: str
    

class TaskRunErrorEvent(AIEvent):
    """数据契约任务执行失败"""
    task_id: str
    task_name: str
    error: Exception


class TeamTaskCreatedEvent(AIEvent):
    """团队自主任务：创建子任务"""
    team_name: str
    task_id: str
    title: str
    assignee: str | None


class TeamTaskUpdatedEvent(AIEvent):
    """团队自主任务：子任务状态变更"""
    team_name: str
    task_id: str
    title: str
    status: str
    result: str | None = None


class WorkflowStartedEvent(AIEvent):
    """工作流开始"""

    workflow_name: str


class WorkflowCompletedEvent(AIEvent):
    """工作流结束"""

    workflow_name: str
    result: Any


class WorkflowCancelledEvent(AIEvent):
    """工作流被取消"""

    workflow_name: str
    reason: str


class WorkflowErrorEvent(AIEvent):
    """工作流发生异常"""

    workflow_name: str
    error: BaseException


class StepStartedEvent(AIEvent):
    """单步/图元开始执行"""

    step_name: str
    step_type: str


class StepCompletedEvent(AIEvent):
    """单步/图元执行完毕"""

    step_name: str
    step_type: str
    result: Any


class StepPausedEvent(AIEvent):
    """单步触发 HITL 挂起"""

    step_name: str
    step_type: str
    reason: str


class StepRetryEvent(AIEvent):
    """单步节点执行失败，触发重试策略"""

    step_name: str
    attempt: int
    reason: str
    delay: float


class StepHealingEvent(AIEvent):
    """单步节点执行失败，触发AI自愈并篡改输入"""

    step_name: str
    original_error: str
    healer_agent_name: str | None = None


class StepFallbackEvent(AIEvent):
    """单步节点重试达上限，触发降级路由"""

    step_name: str
    fallback_node_name: str


class ConditionExecutionStartedEvent(AIEvent):
    step_name: str


class ConditionExecutionCompletedEvent(AIEvent):
    step_name: str
    branch: str
    result: Any


class RouterExecutionStartedEvent(AIEvent):
    step_name: str


class RouterExecutionCompletedEvent(AIEvent):
    step_name: str
    selected_steps: list[str]
    result: Any


class LoopExecutionStartedEvent(AIEvent):
    step_name: str
    max_iterations: int


class LoopIterationStartedEvent(AIEvent):
    step_name: str
    iteration: int


class LoopIterationCompletedEvent(AIEvent):
    step_name: str
    iteration: int


class LoopExecutionCompletedEvent(AIEvent):
    step_name: str
    total_iterations: int


class ParallelExecutionStartedEvent(AIEvent):
    step_name: str
    parallel_step_count: int


class ParallelExecutionCompletedEvent(AIEvent):
    step_name: str
    parallel_step_count: int
    step_results: list[Any]
