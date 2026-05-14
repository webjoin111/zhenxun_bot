from typing import Any

from pydantic import BaseModel, ConfigDict

from zhenxun.services.ai.core.messages import LLMMessage


class TeamAction(BaseModel):
    """多智能体团队协作动作基类"""
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CallAction(TeamAction):
    """
    调度动作：呼叫指定的 Agent 执行任务
    """
    agent: str | Any
    """目标 Agent 的名称（字符串）或动态生成的 Agent 实例"""
    task: str | Any
    """派发给该 Agent 的具体任务或提示词"""
    history: list[LLMMessage] | None = None
    """需要传递给该 Agent 的上下文历史记录（可选）"""
    kwargs: dict[str, Any] | None = None
    """其他透传给 Agent.run_stream 的 kwargs（可选）"""


class ConcurrentCallAction(TeamAction):
    """
    并发调度动作：同时呼叫多个 Agent 执行任务
    """
    actions: list[CallAction]


class FinishAction(TeamAction):
    """
    结束动作：团队协作完成，返回最终结果
    """
    result: Any
    """团队协作的最终产出"""
