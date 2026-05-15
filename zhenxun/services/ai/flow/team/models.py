from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict
from zhenxun.services.ai.core.configs import BaseOutputDefinition

from zhenxun.services.ai.core.messages import LLMMessage


class TeamMode(str, Enum):
    """Team 的多智能体协作模式"""

    COORDINATE = "coordinate"
    """委派协作模式：Leader 主动拆解任务，调用 DelegateTool 将子任务委派给 Member，最终汇总结果返回。"""

    ROUTE = "route"
    """状态路由模式：Router 评估问题，利用 HandoffTool 将控制流直接转交并物理转移给最匹配的 Member。"""

    BROADCAST = "broadcast"
    """并发广播模式：Leader 将同一问题同时分发给所有 Member 处理，最终整合多方观点。"""

    TASKS = "tasks"
    """自主任务模式：Leader 在共享黑板上拆解目标为子任务，处理前置依赖，并驱动 Member 执行，直至达成目标。"""


class RouteDecision(BaseModel):
    """大模型动态路由决策的数据契约"""

    target_name: str
    """选定的最合适的团队成员名称"""
    reason: str = ""
    """选择该成员的详细理由"""
    context_data: Any = ""
    """传递的上下文载荷"""


class Transition(BaseModel):
    """
    声明式移交契约。
    用于定义 Team 模式下，智能体之间转移控制权的条件和目标。
    """

    target: str
    """目标智能体的名称"""
    description: str = ""
    """自然语言描述的移交条件（提供给大模型 LLMRouter 思考时使用）"""
    input_schema: type[BaseModel] | BaseOutputDefinition | None = None
    """(可选) 强类型的输入约束。如果设置，LLMRouter 决定移交时必须且只能生成符合该 Schema 的 JSON 参数，并作为 context_data 传递。"""
    trigger_regex: str | None = None
    """(可选) 正则表达式。如果用户的输入匹配此正则，将触发极速硬路由，跳过大模型思考。"""
    trigger_func: Callable[..., Any] | None = None
    """(可选) 自定义校验函数。返回 True 或目标名称时触发硬路由。支持依赖注入。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
