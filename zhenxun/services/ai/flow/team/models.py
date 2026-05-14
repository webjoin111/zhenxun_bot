from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class TeamMode(str, Enum):
    """Team 的多智能体协作模式"""

    COORDINATE = "coordinate"
    """委派协作模式：Leader 主动拆解任务，调用 DelegateTool 将子任务委派给 Member，最终汇总结果返回。"""

    ROUTE = "route"
    """状态路由模式：Router 评估问题，利用 HandoffTool 将控制流直接转交并物理转移给最匹配的 Member。"""

    BROADCAST = "broadcast"
    """并发广播模式：Leader 将同一问题同时分发给所有 Member 处理，最终整合多方观点。"""


class RouteDecision(BaseModel):
    """大模型动态路由决策的数据契约"""

    target_name: str
    """选定的最合适的团队成员名称"""
    reason: str = ""
    """选择该成员的详细理由"""
    context_data: str = ""
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
    trigger_regex: str | None = None
    """(可选) 正则表达式。如果用户的输入匹配此正则，将触发极速硬路由，跳过大模型思考。"""
    trigger_func: Callable[..., Any] | None = None
    """(可选) 自定义校验函数。返回 True 或目标名称时触发硬路由。支持依赖注入。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
