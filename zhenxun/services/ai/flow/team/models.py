from enum import Enum

from pydantic import BaseModel


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
