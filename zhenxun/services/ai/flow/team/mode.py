from enum import Enum


class TeamMode(str, Enum):
    """Team 的多智能体协作模式"""

    COORDINATE = "coordinate"
    """委派协作模式：Leader 主动拆解任务，调用 DelegateTool 将子任务委派给 Member，最终汇总结果返回。"""

    ROUTE = "route"
    """状态路由模式：Router 评估问题，利用 HandoffTool 将控制流直接转交并物理转移给最匹配的 Member。"""

    BROADCAST = "broadcast"
    """并发广播模式：Leader 将同一问题同时分发给所有 Member 处理，最终整合多方观点。"""
