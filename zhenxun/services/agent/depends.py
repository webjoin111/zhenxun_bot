"""
zhenxun/services/agent/depends.py
定义 Agent 环境下的通用依赖注入项
"""

from nonebot.params import Depends
from nonebot.typing import T_State

from zhenxun.services.llm.tools import RunContext


async def get_agent_context(state: T_State) -> RunContext:
    """从 State 中获取 Agent 运行上下文"""
    return state["_agent_context"]


async def get_session_id(
    ctx: RunContext = Depends(get_agent_context),
) -> str | None:
    """获取当前 Agent 会话 ID"""
    return ctx.session_id


async def get_user_input(ctx: RunContext = Depends(get_agent_context)) -> str:
    """获取当前轮次的用户输入文本"""
    return ctx.extra.get("user_input", "")


GetAgentContext = get_agent_context
GetSessionID = get_session_id
GetUserInput = get_user_input
