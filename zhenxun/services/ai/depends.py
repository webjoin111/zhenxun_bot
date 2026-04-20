from nonebot.adapters import Bot, Event
from nonebot.params import Depends

from zhenxun.services.ai.chat_session import ChatSession
from zhenxun.services.ai.protocols.memory import MemoryIsolationLevel


def StatefulAI(
    isolation_level: MemoryIsolationLevel = MemoryIsolationLevel.GROUP_USER,
):
    """
    真寻 LLM 模块依赖注入 (Dependency Injection) 快捷函数。
    """

    async def dependency(bot: Bot, event: Event) -> ChatSession:
        return ChatSession(bot=bot, event=event, isolation_level=isolation_level)

    return Depends(dependency)
