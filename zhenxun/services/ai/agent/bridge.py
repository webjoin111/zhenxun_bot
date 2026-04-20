import time
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot_plugin_alconna.uniseg import UniMessage

from zhenxun.services.ai.agent.core.agent import Agent
from zhenxun.services.ai.events import EventCenter, ToolStreamEvent
from zhenxun.services.ai.types.agent import AgentRunResult
from zhenxun.services.log import logger
from zhenxun.utils.message import MessageUtils

_session_context_map = {}
_stream_throttle_map: dict[str, float] = {}


@EventCenter.subscribe(ToolStreamEvent)
async def handle_tool_stream(event: ToolStreamEvent):
    if not event.session_id or event.session_id not in _session_context_map:
        return

    is_finished = getattr(event.chunk, "status", "running") == "finished"
    now = time.time()
    last_send = _stream_throttle_map.get(event.session_id, 0)

    display_msg = event.chunk.metadata.get("display") if event.chunk.metadata else None

    if not is_finished and not display_msg and (now - last_send < 1.5):
        return

    _stream_throttle_map[event.session_id] = now
    bot, nonebot_event = _session_context_map[event.session_id]

    if event.chunk.content:
        tool_prefix = (
            f"⏳ [{event.tool_name}] "
            if event.tool_name and event.tool_name != "RunContext"
            else "⏳ "
        )
        await bot.send(nonebot_event, f"{tool_prefix}{event.chunk.content}")

    if display_msg:
        if isinstance(display_msg, UniMessage):
            try:
                await display_msg.send(nonebot_event, bot=bot, reply_to=False)
            except Exception as e:
                logger.error(f"Output Bridge 渲染 UniMessage 失败: {e}")
        elif str(type(display_msg)).find("Message") != -1:
            await bot.send(nonebot_event, display_msg)
        else:
            out_msg = UniMessage() + str(display_msg)
            await out_msg.send(nonebot_event, bot=bot, reply_to=False)


async def run_agent(
    agent: Agent,
    prompt: str,
    matcher: Matcher,
    bot: Bot,
    event: Event,
    deps: Any = None,
    reply: bool = True,
    cancellation_token: Any = None,
    **kwargs: Any,
) -> AgentRunResult:
    """
    NoneBot 桥接层：将会话隔离、上下文注入、自动回复等逻辑封装。
    供插件开发者在 Matcher 的 handle 中直接调用。

    参数:
        agent: 实例化的 Agent 对象
        prompt: 用户的输入
        matcher: 当前事件的 Matcher
        bot: 当前 Bot
        event: 当前 Event
        reply: 是否自动将 Agent 的输出回复给用户 (默认 True)
        kwargs: 传递给 agent.run 的其他参数
    """
    session_id = kwargs.pop("session_id", None)

    if not session_id:
        from .session import active_session_id

        session_id = active_session_id.get()

    if not session_id:
        if getattr(agent, "stateless", True):
            import uuid

            session_id = f"stateless_{uuid.uuid4().hex}"
        else:
            try:
                from nonebot_plugin_session import SessionIdType, extract_session

                session = extract_session(bot, event)
                # 将带空格的 platform name (如 OneBot V11) 的空格替换为下划线
                session_id = session.get_id(SessionIdType.GROUP_USER).replace(" ", "_")
            except Exception:
                try:
                    session_id = f"default_{event.get_user_id()}"
                except Exception:
                    session_id = "default_session"

    tool_filter = kwargs.pop("tool_filter", None)

    _session_context_map[session_id] = (bot, event)
    try:
        result = await agent.run(
            prompt=prompt,
            deps=deps,
            session_id=session_id,
            bot=bot,
            event=event,
            matcher=matcher,
            tool_filter=tool_filter,
            cancellation_token=cancellation_token,
            **kwargs,
        )

        if reply and result.output:
            await MessageUtils.build_message(str(result.output)).send()

        return result
    except Exception as e:
        logger.error(f"Agent {agent.name} 运行失败: {e}", e=e)
        if reply:
            await MessageUtils.build_message(f"❌ 智能体运行发生错误: {e}").send()
        raise e
    finally:
        _session_context_map.pop(session_id, None)
