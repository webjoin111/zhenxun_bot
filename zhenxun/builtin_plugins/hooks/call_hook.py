import re
from typing import Any

from nonebot.adapters import Bot, Message

from zhenxun.configs.config import Config
from zhenxun.models.bot_message_store import BotMessageStore
from zhenxun.services.log import logger
from zhenxun.utils.enum import BotSentType
from zhenxun.utils.manager.message_manager import MessageManager
from zhenxun.utils.platform import PlatformUtils


def replace_message(message: Message) -> str:
    """将消息中的at、image、record、face替换为字符串

    参数:
        message: Message

    返回:
        str: 文本消息
    """
    result = ""
    for msg in message:
        if isinstance(msg, str):
            result += msg
        elif msg.type == "at":
            result += f"@{msg.data['qq']}"
        elif msg.type == "image":
            result += "[image]"
        elif msg.type == "record":
            result += "[record]"
        elif msg.type == "face":
            result += f"[face:{msg.data['id']}]"
        elif msg.type == "reply":
            result += ""
        else:
            result += str(msg)
    return result


def format_message_for_log(message: Any) -> str:
    """格式化消息内容用于日志记录，避免显示过长的base64内容

    参数:
        message: 任意类型的消息对象

    返回:
        str: 格式化后的消息文本，适合日志记录
    """
    if isinstance(message, Message):
        return replace_message(message)

    message_str = str(message)

    base64_pattern = r"(file=)?base64://[A-Za-z0-9+/=]{50,}"
    if re.search(base64_pattern, message_str):
        original_length = len(message_str)
        message_str = re.sub(
            base64_pattern,
            lambda m: f"[图片:base64,长度:{len(m.group())}]",
            message_str,
        )
        if len(message_str) > 500:
            message_str = message_str[:500] + f"...(总长度:{original_length})"
    elif len(message_str) > 500:
        message_str = message_str[:500] + f"...(总长度:{len(str(message))})"

    return message_str


@Bot.on_called_api
async def handle_api_result(
    bot: Bot, exception: Exception | None, api: str, data: dict[str, Any], result: Any
):
    if exception or api != "send_msg":
        return
    user_id = data.get("user_id")
    group_id = data.get("group_id")
    message_id = result.get("message_id")
    message: Message = data.get("message", "")
    message_type = data.get("message_type")
    try:
        # 记录消息id
        if user_id and message_id:
            MessageManager.add(str(user_id), str(message_id))
            logger.debug(
                f"收集消息id，user_id: {user_id}, msg_id: {message_id}", "msg_hook"
            )
    except Exception as e:
        logger.warning(
            f"收集消息id发生错误...data: {data}, result: {result}", "msg_hook", e=e
        )
    if not Config.get_config("hook", "RECORD_BOT_SENT_MESSAGES"):
        return
    try:
        await BotMessageStore.create(
            bot_id=bot.self_id,
            user_id=user_id,
            group_id=group_id,
            sent_type=BotSentType.GROUP
            if message_type == "group"
            else BotSentType.PRIVATE,
            text=replace_message(message),
            plain_text=message.extract_plain_text()
            if isinstance(message, Message)
            else replace_message(message),
            platform=PlatformUtils.get_platform(bot),
        )
        logger.debug(f"消息发送记录，message: {format_message_for_log(message)}")
    except Exception as e:
        logger.warning(
            f"消息发送记录发生错误...data: {data}, result: {result}",
            "msg_hook",
            e=e,
        )
