import asyncio
import time

from nonebot.adapters import Bot
from nonebot.matcher import Matcher
from nonebot_plugin_alconna import At
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import Config
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.services.auth_service import auth_cache
from zhenxun.services.db_context import DB_TIMEOUT_SECONDS
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.utils import EntityIDs, get_entity_ids

from .config import LOGGER_COMMAND, WARNING_THRESHOLD
from .exception import SkipPluginException
from .utils import freq, send_message

Config.add_plugin_config(
    "hook",
    "BAN_RESULT",
    "才不会给你发消息.",
    help="对被ban用户发送的消息",
)


async def is_ban(user_id: str | None, group_id: str | None) -> int:
    """检查用户或群组是否被ban (纯内存操作，极快)

    参数:
        user_id: 用户ID
        group_id: 群组ID

    返回:
        int: ban的剩余时间，-1表示永久ban，>0表示剩余秒数，0表示未被ban
    """
    if not user_id and not group_id:
        return 0

    now = time.time()

    # 优先检查群组黑名单
    if group_id:
        expire = auth_cache.get_group_ban_expire(group_id)
        if expire is not None and (expire == -1 or expire > now):
            return -1 if expire == -1 else int(expire - now)

    # 检查用户黑名单
    if user_id:
        expire = auth_cache.get_user_ban_expire(user_id)
        if expire is not None and (expire == -1 or expire > now):
            return -1 if expire == -1 else int(expire - now)

    return 0


def check_plugin_type(matcher: Matcher) -> bool:
    """判断插件类型是否是隐藏插件

    参数:
        matcher: Matcher

    返回:
        bool: 是否为隐藏插件
    """
    if plugin := matcher.plugin:
        if metadata := plugin.metadata:
            extra = metadata.extra
            if extra.get("plugin_type") in [PluginType.HIDDEN]:
                return False
    return True


def format_time(time_val: float) -> str:
    """格式化时间

    参数:
        time_val: ban时长

    返回:
        str: 格式化时间文本
    """
    if time_val == -1:
        return "∞"
    time_val = abs(int(time_val))
    if time_val < 60:
        time_str = f"{time_val!s} 秒"
    else:
        minute = int(time_val / 60)
        if minute > 60:
            hours = minute // 60
            minute %= 60
            time_str = f"{hours} 小时 {minute}分钟"
        else:
            time_str = f"{minute} 分钟"
    return time_str


async def group_handle(group_id: str) -> None:
    """群组ban检查

    参数:
        group_id: 群组id

    异常:
        SkipPluginException: 群组处于黑名单
    """
    start_time = time.time()
    try:
        if await is_ban(None, group_id):
            raise SkipPluginException("群组处于黑名单中...")
    finally:
        # 记录执行时间
        elapsed = time.time() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"group_handle 耗时: {elapsed:.3f}s",
                LOGGER_COMMAND,
                group_id=group_id,
            )


async def user_handle(plugin: PluginInfo, entity: EntityIDs, session: Uninfo) -> None:
    """用户ban检查

    参数:
        module: 插件模块名
        entity: 实体ID信息
        session: Uninfo

    异常:
        SkipPluginException: 用户处于黑名单
    """
    start_time = time.time()
    try:
        ban_result = Config.get_config("hook", "BAN_RESULT")
        time_val = await is_ban(entity.user_id, entity.group_id)
        if not time_val:
            return
        time_str = format_time(time_val)

        if (
            plugin
            and time_val != -1
            and ban_result
            and freq.is_send_limit_message(plugin, entity.user_id, False)
        ):
            try:
                await asyncio.wait_for(
                    send_message(
                        session,
                        [
                            At(flag="user", target=entity.user_id),
                            f"{ban_result}\n在..在 {time_str} 后才会理你喔",
                        ],
                        entity.user_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(f"发送消息超时: {entity.user_id}", LOGGER_COMMAND)
        raise SkipPluginException("用户处于黑名单中...")
    finally:
        # 记录执行时间
        elapsed = time.time() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"user_handle 耗时: {elapsed:.3f}s",
                LOGGER_COMMAND,
                session=session,
            )


async def auth_ban(
    matcher: Matcher, bot: Bot, session: Uninfo, plugin: PluginInfo
) -> None:
    """权限检查 - ban 检查

    参数:
        matcher: Matcher
        bot: Bot
        session: Uninfo
    """
    start_time = time.time()
    try:
        if not check_plugin_type(matcher):
            return
        if not matcher.plugin_name:
            return
        entity = get_entity_ids(session)
        if entity.user_id in bot.config.superusers:
            return
        if entity.group_id:
            try:
                await asyncio.wait_for(
                    group_handle(entity.group_id), timeout=DB_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(f"群组ban检查超时: {entity.group_id}", LOGGER_COMMAND)
                # 超时时不阻塞，继续执行

        if entity.user_id:
            try:
                await asyncio.wait_for(
                    user_handle(plugin, entity, session),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.error(f"用户ban检查超时: {entity.user_id}", LOGGER_COMMAND)
                # 超时时不阻塞，继续执行
    finally:
        # 记录总执行时间
        elapsed = time.time() - start_time
        if elapsed > WARNING_THRESHOLD:  # 记录耗时超过500ms的检查
            logger.warning(
                f"auth_ban 总耗时: {elapsed:.3f}s, plugin={matcher.plugin_name}",
                LOGGER_COMMAND,
                session=session,
            )
