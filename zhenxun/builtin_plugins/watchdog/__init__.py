# import asyncio  # 不再需要
from datetime import datetime
import time

import nonebot
from nonebot import on_command
from nonebot.adapters import Bot, Event
from nonebot.message import run_preprocessor
from nonebot.plugin import PluginMetadata
from nonebot_plugin_apscheduler import scheduler

# from nonebot_plugin_session import EventSession  # 不再需要
from zhenxun.configs.utils import PluginExtraData, RegisterConfig
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.platform import PlatformUtils

__plugin_meta__ = PluginMetadata(
    name="机器人监控",
    description="监控机器人状态，自动恢复无响应状态",
    usage="",
    extra=PluginExtraData(
        author="Admin",
        version="0.1",
        plugin_type=PluginType.HIDDEN,
        configs=[
            RegisterConfig(
                key="CHECK_INTERVAL",
                value=5,
                help="检查间隔（分钟）",
                default_value=5,
                type=int,
            ),
            RegisterConfig(
                key="MAX_NO_RESPONSE_TIME",
                value=30,
                help="最大无响应时间（分钟）",
                default_value=30,
                type=int,
            ),
            RegisterConfig(
                key="ENABLE_AUTO_RESTART",
                value=False,
                help="启用自动重启",
                default_value=False,
                type=bool,
            ),
        ],
    ).to_dict(),
)

# 记录最后一次命令处理时间
last_command_time = time.time()

# 创建一个命令用于更新时间戳
_ping = on_command("ping", priority=1, block=False)


@_ping.handle()
async def handle_ping():
    """处理ping命令，更新最后命令处理时间"""
    global last_command_time
    last_command_time = time.time()
    await _ping.finish("pong")


# 定时检查机器人状态
@scheduler.scheduled_job("interval", minutes=5)
async def check_bot_status():
    """定时检查机器人状态"""
    from zhenxun.configs.config import Config

    global last_command_time
    current_time = time.time()

    # 获取配置
    # check_interval = Config.get_config("watchdog", "CHECK_INTERVAL") or 5  # 不再需要
    max_no_response_time = Config.get_config("watchdog", "MAX_NO_RESPONSE_TIME") or 30
    enable_auto_restart = Config.get_config("watchdog", "ENABLE_AUTO_RESTART") or False

    # 如果超过指定时间没有处理命令，认为机器人无响应
    if current_time - last_command_time > max_no_response_time * 60:
        logger.error(
            (
                f"机器人可能无响应，"
                f"最后一次命令处理时间: {datetime.fromtimestamp(last_command_time)}"
            ),
            "Watchdog",
        )

        # 如果启用了自动重启
        if enable_auto_restart:
            logger.warning("准备执行自动重启...", "Watchdog")
            # 这里可以添加重启逻辑
            # 例如，可以调用重启插件的功能
            try:
                from zhenxun.builtin_plugins.restart import perform_restart

                await perform_restart()
                logger.info("已执行自动重启", "Watchdog")
            except Exception as e:
                logger.error(f"自动重启失败: {e}", "Watchdog", e=e)
        else:
            # 发送警告消息给超级用户
            try:
                bot = nonebot.get_bot()
                for superuser in bot.config.superusers:
                    await PlatformUtils.send_message(
                        bot,
                        superuser,
                        None,
                        (
                            f"警告：机器人可能无响应，"
                            f"最后一次命令处理时间: "
                            f"{datetime.fromtimestamp(last_command_time)}"
                        ),
                    )
            except Exception as e:
                logger.error(f"发送警告消息失败: {e}", "Watchdog", e=e)


# 更新命令处理时间的钩子
@run_preprocessor
async def update_command_time(
    _bot: Bot, _event: Event
):  # 使用下划线前缀表示未使用的参数
    """更新最后命令处理时间"""
    global last_command_time
    last_command_time = time.time()
