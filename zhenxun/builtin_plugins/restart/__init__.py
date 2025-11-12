import asyncio
from datetime import datetime
import os
from pathlib import Path
import platform
import subprocess
import sys

import aiofiles
import nonebot
from nonebot import on_command
from nonebot.adapters import Bot
from nonebot.params import ArgStr
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig, Config
from zhenxun.configs.utils import PluginExtraData, RegisterConfig
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils
from zhenxun.utils.platform import PlatformUtils

__plugin_meta__ = PluginMetadata(
    name="重启",
    description="执行脚本重启真寻",
    usage="""
    重启
    """.strip(),
    extra=PluginExtraData(
        author="HibiKier",
        version="0.2",
        plugin_type=PluginType.SUPERUSER,
        configs=[
            RegisterConfig(
                key="need_confirm",
                value=True,
                help="重启命令是否需要确认",
                default_value=True,
                type=bool,
            ),
            RegisterConfig(
                key="is_interval_restart",
                value=False,
                help="是否启用定时自动重启（每隔一定时间自动重启）",
                default_value=False,
                type=bool,
            ),
            RegisterConfig(
                key="interval_time",
                value=60,
                help="自动重启的间隔时间(分钟)，每隔多少分钟重启一次",
                default_value=60,
                type=int,
            ),
        ],
    ).to_dict(),
)


_matcher = on_command(
    "重启",
    permission=SUPERUSER,
    rule=to_me(),
    priority=1,
    block=True,
)

driver = nonebot.get_driver()


RESTART_MARK = Path() / "is_restart"

RESTART_FILE = Path() / "restart.sh"


_interval_restart_task = None


def perform_restart_sync():
    """同步执行重启操作"""
    try:
        logger.info("开始定时自动重启真寻...")

        with open(RESTART_MARK, "w", encoding="utf8") as f:
            f.write("auto_restart")

        logger.info("即将执行重启命令...")

        if str(platform.system()).lower() == "windows":
            python = sys.executable
            subprocess.Popen([python, *sys.argv])
            os._exit(0)
        else:
            os.system("./restart.sh")
    except Exception as e:
        logger.error(f"执行重启操作时发生错误: {e}")


async def perform_restart():
    """异步执行重启操作，不需要bot和session参数"""
    logger.info("开始定时自动重启真寻...")
    perform_restart_sync()


async def interval_restart_scheduler():
    """定时重启调度器"""
    try:
        interval_time = Config.get_config("restart", "interval_time")
        if interval_time <= 0:
            interval_time = 60

        seconds = interval_time * 60
        logger.info(f"已启动定时重启功能，每{interval_time}分钟重启一次")
        logger.info(f"下次重启将在 {seconds} 秒后执行")

        while True:
            try:
                for i in range(seconds):
                    await asyncio.sleep(1)
                    if i > 0 and i % 60 == 0:
                        minutes_left = (seconds - i) // 60
                        logger.info(f"距离下次重启还有 {minutes_left} 分钟")

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"定时重启触发，当前时间：{current_time}")
                logger.info("即将执行重启操作...")
                perform_restart_sync()
                logger.warning("重启操作可能失败，等待一分钟后重试...")
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("定时重启任务被取消")
                raise
            except Exception as e:
                logger.error(f"定时重启过程中发生错误: {e}")
                await asyncio.sleep(60)
    except Exception as e:
        logger.error(f"定时重启调度器发生错误: {e}")


async def restart_bot(bot: Bot, session: Uninfo):
    """执行重启操作"""
    await MessageUtils.build_message(
        f"开始重启{BotConfig.self_nickname}..请稍等..."
    ).send()
    async with aiofiles.open(RESTART_MARK, "w", encoding="utf8") as f:
        await f.write(f"{bot.self_id} {session.user.id}")
    logger.info("开始重启真寻...", "重启", session=session)

    if str(platform.system()).lower() == "windows":
        import sys

        python = sys.executable
        os.execl(python, python, *sys.argv)
    else:
        os.system("./restart.sh")  # noqa: ASYNC221


@_matcher.handle()
async def handle_restart(bot: Bot, session: Uninfo):
    need_confirm = Config.get_config("restart", "need_confirm")

    if not need_confirm:
        await restart_bot(bot, session)
    else:
        await _matcher.send(
            f"确定是否重启{BotConfig.self_nickname}？\n确定请回复[是|好|确定]\n（重启失败咱们将失去联系，请谨慎！）"
        )


@_matcher.got("flag")
async def _(bot: Bot, session: Uninfo, flag: str = ArgStr("flag")):
    if flag.lower() in {"true", "是", "好", "确定", "确定是"}:
        await restart_bot(bot, session)
    else:
        await MessageUtils.build_message("已取消操作...").send()


@driver.on_bot_connect
async def _(bot: Bot):
    if str(platform.system()).lower() != "windows" and not RESTART_FILE.exists():
        async with aiofiles.open(RESTART_FILE, "w", encoding="utf8") as f:
            await f.write(
                "pid=$(netstat -tunlp | grep "
                + str(bot.config.port)
                + " | awk '{print $7}')\n"
                "pid=${pid%/*}\n"
                "kill -9 $pid\n"
                "sleep 3\n"
                "python3 bot.py"
            )
        os.system("chmod +x ./restart.sh")  # noqa: ASYNC221
        logger.info("已自动生成 restart.sh(重启) 文件，请检查脚本是否与本地指令符合...")
    if RESTART_MARK.exists():
        async with aiofiles.open(RESTART_MARK, encoding="utf8") as f:
            content = await f.read()

        if content == "auto_restart":
            logger.info(f"{BotConfig.self_nickname}已成功自动重启！")
        else:
            try:
                bot_id, user_id = content.split()
                if bot := nonebot.get_bot(bot_id):
                    if target := PlatformUtils.get_target(user_id=user_id):
                        await MessageUtils.build_message(
                            f"{BotConfig.self_nickname}已成功重启！"
                        ).send(target, bot=bot)
            except Exception as e:
                logger.error(f"处理重启标记文件时发生错误: {e}")

        RESTART_MARK.unlink()

    global _interval_restart_task
    is_interval_restart = Config.get_config("restart", "is_interval_restart")
    if is_interval_restart and _interval_restart_task is None:
        _interval_restart_task = asyncio.create_task(interval_restart_scheduler())
        logger.info("已启动定时重启功能")


@driver.on_bot_disconnect
async def _(_: Bot):
    global _interval_restart_task
    if _interval_restart_task is not None:
        _interval_restart_task.cancel()
        _interval_restart_task = None
        logger.info("已停止定时重启功能")
