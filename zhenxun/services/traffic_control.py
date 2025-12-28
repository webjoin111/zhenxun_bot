import asyncio
from dataclasses import dataclass, field
import re
import time
from typing import Any, ClassVar, cast

from arclet.alconna import command_manager
import nonebot
from nonebot.adapters import Bot, Event
from nonebot.adapters.onebot.v11 import (
    Bot as V11Bot,
)
from nonebot.adapters.onebot.v11 import (
    GroupAdminNoticeEvent,
    GroupDecreaseNoticeEvent,
    GroupIncreaseNoticeEvent,
    MessageEvent,
    MetaEvent,
    NoticeEvent,
    PokeNotifyEvent,
    PrivateMessageEvent,
    RequestEvent,
)
from nonebot.rule import TrieRule

from zhenxun.configs.config import BotConfig, Config
from zhenxun.services.log import logger

Config.add_plugin_config(
    "traffic_control",
    "MAX_WORKERS",
    5,
    help="事件处理最大并发工作线程数，用于解决高并发下的阻塞问题",
    default_value=5,
    type=int,
)

Config.add_plugin_config(
    "traffic_control",
    "ENABLE",
    True,
    help="是否开启流量控制",
    default_value=True,
    type=bool,
)

Config.add_plugin_config(
    "traffic_control",
    "MAX_QUEUE_SIZE",
    1000,
    help="消息队列最大积压数,超过此数量且非命令消息将被直接丢弃.设置为 0 表示不限制。",
    default_value=1000,
    type=int,
)


@dataclass(order=True)
class QueueItem:
    """优先级队列项封装。

    使用 dataclass 的 order=True 自动实现比较。

    参数:
        priority: 优先级，数值越小越优先。
        timestamp: 时间戳，相同优先级下，先进入的先出。
        bot: Bot 实例。
        event: 事件对象。
    """

    priority: int
    timestamp: float
    bot: Bot = field(compare=False)
    event: Event = field(compare=False)


class CommandTrie:
    """高效命令前缀匹配树。"""

    def __init__(self):
        self.root = {}
        self.end_symbol = "__END__"

    def insert(self, text: str):
        """插入一个命令前缀到树中。

        参数:
            text: 命令前缀字符串。
        """
        node = self.root
        for char in text:
            node = node.setdefault(char, {})
        node[self.end_symbol] = True

    def match(self, text: str) -> str | None:
        """检查文本是否以任何已注册的命令开头。

        该方法会尽可能匹配最长的前缀，并确保前缀后紧跟空格或换行符，
        或者是文本的结尾。

        参数:
            text: 待检查的文本。

        返回:
            str | None: 匹配到的命令前缀，如果未匹配则返回 None。
        """
        if not text:
            return None

        node = self.root
        for i, char in enumerate(text):
            if char not in node:
                return None
            node = node[char]

            if self.end_symbol in node:
                if i == len(text) - 1 or text[i + 1] in (" ", "\n"):
                    return text[: i + 1]

        return None


class TrafficController:
    """流量控制管理器 (单例)。

    负责对 Bot 接收到的事件进行优先级调度和流量削峰，
    防止高并发下系统过载。
    """

    _instance: ClassVar["TrafficController | None"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self.queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue()

        self.max_workers = Config.get_config("traffic_control", "MAX_WORKERS", 5)
        self.max_queue_size = Config.get_config(
            "traffic_control", "MAX_QUEUE_SIZE", 1000
        )

        self._original_handle_event: Any = V11Bot.handle_event

        self._worker_tasks: list[asyncio.Task] = []
        self._command_index = CommandTrie()
        self._combined_regex: re.Pattern | None = None
        self._initialized = True

    def _build_command_index(self):
        """构建命令索引，扫描所有 Alconna 命令和原生 Matcher。"""
        count = 0

        try:
            for prefix in TrieRule.prefix.keys():
                self._command_index.insert(prefix)
                count += 1
            logger.debug(
                f"扫描原生命令索引: {list(TrieRule.prefix.keys())}", "TrafficControl"
            )
        except Exception as e:
            logger.warning(f"扫描原生命令索引失败: {e}", "TrafficControl")

        regex_list: list[str] = []

        logger.info("开始扫描 Alconna Manager 以构建命令索引...", "TrafficControl")

        alc_count = 0
        for cmd in command_manager.get_commands():
            try:
                cmd_name = str(cmd.command)

                if cmd_name.startswith("re:"):
                    regex_list.append(cmd_name[3:])
                else:
                    prefixes = cmd.prefixes
                    triggers = []
                    if not prefixes:
                        triggers.append(cmd_name)
                    else:
                        for p in prefixes:
                            if isinstance(p, str):
                                triggers.append(f"{p}{cmd_name}")

                    for t in triggers:
                        if t.strip():
                            self._command_index.insert(t)
                            count += 1
                            alc_count += 1

                shortcuts = getattr(cmd, "_get_shortcuts", lambda: {})()
                if shortcuts:
                    for key in shortcuts.keys():
                        if isinstance(key, str) and key.strip():
                            regex_list.append(key)
            except Exception as e:
                logger.warning(f"索引 Alconna 命令失败: {e}", "TrafficControl")

        if regex_list:
            try:
                named_group_pattern = r"\?P<[^>]+>"
                cleaned_patterns = [
                    f"(?:{re.sub(named_group_pattern, '?:', p)})" for p in regex_list
                ]
                self._combined_regex = re.compile("|".join(cleaned_patterns))
                logger.info(
                    f"已合并 {len(cleaned_patterns)} 个正则规则为 Mega-Regex",
                    "TrafficControl",
                )
            except Exception as e:
                logger.error(f"合并正则失败: {e}", "TrafficControl")
                self._combined_regex = None

        logger.info(
            f"命令索引构建完成，共索引 {count} 个前缀触发词",
            "TrafficControl",
        )

        logger.info(
            f"流量控制器已初始化 | 并发限制: {self.max_workers}", "TrafficControl"
        )

    def get_event_priority(self, bot: Bot, event: Event) -> int:
        """获取事件的优先级。

        参数:
            bot: Bot 实例。
            event: 事件对象。

        返回:
            int: 优先级数值，越小越优先。
        """
        if isinstance(event, MetaEvent):
            return 0

        user_id = getattr(event, "user_id", None)

        if user_id and str(user_id) in bot.config.superusers:
            return 10

        if isinstance(event, PrivateMessageEvent):
            return 20

        if isinstance(event, MessageEvent):
            is_tome = event.is_tome()

            if not is_tome:
                for seg in event.get_message():
                    if seg.type == "at" and str(seg.data.get("qq")) == str(bot.self_id):
                        is_tome = True
                        break

            text = event.get_plaintext().lstrip()

            if not is_tome:
                try:
                    nicknames = set(nonebot.get_driver().config.nickname)
                    if BotConfig.self_nickname:
                        nicknames.add(BotConfig.self_nickname)

                    for nick in nicknames:
                        if text.startswith(nick):
                            is_tome = True
                            break
                except Exception:
                    pass

            matched_prefix = self._command_index.match(text)
            if matched_prefix:
                logger.debug(
                    f"[流量控制] 命中前缀树: '{matched_prefix}' -> 消息:"
                    f"'{text[:20]}...'",
                    "TrafficControl",
                )
                return 20

            if self._combined_regex and self._combined_regex.match(text):
                logger.debug(
                    f"[流量控制] 命中超大正则 -> 消息: '{text[:20]}...'",
                    "TrafficControl",
                )
                return 20

            if is_tome:
                return 22
            return 30

        if isinstance(event, RequestEvent):
            return 25

        if isinstance(event, NoticeEvent):
            if isinstance(
                event,
                (
                    GroupDecreaseNoticeEvent,
                    GroupIncreaseNoticeEvent,
                    GroupAdminNoticeEvent,
                ),
            ):
                return 25

            if isinstance(event, PokeNotifyEvent):
                return 29

            return 40

        return 50

    async def _worker(self, worker_id: int):
        """消费者工作循环，从队列中取出事件并处理。"""
        logger.trace(f"流量控制工作线程 {worker_id} 已启动", "TrafficControl")
        while True:
            try:
                item: QueueItem = await self.queue.get()

                bot = cast(V11Bot, item.bot)
                await self._original_handle_event(bot, item.event)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 错误: {e}", "TrafficControl")
            finally:
                self.queue.task_done()

    async def start(self):
        """启动流量控制器，初始化索引并启动工作线程。"""
        if not Config.get_config("traffic_control", "ENABLE", True):
            logger.info("流量控制已关闭", "TrafficControl")
            return

        self._build_command_index()

        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._worker_tasks.append(task)

        async def patched_handle_event(self: V11Bot, event: Event):
            priority = traffic_controller.get_event_priority(self, event)

            current_qsize = traffic_controller.queue.qsize()

            if (
                traffic_controller.max_queue_size > 0
                and priority >= 30
                and current_qsize > traffic_controller.max_queue_size
            ):
                logger.warning(
                    f"队列已满 ({current_qsize}/{traffic_controller.max_queue_size})，"
                    f"丢弃低优先级消息 (P{priority})",
                    "TrafficControl",
                )
                return

            uid = getattr(event, "user_id", "N/A")
            gid = getattr(event, "group_id", "private")
            logger.debug(
                f"入队 | 优先级: <y>{priority}</y> | {event.get_event_name()} |"
                f" 用户:{uid} 群组:{gid}",
                "TrafficControl",
            )

            traffic_controller.queue.put_nowait(
                QueueItem(
                    priority=priority, timestamp=time.time(), bot=self, event=event
                )
            )

        V11Bot.handle_event = patched_handle_event
        logger.success("全局事件优先队列已启动", "TrafficControl")

    async def stop(self):
        """停止流量控制器，还原 handle_event 方法并取消工作线程。"""
        V11Bot.handle_event = self._original_handle_event

        for task in self._worker_tasks:
            if not task.done():
                task.cancel()

        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        logger.info("流量控制工作线程已停止", "TrafficControl")


traffic_controller = TrafficController()

driver = nonebot.get_driver()
driver.on_startup(traffic_controller.start)
driver.on_shutdown(traffic_controller.stop)
