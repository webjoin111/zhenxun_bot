import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
import traceback
from typing import Any, TypeVar

from zhenxun.services.log import logger
from zhenxun.utils.utils import infer_plugin_namespace

from .base import AIEvent

E = TypeVar("E", bound=AIEvent)


class _EventCenter:
    """
    真寻 AI 事件发布-订阅中心。
    完全异步无阻塞，解耦各模块组件。
    """

    def __init__(self):
        self._subscribers: dict[type[AIEvent], dict[str, list[tuple[int, Any]]]] = (
            defaultdict(lambda: defaultdict(list))
        )

    def subscribe(
        self, event_type: type[E], priority: int = 10, scope: str | None = None
    ) -> Callable[[Callable[[E], Awaitable[None]]], Callable[[E], Awaitable[None]]]:
        """
        订阅装饰器。
        priority 越小越先执行。遇到异常会中断后续执行。
        scope: 指定监听的命名空间，默认隐式推断当前插件。传入 "*" 表示监听所有插件的事件。
        用法:
            @EventCenter.subscribe(ToolCallEvent, priority=1)
            async def my_handler(event: ToolCallEvent): ...
        """
        ns = scope if scope is not None else infer_plugin_namespace()

        def decorator(
            func: Callable[[E], Awaitable[None]],
        ) -> Callable[[E], Awaitable[None]]:
            self._subscribers[event_type][ns].append((priority, func))
            self._subscribers[event_type][ns].sort(key=lambda x: x[0])
            return func

        return decorator

    async def publish(self, event: AIEvent) -> None:
        """
        发布事件。按优先级**顺序**触发所有订阅了该事件类型（及其父类）的处理函数。
        如果任何一个监听器抛出异常，将立刻中断执行并向上抛出。
        """
        event_type = type(event)
        all_handlers = []

        for registered_type, registered_handlers_dict in self._subscribers.items():
            if issubclass(event_type, registered_type):
                all_handlers.extend(registered_handlers_dict.get("*", []))
                if event.namespace != "*":
                    all_handlers.extend(
                        registered_handlers_dict.get(event.namespace, [])
                    )

        if not all_handlers:
            return

        all_handlers.sort(key=lambda x: x[0])

        async def _run_handler(h: Callable, evt: AIEvent):
            try:
                await h(evt)
            except Exception as e:
                logger.error(
                    f"Event Listener '{h.__name__}' 处理事件 '{type(evt).__name__}' 时抛出异步异常: {e}\n{traceback.format_exc()}"
                )

        for _, handler in all_handlers:
            asyncio.create_task(_run_handler(handler, event))


EventCenter = _EventCenter()
