import asyncio
from collections.abc import Callable
import functools
import inspect
from typing import Any


def wrap_to_async(func: Callable) -> Callable:
    """
    将任意类型的函数（同步、异步、同步生成器、异步生成器）统一包装为异步执行形态。
    确保不会阻塞主线程，并保留原函数的元数据和签名。
    """
    if getattr(func, "__is_async_wrapper__", False):
        return func

    if inspect.isasyncgenfunction(func) or inspect.iscoroutinefunction(func):
        return func

    if inspect.isgeneratorfunction(func):

        async def async_gen_wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            _sentinel = object()

            def _sync_runner():
                try:
                    for item in func(*args, **kwargs):
                        asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
                    asyncio.run_coroutine_threadsafe(q.put(_sentinel), loop).result()
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(e), loop).result()

            import threading

            t = threading.Thread(target=_sync_runner)
            t.start()

            while True:
                item = await q.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        functools.update_wrapper(async_gen_wrapper, func)
        setattr(async_gen_wrapper, "__is_async_wrapper__", True)
        setattr(async_gen_wrapper, "_is_async_gen", True)
        return async_gen_wrapper

    async def async_wrapper(*args, **kwargs):
        p_func = functools.partial(func, *args, **kwargs)
        return await asyncio.to_thread(p_func)

    functools.update_wrapper(async_wrapper, func)
    setattr(async_wrapper, "__is_async_wrapper__", True)
    setattr(async_wrapper, "_is_coroutine", True)
    return async_wrapper


class ContextUtils:
    """
    从底层依赖容器 (deps) 中提取运行环境信息的纯静态工具类。

    """

    @staticmethod
    def extract_user_id(deps: Any) -> str | None:
        if not deps:
            return None
        if hasattr(deps, "user_id") and getattr(deps, "user_id") is not None:
            return str(getattr(deps, "user_id"))
        event = getattr(deps, "event", None)
        if event:
            try:
                return str(event.get_user_id())
            except Exception:
                return (
                    str(getattr(event, "user_id", ""))
                    or str(getattr(event, "sender_id", ""))
                    or None
                )
        return None

    @staticmethod
    def extract_group_id(deps: Any) -> str | None:
        if not deps:
            return None
        if hasattr(deps, "group_id") and getattr(deps, "group_id") is not None:
            return str(getattr(deps, "group_id"))
        event = getattr(deps, "event", None)
        if event:
            return str(getattr(event, "group_id", "")) or None
        return None

    @staticmethod
    def extract_platform(deps: Any) -> str:
        if not deps:
            return "unknown"
        if hasattr(deps, "platform") and getattr(deps, "platform") is not None:
            return str(getattr(deps, "platform"))
        bot = getattr(deps, "bot", None)
        if bot:
            from zhenxun.utils.platform import PlatformUtils

            return PlatformUtils.get_platform(bot)
        return "unknown"
