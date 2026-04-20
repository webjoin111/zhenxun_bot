from collections.abc import Callable
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.tool import LazyToolProxy
from zhenxun.services.ai.tools.engine.registry import tool_provider_manager
from zhenxun.services.ai.types.tools import ToolOptions
from zhenxun.services.cache.runtime_cache import LevelUserMemoryCache
from zhenxun.services.log import logger


def _update_settings(func: Callable, **kwargs) -> Callable:
    """叠加协议底层：创建或更新函数的 __tool_settings__"""
    if not hasattr(func, "__tool_settings__"):
        setattr(func, "__tool_settings__", ToolOptions())
    settings: ToolOptions = getattr(func, "__tool_settings__")
    for k, v in kwargs.items():
        if k == "middlewares":
            settings.middlewares.extend(v)
        elif k == "prepare":
            old_prepare = settings.prepare
            if old_prepare:

                def make_chained(p1, p2):
                    async def chained(ctx, tdef):
                        from nonebot.utils import is_coroutine_callable

                        res1 = (
                            await p1(ctx, tdef)
                            if is_coroutine_callable(p1)
                            else p1(ctx, tdef)
                        )
                        if res1 is None:
                            return None
                        res2 = (
                            await p2(ctx, res1)
                            if is_coroutine_callable(p2)
                            else p2(ctx, res1)
                        )
                        return res2

                    return chained

                settings.prepare = make_chained(old_prepare, v)
            else:
                settings.prepare = v
        elif k == "metadata":
            settings.metadata.update(v)
        else:
            setattr(settings, k, v)
    return func


def require_sandbox(
    python_packages: list[str] | None = None,
    node_packages: list[str] | None = None,
    system_packages: list[str] | None = None,
):
    """显式声明该工具所需安装的沙箱环境依赖"""

    def decorator(func):
        setattr(
            func,
            "__sandbox_requirements__",
            {
                "python": python_packages or [],
                "node": node_packages or [],
                "system": system_packages or [],
            },
        )
        return func

    return decorator


def tool(
    name: str | None = None,
    description: str | None = None,
    max_retries: int | None = None,
    settings: ToolOptions | None = None,
):
    """
    将普通函数注册为 LLM 工具的装饰器。

    Args:
        name: 工具的名称(英文字母及下划线)。大模型将看到此名称。如果为空则默认使用函数名。
        description: 工具描述。大模型将基于此决定何时、如何使用该工具。如果为空，将读取函数 docstring。
        max_retries: 工具的最大重试次数。超过此次数后将阻断大模型继续尝试。
        settings: 工具的高阶配置对象 (ToolOptions)。用于控制极速缓存、人工审批拦截、静默执行等扩展能力。
    """

    def decorator(func: Callable):
        base_settings = settings or ToolOptions()
        if max_retries is not None:
            base_settings.max_retries = max_retries
        existing_settings = getattr(func, "__tool_settings__", None)
        if existing_settings and isinstance(existing_settings, ToolOptions):
            base_settings = existing_settings.merge(base_settings)

        reqs = getattr(func, "__sandbox_requirements__", None)
        if reqs:
            base_settings.sandbox_requirements = reqs

        tool_name = name or func.__name__
        tool_desc = description or "未提供描述"

        def tool_factory():
            from zhenxun.services.ai.tools.core.tool import FunctionTool

            return FunctionTool(func, tool_name, tool_desc, base_settings)

        proxy = LazyToolProxy(
            name=tool_name, description=tool_desc, factory=tool_factory
        )
        tool_provider_manager.register_tool(proxy)
        logger.info(f"已注册全局工具(懒加载): '{tool_name}'")

        setattr(func, "__tool_settings__", base_settings)
        return func

    return decorator


def toolkit_tool(
    name: str | None = None,
    description: str | None = None,
    max_retries: int | None = None,
    settings: ToolOptions | None = None,
):
    """
    用于 Toolkit 类内部方法的工具注册装饰器。

    Args:
        name: 工具的名称。如果为空则默认使用函数名。
        description: 工具描述。
        max_retries: 工具的最大重试次数。超过此次数后将阻断大模型继续尝试。
        settings: 工具的高阶配置对象 (ToolOptions)。
    """

    def decorator(func: Callable):
        base_settings = settings or ToolOptions()
        if max_retries is not None:
            base_settings.max_retries = max_retries
        existing_settings = getattr(func, "__tool_settings__", None)
        if existing_settings and isinstance(existing_settings, ToolOptions):
            base_settings = existing_settings.merge(base_settings)

        reqs = getattr(func, "__sandbox_requirements__", None)
        if reqs:
            base_settings.sandbox_requirements = reqs

        setattr(func, "__toolkit_tool__", True)
        setattr(func, "__tool_name__", name or func.__name__)
        setattr(func, "__tool_desc__", description or func.__doc__ or "未提供描述")
        setattr(func, "__tool_settings__", base_settings)
        return func

    return decorator


def with_cache(ttl: int = 3600, cache_function: Callable | None = None):
    """开启极速缓存，阻止参数相同的重复请求发往底层。"""

    def decorator(func: Callable):
        return _update_settings(
            func, cache=True, cache_ttl=ttl, cache_function=cache_function
        )

    return decorator


def silent():
    """静默执行。工具执行过程与结果不会作为界面流渲染给用户，仅作大模型内部参考。"""

    def decorator(func: Callable):
        return _update_settings(func, silent=True)

    return decorator


def direct_reply():
    """直出模式。工具执行完毕后强制中断大模型的思考循环，将工具输出作为最终回答返回。"""

    def decorator(func: Callable):
        return _update_settings(func, direct_reply=True)

    return decorator


def require_approval():
    """高危操作标记。调用前将拦截并发送至群组要求超管人工审核。"""

    def decorator(func: Callable):
        return _update_settings(func, require_approval=True)

    return decorator


def require_superuser():
    """仅限超级管理员使用。若调用者无权限，该工具将直接对大模型隐藏。"""

    async def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        bot = ctx.bot
        event = ctx.event
        if not isinstance(bot, Bot) or not isinstance(event, Event):
            return None
        if await SUPERUSER(bot, event):
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(func, prepare=_prepare)

    return decorator


def require_admin_level(min_level: int = 1):
    """仅限满足指定群聊权限等级的用户使用。"""

    async def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        user_id, group_id = ctx.get_user_id(), ctx.get_group_id()
        bot = ctx.bot
        event = ctx.event
        if (
            isinstance(bot, Bot)
            and isinstance(event, Event)
            and await SUPERUSER(bot, event)
        ):
            return tool_def
        if not user_id or not group_id:
            return None
        global_user, group_users = await LevelUserMemoryCache.get_levels(
            user_id, group_id
        )
        user_level = global_user.user_level if global_user else 0
        if group_users:
            user_level = max(user_level, group_users.user_level)
        if user_level >= min_level:
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(
            func, prepare=_prepare, metadata={"admin_level": min_level}
        )

    return decorator


def require_group():
    """限制该工具仅能在群聊环境中被大模型调用。"""

    def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        if ctx.get_group_id():
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(func, prepare=_prepare)

    return decorator


def require_minimum_gold(amount: int):
    """需满足一定的金币余额才可调用，同时执行后会自动扣除该金币。"""

    async def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        user_id = ctx.get_user_id()
        if not user_id:
            return None
        bot = ctx.bot
        event = ctx.event
        if (
            isinstance(bot, Bot)
            and isinstance(event, Event)
            and await SUPERUSER(bot, event)
        ):
            return tool_def
        from zhenxun.models.user_console import UserConsole

        user = await UserConsole.get_or_none(user_id=user_id)
        if user is not None and user.gold >= amount:
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(func, prepare=_prepare, metadata={"cost_gold": amount})

    return decorator


def require_session_state(key: str, expected_value: Any = None):
    """需要当前执行上下文中包含指定状态变量，否则隐藏工具"""

    def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        val = ctx.extra.get(key)
        if expected_value is not None:
            if val == expected_value:
                return tool_def
        elif val is not None and val is not False:
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(func, prepare=_prepare)

    return decorator


def require_config(module: str, key: str, expected_value: Any = True):
    """需要 WebUI 中指定模块开关打开才可被大模型调用。"""

    def _prepare(ctx: RunContext, tool_def: Any) -> Any | None:
        from zhenxun.configs.config import Config

        if Config.get_config(module, key) == expected_value:
            return tool_def
        return None

    def decorator(func: Callable):
        return _update_settings(func, prepare=_prepare)

    return decorator
