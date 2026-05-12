from collections.abc import Callable
import inspect
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.protocols.capabilities import (
    AbstractCapability,
    WrapToolExecuteHandler,
)
from zhenxun.services.ai.run import DependencyInjector, RunContext
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.cache.cache_containers import CacheDict
from zhenxun.services.log import logger

TOOL_RESULT_CACHE = CacheDict("TOOL_RESULT", expire=0)


class CacheCapability(AbstractCapability):
    """
    极速缓存能力中间件 (Tool-Level)
    拦截相同的参数调用，直接返回缓存结果，无需请求底层。
    """

    def __init__(self, ttl: int = 3600, cache_function: Callable | None = None):
        self.ttl = ttl
        self.cache_function = cache_function

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        tool = context.call.current_tool
        if tool is None:
            return await handler(arguments)
        if not hasattr(tool, "_generate_cache_key"):
            return await handler(arguments)

        cache_key = tool._generate_cache_key(arguments)
        try:
            cached_data = TOOL_RESULT_CACHE[cache_key]
            from zhenxun.utils.pydantic_compat import parse_as

            cached_result = (
                parse_as(ToolResult, cached_data)
                if isinstance(cached_data, dict)
                else cached_data
            )
            logger.info(
                f"⚡ [Cache Hit] 工具 {tool_name} 命中极速缓存, "
                f"熔断执行! Key: {cache_key}"
            )
            return cached_result
        except KeyError:
            pass

        result = await handler(arguments)

        if result and not getattr(result, "is_error", False):
            should_cache = True
            if self.cache_function:
                try:
                    should_cache = self.cache_function(arguments, result)
                except Exception as ce:
                    logger.warning(f"Cache function 执行失败: {ce}")
                    should_cache = False
            if should_cache:
                from zhenxun.utils.pydantic_compat import model_dump

                TOOL_RESULT_CACHE.set(cache_key, model_dump(result), expire=self.ttl)

        return result


class ApprovalCapability(AbstractCapability):
    """
    人工审批能力中间件 (Tool-Level / HITL)
    执行高危操作前在群聊中发起授权确认，拒绝则抛出取消异常。
    """

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        tool = context.call.current_tool
        if tool is None:
            return arguments

        if hasattr(tool, "should_confirm"):
            confirm_msg = await tool.should_confirm(context=context, **arguments)
            if confirm_msg:
                import asyncio

                if "global" not in context.run.hitl_locks:
                    context.run.hitl_locks["global"] = asyncio.Lock()
                hitl_lock = context.run.hitl_locks["global"]

                await hitl_lock.acquire()
                try:
                    from zhenxun.services.ai.run.hitl import HITLController
                    hitl = HITLController(context)
                    await hitl.ask_confirm(
                        f"⚠️ **安全交互审批**\n\n{confirm_msg}", timeout=60.0
                    )
                    logger.info(f"🛡️ [HITL] 工具 {tool_name} 审批通过。")
                finally:
                    hitl_lock.release()

        return arguments


class LifecycleCapability(AbstractCapability):
    """
    通用生命周期能力中间件 (Generic Lifecycle Capability)
    接收开发者传入的简单函数，自动为其提供依赖注入 (DI) 和洋葱模型拦截。
    """

    def __init__(
        self,
        before_execute: Callable | None = None,
        after_execute: Callable | None = None,
        validate_args: Callable | None = None,
        prepare_tool: Callable | None = None,
    ):
        self.before_execute_hook = before_execute
        self.after_execute_hook = after_execute
        self.validate_args_hook = validate_args
        self.prepare_tool_hook = prepare_tool

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        if not self.prepare_tool_hook:
            return tool_defs
        new_defs = []
        for tdef in tool_defs:
            sig = inspect.signature(self.prepare_tool_hook)
            kwargs = await DependencyInjector.resolve_all(
                sig, {"tool_def": tdef}, context
            )
            if is_coroutine_callable(self.prepare_tool_hook):
                res = await self.prepare_tool_hook(**kwargs)
            else:
                res = self.prepare_tool_hook(**kwargs)
            if res is not None:
                new_defs.append(res)
        return new_defs

    async def before_tool_validate(
        self, context: RunContext, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        if not self.validate_args_hook or not isinstance(args, dict):
            return args
        sig = inspect.signature(self.validate_args_hook)
        call_kwargs = dict(args)
        call_kwargs["args"] = args
        call_kwargs["tool_args"] = args
        call_kwargs["tool_name"] = tool_name
        resolved_kwargs = await DependencyInjector.resolve_all(
            sig, call_kwargs, context
        )
        filtered_kwargs = {
            k: v for k, v in resolved_kwargs.items() if k in sig.parameters
        }
        if is_coroutine_callable(self.validate_args_hook):
            await self.validate_args_hook(**filtered_kwargs)
        else:
            self.validate_args_hook(**filtered_kwargs)
        return args

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.before_execute_hook:
            return arguments
        sig = inspect.signature(self.before_execute_hook)
        call_kwargs = dict(arguments)
        call_kwargs["args"] = arguments
        call_kwargs["tool_args"] = arguments
        call_kwargs["tool_name"] = tool_name
        resolved_kwargs = await DependencyInjector.resolve_all(
            sig, call_kwargs, context
        )
        filtered_kwargs = {
            k: v for k, v in resolved_kwargs.items() if k in sig.parameters
        }
        if is_coroutine_callable(self.before_execute_hook):
            await self.before_execute_hook(**filtered_kwargs)
        else:
            self.before_execute_hook(**filtered_kwargs)
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        if not self.after_execute_hook:
            return result
        sig = inspect.signature(self.after_execute_hook)
        call_kwargs = dict(arguments)
        call_kwargs["args"] = arguments
        call_kwargs["tool_args"] = arguments
        call_kwargs["tool_name"] = tool_name
        call_kwargs["result"] = result
        call_kwargs["tool_result"] = result
        resolved_kwargs = await DependencyInjector.resolve_all(
            sig, call_kwargs, context
        )
        filtered_kwargs = {
            k: v for k, v in resolved_kwargs.items() if k in sig.parameters
        }
        if is_coroutine_callable(self.after_execute_hook):
            res = await self.after_execute_hook(**filtered_kwargs)
        else:
            res = self.after_execute_hook(**filtered_kwargs)
        return res if res is not None else result


class SuperuserCapability(AbstractCapability):
    """仅超级管理员可见"""

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        bot = getattr(context.deps, "bot", None)
        event = getattr(context.deps, "event", None)
        if not isinstance(bot, Bot) or not isinstance(event, Event):
            return []
        if await SUPERUSER(bot, event):
            return tool_defs
        return []


class AdminLevelCapability(AbstractCapability):
    """需要指定群聊权限等级"""

    def __init__(self, min_level: int):
        self.min_level = min_level

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        from zhenxun.services.ai.utils.runtime_utils import ContextUtils
        from zhenxun.services.cache.runtime_cache import LevelUserMemoryCache

        user_id = ContextUtils.extract_user_id(context.deps)
        group_id = ContextUtils.extract_group_id(context.deps)
        bot = getattr(context.deps, "bot", None)
        event = getattr(context.deps, "event", None)

        if (
            isinstance(bot, Bot)
            and isinstance(event, Event)
            and await SUPERUSER(bot, event)
        ):
            return tool_defs

        if not user_id or not group_id:
            return []
        global_user, group_users = await LevelUserMemoryCache.get_levels(
            user_id, group_id
        )
        user_level = global_user.user_level if global_user else 0
        if group_users:
            user_level = max(user_level, group_users.user_level)

        if user_level >= self.min_level:
            return tool_defs
        return []


class GroupOnlyCapability(AbstractCapability):
    """仅群聊可用"""

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        from zhenxun.services.ai.utils.runtime_utils import ContextUtils

        if ContextUtils.extract_group_id(context.deps):
            return tool_defs
        return []


class ConfigDependencyCapability(AbstractCapability):
    """依赖特定配置开关"""

    def __init__(self, module: str, key: str, expected_value: Any = True):
        self.module = module
        self.key = key
        self.expected_value = expected_value

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        from zhenxun.configs.config import Config

        if Config.get_config(self.module, self.key) == self.expected_value:
            return tool_defs
        return []


class InteractiveCapability(AbstractCapability):
    """交互式补全局部中间件：在验证阶段拦截参数缺失异常并向用户提问"""

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: Callable,
    ) -> dict[str, Any]:
        from zhenxun.services.ai.core.exceptions import (
            NeedsInputException,
            ToolFatalError,
            ToolRetryError,
        )

        tool = context.call.current_tool
        current_kwargs = dict(args) if isinstance(args, dict) else args

        while True:
            try:
                return await handler(current_kwargs)
            except NeedsInputException as e:
                bot = getattr(context.deps, "bot", None)
                event = getattr(context.deps, "event", None)

                if not isinstance(bot, Bot) or not isinstance(event, Event):
                    logger.warning(
                        f"交互式工具 {getattr(tool, 'name', 'unknown')} 缺少参数，但处于非交互环境。"
                    )
                    raise ToolRetryError(f"缺少必填参数: {e.missing_description}。")

                prompt_msg = f"执行 {getattr(tool, 'name', '该操作')} 需要补充参数：\n[{e.missing_description}]\n请发送文本补充，或回复“取消”中止。"
                try:
                    from zhenxun.services.ai.run.hitl import HITLController
                    hitl = HITLController(context)
                    user_input = await hitl.ask_text(prompt_msg, timeout=60.0)
                except ToolFatalError:
                    raise ToolFatalError(
                        "参数收集超时，用户已离开，任务已中止。",
                        display_content=f"❌ 工具 '{getattr(tool, 'name', '')}' 等待参数输入超时被取消。",
                    )
                if not isinstance(current_kwargs, dict):
                    current_kwargs = {}
                current_kwargs[e.missing_field] = user_input


class FallbackCapability(AbstractCapability):
    """
    降级路由器局部中间件。
    拦截执行结果，若发现报错且配置了备用工具，则透明重定向执行流。
    """

    def __init__(self, fallback_tool_name: str):
        self.fallback_tool_name = fallback_tool_name

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        if result and getattr(result, "is_error", False):
            available_tools = context.state.get("__available_tools", {})
            fallback_executable = available_tools.get(self.fallback_tool_name)
            if fallback_executable:
                logger.info(
                    f"🔄 [语义容错路由] 主工具 '{tool_name}' 故障，正在透明重定向至备用工具 '{self.fallback_tool_name}'..."
                )
                try:
                    fallback_result = await fallback_executable.execute(
                        context=context, **arguments
                    )
                    notice = f"[系统底座：由于主工具 '{tool_name}' 发生故障，系统已自动路由至备用工具 '{self.fallback_tool_name}' 并执行成功]\n"
                    if isinstance(fallback_result.output, str):
                        fallback_result.output = notice + fallback_result.output
                    return fallback_result
                except Exception as fallback_e:
                    logger.error(
                        f"容错路由工具 '{self.fallback_tool_name}' 也执行失败: {fallback_e}"
                    )
                    context.run.add_system_prompt(
                        f"🚨 [系统警告] 原工具和备用工具({self.fallback_tool_name})均执行失败，请停止尝试并告知用户。"
                    )
        return result
