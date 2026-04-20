import json
import time
from typing import Any

from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot_plugin_waiter import waiter

from zhenxun.models.level_user import LevelUser
from zhenxun.models.user_console import UserConsole
from zhenxun.services.ai.events import (
    EventCenter,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
from zhenxun.services.ai.protocols.tool import (
    ToolExecutable,
    ToolMiddleware,
    ToolNextCall,
)
from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.types.exceptions import NeedsAuthException, NeedsInputException
from zhenxun.services.ai.types.tools import ToolResult
from zhenxun.services.cache.cache_containers import CacheDict
from zhenxun.services.cache.runtime_cache import LevelUserMemoryCache
from zhenxun.services.log import logger
from zhenxun.utils.enum import GoldHandle
from zhenxun.utils.exception import InsufficientGold
from zhenxun.utils.pydantic_compat import model_dump


class DummyCredentialManager:
    """一个模拟的全局凭证管理器"""

    _tokens: dict[str, dict[str, str]] = {}

    @classmethod
    def get_token(cls, user_id: str, provider: str) -> str | None:
        return cls._tokens.get(user_id, {}).get(provider)

    @classmethod
    def set_token(cls, user_id: str, provider: str, token: str) -> None:
        if user_id not in cls._tokens:
            cls._tokens[user_id] = {}
        cls._tokens[user_id][provider] = token

    @classmethod
    def clear_token(cls, user_id: str, provider: str) -> None:
        if user_id in cls._tokens and provider in cls._tokens[user_id]:
            del cls._tokens[user_id][provider]


TOOL_RESULT_CACHE = CacheDict("TOOL_RESULT", expire=0)


class ToolCacheMiddleware:
    """工具极速缓存中间件：根据参数生成Hash拦截执行，复用历史结果"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        settings = getattr(tool, "settings", None)
        is_cacheable = settings.cache if settings else False
        cache_key = None

        if is_cacheable and hasattr(tool, "_generate_cache_key"):
            cache_key = tool._generate_cache_key(kwargs)  # type: ignore
            try:
                cached_data = TOOL_RESULT_CACHE[cache_key]
                from zhenxun.utils.pydantic_compat import parse_as

                cached_result = (
                    parse_as(ToolResult, cached_data)
                    if isinstance(cached_data, dict)
                    else cached_data
                )
                logger.info(
                    f"⚡ [Cache Hit] 工具 {getattr(tool, 'name', 'unknown')} 命中极速缓存, 熔断执行! Key: {cache_key}",
                    "ToolCacheMiddleware",
                )
                return cached_result
            except KeyError:
                pass

        result = await next_call(kwargs, context)

        if (
            is_cacheable
            and cache_key
            and result
            and not getattr(result, "is_error", False)
        ):
            cache_func = settings.cache_function if settings else None
            should_cache = True
            if cache_func:
                try:
                    should_cache = cache_func(kwargs, result)
                except Exception as ce:
                    logger.warning(
                        f"Cache function 执行失败: {ce}", "ToolCacheMiddleware"
                    )
                    should_cache = False

            if should_cache:
                ttl = settings.cache_ttl if settings else 3600
                try:
                    TOOL_RESULT_CACHE.set(cache_key, model_dump(result), expire=ttl)
                    logger.debug(
                        f"💾 [Cache Store] 工具 {getattr(tool, 'name', 'unknown')} 结果已存入全局缓存池, TTL: {ttl}s",
                        "ToolCacheMiddleware",
                    )
                except Exception as e:
                    logger.warning(f"写入工具缓存失败: {e}", "ToolCacheMiddleware")

        return result


class ManualConfirmMiddleware:
    """人机交互审批中间件 (HITL)：拦截高危操作并在群内发起授权确认"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        if not hasattr(tool, "should_confirm"):
            return await next_call(kwargs, context)

        confirm_msg = await tool.should_confirm(**kwargs)
        if not confirm_msg:
            return await next_call(kwargs, context)

        if "hitl_lock" not in context.extra:
            import asyncio

            context.extra["hitl_lock"] = asyncio.Lock()
        hitl_lock = context.extra["hitl_lock"]

        await hitl_lock.acquire()
        try:
            bot = context.bot
            event = context.event

            if bot and event and isinstance(bot, Bot) and isinstance(event, Event):
                settings = getattr(tool, "settings", None)
                admin_level = (
                    settings.metadata.get("admin_level", 0) if settings else 0
                ) or getattr(tool, "metadata", {}).get("admin_level", 0)
                auth_notice = "\n(群管/超管可代为审批)" if admin_level > 0 else ""

                await bot.send(
                    event,
                    f"⚠️ **安全交互审批**\n\n{confirm_msg}{auth_notice}\n\n请在 60 秒内回复 [Y/是] 确认，或 [N/否] 拒绝。",
                )
                logger.info(
                    f"🛡️ [HITL] 工具 {getattr(tool, 'name', 'unknown')} 已挂起，等待用户交互审批..."
                )

                original_user_id = context.get_user_id()
                original_group_id = context.get_group_id()

                @waiter(waits=["message"], keep_session=False)
                async def confirm_waiter(e: Event, b: Bot):
                    raw_curr_group = getattr(
                        e, "group_id", getattr(e, "channel_id", None)
                    )
                    curr_group_id = str(raw_curr_group) if raw_curr_group else None
                    orig_group_id = (
                        str(original_group_id) if original_group_id else None
                    )

                    if curr_group_id != orig_group_id:
                        return None

                    text = e.get_plaintext().strip().lower()
                    is_command = text in [
                        "y",
                        "yes",
                        "是",
                        "1",
                        "ok",
                        "确认",
                        "n",
                        "no",
                        "否",
                        "0",
                        "取消",
                        "cancel",
                        "拒绝",
                    ]
                    if not is_command:
                        return None

                    curr_user_id = str(e.get_user_id())
                    is_authorized = False

                    if curr_user_id == original_user_id:
                        is_authorized = True
                    elif await SUPERUSER(b, e):
                        is_authorized = True
                    elif admin_level > 0 and original_group_id:
                        if await LevelUser.check_level(
                            curr_user_id, original_group_id, admin_level
                        ):
                            is_authorized = True

                    if not is_authorized:
                        await b.send(e, "⚠️ 权限不足，你无权审批此高危操作。")
                        return None

                    return text in ["y", "yes", "是", "1", "ok", "确认"]

                is_confirmed = await confirm_waiter.wait(timeout=60)
                if is_confirmed is None:
                    logger.warning(
                        f"🛡️ [HITL] 审批超时，已自动取消工具 {getattr(tool, 'name', 'unknown')}。"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "审批超时，已取消。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )
                elif is_confirmed is False:
                    logger.warning(
                        f"🛡️ [HITL] 用户明确拒绝了工具 {getattr(tool, 'name', 'unknown')}。"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "用户拒绝了操作。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )
            else:
                return ToolResult(
                    output=json.dumps(
                        {
                            "error_type": "UserCancellation",
                            "message": "环境不支持审批，已拦截。",
                            "is_retryable": False,
                        },
                        ensure_ascii=False,
                    ),
                    is_error=True,
                )
        finally:
            hitl_lock.release()

        return await next_call(kwargs, context)


class RequireAuthMiddleware:
    """声明式鉴权与动态授权(HITL)中间件"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        user_id = context.get_user_id()

        settings = getattr(tool, "settings", None)
        auth_provider = (
            settings.metadata.get("auth_provider") if settings else None
        ) or getattr(tool, "metadata", {}).get("auth_provider")
        if auth_provider and user_id:
            if token := DummyCredentialManager.get_token(user_id, auth_provider):
                if "auth_tokens" not in context.extra:
                    context.extra["auth_tokens"] = {}
                context.extra["auth_tokens"][auth_provider] = token

        current_kwargs = dict(kwargs)
        while True:
            try:
                return await next_call(current_kwargs, context)
            except NeedsAuthException as e:
                provider = e.provider
                bot = context.bot
                event = context.event

                if (
                    not bot
                    or not event
                    or not user_id
                    or not isinstance(bot, Bot)
                    or not isinstance(event, Event)
                ):
                    logger.warning(
                        f"由于缺少 {provider} 凭证，工具执行被拦截（非交互环境）。"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "AuthFailed",
                                "message": f"缺失 {provider} 的授权凭证。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        display=f"❌ 执行失败: 缺少 {provider} 授权",
                        is_error=True,
                    )

                prompt_msg = (
                    f"⚠️ 该功能需要绑定您的 [{provider}] 账号以初始化连接。\n\n"
                    f"请点击链接完成 OAuth 授权，或直接在此回复您的 Token (回复'取消'中止操作)。"
                )
                await bot.send(event, prompt_msg)

                original_group_id = context.get_group_id()

                @waiter(waits=["message"], keep_session=False)
                async def auth_waiter(e_in: Event):
                    raw_curr_group = getattr(
                        e_in, "group_id", getattr(e_in, "channel_id", None)
                    )
                    if (
                        str(raw_curr_group) != str(original_group_id)
                        or str(e_in.get_user_id()) != user_id
                    ):
                        return None
                    return e_in.get_plaintext().strip()

                user_input = await auth_waiter.wait(timeout=60)

                if user_input is None or user_input.lower() in ["取消", "cancel", "0"]:
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "用户拒绝了授权绑定。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )

                DummyCredentialManager.set_token(user_id, provider, user_input)
                if "auth_tokens" not in context.extra:
                    context.extra["auth_tokens"] = {}
                context.extra["auth_tokens"][provider] = user_input
                await bot.send(
                    event, f"✅ [{provider}] 凭证已加载，正在恢复任务执行..."
                )


class MissingParamPromptMiddleware:
    """交互式补全中间件：拦截参数缺失异常"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        current_kwargs = dict(kwargs)

        while True:
            try:
                return await next_call(current_kwargs, context)

            except NeedsInputException as e:
                bot = context.bot
                event = context.event

                if not isinstance(bot, Bot) or not isinstance(event, Event):
                    logger.warning(
                        f"交互式工具 {getattr(tool, 'name', 'unknown')} 缺少参数，但处于非交互环境。"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "MissingParameter",
                                "message": f"缺少参数或格式错误: {e.missing_description}",
                                "is_retryable": True,
                            },
                            ensure_ascii=False,
                        ),
                        display=f"❌ 执行缺少参数: {e.missing_field}",
                        is_error=True,
                    )

                prompt_msg = f"执行 {getattr(tool, 'name', '该操作')} 需要补充参数：\n[{e.missing_description}]\n请发送文本补充，或回复“取消”中止。"
                await bot.send(event, prompt_msg)

                original_user_id = context.get_user_id()
                original_group_id = context.get_group_id()

                @waiter(waits=["message"], keep_session=False)
                async def input_waiter(e_in: Event):
                    raw_curr_group = getattr(
                        e_in, "group_id", getattr(e_in, "channel_id", None)
                    )
                    curr_group_id = str(raw_curr_group) if raw_curr_group else None
                    orig_group_id = (
                        str(original_group_id) if original_group_id else None
                    )

                    if curr_group_id != orig_group_id:
                        return None
                    if str(e_in.get_user_id()) != original_user_id:
                        return None

                    return e_in.get_plaintext().strip()

                user_input = await input_waiter.wait(timeout=60)

                if user_input is None:
                    logger.warning(
                        f"🛡️ [HITL] 参数收集超时，已自动取消工具 {getattr(tool, 'name', '')}。"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "参数收集超时，用户已离开。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )

                if user_input.lower() in ["取消", "cancel", "退出", "quit", "0"]:
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "用户主动取消了操作。",
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )

                current_kwargs[e.missing_field] = user_input


class PermissionMiddleware:
    """权限校验中间件：拦截权限不足的工具调用"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        admin_level = getattr(tool, "metadata", {}).get("admin_level", 0)
        if admin_level > 0:
            user_id = context.get_user_id()
            group_id = context.get_group_id()

            bot = context.bot
            event = context.event
            if (
                bot
                and event
                and isinstance(bot, Bot)
                and isinstance(event, Event)
                and await SUPERUSER(bot, event)
            ):
                return await next_call(kwargs, context)

            if user_id:
                global_user, group_users = await LevelUserMemoryCache.get_levels(
                    user_id, group_id
                )
                user_level = global_user.user_level if global_user else 0
                if group_id and group_users:
                    user_level = max(user_level, group_users.user_level)

                if user_level < admin_level:
                    msg = (
                        "系统警告：用户权限不足（需要等级 "
                        f"{admin_level}，用户仅有 {user_level}）。"
                        "请温和地向用户解释权限不足，并拒绝执行。"
                    )
                    logger.warning(
                        f"🛡️ [Middleware] 权限拦截: 用户 {user_id} 尝试调用 "
                        f"{getattr(tool, 'name', 'unknown')}"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "PermissionDenied",
                                "message": msg,
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        display=f"❌ 权限不足: 需要等级 {admin_level}",
                        is_error=True,
                    )
        return await next_call(kwargs, context)


class BillingMiddleware:
    """经济系统中间件：执行工具前扣除金币"""

    async def __call__(
        self,
        tool: ToolExecutable,
        kwargs: dict[str, Any],
        context: RunContext,
        next_call: ToolNextCall,
    ) -> ToolResult:
        settings = getattr(tool, "settings", None)
        cost_gold = (
            settings.metadata.get("cost_gold", 0) if settings else 0
        ) or getattr(tool, "metadata", {}).get("cost_gold", 0)
        if cost_gold > 0:
            user_id = context.get_user_id()
            platform = context.get_platform()
            if user_id:
                try:
                    await UserConsole.reduce_gold(
                        user_id,
                        cost_gold,
                        GoldHandle.PLUGIN,
                        f"agent_tool:{getattr(tool, 'name', 'unknown')}",
                        platform,
                    )
                except InsufficientGold:
                    msg = (
                        f"系统警告：用户金币不足（需要 {cost_gold} 金币，但余额不够）。"
                        "请向用户解释金币不足，提醒可通过签到赚取，并拒绝执行。"
                    )
                    logger.warning(
                        f"💰 [Middleware] 金币拦截: 用户 {user_id} 尝试调用 "
                        f"{getattr(tool, 'name', 'unknown')}"
                    )
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "InsufficientGold",
                                "message": msg,
                                "is_retryable": False,
                            },
                            ensure_ascii=False,
                        ),
                        display=f"❌ 余额不足: 需要 {cost_gold} 金币",
                        is_error=True,
                    )
        return await next_call(kwargs, context)


GLOBAL_MIDDLEWARES: list[ToolMiddleware] = [
    ToolCacheMiddleware(),
    PermissionMiddleware(),
    BillingMiddleware(),
    ManualConfirmMiddleware(),
    RequireAuthMiddleware(),
    MissingParamPromptMiddleware(),
]


def register_global_middleware(middleware: ToolMiddleware) -> None:
    GLOBAL_MIDDLEWARES.append(middleware)
    logger.debug(f"已注册全局工具中间件: {middleware.__class__.__name__}")


_SESSION_UI_STATES: dict[str, list[str]] = {}


class UIStreamerContext:
    """状态流收集上下文管理器，用于渲染 Markdown 战报"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = 0.0

    async def __aenter__(self):
        self.start_time = time.monotonic()
        _SESSION_UI_STATES[self.session_id] = []
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def render(self) -> str:
        lines = _SESSION_UI_STATES.get(self.session_id, [])
        if not lines:
            return ""
        duration = time.monotonic() - self.start_time
        header = f"🚀 **Agent 思考与执行流** (耗时 {duration:.1f}s)\n" + "-" * 20 + "\n"
        body = "\n".join(lines)
        footer = "\n" + "-" * 20
        return header + body + footer


@EventCenter.subscribe(ToolCallEvent, priority=10)
async def _on_tool_call_for_ui(event: ToolCallEvent):
    if event.session_id and event.session_id in _SESSION_UI_STATES:
        _SESSION_UI_STATES[event.session_id].append(f"🔄 正在调用: `{event.tool_name}`")


@EventCenter.subscribe(ToolResultEvent, priority=10)
async def _on_tool_result_for_ui(event: ToolResultEvent):
    if event.session_id and event.session_id in _SESSION_UI_STATES:
        lines = _SESSION_UI_STATES[event.session_id]
        if event.error or (event.result and event.result.is_error):
            lines.append(f"❌ 调用失败: `{event.tool_name}`")
        else:
            lines.append(f"✅ 调用成功: `{event.tool_name}` ({event.duration_ms:.0f}ms)")


@EventCenter.subscribe(ToolStreamEvent, priority=10)
async def _on_tool_stream_for_ui(event: ToolStreamEvent):
    if event.session_id and event.session_id in _SESSION_UI_STATES:
        _SESSION_UI_STATES[event.session_id].append(f"  └ ⏳ {event.chunk.content}")

