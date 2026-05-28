from collections import defaultdict
import json
import time
from typing import Any, ClassVar

from nonebot.permission import SUPERUSER

from zhenxun.models.user_console import UserConsole
from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    AgentEndEvent,
    AgentStartEvent,
    ModelEndEvent,
    ModelStartEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from zhenxun.services.ai.core.exceptions import (
    AbortException,
    NeedsAuthException,
    ToolFatalError,
)
from zhenxun.services.ai.core.messages import LLMResponse
from zhenxun.services.ai.protocols.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from zhenxun.services.ai.protocols.middleware import LLMContext
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.cache.runtime_cache import LevelUserMemoryCache
from zhenxun.services.log import logger
from zhenxun.utils.enum import GoldHandle
from zhenxun.utils.exception import InsufficientGold
from zhenxun.utils.utils import infer_plugin_namespace


class DummyCredentialManager:
    """一个模拟的全局凭证管理器"""

    _tokens: ClassVar[dict[str, dict[str, str]]] = {}

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


class StuckDetectionCapability(AbstractCapability):
    """死循环检测：替代原有的 Event Listener，使用前置请求拦截防止 LLM 陷入无限重试"""

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        import hashlib

        from zhenxun.services.ai.core.exceptions import ToolFatalError
        from zhenxun.services.ai.core.messages import ToolCallPart

        max_repeated_errors = 3
        action_hashes = []
        messages = list(llm_context.messages)
        idx = len(messages) - 1

        while idx >= 0:
            msg = messages[idx]
            if msg.role == "tool":
                batch_tool_contents = []
                while idx >= 0 and messages[idx].role == "tool":
                    for tr in messages[idx].tool_returns:
                        batch_tool_contents.append(f"{tr.tool_name}:{tr.output}")
                    idx -= 1

                if (
                    idx >= 0
                    and messages[idx].role == "assistant"
                    and messages[idx].tool_calls
                ):
                    assistant_msg = messages[idx]
                    batch_tool_calls = []
                    for tc in assistant_msg.tool_calls:
                        if isinstance(tc, ToolCallPart):
                            args_str = (
                                tc.args
                                if isinstance(tc.args, str)
                                else json.dumps(tc.args, ensure_ascii=False)
                            )
                            batch_tool_calls.append(f"{tc.tool_name}:{args_str}")

                    batch_tool_calls.sort()
                    batch_tool_contents.sort()

                    state_str = (
                        "|".join(batch_tool_calls)
                        + "||"
                        + "|".join(batch_tool_contents)
                    )
                    state_hash = hashlib.md5(state_str.encode("utf-8")).hexdigest()
                    action_hashes.append(state_hash)
                    idx -= 1
                else:
                    break
            elif msg.role == "assistant":
                idx -= 1
            else:
                break

        if len(action_hashes) >= max_repeated_errors:
            recent_hashes = action_hashes[:max_repeated_errors]
            if len(set(recent_hashes)) == 1:
                logger.warning(
                    f"[StuckDetection] 拦截到死循环：连续 {max_repeated_errors} 次产生完全相同的状态哈希碰撞。"
                )
                raise ToolFatalError(
                    f"Agent 触发终极防呆机制：连续 {max_repeated_errors} 次产生完全相同的无效工具调用状态，已物理阻断以节省 Token。"
                )

        return llm_context


class RequireAuthCapability(AbstractCapability):
    """声明式鉴权与动态授权(HITL)中间件"""

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        tool = context.call.current_tool
        user_id = context.get_user_id()

        settings = getattr(tool, "settings", None)
        auth_provider = (
            settings.metadata.get("auth_provider") if settings else None
        ) or getattr(tool, "metadata", {}).get("auth_provider")
        if auth_provider and user_id:
            if token := DummyCredentialManager.get_token(user_id, auth_provider):
                context.session.auth_tokens[auth_provider] = token

        current_kwargs = dict(arguments)
        while True:
            try:
                return await handler(current_kwargs)
            except NeedsAuthException as e:
                provider = e.provider
                bot = context.get_bot()
                event = context.get_event()

                if not bot or not event or not user_id:
                    logger.warning(
                        f"由于缺少 {provider} 凭证，工具执行被拦截（非交互环境）。"
                    )
                    return (
                        ToolResult(
                            output=json.dumps(
                                {
                                    "error_type": "AuthFailed",
                                    "message": f"缺失 {provider} 的授权凭证。",
                                },
                                ensure_ascii=False,
                            ),
                        )
                        .show_to_user(f"❌ 执行失败: 缺少 {provider} 授权")
                        .as_error(is_retryable=False)
                    )

                prompt_msg = (
                    f"⚠️ 该功能需要绑定您的 [{provider}] 账号以初始化连接。\n\n"
                    f"请点击链接完成 OAuth 授权，或直接在此回复您的 Token (回复'取消'中止操作)。"  # noqa: E501
                )

                try:
                    from zhenxun.services.ai.run.hitl import HITLController

                    hitl = HITLController(context)
                    user_input = await hitl.ask_text(prompt_msg, timeout=60.0)
                except (AbortException, ToolFatalError):
                    return ToolResult(
                        output=json.dumps(
                            {
                                "error_type": "UserCancellation",
                                "message": "用户拒绝了授权绑定。",
                            },
                            ensure_ascii=False,
                        ),
                    ).as_error(is_retryable=False)

                DummyCredentialManager.set_token(user_id, provider, user_input)
                context.session.auth_tokens[provider] = user_input
                await bot.send(
                    event, f"✅ [{provider}] 凭证已加载，正在恢复任务执行..."
                )


class PermissionCapability(AbstractCapability):
    """权限校验中间件：在执行前根据确定参数进行动态鉴权"""

    async def before_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        tool = context.call.current_tool
        admin_level = getattr(tool, "metadata", {}).get("admin_level", 0)
        if admin_level > 0:
            user_id = context.get_user_id()
            group_id = context.get_group_id()

            bot = context.get_bot()
            event = context.get_event()
            if bot and event and await SUPERUSER(bot, event):
                return arguments

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
                        f"🛡️ [Capability] 权限拦截: 用户 {user_id} 尝试调用 "
                        f"{getattr(tool, 'name', 'unknown')}"
                    )
                    from zhenxun.services.ai.core.exceptions import ToolFatalError

                    raise ToolFatalError(
                        msg, display_content=f"❌ 权限不足: 需要等级 {admin_level}"
                    )
        return arguments


class BillingCapability(AbstractCapability):
    """经济系统中间件：执行前扣除金币"""

    async def before_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        tool = context.call.current_tool
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
                        f"💰 [Capability] 金币拦截: 用户 {user_id} 尝试调用 "
                        f"{getattr(tool, 'name', 'unknown')}"
                    )
                    from zhenxun.services.ai.core.exceptions import ToolFatalError

                    raise ToolFatalError(
                        msg, display_content=f"❌ 余额不足: 需要 {cost_gold} 金币"
                    )
        return arguments


class EventDispatcherCapability(AbstractCapability):
    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        agent_name = context.run.agent_name or "unknown"
        prompt = context.run.user_input or ""
        ns = getattr(context.session, "namespace", "global")

        await EventCenter.publish(
            AgentStartEvent(
                session_id=context.session_id,
                agent_name=agent_name,
                prompt=prompt,
                namespace=ns,
            )
        )
        start_t = time.monotonic()
        try:
            res = await handler()
            dur = (time.monotonic() - start_t) * 1000
            await EventCenter.publish(
                AgentEndEvent(
                    session_id=context.session_id,
                    agent_name=agent_name,
                    result=res,
                    duration_ms=dur,
                    namespace=ns,
                )
            )
            return res
        except Exception as e:
            raise e

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        ns = getattr(context.session, "namespace", "global")
        await EventCenter.publish(
            ModelStartEvent(
                session_id=context.session_id,
                model_name=context.run.current_model or "model_instance",
                messages=list(llm_context.messages),
                namespace=ns,
            )
        )
        start_t = time.monotonic()
        try:
            response = await handler(llm_context)
            dur = (time.monotonic() - start_t) * 1000
            await EventCenter.publish(
                ModelEndEvent(
                    session_id=context.session_id,
                    response=response,
                    duration_ms=dur,
                    namespace=ns,
                )
            )
            return response
        except Exception as e:
            raise e

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        ns = getattr(context.session, "namespace", "global")
        await EventCenter.publish(
            ToolCallEvent(
                session_id=context.session_id,
                tool_call_id="dynamic",
                tool_name=tool_name,
                arguments=arguments.copy(),
                namespace=ns,
            )
        )
        start_t = time.monotonic()
        try:
            result = await handler(arguments)
            dur = (time.monotonic() - start_t) * 1000
            await EventCenter.publish(
                ToolResultEvent(
                    session_id=context.session_id,
                    tool_call_id="dynamic",
                    tool_name=tool_name,
                    result=result,
                    error=None,
                    duration_ms=dur,
                    namespace=ns,
                )
            )
            return result
        except Exception as e:
            from zhenxun.services.ai.core.exceptions import ControlFlowException

            if isinstance(e, ControlFlowException):
                raise e

            dur = (time.monotonic() - start_t) * 1000
            await EventCenter.publish(
                ToolErrorEvent(
                    session_id=context.session_id,
                    tool_call_id="dynamic",
                    tool_name=tool_name,
                    error=e,
                    namespace=ns,
                )
            )
            await EventCenter.publish(
                ToolResultEvent(
                    session_id=context.session_id,
                    tool_call_id="dynamic",
                    tool_name=tool_name,
                    result=None,
                    error=e,
                    duration_ms=dur,
                    namespace=ns,
                )
            )
            raise e


class ToolSideEffectCapability(AbstractCapability):
    """
    副作用处理中间件。
    代理执行遗留的 ToolResult 副作用 (UI展现、状态流转、Prompt追加)，
    将 AgentExecutor 从杂项中解放出来。
    """

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        from zhenxun.services.ai.tools.models import StateSyncResult, ToolResult

        if isinstance(result, ToolResult):
            if result.ui_display is not None:
                from zhenxun.services.ai.run.ui_controller import UIController

                ui = UIController(context)
                await ui.send_display(result.ui_display)

            if isinstance(result, StateSyncResult) and result.state_notice:
                context.run.add_system_prompt(
                    f"[系统通知(状态同步)]：{result.state_notice}"
                )

        return result


class ToolRetryAndReflectionCapability(AbstractCapability):
    """
    重试与自愈反思中间件。
    接管原执行器中的重试计数与致命异常熔断。将 Python 异常优雅地转化为大模型的反思 Prompt。
    """

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            return await handler(arguments)
        except Exception as e:
            from zhenxun.services.ai.core.exceptions import (
                AbortException,
                ControlFlowException,
                ToolFatalError,
                ToolFinishException,
            )
            from zhenxun.services.ai.tools.engine.policy import ToolExecutionPolicy
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(e, ControlFlowException):
                raise e

            retries = context.run.tool_retries.get(tool_name, 0)
            retries += 1
            context.run.tool_retries[tool_name] = retries

            from typing import cast

            from zhenxun.services.ai.tools.core.tool import BaseTool

            tool = cast(BaseTool, context.call.current_tool)
            policy = ToolExecutionPolicy(tool)
            max_retries_limit = policy.max_retries

            if isinstance(e, (ToolFatalError, ToolFinishException)):
                display_msg = getattr(e, "display_content", f"❌ 系统致命错误: {e}")
                raise AbortException(reason=str(e), display=display_msg)

            if retries > max_retries_limit:
                raise AbortException(
                    reason=f"工具 '{tool_name}' 连续出错达 {retries} 次，超出上限。",
                    display=f"🚨 工具 '{tool_name}' 已达最大重试次数，执行阻断。",
                )

            return ToolResult(output=f"执行发生异常: {e}").as_error()


GLOBAL_CAPABILITIES: dict[str, list[AbstractCapability]] = defaultdict(list)

for _cap in [
    StuckDetectionCapability(),
    PermissionCapability(),
    BillingCapability(),
    RequireAuthCapability(),
    EventDispatcherCapability(),
    ToolSideEffectCapability(),
    ToolRetryAndReflectionCapability(),
]:
    GLOBAL_CAPABILITIES["global"].append(_cap)


def register_global_capability(
    capability: AbstractCapability, scope: str | None = None
) -> None:
    ns = scope if scope is not None else infer_plugin_namespace()
    GLOBAL_CAPABILITIES[ns].append(capability)
    logger.debug(
        f"已注册全局 Capability: {capability.__class__.__name__} -> Namespace: {ns}"
    )
