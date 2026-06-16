from __future__ import annotations

from collections import defaultdict
import json
import time
from typing import Any, ClassVar

from nonebot.permission import SUPERUSER

from zhenxun.models.user_console import UserConsole
from zhenxun.services.ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from zhenxun.services.ai.core.exceptions import (
    AbortException,
    ControlFlowExit,
    GuardrailViolationError,
    LLMErrorCode,
    LLMException,
    ModelRetry,
    NeedsAuthException,
    SchemaParseError,
    ToolFatalError,
)
from zhenxun.services.ai.core.messages import LLMResponse
from zhenxun.services.ai.core.protocols.middleware import LLMContext
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.run.models import AgentRunSummary
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

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
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
                    "[StuckDetection] 拦截到死循环：连续 "
                    f"{max_repeated_errors} 次产生完全相同的状态哈希碰撞。"
                )
                raise ToolFatalError(
                    "Agent 触发终极防呆机制：连续 "
                    f"{max_repeated_errors} 次产生完全相同的"
                    "无效工具调用状态，已物理阻断以节省 Token。"
                )

        return await handler(llm_context)


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

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> dict[str, Any]:
        tool = context.call.current_tool
        admin_level = getattr(tool, "metadata", {}).get("admin_level", 0)
        if admin_level > 0:
            user_id = context.get_user_id()
            group_id = context.get_group_id()

            bot = context.get_bot()
            event = context.get_event()
            if bot and event and await SUPERUSER(bot, event):
                return await handler(arguments)

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
        return await handler(arguments)


class BillingCapability(AbstractCapability):
    """经济系统中间件：执行前扣除金币"""

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
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
        return await handler(arguments)


class TelemetryCapability(AbstractCapability):
    """
    核心可观测性与遥测 拦截器。
    利用洋葱模型接管完整的 Agent 生命周期，计算瀑布流耗时，并聚合生成全局运行摘要
    """

    def __init__(self):
        self.summary = AgentRunSummary()
        self.start_t = 0.0

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        return TelemetryCapability()

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        agent_name = context.run.agent_name or "unknown"

        logger.debug(f"🚀 [Telemetry] 智能体 {agent_name} 开始运行")
        self.start_t = time.monotonic()
        try:
            res = await handler()
        except ControlFlowExit as e:
            self.summary.total_latency_ms = (time.monotonic() - self.start_t) * 1000
            latency = self.summary.total_latency_ms
            logger.debug(
                f"🛑 [Telemetry] 智能体 {agent_name} 正常中止/控制流转移: "
                f"{type(e).__name__} (耗时: {latency:.2f}ms)"
            )
            raise e.with_traceback(None) from None
        except Exception as e:
            self.summary.total_latency_ms = (time.monotonic() - self.start_t) * 1000
            latency = self.summary.total_latency_ms
            raise e.with_traceback(None) from None

        self.summary.total_latency_ms = (time.monotonic() - self.start_t) * 1000
        self.summary.usage = res.usage
        res.telemetry = self.summary

        latency = self.summary.total_latency_ms
        logger.debug(
            f"🏁 [Telemetry] 智能体 {agent_name} 运行结束 (总耗时: {latency:.2f}ms)"
        )
        return res

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        model_name = context.run.current_model or "model_instance"
        start_t = time.monotonic()
        try:
            response = await handler(llm_context)
            dur = (time.monotonic() - start_t) * 1000

            self.summary.chats.total += 1
            self.summary.chats.total_latency_ms += dur

            stop_reason = "tool_calls" if response.tool_calls else "stop"
            self.summary.chats.by_stop_reason[stop_reason] = (
                self.summary.chats.by_stop_reason.get(stop_reason, 0) + 1
            )

            for call in response.tool_calls:
                if llm_context.tools:
                    tool_inst = next(
                        (
                            t
                            for t in llm_context.tools
                            if getattr(t, "name", "") == call.tool_name
                        ),
                        None,
                    )
                    if (
                        tool_inst
                        and getattr(tool_inst, "execution_side", "client") == "server"
                    ):
                        self.summary.tools.total += 1
                        self.summary.tools.ok += 1
                        tool_stat = self.summary.tools.by_name.setdefault(
                            call.tool_name,
                            {"total": 0, "ok": 0, "error": 0, "latency_ms": 0.0},
                        )
                        tool_stat["total"] += 1
                        tool_stat["ok"] += 1

            logger.debug(
                f"🧠 [Telemetry] 模型 {model_name} 调用完成 (耗时: {dur:.2f}ms)"
            )
            return response
        except Exception as e:
            dur = (time.monotonic() - start_t) * 1000
            self.summary.chats.total += 1
            self.summary.chats.total_latency_ms += dur
            self.summary.chats.by_stop_reason["error"] = (
                self.summary.chats.by_stop_reason.get("error", 0) + 1
            )
            raise e.with_traceback(None) from None

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        start_t = time.monotonic()
        try:
            result = await handler(arguments)
            dur = (time.monotonic() - start_t) * 1000

            self.summary.tools.total += 1
            self.summary.tools.total_latency_ms += dur

            tool_stat = self.summary.tools.by_name.setdefault(
                tool_name, {"total": 0, "ok": 0, "error": 0, "latency_ms": 0.0}
            )
            tool_stat["total"] += 1
            tool_stat["latency_ms"] += dur

            if getattr(result, "is_error", False):
                self.summary.tools.error += 1
                tool_stat["error"] += 1
            else:
                self.summary.tools.ok += 1
                tool_stat["ok"] += 1

            logger.debug(f"🛠️ [Telemetry] 工具 {tool_name} 执行完毕 (耗时: {dur:.2f}ms)")
            return result
        except ControlFlowExit as e:
            dur = (time.monotonic() - start_t) * 1000
            self.summary.tools.total += 1
            self.summary.tools.total_latency_ms += dur

            tool_stat = self.summary.tools.by_name.setdefault(
                tool_name, {"total": 0, "ok": 0, "error": 0, "latency_ms": 0.0}
            )
            tool_stat["total"] += 1
            tool_stat["latency_ms"] += dur
            raise e.with_traceback(None) from None
        except Exception as e:
            dur = (time.monotonic() - start_t) * 1000
            self.summary.tools.total += 1
            self.summary.tools.total_latency_ms += dur
            self.summary.tools.error += 1

            tool_stat = self.summary.tools.by_name.setdefault(
                tool_name, {"total": 0, "ok": 0, "error": 0, "latency_ms": 0.0}
            )
            tool_stat["total"] += 1
            tool_stat["latency_ms"] += dur
            tool_stat["error"] += 1
            raise e.with_traceback(None) from None


class ToolSideEffectCapability(AbstractCapability):
    """
    副作用处理中间件。
    代理执行遗留的 ToolResult 副作用 (UI展现、状态流转、Prompt追加)，
    将 AgentExecutor 从杂项中解放出来。
    """

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        result = await handler(arguments)
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
    接管原执行器中的重试计数与致命异常熔断。
    将 Python 异常优雅地转化为大模型的反思 Prompt。
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
                ControlFlowExit,
                ToolFatalError,
                ToolFinishException,
            )
            from zhenxun.services.ai.tools.engine.policy import ToolExecutionPolicy
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(e, ControlFlowExit):
                raise e

            retries = context.run.tool_retries.get(tool_name, 0)
            retries += 1
            context.run.tool_retries[tool_name] = retries

            from typing import cast

            from zhenxun.services.ai.tools.core.tool import BaseTool

            tool = cast(BaseTool, context.call.current_tool)
            policy = ToolExecutionPolicy(tool)
            max_retries_limit = policy.max_retries

            if isinstance(e, ToolFatalError | ToolFinishException):
                display_msg = getattr(e, "display_content", f"❌ 系统致命错误: {e}")
                raise AbortException(reason=str(e), display=display_msg)

            if retries > max_retries_limit:
                raise AbortException(
                    reason=f"工具 '{tool_name}' 连续出错达 {retries} 次，超出上限。",
                    display=f"🚨 工具 '{tool_name}' 已达最大重试次数，执行阻断。",
                )

            return ToolResult(output=f"执行发生异常: {e}").as_error()


class ReflexionCapability(AbstractCapability):
    """自愈反思与验证引擎 (Reflexion Engine)。
    统一处理结构化解析失败 and 语义护栏拦截。"""

    async def wrap_tool_execute(self, context, tool_name, arguments, handler):
        try:
            return await handler(arguments)
        except Exception as error:
            from zhenxun.services.ai.core.engine.structured_parser import (
                DEFAULT_IVR_TEMPLATE,
            )
            from zhenxun.services.ai.core.exceptions import ModelRetry, ToolRetryError
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(error, ToolRetryError | ModelRetry):
                error_msg = getattr(error, "message", str(error))
                feedback_prompt = DEFAULT_IVR_TEMPLATE.format(error_msg=error_msg)
                context.run.add_system_prompt(feedback_prompt)
                return ToolResult(
                    output=f"执行失败：{error_msg}",
                ).as_error()
            raise error.with_traceback(None) from None

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        output_processor = llm_context.extra.get("output_processor")
        guardrails = llm_context.extra.get("guardrails", [])

        if not output_processor and not guardrails:
            return await handler(llm_context)

        max_retries = llm_context.extra.get("max_retries", 3)
        error_template = (
            output_processor.error_template if output_processor else "{error_msg}"
        )

        ivr_messages = list(llm_context.messages)
        last_exception: Exception | None = None

        from zhenxun.services.ai.guardrails import GuardrailPipeline

        pipeline = GuardrailPipeline(guardrails) if guardrails else None

        for attempt in range(max_retries + 1):
            llm_context.messages = list(ivr_messages)
            current_response_text: str = ""

            try:
                if pipeline:
                    llm_context.messages = await pipeline.run_input_pipeline(
                        llm_context.messages, context
                    )

                response = await handler(llm_context)
                current_response_text = response.text

                if response.tool_calls:
                    return response

                if output_processor:
                    final_obj = await output_processor.validate_and_parse(
                        current_response_text, context=context
                    )
                else:
                    final_obj = current_response_text

                if pipeline:
                    resp_out, final_obj_out = await pipeline.run_output_pipeline(
                        response, final_obj, context
                    )
                    from typing import cast

                    response = cast("LLMResponse", resp_out)
                    final_obj = final_obj_out
                    current_response_text = response.text

                response.parsed_obj = final_obj
                return response

            except Exception as e:
                from typing import cast

                from zhenxun.services.ai.core.messages import LLMMessage

                is_model_retry = isinstance(e, ModelRetry)
                is_llm_error = isinstance(e, LLMException)
                llm_error: LLMException | None = (
                    cast(LLMException, e) if is_llm_error else None
                )
                last_exception = e

                if (
                    not is_model_retry
                    and llm_error
                    and llm_error.code
                    not in (
                        LLMErrorCode.RESPONSE_PARSE_ERROR,
                        LLMErrorCode.API_RESPONSE_INVALID,
                    )
                ):
                    raise e

                if attempt < max_retries:
                    if is_model_retry:
                        error_msg = getattr(e, "message", str(e))
                        raw_response = current_response_text
                    else:
                        error_msg = (
                            llm_error.details.get("validation_error", str(e))
                            if llm_error
                            else str(e)
                        )
                        raw_response = current_response_text or (
                            llm_error.details.get("raw_response", "")
                            if llm_error
                            else ""
                        )

                    logger.warning(
                        "输出校验未通过 "
                        f"(尝试 {attempt + 1}/{max_retries + 1})。"
                        f"启动反思修复闭环... 失败原因: {error_msg}"
                    )

                    if raw_response:
                        ivr_messages.append(
                            cast(
                                LLMMessage,
                                LLMMessage.assistant_text_response(raw_response),
                            )
                        )

                    if isinstance(e, SchemaParseError):
                        feedback_prompt = (
                            "### ❌ [格式解析失败]\n"
                            "你输出的结构化数据（JSON）格式损坏或字段不匹配，"
                            "未能通过 Schema 校验。\n\n"
                            "**解析错误报告：**\n"
                            f"> {error_msg}\n\n"
                            "**修正要求：** 请仔细检查缺失的必填字段、错误的数据类型或"
                            "未闭合的括号，严格参考你可用的工具 Schema 定义，"
                            "重新输出正确格式的数据。"
                        )
                    elif isinstance(e, GuardrailViolationError):
                        feedback_prompt = (
                            "### 🛡️ [业务护栏违规]\n"
                            "你输出的数据格式完全正确，但在业务逻辑层触发了合规/风控护栏。\n\n"
                            "**拦截原因报告：**\n"
                            f"> {error_msg}\n\n"
                            "**修正要求：** 请结合上述反馈报告，"
                            "反思你的决策逻辑或内容生成，"
                            "在保持数据格式正确的前提下，重新生成符合护栏规范的内容。"
                        )
                    else:
                        if output_processor and error_template:
                            feedback_prompt = error_template.format(error_msg=error_msg)
                        else:
                            from zhenxun.services.ai.core.engine import (
                                structured_parser as sp,
                            )

                            feedback_prompt = sp.DEFAULT_IVR_TEMPLATE.format(
                                error_msg=error_msg
                            )
                    ivr_messages.append(
                        cast(LLMMessage, LLMMessage.user(feedback_prompt))
                    )
                    continue

                if llm_error and not getattr(llm_error, "recoverable", True):
                    raise llm_error.with_traceback(None) from None

        if last_exception:
            raise last_exception.with_traceback(None) from None
        raise LLMException(
            "反思循环耗尽，未能生成符合所有校验规则的合法结果。",
            code=LLMErrorCode.GENERATION_FAILED,
        ).with_traceback(None) from None


GLOBAL_CAPABILITIES: dict[str, list[AbstractCapability]] = defaultdict(list)

for _cap in [
    StuckDetectionCapability(),
    PermissionCapability(),
    BillingCapability(),
    RequireAuthCapability(),
    TelemetryCapability(),
    ToolSideEffectCapability(),
    ToolRetryAndReflectionCapability(),
    ReflexionCapability(),
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
