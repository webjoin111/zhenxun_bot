import ast
import asyncio
import json
import time
from typing import Any, cast

import json_repair

try:
    import ujson as fast_json
except ImportError:
    fast_json = json


from zhenxun.services.ai.events.center import EventCenter
from zhenxun.services.ai.events.event_types import (
    ToolCallEvent,
    ToolResultEvent,
)
from zhenxun.services.ai.types.messages import ToolCallPart
from zhenxun.services.ai.types.exceptions import (
    LLMException,
    ToolFatalError,
    ToolFinishException,
    ToolRetryError,
)
from zhenxun.services.ai.types.messages import AnyLLMMessage, LLMMessage
from zhenxun.services.ai.types.tools import (
    ToolResult,
)
from zhenxun.services.log import logger
from zhenxun.utils.decorator.retry import Retry


class ToolExecutor:
    """
    全能工具执行器。
    负责接收工具调用请求，解析参数，触发回调，执行工具，并返回标准化的结果。
    """

    def __init__(self):
        pass

    async def execute_tool_call(
        self,
        tool_call: ToolCallPart,
        available_tools: Any,
        context: Any | None = None,
        model_name: str | None = None,
        history_messages: list[Any] | None = None,
        retry_count: int = 0,
        max_retries: int = 0,
    ) -> tuple[ToolCallPart, ToolResult]:
        if not tool_call.tool_name:
            err = "tool_call.tool_name 不能为空"
            return tool_call, ToolResult(output=err, is_error=True, terminate_run=True)

        tool_name = tool_call.tool_name
        arguments_str = tool_call.args if isinstance(tool_call.args, str) else json.dumps(tool_call.args, ensure_ascii=False)
        arguments: dict[str, Any] = {}

        if arguments_str:
            parsed_successfully = False

            try:
                arguments = json.loads(arguments_str)
                if isinstance(arguments, dict):
                    parsed_successfully = True
            except json.JSONDecodeError:
                pass

            if not parsed_successfully:
                try:
                    arguments = ast.literal_eval(arguments_str)
                    if isinstance(arguments, dict):
                        parsed_successfully = True
                except (ValueError, SyntaxError):
                    pass

            if not parsed_successfully:
                try:
                    repaired_str = str(
                        json_repair.repair_json(arguments_str, skip_json_loads=True)
                    )
                    arguments = json.loads(repaired_str)
                    if isinstance(arguments, dict):
                        parsed_successfully = True
                        logger.debug(
                            f"⚒️ [Cascade Parse] 成功修复损坏的工具参数: "
                            f"{arguments_str} -> {repaired_str}",
                            "ToolExecutor",
                        )
                except Exception:
                    pass

            if not parsed_successfully:
                return tool_call, ToolResult(
                    output=f"参数JSON解析失败: {arguments_str}",
                    is_error=True,
                    system_prompt_append="💡 [引导] 你刚才提供的工具 JSON 参数格式严重错误，请严格输出标准 JSON 格式后再试！",
                )

        executable = available_tools.get(tool_name)
        has_req_methods = hasattr(executable, "get_definition") and hasattr(
            executable, "execute"
        )
        if not executable or not has_req_methods:
            return tool_call, ToolResult(
                output=f"Tool '{tool_name}' not found.",
                is_error=True,
                terminate_run=True,
            )

        tool_def = await executable.get_definition()
        metadata = tool_def.metadata if tool_def else {}

        from zhenxun.services.ai.tools.core.context import (
            ModelExecutionInfo,
            RunContext,
        )

        if context:
            safe_context = context.clone_for_execution(
                model_name=model_name,
                history_messages=history_messages or [],
                retry_count=retry_count,
                max_retries=max_retries,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
            )
        else:
            safe_context = RunContext(
                _model_execution_info=ModelExecutionInfo(
                    model_name=model_name,
                    history_messages=history_messages or [],
                    retry_count=retry_count,
                    max_retries=max_retries,
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                )
            )

        context = safe_context

        call_event = ToolCallEvent(
            session_id=context.session_id if context else None,
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            arguments=arguments,
            context=context,
            metadata=metadata,
        )
        from zhenxun.services.ai.config import get_llm_config

        if not get_llm_config().debug_log:
            try:
                definition = await executable.get_definition()
                schema_payload = getattr(definition, "parameters", {})
                schema_json = fast_json.dumps(
                    schema_payload,
                    ensure_ascii=False,
                )
                logger.debug(
                    f"🔍 [JIT Schema] {tool_name}: {schema_json}",
                    "ToolExecutor",
                )
            except Exception as e:
                logger.trace(f"JIT Schema logging failed: {e}")

        start_t = time.monotonic()
        result: ToolResult | None = None
        error: Exception | None = None

        try:
            await EventCenter.publish(call_event)
            arguments = call_event.arguments
            tool_call.args = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)

            from zhenxun.services.ai.tools.core.context import set_current_context

            @Retry.simple(stop_max_attempt=2, wait_fixed_seconds=1)
            async def execute_with_retry():
                with set_current_context(safe_context):
                    return await executable.execute(context=safe_context, **arguments)

            result = await execute_with_retry()

            if result and not getattr(result, "is_error", False):
                retry_key = f"__tool_retries_{tool_name}"
                if context and retry_key in context.extra:
                    context.extra[retry_key] = 0
        except Exception as e:
            error = e
            retry_key = f"__tool_retries_{tool_name}"
            retries = context.extra.get(retry_key, 0) if context else 0
            retries += 1
            if context:
                context.extra[retry_key] = retries

            settings = getattr(executable, "settings", None)
            max_retries_limit = (
                getattr(settings, "max_retries", None) or max_retries or 1
            )

            if isinstance(e, ToolRetryError) and retries <= max_retries_limit:
                result = ToolResult(
                    output=str(e),
                    is_error=True,
                    system_prompt_append=f"💡 [系统引导] 刚才的调用触发了可恢复的业务异常：{e}。请自我反思错误，修正参数后重试！",
                )
            elif (
                isinstance(e, ToolFatalError)
                or isinstance(e, ToolFinishException)
                or retries > max_retries_limit
            ):
                display_msg = getattr(e, "display_content", f"❌ 系统致命错误: {e}")
                result = ToolResult(
                    output=f"🚨 [强制熔断] 无法修复的错误或重试耗尽：{e}。禁止再次调用此工具！",
                    is_error=True,
                    terminate_run=True,
                    display=display_msg,
                )
            else:
                result = ToolResult(
                    output=f"System Error ({type(e).__name__}): {e}",
                    is_error=True,
                    terminate_run=True,
                )

        if result and getattr(result, "is_error", False):
            fallback_tool_name = metadata.get("fallback_tool")
            if fallback_tool_name:
                fallback_executable = available_tools.get(fallback_tool_name)
                if fallback_executable:
                    logger.info(
                        f"🔄 [语义容错路由] 主工具 '{tool_name}' 故障，正在底层透明重定向至备用工具 '{fallback_tool_name}'..."
                    )
                    try:
                        fallback_result = await fallback_executable.execute(
                            context=context, **arguments
                        )
                        notice = f"[系统底座：由于主工具 '{tool_name}' 发生故障，系统已自动路由至备用工具 '{fallback_tool_name}' 并执行成功]\n"

                        if isinstance(fallback_result.output, str):
                            fallback_result.output = notice + fallback_result.output

                        result = fallback_result
                        error = None
                    except Exception as fallback_e:
                        logger.error(
                            f"容错路由工具 '{fallback_tool_name}' 也执行失败: {fallback_e}"
                        )
                        result.system_prompt_append = f"🚨 [系统警告] 原工具和备用工具({fallback_tool_name})均执行失败，请停止尝试并告知用户。"

        duration = time.monotonic() - start_t

        await EventCenter.publish(
            ToolResultEvent(
                session_id=context.session_id if context else None,
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                result=result,
                error=error,
                duration_ms=duration * 1000,
            )
        )

        if result is None:
            raise LLMException("工具执行未返回任何结果。")

        return tool_call, result

    async def execute_batch(
        self,
        tool_calls: list[ToolCallPart],
        available_tools: Any,
        context: Any | None = None,
        model_name: str | None = None,
        history_messages: list[Any] | None = None,
        retry_count: int = 0,
        max_retries: int = 0,
    ) -> list[AnyLLMMessage]:
        if not tool_calls:
            return []

        tasks = [
            self.execute_tool_call(
                call,
                available_tools,
                context,
                model_name=model_name,
                history_messages=history_messages,
                retry_count=retry_count,
                max_retries=max_retries,
            )
            for call in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_messages: list[AnyLLMMessage] = []
        for index, result_pair in enumerate(results):
            original_call = tool_calls[index]
            func_name = original_call.tool_name

            if isinstance(result_pair, Exception):
                logger.error(
                    f"工具执行发生未捕获异常: {func_name}, 错误: {result_pair}"
                )
                tool_messages.append(
                    LLMMessage.tool_response(
                        tool_call_id=original_call.id,
                        function_name=func_name,
                        result={
                            "error": f"System Execution Error: {result_pair}",
                            "status": "failed",
                        },
                    )
                )
                continue

            tool_call_result = cast(tuple[ToolCallPart, ToolResult], result_pair)
            _, tool_result = tool_call_result
            tool_messages.append(
                LLMMessage.tool_response(
                    tool_call_id=original_call.id,
                    function_name=func_name,
                    result=tool_result.output,
                )
            )
        return tool_messages

