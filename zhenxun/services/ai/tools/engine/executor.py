import ast
import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import json_repair

from zhenxun.services.ai.core.exceptions import (
    ToolRetryError,
)
from zhenxun.services.ai.core.messages import AnyLLMMessage, LLMMessage, ToolCallPart
from zhenxun.services.ai.core.stream_events import (
    EventStreamer,
    ToolCallResultEvent,
    ToolCallStart,
)
from zhenxun.services.ai.protocols.capabilities import CombinedCapability
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.models import (
    ToolResult,
    ValidatedToolCall,
)
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.tools.engine.registry import ToolCollection


class ToolExecutor:
    """
    全能工具执行器。
    负责接收工具调用请求，解析参数，触发回调，执行工具，并返回标准化的结果。
    """

    def __init__(self):
        pass

    def _get_combined_capability(
        self, executable: Any, context: RunContext
    ) -> CombinedCapability:
        """合并 Agent 上下文 (已包含 Global) 与 Tool 私有的 Capability"""
        tool_caps = getattr(getattr(executable, "settings", None), "capabilities", [])
        agent_caps = getattr(context, "capabilities", [])

        all_caps = list(agent_caps)
        for c in tool_caps:
            if c not in all_caps:
                all_caps.append(c)
        return CombinedCapability(all_caps)

    async def validate_tool_call(
        self,
        tool_call: ToolCallPart,
        available_tools: "ToolCollection | dict[str, Any] | None",
        context: RunContext | None = None,
        event_streamer: EventStreamer | None = None,
    ) -> ValidatedToolCall:
        """验证单一工具调用，完成参数解析、类型检查与交互式补全(如果触发)。"""
        if not available_tools:
            available_tools = {}
        if not tool_call.tool_name:
            return ValidatedToolCall(
                call=tool_call,
                args_valid=False,
                validation_error=ToolRetryError("tool_call.tool_name 不能为空"),
            )

        tool_name = tool_call.tool_name
        arguments_str = (
            tool_call.args
            if isinstance(tool_call.args, str)
            else json.dumps(tool_call.args, ensure_ascii=False)
        )
        arguments: dict[str, Any] = {}
        intent_str: str | None = None

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

            if parsed_successfully and isinstance(arguments, dict):
                _intent = arguments.pop("_intent", None)
                if _intent:
                    logger.info(
                        f"🧠 [Agent Intent] 调用工具 {tool_name} 的意图: {_intent}"
                    )
                    intent_str = str(_intent)

            if not parsed_successfully:
                if context:
                    context.run.tool_retries[tool_name] = (
                        context.run.tool_retries.get(tool_name, 0) + 1
                    )
                return ValidatedToolCall(
                    call=tool_call,
                    args_valid=False,
                    validation_error=ToolRetryError(
                        f"参数JSON解析失败，请检查 JSON 语法是否合法: {arguments_str}"
                    ),
                )

        executable = available_tools.get(tool_name)
        if not executable or not hasattr(executable, "execute"):
            return ValidatedToolCall(
                call=tool_call,
                args_valid=False,
                validation_error=ToolRetryError(f"Tool '{tool_name}' not found."),
            )

        from zhenxun.services.ai.run import RunContext

        if context:
            safe_context = context.clone_for_tool_call(tool_call.id, tool_name)
            safe_context.run.streamer = event_streamer
            safe_context.call.tool_name = tool_name
        else:
            safe_context = RunContext()
            safe_context.run.streamer = event_streamer
            safe_context.call.tool_name = tool_name

        safe_context.call.current_tool = executable

        combined_cap = self._get_combined_capability(executable, safe_context)

        async def inner_validate(args_inner):
            if isinstance(args_inner, dict) and hasattr(executable, "validate_args"):
                import inspect

                sig = inspect.signature(executable.validate_args)
                if "context" in sig.parameters:
                    return await executable.validate_args(
                        args_inner, context=safe_context
                    )
                return await executable.validate_args(args_inner)
            return args_inner

        try:
            validated_args = await combined_cap.wrap_tool_validate(
                safe_context, tool_name, arguments, inner_validate
            )
            return ValidatedToolCall(
                call=tool_call,
                tool=executable,
                args_valid=True,
                validated_args=validated_args,
                intent=intent_str,
            )
        except BaseException as e:
            return ValidatedToolCall(
                call=tool_call,
                tool=executable,
                args_valid=False,
                validation_error=e,
            )

    async def execute_tool_call(
        self,
        validated: ValidatedToolCall,
        available_tools: "ToolCollection | dict[str, Any] | None",
        context: RunContext | None = None,
        model_name: str | None = None,
        max_retries: int = 0,
        event_streamer: EventStreamer | None = None,
    ) -> tuple[ToolCallPart, ToolResult]:
        """核心执行阶段。只接收已经通过 Validation 阶段的 ValidatedToolCall 载体。"""
        if not available_tools:
            available_tools = {}
        tool_name = validated.call.tool_name
        if not validated.args_valid or validated.tool is None:
            from zhenxun.services.ai.core.exceptions import ControlFlowExit

            if isinstance(validated.validation_error, ControlFlowExit):
                raise validated.validation_error

            err_msg = getattr(
                validated.validation_error, "message", str(validated.validation_error)
            )
            res = ToolResult(output=f"执行被拦截或参数错误: {err_msg}").as_error()
            if event_streamer:
                await event_streamer.send(
                    ToolCallResultEvent(
                        tool_name=tool_name, result=res, is_error=res.is_error
                    )
                )
            return validated.call, res

        executable = validated.tool
        arguments = validated.validated_args or {}

        from zhenxun.services.ai.run import RunContext

        safe_context = (
            context.clone_for_tool_call(validated.call.id, tool_name)
            if context
            else RunContext()
        )
        safe_context.run.streamer = event_streamer
        safe_context.call.tool_name = tool_name
        safe_context.call.current_tool = executable
        safe_context.state["__available_tools"] = available_tools

        call_event = ToolCallStart(
            tool_name=tool_name, arguments=arguments, intent=validated.intent
        )
        if event_streamer:
            await event_streamer.send(call_event)

        from zhenxun.services.ai.run.context import set_run_context

        combined_cap = self._get_combined_capability(executable, safe_context)

        async def inner_handler(args_inner: dict) -> Any:
            return await executable.execute(context=safe_context, **args_inner)

        with set_run_context(safe_context):
            try:
                result = await combined_cap.wrap_tool_execute(
                    safe_context, tool_name, arguments, inner_handler
                )

                if not isinstance(result, ToolResult):
                    result = ToolResult(output=result)
            except BaseException as e:
                from zhenxun.services.ai.core.exceptions import ControlFlowExit

                if isinstance(e, ControlFlowExit):
                    raise e
                logger.error(f"洋葱模型异常穿透: {e}")
                result = ToolResult(output=f"System Fatal Error: {e}").as_error()

        if event_streamer:
            await event_streamer.send(
                ToolCallResultEvent(
                    tool_name=tool_name,
                    result=result,
                    is_error=result.is_error,
                )
            )

        return validated.call, result

    async def execute_batch(
        self,
        tool_calls: list[ToolCallPart],
        available_tools: "ToolCollection | dict[str, Any] | None",
        context: RunContext | None = None,
        model_name: str | None = None,
        max_retries: int = 0,
        event_streamer: EventStreamer | None = None,
    ) -> list[AnyLLMMessage]:
        """批量并发执行多个工具调用。"""
        if not available_tools:
            available_tools = {}
        if not tool_calls:
            return []

        val_tasks = [
            self.validate_tool_call(
                call,
                available_tools,
                context,
                event_streamer=event_streamer,
            )
            for call in tool_calls
        ]
        validated_calls = await asyncio.gather(*val_tasks)

        results: list[Any] = [None] * len(validated_calls)
        shared_tasks = []

        loop = asyncio.get_running_loop()
        last_exclusive_task = loop.create_future()
        last_exclusive_task.set_result(None)

        async def _run_tool(index: int, val_call: ValidatedToolCall):
            try:
                res = await self.execute_tool_call(
                    val_call,
                    available_tools,
                    context,
                    model_name=model_name,
                    max_retries=max_retries,
                    event_streamer=event_streamer,
                )
                results[index] = res
            except Exception as e:
                results[index] = e

        try:
            for i, val_call in enumerate(validated_calls):
                executable = getattr(val_call, "tool", None)
                concurrency_mode = getattr(
                    getattr(executable, "settings", None), "concurrency", "shared"
                )

                if concurrency_mode == "exclusive":
                    await last_exclusive_task
                    if shared_tasks:
                        await asyncio.gather(*shared_tasks, return_exceptions=True)
                        shared_tasks.clear()

                    task = asyncio.create_task(_run_tool(i, val_call))
                    last_exclusive_task = task
                else:

                    async def _run_shared(
                        idx=i, vc=val_call, barrier=last_exclusive_task
                    ):
                        await barrier
                        await _run_tool(idx, vc)

                    task = asyncio.create_task(_run_shared())
                    shared_tasks.append(task)

            await last_exclusive_task
            if shared_tasks:
                await asyncio.gather(*shared_tasks, return_exceptions=True)
        finally:
            if not last_exclusive_task.done():
                last_exclusive_task.cancel()
            for t in shared_tasks:
                if not t.done():
                    t.cancel()

        tool_messages: list[AnyLLMMessage] = []
        for index, result_pair in enumerate(results):
            original_call = tool_calls[index]
            func_name = original_call.tool_name

            if isinstance(result_pair, BaseException):
                from zhenxun.services.ai.core.exceptions import ControlFlowExit

                if isinstance(result_pair, ControlFlowExit):
                    raise result_pair

                logger.error(f"工具批量执行并发崩溃: {func_name}, 错误: {result_pair}")
                result_pair = (
                    original_call,
                    ToolResult(output=f"Crash: {result_pair}").as_error(),
                )

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
