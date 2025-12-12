"""
LLM 执行器模块

包含 AgentExecutor 类，在 LLMToolExecutor 基础上增加了 Agent 特定的功能，
如 MCP 工具、HIL 等。
"""

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from zhenxun.services.llm.config import LLMGenerationConfig
from zhenxun.services.llm.config.generation import OutputConfig
from zhenxun.services.llm.service import LLMModel
from zhenxun.services.llm.tools import (
    RunContext,
    ToolErrorResult,
    ToolInvoker,
    tool_provider_manager,
)
from zhenxun.services.llm.types import (
    LLMContentPart,
    LLMErrorCode,
    LLMException,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ModelName,
    ResponseFormat,
    ToolExecutable,
    ToolResult,
)
from zhenxun.services.llm.types.models import ToolChoice
from zhenxun.services.llm.utils import (
    extract_text_from_content,
    parse_and_validate_json,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import (
    model_copy,
    model_dump,
    model_json_schema,
    model_validate,
)

from .callbacks import BaseCallbackHandler, LoggingCallbackHandler
from .context import LLMInterface
from .types import ExecutionConfig, ToolFilter

if TYPE_CHECKING:
    from zhenxun.services.llm.session import AI

MAX_RECURSION_DEPTH = 2
T = TypeVar("T", bound=BaseModel)


class AgentExecutor(LLMInterface):
    """
    一个面向 Agent 的高级执行器 (Runner)。
    它不再继承 LLMToolExecutor，而是采用组合模式。
    负责维护思考-行动循环以及处理 Hook 回调。
    """

    def __init__(
        self,
        model: LLMModel,
        session: "AI",
        recursion_depth: int = 0,
        callbacks: list[BaseCallbackHandler] | None = None,
        tool_filter: ToolFilter | None = None,
        tools_map: dict[str, ToolExecutable] | None = None,
    ):
        self.model = model
        self.session = session
        self.recursion_depth = recursion_depth
        self.callbacks: list[BaseCallbackHandler] = (
            callbacks if callbacks is not None else [LoggingCallbackHandler()]
        )
        self.tool_invoker = ToolInvoker(callbacks=self.callbacks)
        self.tool_filter = tool_filter
        self.tools_map = tools_map

    async def _trigger_callbacks(self, event_name: str, *args, **kwargs: Any) -> None:
        """安全地触发所有回调处理器上的指定事件。"""
        if not self.callbacks:
            return

        tasks = [
            getattr(handler, event_name)(*args, **kwargs)
            for handler in self.callbacks
            if hasattr(handler, event_name)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    def _filter_discovered_tools(
        self,
        tools: dict[str, ToolExecutable],
        tool_filter: ToolFilter | None,
    ) -> dict[str, ToolExecutable]:
        """在工具发现后，根据 ToolFilter 对工具列表进行内存过滤。"""
        if not tool_filter:
            return tools

        filtered = tools.copy()
        if tool_filter.allowed:
            filtered = {k: v for k, v in filtered.items() if k in tool_filter.allowed}
        if tool_filter.excluded:
            filtered = {
                k: v for k, v in filtered.items() if k not in tool_filter.excluded
            }

        return filtered

    async def _get_effective_tools(self) -> dict[str, ToolExecutable]:
        """根据过滤规则发现并返回最终生效的工具集。"""
        if self.tools_map is not None:
            return self._filter_discovered_tools(self.tools_map, self.tool_filter)

        allowed_servers = self.tool_filter.allowed_servers if self.tool_filter else None
        excluded_servers = (
            self.tool_filter.excluded_servers if self.tool_filter else None
        )

        if self.tool_filter and self.tool_filter.allowed is not None:
            allowed_tools = await tool_provider_manager.resolve_specific_tools(
                self.tool_filter.allowed
            )
            effective_tools = self._filter_discovered_tools(
                allowed_tools, self.tool_filter
            )
        else:
            global_tools = await tool_provider_manager.get_resolved_tools(
                allowed_servers=allowed_servers,
                excluded_servers=excluded_servers,
            )
            effective_tools = self._filter_discovered_tools(
                global_tools, self.tool_filter
            )
        return effective_tools

    async def _reflexion_loop(
        self,
        original_call: "LLMToolCall",
        error_result: ToolResult,
        history: list[LLMMessage],
        context: RunContext | None,
        config: ExecutionConfig,
        generation_config: LLMGenerationConfig | None,
    ) -> list[LLMMessage]:
        """
        构造一个影子上下文，让 LLM 自我分析错误并尝试修复工具调用。
        """
        effective_tools = await self._get_effective_tools()
        shadow_history = list(history)

        shadow_history.append(
            LLMMessage(role="assistant", content="", tool_calls=[original_call])
        )

        error_hint = error_result.display_content or str(error_result.output)
        error_payload: dict[str, Any] | None = None
        if isinstance(error_result.output, dict):
            error_payload = error_result.output
        elif isinstance(error_result.output, str):
            try:
                error_payload = json.loads(error_result.output)
            except Exception:
                pass

        if error_payload:
            try:
                parsed_error = ToolErrorResult.model_validate(error_payload)
                error_hint = parsed_error.message
            except Exception:
                pass

        shadow_history.append(
            LLMMessage.tool_response(
                tool_call_id=original_call.id,
                function_name=original_call.function.name,
                result=error_result.output,
            )
        )

        reflexion_prompt = (
            f"工具调用失败。错误信息: {error_hint}\n"
            "请分析错误原因（参数格式错误？逻辑错误？前置条件未满足？），"
            "并生成一个新的、修正后的工具调用。不要解释，直接调用工具。"
        )
        shadow_history.append(LLMMessage.user(reflexion_prompt))

        logger.info(f"🔄 [Reflexion] 触发反思循环，错误: {error_hint[:50]}...")

        try:
            response = await self.session.generate_internal(
                messages=shadow_history,
                config=generation_config,
                tools=list(effective_tools.values()) if effective_tools else None,
                model_instance=self.model,
            )

            if response.tool_calls:
                new_results = await self.tool_invoker.execute_batch(
                    response.tool_calls, effective_tools, context
                )
                assistant_msg = LLMMessage(
                    role="assistant",
                    content=response.content_parts or response.text,
                    tool_calls=response.tool_calls,
                )
                return [assistant_msg, *new_results]
            return [LLMMessage(role="assistant", content=response.text)]
        except Exception as e:
            logger.warning(f"🔄 [Reflexion] 尝试修复失败: {e}")
            return [
                LLMMessage.tool_response(
                    tool_call_id=original_call.id,
                    function_name=original_call.function.name,
                    result=error_result.output,
                )
            ]

    def _is_tool_error(self, result: ToolResult) -> bool:
        """
        判断工具执行结果是否包含结构化错误。
        """
        payload = result.output
        if isinstance(payload, dict):
            try:
                model_validate(ToolErrorResult, payload)
                return True
            except Exception:
                return False

        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                model_validate(ToolErrorResult, parsed)
                return True
            except Exception:
                return False

        return False

    def _can_retry_via_llm(self, result: ToolResult) -> bool:
        """判断错误是否值得让 LLM 尝试修复 (基于 ToolResult 中的 is_retryable 标志)"""
        payload = result.output
        if isinstance(payload, dict):
            return payload.get("is_retryable", False)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed.get("is_retryable", False)
            except Exception:
                pass
        return False

    async def run(
        self,
        messages: list[LLMMessage],
        context: RunContext | None = None,
        config: ExecutionConfig | None = None,
        generation_config: LLMGenerationConfig | None = None,
    ) -> list[LLMMessage]:
        """
        准备环境并初始 Runner 模式的执行循环。
        """
        effective_config = config or ExecutionConfig()
        effective_tools = await self._get_effective_tools()

        logger.info(
            "✅ AgentExecutor: 准备完成，将使用 "
            f"{len(effective_tools)} 个工具并将任务委托给引擎。"
        )

        execution_history = list(messages)
        start_time = time.monotonic()
        await self._trigger_callbacks("on_agent_start", messages=messages)
        try:
            for _ in range(effective_config.max_cycles):
                await self._trigger_callbacks(
                    "on_model_start",
                    model_name=self.model.model_name,
                    messages=execution_history,
                )
                model_start = time.monotonic()

                response = await self.session.generate_internal(
                    messages=execution_history,
                    config=generation_config,
                    tools=list(effective_tools.values()) if effective_tools else None,
                    model_instance=self.model,
                )

                await self._trigger_callbacks(
                    "on_model_end",
                    response=response,
                    duration=time.monotonic() - model_start,
                )

                assistant_content = (
                    response.content_parts if response.content_parts else response.text
                )

                if response.thought_signature and isinstance(assistant_content, list):
                    for part in assistant_content:
                        if part.type == "thought":
                            if part.metadata is None:
                                part.metadata = {}
                            part.metadata["thought_signature"] = (
                                response.thought_signature
                            )
                            break
                assistant_message = LLMMessage(
                    role="assistant",
                    content=assistant_content,
                    tool_calls=response.tool_calls,
                )
                execution_history.append(assistant_message)

                if not response.tool_calls:
                    has_thought = bool(
                        response.thought_text or response.thought_signature
                    )
                    has_text_response = bool(response.text and response.text.strip())

                    if has_thought and not has_text_response:
                        logger.warning(
                            "⚠️ [AgentExecutor] 检测到模型思考后中断 "
                            "(Thinking Halt)，触发强制行动指令。"
                        )
                        execution_history.append(
                            LLMMessage.user(
                                "检测到你进行了思考但未输出行动。请根据你的思考结果，"
                                "立即生成工具调用请求或最终回复。"
                            )
                        )
                        continue

                    logger.info("✅ AgentExecutor：模型未请求工具调用，执行结束。")
                    return execution_history

                logger.info(
                    "🛠️ AgentExecutor：模型请求并行调用 "
                    f"{len(response.tool_calls)} 个工具"
                )

                async def _execute_single_managed(call) -> str:
                    """执行单个工具（含反思），返回最终的字符串结果"""
                    _, result = await self.tool_invoker.execute_tool_call(
                        call, effective_tools, context
                    )

                    if (
                        self._is_tool_error(result)
                        and effective_config.reflexion_retries > 0
                    ):
                        if self._can_retry_via_llm(result):
                            reflexion_msgs = await self._reflexion_loop(
                                call,
                                result,
                                list(execution_history[:-1]),
                                context,
                                effective_config,
                                generation_config,
                            )

                            if reflexion_msgs and reflexion_msgs[-1].role == "tool":
                                content = reflexion_msgs[-1].content
                                if isinstance(content, str):
                                    return content
                                return str(content)

                    return str(result.output)

                tasks = [_execute_single_managed(call) for call in response.tool_calls]

                tool_results_msgs: list[LLMMessage] = []
                results_contents = await asyncio.gather(*tasks, return_exceptions=True)
                for i, content_or_exc in enumerate(results_contents):
                    original_call = response.tool_calls[i]

                    if isinstance(content_or_exc, Exception):
                        logger.error(
                            f"工具 {original_call.function.name} "
                            f"执行异常: {content_or_exc}"
                        )
                        final_content = json.dumps(
                            {"error": str(content_or_exc), "status": "failed"}
                        )
                    else:
                        final_content = content_or_exc

                    msg = LLMMessage.tool_response(
                        tool_call_id=original_call.id,
                        function_name=original_call.function.name,
                        result=final_content,
                    )

                    if original_call.thought_signature:
                        msg.thought_signature = original_call.thought_signature

                    tool_results_msgs.append(msg)

                execution_history.extend(tool_results_msgs)

            raise LLMException(
                f"超过最大工具调用循环次数 ({effective_config.max_cycles})。",
                code=LLMErrorCode.GENERATION_FAILED,
            )
        finally:
            duration = time.monotonic() - start_time
            await self._trigger_callbacks(
                "on_agent_end", final_history=execution_history, duration=duration
            )

    async def run_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[T],
        *,
        context: RunContext | None = None,
        config: ExecutionConfig | None = None,
        generation_config: LLMGenerationConfig | None = None,
    ) -> T:
        """
        执行完整的思考-行动循环，并将最终结果解析为指定的Pydantic模型。

        参数:
            messages: 初始消息列表。
            response_model: 用于解析和验证响应的Pydantic模型类。

        返回:
            T: 解析后的Pydantic模型实例。
        """
        try:
            json_schema = model_json_schema(response_model)
        except AttributeError:
            json_schema = response_model.schema()

        base_schema_config = LLMGenerationConfig(
            output=OutputConfig(
                response_format=ResponseFormat.JSON, response_schema=json_schema
            )
        )
        final_gen_config = base_schema_config
        if generation_config:
            update_dict = model_dump(generation_config, exclude_unset=True)
            update_dict.update(
                {
                    "response_format": ResponseFormat.JSON,
                    "response_schema": json_schema,
                }
            )
            final_gen_config = model_copy(base_schema_config, update=update_dict)

        logger.info("🔄 AgentExecutor：以结构化模式启动执行循环 (Schema-Driven)...")
        final_history = await self.run(
            messages,
            context=context,
            config=config,
            generation_config=final_gen_config,
        )

        final_assistant_message = None
        for msg in reversed(final_history):
            if msg.role == "assistant":
                final_assistant_message = msg
                break

        final_text_content = extract_text_from_content(
            final_assistant_message.content if final_assistant_message else None
        )

        if not final_assistant_message or not final_text_content:
            raise LLMException(
                "Agent 执行完毕但未能生成最终的文本回复以供结构化解析。",
                code=LLMErrorCode.GENERATION_FAILED,
            )

        return parse_and_validate_json(final_text_content, response_model)

    async def chat(
        self,
        message: str | LLMMessage | list[LLMContentPart],
        *,
        model: ModelName = None,
        tools: list[dict[str, Any] | str] | None = None,
    ) -> LLMResponse:
        """
        LLMInterface.chat 的实现。
        执行一个无状态的、一次性的子 Agent 循环。
        """
        from zhenxun.services.llm.session import AI

        if self.recursion_depth >= MAX_RECURSION_DEPTH:
            logger.warning(
                f"工具调用递归深度达到 {self.recursion_depth}"
                f" (Max: {MAX_RECURSION_DEPTH})"
            )

        logger.info(f"工具内部发起了一次子聊天... [Depth: {self.recursion_depth + 1}]")

        ai = AI()
        current_message: LLMMessage
        if isinstance(message, str):
            current_message = LLMMessage.user(message)
        elif isinstance(message, list):
            current_message = LLMMessage.user(message)
        else:
            current_message = message

        resolved_model_name = model or self.model.model_name

        tool_filter = None
        if tools:
            ad_hoc_tools = await ai._resolve_tools(tools)
            tool_filter = ToolFilter(allowed=list(ad_hoc_tools.keys()))

        from zhenxun.services.llm.manager import get_model_instance

        async with await get_model_instance(
            resolved_model_name, override_config=None
        ) as model_instance:
            sub_executor = AgentExecutor(
                model_instance,
                session=ai,
                tool_filter=tool_filter,
                recursion_depth=self.recursion_depth + 1,
                callbacks=self.callbacks,
            )
            final_history = await sub_executor.run([current_message])

        for msg in reversed(final_history):
            if msg.role == "assistant":
                text = extract_text_from_content(msg.content)
                return LLMResponse(text=text, tool_calls=msg.tool_calls)

        raise LLMException(
            "子聊天任务未能产生有效的助手回复。", code=LLMErrorCode.GENERATION_FAILED
        )

    async def generate_structured(
        self,
        message: str | LLMMessage | list[LLMContentPart],
        response_model: type[T],
        *,
        model: ModelName = None,
        tools: list[dict[str, Any] | str] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        instruction: str | None = None,
    ) -> T:
        """LLMInterface.generate_structured 的实现。"""
        from zhenxun.services.llm.api import (
            generate_structured as generate_structured_api,
        )

        logger.info(f"工具内部发起了一次结构化生成... [Depth: {self.recursion_depth}]")

        return await generate_structured_api(
            message,
            response_model,
            model=model or self.model.model_name,
            tools=tools,
            tool_choice=tool_choice,
            instruction=instruction,
        )
