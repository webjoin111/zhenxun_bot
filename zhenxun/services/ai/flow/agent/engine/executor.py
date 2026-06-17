from abc import ABC, abstractmethod
import asyncio
import json
from typing import Any, Protocol, cast

from zhenxun.services.ai.core.engine.token_counter import (
    parse_usage_info,
    token_counter,
)
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
    LLMErrorCode,
    LLMException,
)
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    LLMMessage,
    LLMResponse,
    ToolCallPart,
    ToolReturnPart,
)
from zhenxun.services.ai.core.options import GenerationConfig
from zhenxun.services.ai.flow.agent.models import AgentSettings, AgentState
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_construct


class DirectiveHandler(Protocol):
    async def handle(
        self, state: AgentState, tool_res: ToolResult
    ) -> tuple[Any, str, bool]:
        """
        处理特定的工具指令策略接口。
        返回: (display_msg, final_content, should_consume)
        - display_msg: 发送给 UI 的消息
        - final_content: 追加给 LLM 的文本回复
        - should_consume: 如果为 True，原 tool_res 将被置为 None，不再进行多模态解析
        """
        ...


class SubmitStructuredDirectiveHandler:
    async def handle(
        self, state: AgentState, tool_res: ToolResult
    ) -> tuple[Any, str, bool]:
        state.structured_result = tool_res.output
        return tool_res.ui_display, "✅ 结构化结果处理完毕。", True


class EndRunDirectiveHandler:
    async def handle(
        self, state: AgentState, tool_res: ToolResult
    ) -> tuple[Any, str, bool]:
        state.should_terminate = True
        state.early_result_output = tool_res.output
        return tool_res.ui_display, "✅ 已获取最终结果，结束当前任务。", True


class HandoffDirectiveHandler:
    async def handle(
        self, state: AgentState, tool_res: ToolResult
    ) -> tuple[Any, str, bool]:
        state.should_terminate = True
        state.early_result_output = tool_res.output
        state.handoff_triggered = tool_res
        target = getattr(tool_res, "target", "unknown")
        return tool_res.ui_display, f"✅ 已决定移交控制权至 {target}。", True


class BaseAgentExecutor(ABC):
    """
    Agent 核心执行器基类 (Template Method Pattern)。
    定义了基于生命周期的大模型控制流。第三方开发者可通过重写特定钩子，
    """

    async def run(
        self,
        state: AgentState,
        settings: AgentSettings,
        generation_config: GenerationConfig,
        model_instance: Any,
    ) -> AgentRunResult[Any]:
        """
        核心模板方法 (Template Method)。
        组织整个大模型推导与工具调用的生命周期循环。如无必要，请勿重写此方法。
        """
        await self.on_start(state, settings, model_instance)

        try:
            for cycle_index in range(settings.max_cycles):
                await self.on_cycle_start(state, cycle_index, model_instance)

                messages, extra = await self.build_llm_request(state)
                response = await self.execute_llm(
                    state, messages, extra, generation_config, model_instance
                )

                await self.handle_llm_response(state, response)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

                tool_calls = await self.filter_tool_calls(state, response)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

                tool_results = await self.execute_tools(
                    state, tool_calls, model_instance
                )

                await self.handle_tool_results(state, tool_calls, tool_results)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

            return await self.on_fallback(
                state, settings, generation_config, model_instance
            )
        except Exception as e:
            raise e

    @abstractmethod
    async def on_start(
        self, state: AgentState, settings: AgentSettings, model_instance: Any
    ) -> None:
        """生命周期: Agent 启动时调用，用于初始化状态或资源。"""
        pass

    @abstractmethod
    async def on_cycle_start(
        self, state: AgentState, cycle_index: int, model_instance: Any
    ) -> None:
        """生命周期: 每次推理循环开始时调用。可用于 Token 预估或防死循环检测。"""
        pass

    @abstractmethod
    async def build_llm_request(
        self, state: AgentState
    ) -> tuple[list[LLMMessage], dict[str, Any]]:
        """生命周期: 构造请求大模型的 Messages 上下文和 Extra 参数。"""
        pass

    @abstractmethod
    async def execute_llm(
        self,
        state: AgentState,
        messages: list[LLMMessage],
        extra: dict[str, Any],
        config: GenerationConfig,
        model_instance: Any,
    ) -> LLMResponse:
        """生命周期: 触发大模型 API 请求并返回响应。"""
        pass

    @abstractmethod
    async def handle_llm_response(
        self, state: AgentState, response: LLMResponse
    ) -> None:
        """
        生命周期: 处理大模型返回的结果，解析 Token 用量，
        并将模型回复追加至对话历史。
        """
        pass

    @abstractmethod
    async def filter_tool_calls(
        self, state: AgentState, response: LLMResponse
    ) -> list[ToolCallPart]:
        """生命周期: 从大模型的响应中提取并过滤出需要在本地客户端执行的工具调用请求。"""
        pass

    @abstractmethod
    async def execute_tools(
        self, state: AgentState, tool_calls: list[ToolCallPart], model_instance: Any
    ) -> list[Any]:
        """生命周期: 并发执行提取出的工具，并收集结果或异常。"""
        pass

    @abstractmethod
    async def handle_tool_results(
        self, state: AgentState, tool_calls: list[ToolCallPart], tool_results: list[Any]
    ) -> None:
        """
        生命周期: 处理工具返回的结果。
        包括异常拦截、UI 渲染、Handoff 移交指令以及将结果追加至对话历史。
        """
        pass

    @abstractmethod
    async def on_fallback(
        self,
        state: AgentState,
        settings: AgentSettings,
        config: GenerationConfig,
        model_instance: Any,
    ) -> AgentRunResult[Any]:
        """生命周期: 当大模型思考循环达到 max_cycles 时触发，执行兜底策略。"""
        pass


class StandardAgentExecutor(BaseAgentExecutor):
    """
    LLM 任务执行器（核心推理引擎）。
    负责：生命周期回调触发、工具循环调用、
    错误反思(Reflexion)、Token消耗追踪。
    """

    def __init__(self):
        self.tool_executor = ToolExecutor()
        self._directive_handlers: dict[str, DirectiveHandler] = {
            "submit_structured": SubmitStructuredDirectiveHandler(),
            "end_run": EndRunDirectiveHandler(),
            "handoff": HandoffDirectiveHandler(),
        }

    def _is_tool_error(self, result: ToolResult) -> bool:
        """通过新版的专属字段直接判断"""
        return result.is_error

    def _can_retry_via_llm(self, result: ToolResult) -> bool:
        """通过新版的专属字段直接判断是否允许重试"""
        return result.is_retryable

    async def _execute_model_request(
        self,
        model_instance: Any,
        messages: list[LLMMessage],
        config: GenerationConfig,
        run_context: RunContext,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any = None,
    ) -> LLMResponse:
        """
        不再在执行器层重复包裹中间件，
        直接透传给底层模型实例
        """
        return await model_instance.generate_response(
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            timeout=None,
            extra=extra or {},
            cancellation_token=cancellation_token,
        )

    async def on_start(
        self, state: AgentState, settings: AgentSettings, model_instance: Any
    ) -> None:
        state.run_context.run.messages = state.messages

    async def on_cycle_start(
        self,
        state: AgentState,
        cycle_index: int,
        model_instance: Any,
    ) -> None:
        cancellation_token = state.run_context.run.cancellation_token
        if cancellation_token:
            cancellation_token.raise_if_cancelled()

        try:
            est_tokens = token_counter.count_context(
                state.messages, model_instance.model_name, base_overhead=0
            )
            logger.debug(
                f"[TokenTracker] (Iter {cycle_index + 1}) "
                f"预估将消耗 {est_tokens} Token "
                f"(Model: {model_instance.model_name})"
            )
        except Exception:
            pass

    async def build_llm_request(
        self,
        state: AgentState,
    ) -> tuple[list[LLMMessage], dict[str, Any]]:
        run_context = state.run_context
        current_extra = run_context.state.copy()
        current_extra["__sys_capabilities"] = getattr(run_context, "capabilities", [])
        current_extra["run_context"] = run_context

        messages_to_send = []
        if state.static_system_prompt:
            if isinstance(state.static_system_prompt, list):
                for sp in state.static_system_prompt:
                    if sp and sp.strip():
                        messages_to_send.append(LLMMessage.system(sp))
            else:
                if state.static_system_prompt and state.static_system_prompt.strip():
                    messages_to_send.append(
                        LLMMessage.system(state.static_system_prompt)
                    )

        messages_to_send.extend(state.messages)

        dynamic_parts = []
        if state.dynamic_system_prompt and state.dynamic_system_prompt.strip():
            dynamic_parts.append(state.dynamic_system_prompt)
        if (
            hasattr(run_context.run, "dynamic_prompts")
            and run_context.run.dynamic_prompts
        ):
            dynamic_parts.append(
                "### 🔄 [系统实时状态注入]\n"
                + "\n\n".join(run_context.run.dynamic_prompts.values())
            )

        if dynamic_parts:
            messages_to_send.append(LLMMessage.system("\n\n".join(dynamic_parts)))

        return messages_to_send, current_extra

    async def execute_llm(
        self,
        state: AgentState,
        messages: list[LLMMessage],
        extra: dict[str, Any],
        config: GenerationConfig,
        model_instance: Any,
    ) -> LLMResponse:
        run_context = state.run_context
        tools = state.tools
        cancellation_token = run_context.run.cancellation_token

        return await self._execute_model_request(
            model_instance=model_instance,
            messages=messages,
            config=config,
            run_context=run_context,
            tools=list(tools) if tools else None,
            tool_choice=None,
            extra=extra,
            cancellation_token=cancellation_token,
        )

    async def handle_llm_response(
        self,
        state: AgentState,
        response: LLMResponse,
    ) -> None:
        run_context = state.run_context
        assistant_content = (
            response.content_parts if response.content_parts else response.text
        )
        if response.thought_signature and isinstance(assistant_content, list):
            for part in assistant_content:
                if part.type == "thought":
                    if part.metadata is None:
                        part.metadata = {}
                    part.metadata["thought_signature"] = response.thought_signature
                    break

        from zhenxun.services.ai.core.messages import AssistantContentUnion

        assistant_message = AssistantMessage(
            content=cast(list[AssistantContentUnion], response.content_parts)
        )
        if hasattr(response, "parsed_obj") and response.parsed_obj is not None:
            if assistant_message.metadata is None:
                assistant_message.metadata = {}
            assistant_message.metadata["parsed_obj"] = response.parsed_obj

        usage_obj = parse_usage_info(response.usage_info)
        state.usage += usage_obj
        if usage_obj.completion_tokens > 0:
            assistant_message.token_cost = usage_obj.completion_tokens

        state.messages.append(assistant_message)
        run_context.session.append_only_manager.sync_messages(state.messages)

        if not response.tool_calls:
            logger.debug("✅ AgentExecutor：模型未请求工具调用，推理循环结束。")
            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=response.text,
                messages=state.messages,
                usage=state.usage,
            )

    async def filter_tool_calls(
        self,
        state: AgentState,
        response: LLMResponse,
    ) -> list[ToolCallPart]:
        tools = state.tools
        event_streamer = state.run_context.run.streamer

        completed_call_ids = {
            p.tool_call_id
            for p in response.content_parts
            if isinstance(p, ToolReturnPart)
        }
        client_tool_calls = []
        for call in response.tool_calls:
            tool_inst = tools.get(call.tool_name) if tools else None
            is_server_side = call.id in completed_call_ids or (
                tool_inst and getattr(tool_inst, "execution_side", "client") == "server"
            )

            if is_server_side:
                logger.debug(
                    "☁️ [AgentExecutor] 检测到云端工具调用: "
                    f"{call.tool_name}，已跳过本地执行。"
                )
                if event_streamer:
                    from zhenxun.services.ai.core.stream_events import (
                        ToolCallResultEvent,
                        ToolCallStart,
                    )
                    from zhenxun.services.ai.tools.models import ToolResult

                    await event_streamer.send(
                        ToolCallStart(
                            tool_name=call.tool_name,
                            arguments=call.args if isinstance(call.args, dict) else {},
                            intent=getattr(call, "intent", None),
                        )
                    )
                    return_part = next(
                        (
                            p
                            for p in response.content_parts
                            if isinstance(p, ToolReturnPart)
                            and p.tool_call_id == call.id
                        ),
                        None,
                    )
                    if return_part:
                        res_mock = ToolResult(output=return_part.output)
                        await event_streamer.send(
                            ToolCallResultEvent(
                                tool_name=call.tool_name,
                                result=res_mock,
                                is_error=False,
                            )
                        )
            else:
                client_tool_calls.append(call)

        if not client_tool_calls:
            logger.info("✅ AgentExecutor：无本地客户端工具需执行，推理循环平滑结束。")
            from zhenxun.utils.pydantic_compat import model_construct

            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=response.text,
                messages=state.messages,
                usage=state.usage,
            )

        return client_tool_calls

    async def execute_tools(
        self,
        state: AgentState,
        tool_calls: list[ToolCallPart],
        model_instance: Any,
    ) -> list[Any]:
        run_context = state.run_context
        tools = state.tools
        event_streamer = run_context.run.streamer

        val_tasks = [
            self.tool_executor.validate_tool_call(
                call,
                tools,
                run_context,
                event_streamer=event_streamer,
            )
            for call in tool_calls
        ]
        validated_calls = await asyncio.gather(*val_tasks)

        exec_tasks = [
            self.tool_executor.execute_tool_call(
                val_call,
                tools,
                run_context,
                event_streamer=event_streamer,
            )
            for val_call in validated_calls
        ]
        tool_results = await asyncio.gather(*exec_tasks, return_exceptions=True)
        return tool_results

    async def handle_tool_results(
        self,
        state: AgentState,
        tool_calls: list[ToolCallPart],
        tool_results: list[Any],
    ) -> None:
        run_context = state.run_context

        from zhenxun.services.ai.run.ui_controller import UIController

        for i, res_or_exc in enumerate(tool_results):
            original_call = tool_calls[i]
            media_parts = []
            display_msg = None
            final_content = "Success"

            if isinstance(res_or_exc, BaseException):
                if isinstance(res_or_exc, ControlFlowExit):
                    raise res_or_exc
                else:
                    final_content = json.dumps(
                        {"error": str(res_or_exc), "status": "failed"},
                        ensure_ascii=False,
                    )
                    tool_res = None
            else:
                _, tool_res = res_or_exc

                log_msg = getattr(tool_res, "log_content", None)
                if log_msg:
                    logger.info(f"📝 [{original_call.tool_name}] {log_msg}")

                if hasattr(tool_res, "directive"):
                    handler = self._directive_handlers.get(str(tool_res.directive))
                    if handler:
                        display_msg, final_content, consume = await handler.handle(
                            state, tool_res
                        )
                        if consume:
                            tool_res = None

                    if display_msg:
                        ui = UIController(run_context)
                        await ui.send_display(display_msg)

                if tool_res is not None:
                    from zhenxun.services.ai.core.messages import (
                        AudioPart,
                        FilePart,
                        ImagePart,
                        TextPart,
                        VideoPart,
                    )
                    from zhenxun.utils.pydantic_compat import dump_json_safely

                    if isinstance(tool_res.output, list):
                        texts = []
                        for item in tool_res.output:
                            if isinstance(
                                item,
                                (ImagePart, AudioPart, VideoPart, FilePart),
                            ):
                                media_parts.append(item)
                            elif isinstance(item, TextPart):
                                texts.append(item.text)
                            else:
                                texts.append(str(item))
                        final_content = " ".join(texts) if texts else "Success"
                    elif isinstance(tool_res.output, str):
                        final_content = tool_res.output
                    else:
                        final_content = dump_json_safely(
                            tool_res.output, ensure_ascii=False
                        )

                    tool_usage = getattr(tool_res, "usage", None)
                    if tool_usage is not None:
                        state.usage += tool_usage

            msg = LLMMessage.tool_response(
                original_call.id, original_call.tool_name, final_content
            )

            if media_parts:
                msg.content.extend(media_parts)

            state.messages.append(msg)

        run_context.session.append_only_manager.sync_messages(state.messages)

        if state.structured_result is not None:
            logger.info("✅ AgentExecutor：拦截到结构化结果提交，结束循环。")
            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=None,
                messages=state.messages,
                structured_data=state.structured_result,
                usage=state.usage,
            )
            return

        if state.handoff_triggered is not None:
            from zhenxun.services.ai.run.models import HandoffPayload

            logger.info("✅ AgentExecutor：拦截到移交(Handoff)信号，结束循环。")
            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=state.early_result_output,
                messages=state.messages,
                usage=state.usage,
                handoff=HandoffPayload(
                    target=getattr(state.handoff_triggered, "target", ""),
                    reason=getattr(state.handoff_triggered, "reason", ""),
                    context_data=getattr(state.handoff_triggered, "context_data", ""),
                ),
            )
            return

        if state.should_terminate:
            logger.debug(
                "✅ AgentExecutor：捕获到工具发出的终止信号，提前结束推理循环。"
            )
            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=state.early_result_output,
                messages=state.messages,
                usage=state.usage,
            )
            return

    async def on_fallback(
        self,
        state: AgentState,
        settings: AgentSettings,
        config: GenerationConfig,
        model_instance: Any,
    ) -> AgentRunResult[Any]:
        run_context = state.run_context
        cancellation_token = run_context.run.cancellation_token
        event_streamer = run_context.run.streamer

        if not settings.enable_fallback_summary:
            raise LLMException(
                f"超过最大工具调用循环次数 ({settings.max_cycles})。",
                code=LLMErrorCode.GENERATION_FAILED,
            )

        logger.warning(
            f"AgentExecutor 达到最大循环次数 ({settings.max_cycles})，"
            "触发兜底总结机制。"
        )

        if event_streamer:
            from zhenxun.services.ai.core.stream_events import ToolStreamChunk

            await event_streamer.send(
                ToolStreamChunk(
                    tool_name="System",
                    content="⏳ 思考过程过于复杂，正在强制生成最终总结...",
                )
            )

        fallback_msg = LLMMessage.user(
            "### 🚨 [系统强制指令]\n"
            "你的任务执行已达到最大工具调用循环次数上限，当前思考流已被框架强制中断。\n"
            "请**诚实地**向用户总结：你目前进行到了哪一步？遇到了什么困难导致循环耗尽？还有哪些预期步骤未能完成？\n"
            "**绝对禁止**对用户撒谎声声称你已经完成了任务。严禁再次尝试调用任何工具！请直接输出纯文本结果。"
        )
        state.messages.append(fallback_msg)

        current_extra = run_context.state.copy()
        current_extra["__sys_capabilities"] = getattr(run_context, "capabilities", [])
        current_extra["run_context"] = run_context

        fallback_response = await self._execute_model_request(
            model_instance=model_instance,
            messages=state.messages,
            config=config,
            run_context=run_context,
            tools=[],
            tool_choice="none",
            extra=current_extra,
            cancellation_token=cancellation_token,
        )

        from zhenxun.services.ai.core.messages import AssistantContentUnion

        assistant_message = AssistantMessage(
            content=cast(list[AssistantContentUnion], fallback_response.content_parts)
        )

        usage_obj = parse_usage_info(fallback_response.usage_info)
        state.usage += usage_obj
        if usage_obj.completion_tokens > 0:
            assistant_message.token_cost = usage_obj.completion_tokens

        state.messages.append(assistant_message)

        return model_construct(
            AgentRunResult,
            output=fallback_response.text,
            messages=state.messages,
            structured_data=None,
            usage=state.usage,
        )
