from abc import ABC, abstractmethod
import asyncio
import json
from typing import Any, cast

from zhenxun.services.ai.core.engine.token_counter import (
    parse_usage_info,
    token_counter,
)
from zhenxun.services.ai.core.events import ToolStreamChunk
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
    UpstreamServerException,
)
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    AudioPart,
    ChatRequest,
    ChatResponse,
    FilePart,
    ImagePart,
    LLMMessage,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    VideoPart,
)
from zhenxun.services.ai.core.options import GenerationConfig
from zhenxun.services.ai.flow.agent.engine.directive import (
    DirectiveHandlerFunc,
    directive_manager,
)
from zhenxun.services.ai.flow.agent.models import AgentRunResources, AgentState
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.run.ui import UIController
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import dump_json_safely, model_construct


class BaseAgentExecutor(ABC):
    """
    Agent 核心执行器基类 (Template Method Pattern)。
    定义了基于生命周期的大模型控制流。第三方开发者可通过重写特定钩子，
    """

    async def run(
        self, state: AgentState, resources: AgentRunResources
    ) -> AgentRunResult[Any]:
        """
        核心模板方法 (Template Method)。
        组织整个大模型推导与工具调用的生命周期循环。如无必要，请勿重写此方法。
        """
        await self.on_start(state, resources)

        try:
            for cycle_index in range(resources.config.max_cycles):
                state.current_cycle = cycle_index
                await self.on_cycle_start(state, resources)

                await self.build_llm_request(state, resources)
                await self.execute_llm(state, resources)

                await self.handle_llm_response(state, resources)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

                await self.filter_tool_calls(state, resources)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

                await self.execute_tools(state, resources)

                await self.handle_tool_results(state, resources)
                if state.is_finished:
                    assert state.final_result is not None
                    return state.final_result

            return await self.on_fallback(state, resources)
        except Exception as e:
            raise e

    @abstractmethod
    async def on_start(self, state: AgentState, resources: AgentRunResources) -> None:
        """生命周期: Agent 启动时调用，用于初始化状态或资源。"""
        pass

    @abstractmethod
    async def on_cycle_start(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """生命周期: 每次推理循环开始时调用。可用于 Token 预估或防死循环检测。"""
        pass

    @abstractmethod
    async def build_llm_request(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """生命周期: 构造请求大模型的 Messages 上下文和 Extra 参数。"""
        pass

    @abstractmethod
    async def execute_llm(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """生命周期: 触发大模型 API 请求并返回响应。"""
        pass

    @abstractmethod
    async def handle_llm_response(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """
        生命周期: 处理大模型返回的结果，解析 Token 用量，
        并将模型回复追加至对话历史。
        """
        pass

    @abstractmethod
    async def filter_tool_calls(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """生命周期: 从大模型的响应中提取并过滤出需要在本地客户端执行的工具调用请求。"""
        pass

    @abstractmethod
    async def execute_tools(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """生命周期: 并发执行提取出的工具，并收集结果或异常。"""
        pass

    @abstractmethod
    async def handle_tool_results(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """
        生命周期: 处理工具返回的结果。
        包括异常拦截、UI 渲染、Handoff 移交指令以及将结果追加至对话历史。
        """
        pass

    @abstractmethod
    async def on_fallback(
        self, state: AgentState, resources: AgentRunResources
    ) -> AgentRunResult[Any]:
        """生命周期: 当大模型思考循环达到 max_cycles 时触发，执行兜底策略。"""
        pass


class StandardAgentExecutor(BaseAgentExecutor):
    """
    LLM 任务执行器（核心推理引擎）。
    负责：生命周期回调触发、工具循环调用、
    错误反思(Reflexion)、Token消耗追踪。
    """

    def __init__(
        self, directive_handlers: dict[str, DirectiveHandlerFunc] | None = None
    ):
        self.tool_executor = ToolExecutor()
        self._directive_handlers: dict[str, DirectiveHandlerFunc] = (
            directive_handlers or {}
        )

    def _can_retry_via_llm(self, result: ToolResult) -> bool:
        """通过新版的专属字段直接判断是否允许重试"""
        return result.is_retryable

    async def _invoke_and_record_llm(
        self,
        state: AgentState,
        resources: AgentRunResources,
        messages: list[LLMMessage],
        tools: list[Any] | None,
        tool_choice: Any = None,
    ) -> ChatResponse:
        """执行 LLM 请求，处理基础指标遥测统计，并将新对话上下文追加到状态流"""
        run_context = resources.run_context
        cancellation_token = run_context.run.cancellation_token

        current_extra = run_context.state.copy()
        current_extra["__sys_capabilities"] = getattr(run_context, "capabilities", [])
        current_extra["run_context"] = run_context

        response = await self._execute_model_request(
            model_name=resources.model_name,
            messages=messages,
            config=resources.generation_config or GenerationConfig(),
            run_context=run_context,
            tools=tools,
            tool_choice=tool_choice,
            extra=current_extra,
            cancellation_token=cancellation_token,
        )

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
            if not isinstance(response.parsed_obj, str):
                if assistant_message.metadata is None:
                    assistant_message.metadata = {}
                assistant_message.metadata["parsed_obj"] = response.parsed_obj

        usage_obj = parse_usage_info(response.usage_info)
        state.usage += usage_obj
        if usage_obj.completion_tokens > 0:
            assistant_message.token_cost = usage_obj.completion_tokens

        state.messages.append(assistant_message)
        run_context.session.append_only_manager.sync_messages(state.messages)

        return response

    async def _execute_model_request(
        self,
        model_name: str | None,
        messages: list[LLMMessage],
        config: GenerationConfig,
        run_context: RunContext,
        tools: list[Any] | None = None,
        tool_choice: Any = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any = None,
    ) -> ChatResponse:
        from zhenxun.services.ai.capabilities import CombinedCapability
        from zhenxun.services.ai.core.models import LLMContext
        from zhenxun.services.ai.llm.engine.router import LLMOrchestrator

        request = ChatRequest(
            messages=messages,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            extra=extra or {},
        )

        sys_caps = request.extra.pop("__sys_capabilities", [])
        llm_context = LLMContext(request=request, cancellation_token=cancellation_token)
        combined_cap = CombinedCapability(sys_caps)

        async def inner_handler(ctx: LLMContext[Any, Any]) -> ChatResponse:
            return await LLMOrchestrator.invoke(
                request=ctx.request,
                model_name=model_name,
                task="chat",
                override_config=config,
                cancellation_token=ctx.cancellation_token,
            )

        return await combined_cap.wrap_model_request(
            run_context, llm_context, inner_handler
        )

    async def on_start(self, state: AgentState, resources: AgentRunResources) -> None:
        resources.run_context.run.messages = state.messages

    async def on_cycle_start(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        cancellation_token = resources.run_context.run.cancellation_token
        if cancellation_token:
            cancellation_token.raise_if_cancelled()

        try:
            est_tokens = token_counter.count_context(
                state.messages, resources.model_name or "", base_overhead=0
            )
            logger.debug(
                f"[TokenTracker] (Iter {state.current_cycle + 1}) "
                f"预估将消耗 {est_tokens} Token "
                f"(Model: {resources.model_name or 'Unknown'})"
            )
        except Exception:
            pass

    async def build_llm_request(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        run_context = resources.run_context

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

        if state.dynamic_system_messages:
            messages_to_send.extend(state.dynamic_system_messages)

        if (
            hasattr(run_context.run, "dynamic_prompts")
            and run_context.run.dynamic_prompts
        ):
            for prompt_text in run_context.run.dynamic_prompts.values():
                if prompt_text and prompt_text.strip():
                    messages_to_send.append(LLMMessage.system(prompt_text))

        messages_to_send.extend(state.messages)

        state.current_request_messages = messages_to_send

    async def execute_llm(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        tools = state.tools

        state.current_response = await self._invoke_and_record_llm(
            state=state,
            resources=resources,
            messages=state.current_request_messages,
            tools=list(tools) if tools else None,
            tool_choice=None,
        )

    async def handle_llm_response(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        response = state.current_response
        if not response:
            return

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
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        response = state.current_response
        if not response:
            return
        tools = state.tools
        event_streamer = resources.run_context.run.streamer

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
                async with self.tool_executor._tool_stream_scope(
                    event_streamer,
                    call.tool_name,
                    call.args if isinstance(call.args, dict) else {},
                    getattr(call, "intent", None),
                ) as box:
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
                        from zhenxun.services.ai.tools.models import ToolResult

                        box["result"] = ToolResult(output=return_part.output)
            else:
                client_tool_calls.append(call)

        if not client_tool_calls:
            logger.info("✅ AgentExecutor：无本地客户端工具需执行，推理循环平滑结束。")

            state.is_finished = True
            state.final_result = model_construct(
                AgentRunResult,
                output=response.text,
                messages=state.messages,
                usage=state.usage,
            )

        state.current_tool_calls = client_tool_calls

    async def execute_tools(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        run_context = resources.run_context
        tools = state.tools
        event_streamer = run_context.run.streamer
        tool_calls = state.current_tool_calls

        if not tool_calls:
            return

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
        state.current_tool_results = tool_results

    async def _process_tool_directive(
        self, state: AgentState, resources: AgentRunResources, tool_res: ToolResult
    ) -> tuple[ToolResult | None, str | None]:
        """处理工具指令与 UI 副作用渲染"""
        if not hasattr(tool_res, "directive"):
            return tool_res, None

        directive_name = str(tool_res.directive)
        namespace = getattr(resources.run_context.session, "namespace", "global")

        handler = self._directive_handlers.get(directive_name)
        if not handler:
            handler = directive_manager.get_handler(directive_name, namespace)

        if not handler:
            return tool_res, None

        display_msg, final_content, consume = await handler(state, resources, tool_res)

        if display_msg:
            ui = UIController(resources.run_context)
            await ui.send_display(display_msg)

        if consume:
            return None, final_content

        return tool_res, None

    def _assemble_tool_message(
        self,
        original_call: ToolCallPart,
        res_or_exc: Any,
        tool_res: ToolResult | None,
        directive_content: str | None,
        state: AgentState,
    ) -> LLMMessage:
        """负责处理异常、解析多模态、序列化，并装配为最终的工具消息载体"""
        media_parts = []
        final_content = "Success"

        if isinstance(res_or_exc, BaseException):
            if isinstance(res_or_exc, ControlFlowExit):
                raise res_or_exc
            final_content = json.dumps(
                {"error": str(res_or_exc), "status": "failed"},
                ensure_ascii=False,
            )
        elif tool_res is None and directive_content is not None:
            final_content = directive_content
        elif tool_res is not None:
            if isinstance(tool_res.output, list):
                texts = []
                for item in tool_res.output:
                    if isinstance(item, ImagePart | AudioPart | VideoPart | FilePart):
                        media_parts.append(item)
                    elif isinstance(item, TextPart):
                        texts.append(item.text)
                    else:
                        texts.append(str(item))
                final_content = " ".join(texts) if texts else "Success"
            elif isinstance(tool_res.output, str):
                final_content = tool_res.output
            else:
                final_content = dump_json_safely(tool_res.output, ensure_ascii=False)

            tool_usage = getattr(tool_res, "usage", None)
            if tool_usage is not None:
                state.usage += tool_usage

        msg = LLMMessage.tool_response(
            original_call.id, original_call.tool_name, final_content
        )
        if media_parts:
            msg.content.extend(media_parts)
        return msg

    async def handle_tool_results(
        self, state: AgentState, resources: AgentRunResources
    ) -> None:
        """处理所有工具执行结果，调度副作用指令并装配对话回传报文。"""
        tool_calls = state.current_tool_calls
        tool_results = state.current_tool_results
        if not tool_calls or not tool_results:
            return

        for i, res_or_exc in enumerate(tool_results):
            original_call = tool_calls[i]
            tool_res = None
            directive_content = None

            if not isinstance(res_or_exc, BaseException):
                _, raw_tool_res = res_or_exc
                log_msg = getattr(raw_tool_res, "log_content", None)
                if log_msg:
                    logger.info(f"📝 [{original_call.tool_name}] {log_msg}")

                tool_res, directive_content = await self._process_tool_directive(
                    state, resources, raw_tool_res
                )

            msg = self._assemble_tool_message(
                original_call, res_or_exc, tool_res, directive_content, state
            )
            state.messages.append(msg)

        resources.run_context.session.append_only_manager.sync_messages(state.messages)

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
        self, state: AgentState, resources: AgentRunResources
    ) -> AgentRunResult[Any]:
        run_context = resources.run_context
        event_streamer = run_context.run.streamer

        if not resources.config.enable_fallback_summary:
            raise UpstreamServerException(
                f"超过最大工具调用循环次数 ({resources.config.max_cycles})。",
            )

        logger.warning(
            f"AgentExecutor 达到最大循环次数 ({resources.config.max_cycles})，"
            "触发兜底总结机制。"
        )

        if event_streamer:
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

        fallback_response = await self._invoke_and_record_llm(
            state=state,
            resources=resources,
            messages=state.messages,
            tools=[],
            tool_choice="none",
        )

        return model_construct(
            AgentRunResult,
            output=fallback_response.text,
            messages=state.messages,
            structured_data=None,
            usage=state.usage,
        )
