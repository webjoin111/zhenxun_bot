import asyncio
import json
from typing import Any, cast

from zhenxun.services.ai.core.configs import GenerationConfig
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
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UsageInfo,
)
from zhenxun.services.ai.flow.agent.models import AgentEngineConfig, AgentLoopContext
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_construct


class AgentExecutor:
    """
    LLM 任务执行器（核心推理引擎）。
    负责：生命周期回调触发、工具循环调用、
    错误反思(Reflexion)、Token消耗追踪。
    此层已重构为纯净无状态结构，
    隔离了所有与 Agent 装配相关的逻辑。
    """

    def __init__(self):
        pass

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

    async def _execute_reflexion(
        self,
        original_call: ToolCallPart,
        error_result: ToolResult,
        history: list[LLMMessage],
        model_instance: Any,
        generation_config: GenerationConfig,
        run_context: RunContext,
        tool_executor: ToolExecutor,
        tools: Any,
    ) -> list[LLMMessage]:
        """
        构造影子上下文，
        让 LLM 自我分析错误并尝试修复工具调用。
        """
        shadow_history = list(history)
        shadow_history.append(AssistantMessage(content=[original_call]))

        error_hint = error_result.ui_display or str(error_result.output)
        error_payload: dict[str, Any] | None = None

        if isinstance(error_result.output, dict):
            error_payload = error_result.output
        elif isinstance(error_result.output, str):
            try:
                error_payload = json.loads(error_result.output)
            except Exception:
                pass

        if error_payload:
            error_hint = str(error_payload.get("message", error_hint))

        shadow_history.append(
            LLMMessage.tool_response(
                original_call.id,
                original_call.tool_name,
                error_result.output,
            )
        )
        reflexion_prompt = (
            "### 🔄 [工具执行异常自愈]\n"
            "检测到工具调用返回了错误结果。"
            "请启动自我诊断流程：\n"
            f"- **错误详情**：> {error_hint}\n"
            "- **分析要求**：对比工具定义，"
            "检查参数格式、逻辑约束或前置条件是否满足。\n"
            "- **操作指令**：请输出一个修正后的工具调用指令。"
            "**禁止进行任何文字解释，直接调用工具。**"
        )
        shadow_history.append(LLMMessage.user(reflexion_prompt))
        logger.info(f"🔄 [Reflexion] 触发反思循环，错误: {error_hint[:50]}...")

        extra = {
            "__sys_capabilities": getattr(run_context, "capabilities", []),
            "run_context": run_context,
        }

        try:
            response = await model_instance.generate_response(
                messages=shadow_history,
                config=generation_config,
                tools=list(tools) if tools else None,
                extra=extra,
            )
            if response.tool_calls:
                new_results = await tool_executor.execute_batch(
                    response.tool_calls,
                    tools,
                    run_context,
                    model_name=model_instance.model_name,
                    max_retries=1,
                )
                from zhenxun.services.ai.core.messages import AssistantContentUnion

                assistant_content = cast(
                    list[AssistantContentUnion],
                    response.content_parts
                    if response.content_parts
                    else [TextPart(text=response.text or "")],
                )
                assistant_msg = AssistantMessage(content=assistant_content)
                assistant_msg.content.extend(
                    cast(list[AssistantContentUnion], response.tool_calls)
                )
                return [assistant_msg, *new_results]
            return [AssistantMessage(content=[TextPart(text=response.text or "")])]
        except Exception as e:
            logger.warning(f"🔄 [Reflexion] 尝试修复失败: {e}")
            return [
                LLMMessage.tool_response(
                    original_call.id,
                    original_call.tool_name,
                    error_result.output,
                )
            ]

    async def run(
        self,
        loop_ctx: AgentLoopContext,
        exec_config: AgentEngineConfig,
        generation_config: GenerationConfig,
        model_instance: Any,
    ) -> AgentRunResult[Any]:
        """
        执行推理管线，包含工具循环与生命周期回调管理。
        返回: 包含运行状态、历史消息和控制流信号的 AgentRunResult 对象
        """
        tool_executor = ToolExecutor()
        run_context = loop_ctx.run_context
        tools = loop_ctx.tools
        cancellation_token = run_context.run.cancellation_token
        event_streamer = run_context.run.streamer

        execution_history = list(loop_ctx.messages)
        run_context.run.messages = execution_history

        cumulative_usage = UsageInfo()

        try:
            for cycle_index in range(exec_config.max_cycles):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()

                try:
                    est_tokens = token_counter.count_context(
                        execution_history, model_instance.model_name, base_overhead=0
                    )
                    logger.debug(
                        f"[TokenTracker] 预估将消耗 {est_tokens} Token "
                        f"(Model: {model_instance.model_name})"
                    )
                except Exception:
                    pass

                current_extra = run_context.state.copy()
                current_extra["__sys_capabilities"] = getattr(
                    run_context, "capabilities", []
                )
                current_extra["run_context"] = run_context

                messages_to_send = []
                if loop_ctx.static_system_prompt:
                    if isinstance(loop_ctx.static_system_prompt, list):
                        for sp in loop_ctx.static_system_prompt:
                            messages_to_send.append(LLMMessage.system(sp))
                    else:
                        messages_to_send.append(
                            LLMMessage.system(loop_ctx.static_system_prompt)
                        )

                messages_to_send.extend(execution_history)

                dynamic_parts = []
                if loop_ctx.dynamic_system_prompt:
                    dynamic_parts.append(loop_ctx.dynamic_system_prompt)
                if (
                    hasattr(run_context.run, "dynamic_prompts")
                    and run_context.run.dynamic_prompts
                ):
                    dynamic_parts.append(
                        "### 🔄 [系统实时状态注入]\n"
                        + "\n\n".join(run_context.run.dynamic_prompts.values())
                    )

                if dynamic_parts:
                    messages_to_send.append(
                        LLMMessage.system("\n\n".join(dynamic_parts))
                    )

                response = await self._execute_model_request(
                    model_instance=model_instance,
                    messages=messages_to_send,
                    config=generation_config,
                    run_context=run_context,
                    tools=list(tools) if tools else None,
                    tool_choice=None,
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
                            part.metadata["thought_signature"] = (
                                response.thought_signature
                            )
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
                cumulative_usage += usage_obj
                if usage_obj.completion_tokens > 0:
                    assistant_message.token_cost = usage_obj.completion_tokens

                execution_history.append(assistant_message)
                run_context.session.append_only_manager.sync_messages(execution_history)

                if not response.tool_calls:
                    logger.info("✅ AgentExecutor：模型未请求工具调用，推理循环结束。")
                    return model_construct(
                        AgentRunResult,
                        output=None,
                        messages=execution_history,
                        usage=cumulative_usage,
                    )

                completed_call_ids = {
                    p.tool_call_id
                    for p in response.content_parts
                    if isinstance(p, ToolReturnPart)
                }
                client_tool_calls = []
                for call in response.tool_calls:
                    tool_inst = tools.get(call.tool_name) if tools else None
                    is_server_side = call.id in completed_call_ids or (
                        tool_inst
                        and getattr(tool_inst, "execution_side", "client") == "server"
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
                                    arguments=call.args
                                    if isinstance(call.args, dict)
                                    else {},
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
                    logger.info(
                        "✅ AgentExecutor：无本地客户端工具需执行，推理循环平滑结束。"
                    )
                    return model_construct(
                        AgentRunResult,
                        output=None,
                        messages=execution_history,
                        usage=cumulative_usage,
                    )

                val_tasks = [
                    tool_executor.validate_tool_call(
                        call,
                        tools,
                        run_context,
                        event_streamer=event_streamer,
                    )
                    for call in client_tool_calls
                ]
                validated_calls = await asyncio.gather(*val_tasks)

                exec_tasks = [
                    tool_executor.execute_tool_call(
                        val_call,
                        tools,
                        run_context,
                        event_streamer=event_streamer,
                    )
                    for val_call in validated_calls
                ]
                tool_results = await asyncio.gather(*exec_tasks, return_exceptions=True)

                structured_result = None
                early_result_output = None
                should_terminate = False
                handoff_triggered = None

                from zhenxun.services.ai.run.ui_controller import UIController

                for i, res_or_exc in enumerate(tool_results):
                    original_call = client_tool_calls[i]
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
                            if tool_res.directive == "submit_structured":
                                structured_result = tool_res.output
                                display_msg = tool_res.ui_display
                                final_content = "✅ 结构化结果处理完毕。"
                                tool_res = None
                            elif tool_res.directive == "end_run":
                                should_terminate = True
                                early_result_output = tool_res.output
                                display_msg = tool_res.ui_display
                                final_content = "✅ 已获取最终结果，结束当前任务。"
                                tool_res = None
                            elif tool_res.directive == "handoff":
                                should_terminate = True
                                early_result_output = tool_res.output
                                handoff_triggered = tool_res
                                display_msg = tool_res.ui_display
                                target = getattr(tool_res, "target", "unknown")
                                final_content = f"✅ 已决定移交控制权至 {target}。"
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
                                cumulative_usage += tool_usage

                    msg = LLMMessage.tool_response(
                        original_call.id, original_call.tool_name, final_content
                    )

                    if media_parts:
                        msg.content.extend(media_parts)

                    execution_history.append(msg)

                run_context.session.append_only_manager.sync_messages(execution_history)

                if structured_result is not None:
                    logger.info("✅ AgentExecutor：拦截到结构化结果提交，结束循环。")
                    return model_construct(
                        AgentRunResult,
                        output=None,
                        messages=execution_history,
                        structured_data=structured_result,
                        usage=cumulative_usage,
                    )

                if handoff_triggered is not None:
                    from zhenxun.services.ai.run.models import HandoffPayload

                    logger.info("✅ AgentExecutor：拦截到移交(Handoff)信号，结束循环。")
                    return model_construct(
                        AgentRunResult,
                        output=early_result_output,
                        messages=execution_history,
                        usage=cumulative_usage,
                        handoff=HandoffPayload(
                            target=getattr(handoff_triggered, "target", ""),
                            reason=getattr(handoff_triggered, "reason", ""),
                            context_data=getattr(handoff_triggered, "context_data", ""),
                        ),
                    )

                if should_terminate:
                    logger.debug(
                        "✅ AgentExecutor：捕获到工具发出的终止信号，提前结束推理循环。"
                    )
                    return model_construct(
                        AgentRunResult,
                        output=early_result_output,
                        messages=execution_history,
                        usage=cumulative_usage,
                    )

            if not exec_config.enable_fallback_summary:
                raise LLMException(
                    f"超过最大工具调用循环次数 ({exec_config.max_cycles})。",
                    code=LLMErrorCode.GENERATION_FAILED,
                )

            logger.warning(
                f"AgentExecutor 达到最大循环次数 ({exec_config.max_cycles})，"
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
                "**绝对禁止**对用户撒谎声称你已经完成了任务。严禁再次尝试调用任何工具！请直接输出纯文本结果。"
            )
            execution_history.append(fallback_msg)

            current_extra = run_context.state.copy()
            current_extra["__sys_capabilities"] = getattr(
                run_context, "capabilities", []
            )
            current_extra["run_context"] = run_context

            fallback_response = await self._execute_model_request(
                model_instance=model_instance,
                messages=execution_history,
                config=generation_config,
                run_context=run_context,
                tools=[],
                tool_choice="none",
                extra=current_extra,
                cancellation_token=cancellation_token,
            )

            from zhenxun.services.ai.core.messages import AssistantContentUnion

            assistant_message = AssistantMessage(
                content=cast(
                    list[AssistantContentUnion], fallback_response.content_parts
                )
            )

            usage_obj = parse_usage_info(fallback_response.usage_info)
            cumulative_usage += usage_obj
            if usage_obj.completion_tokens > 0:
                assistant_message.token_cost = usage_obj.completion_tokens

            execution_history.append(assistant_message)

            return model_construct(
                AgentRunResult,
                output=fallback_response.text,
                messages=execution_history,
                structured_data=None,
                usage=cumulative_usage,
            )
        except Exception as e:
            raise e
