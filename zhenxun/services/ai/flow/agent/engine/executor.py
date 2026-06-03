import asyncio
import json
from typing import Any, cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.engine.token_estimator import (
    global_estimator,
    parse_usage_info,
)
from zhenxun.services.ai.core.exceptions import (
    ControlFlowException,
    EndRunException,
    LLMErrorCode,
    LLMException,
    SubmitStructuredException,
)
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    LLMMessage,
    LLMResponse,
    TextPart,
    ToolCallPart,
    UsageInfo,
)
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_construct


class AgentExecutorConfig(BaseModel):
    """
    核心执行引擎配置。
    用于在单次运行中精细控制多步推理与工具调用的行为。
    """

    max_cycles: int = Field(
        default=10, description="工具调用循环的最大次数，防止无限循环。"
    )
    enable_parallel_calls: bool = Field(
        default=True, description="是否允许LLM在一次思考中请求调用多个工具。"
    )
    reflexion_retries: int = Field(
        default=1,
        description="当工具执行出错时，允许进行自我反思和修正的最大重试次数。",
    )
    enable_fallback_summary: bool = Field(
        default=True,
        description="达到最大循环次数时，是否触发大模型兜底总结（而不是直接报错）。",
    )


class AgentExecutor:
    """
    LLM 任务执行器（核心推理引擎）。
    负责：生命周期回调触发、工具循环调用、错误反思(Reflexion)、Token消耗追踪。
    """

    def __init__(
        self,
        tools: Any,
        config: AgentExecutorConfig | None = None,
    ):
        self.tools = tools
        self.config = config or AgentExecutorConfig()

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
        """不再在执行器层重复包裹中间件，直接透传给底层模型实例"""
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
    ) -> list[LLMMessage]:
        """构造影子上下文，让 LLM 自我分析错误并尝试修复工具调用。"""
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
            f"### 🔄 [工具执行异常自愈]\n"
            f"检测到工具调用返回了错误结果。请启动自我诊断流程：\n"
            f"- **错误详情**：> {error_hint}\n"
            f"- **分析要求**：对比工具定义，检查参数格式、逻辑约束或前置条件是否满足。\n"
            f"- **操作指令**：请输出一个修正后的工具调用指令。**禁止进行任何文字解释，直接调用工具。**"
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
                tools=list(self.tools) if self.tools else None,
                extra=extra,
            )
            if response.tool_calls:
                new_results = await tool_executor.execute_batch(
                    response.tool_calls,
                    self.tools,
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
        messages: list[LLMMessage],
        model_instance: Any,
        run_context: RunContext,
        generation_config: GenerationConfig | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any = None,
        event_streamer: Any | None = None,
    ) -> AgentRunResult[Any]:
        """
        执行推理管线，包含工具循环与生命周期回调管理。
        返回: 包含运行状态、历史消息和控制流信号的 AgentRunResult 对象
        """
        tool_executor = ToolExecutor()
        gen_config = generation_config or GenerationConfig()

        execution_history = list(messages)
        run_context.run.messages = execution_history

        cumulative_usage = UsageInfo()

        try:
            for cycle_index in range(self.config.max_cycles):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()

                try:
                    est_tokens = global_estimator.estimate_context(
                        execution_history, model_instance.model_name, base_overhead=0
                    )
                    logger.debug(
                        f"[TokenTracker] 预估将消耗 {est_tokens} Token "
                        f"(Model: {model_instance.model_name})"
                    )
                except Exception:
                    pass

                current_extra = run_context.state.copy()
                if extra:
                    current_extra.update(extra)
                current_extra["__sys_capabilities"] = getattr(
                    run_context, "capabilities", []
                )
                current_extra["run_context"] = run_context

                messages_to_send = list(execution_history)
                if (
                    hasattr(run_context.run, "dynamic_prompts")
                    and run_context.run.dynamic_prompts
                ):
                    dynamic_text = "\n\n".join(run_context.run.dynamic_prompts.values())
                    injected = False
                    for i, msg in enumerate(messages_to_send):
                        if msg.role == "system":
                            messages_to_send[i] = (
                                msg + f"\n\n### 🔄 [系统动态注入]\n{dynamic_text}"
                            )
                            injected = True
                            break
                    if not injected:
                        messages_to_send.insert(
                            0,
                            LLMMessage.system(f"### 🔄 [系统动态注入]\n{dynamic_text}"),
                        )

                response = await self._execute_model_request(
                    model_instance=model_instance,
                    messages=messages_to_send,
                    config=gen_config,
                    run_context=run_context,
                    tools=list(self.tools) if self.tools else None,
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
                if usage_obj.prompt_tokens > 0:
                    global_estimator.calibrate(
                        usage_obj.prompt_tokens,
                        execution_history,
                        model_instance.model_name,
                    )
                cumulative_usage += usage_obj
                if usage_obj.completion_tokens > 0:
                    assistant_message.token_cost = usage_obj.completion_tokens

                execution_history.append(assistant_message)

                if not response.tool_calls:
                    logger.info("✅ AgentExecutor：模型未请求工具调用，推理循环结束。")
                    return model_construct(
                        AgentRunResult,
                        output=None,
                        messages=execution_history,
                        usage=cumulative_usage,
                    )

                val_tasks = [
                    tool_executor.validate_tool_call(
                        call,
                        self.tools,
                        run_context,
                        event_streamer=event_streamer,
                    )
                    for call in response.tool_calls
                ]
                validated_calls = await asyncio.gather(*val_tasks)

                exec_tasks = [
                    tool_executor.execute_tool_call(
                        val_call,
                        self.tools,
                        run_context,
                        event_streamer=event_streamer,
                    )
                    for val_call in validated_calls
                ]
                tool_results = await asyncio.gather(*exec_tasks, return_exceptions=True)

                structured_result = None
                early_result_output = None

                should_terminate = False

                from zhenxun.services.ai.run.ui_controller import UIController

                for i, res_or_exc in enumerate(tool_results):
                    original_call = response.tool_calls[i]
                    media_parts = []

                    if isinstance(res_or_exc, BaseException):
                        if isinstance(res_or_exc, ControlFlowException):
                            if isinstance(res_or_exc, SubmitStructuredException):
                                structured_result = res_or_exc.data
                                display_msg = None
                                final_content = "✅ 结构化结果校验通过，已提交。"
                            elif isinstance(res_or_exc, EndRunException):
                                should_terminate = True
                                display_msg = res_or_exc.display
                                early_result_output = getattr(
                                    res_or_exc, "result_output", None
                                )
                                final_content = "✅ 已获取最终结果，结束当前任务。"
                                logger.info(
                                    f"🛑 [中断执行] 工具 {original_call.tool_name} 触发了直接返回结果信号。"
                                )
                            else:
                                raise res_or_exc

                            if display_msg:
                                ui = UIController(run_context)
                                await ui.send_display(display_msg)
                                logger.info(
                                    f"📤 已通过 UIController 将控制流 '{original_call.tool_name}' 的展示数据发往前端。"
                                )

                            tool_res = None
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
                                    item, (ImagePart, AudioPart, VideoPart, FilePart)
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

                if structured_result is not None:
                    logger.info("✅ AgentExecutor：拦截到结构化结果提交，结束循环。")
                    return model_construct(
                        AgentRunResult,
                        output=None,
                        messages=execution_history,
                        structured_data=structured_result,
                        usage=cumulative_usage,
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

            if not self.config.enable_fallback_summary:
                raise LLMException(
                    f"超过最大工具调用循环次数 ({self.config.max_cycles})。",
                    code=LLMErrorCode.GENERATION_FAILED,
                )

            logger.warning(
                f"AgentExecutor 达到最大循环次数 ({self.config.max_cycles})，触发兜底总结机制。"
            )

            if event_streamer:
                from zhenxun.services.ai.core.stream_events import ToolStreamChunk

                await event_streamer.send(
                    ToolStreamChunk(
                        tool_name="System",
                        content="⏳ 思考过程过于复杂，正在强制生成最终总结...",
                    )
                )

            fallback_msg = LLMMessage.system(
                "### 🚨 [系统强制指令]\n"
                "你的任务执行已达到最大循环次数上限。请根据以上所有收集到的信息，直接给出一个最终的总结性回复。\n"
                "严禁再次尝试调用任何工具！请直接输出纯文本结果。"
            )
            execution_history.append(fallback_msg)

            current_extra = run_context.state.copy()
            if extra:
                current_extra.update(extra)
            current_extra["__sys_capabilities"] = getattr(
                run_context, "capabilities", []
            )
            current_extra["run_context"] = run_context

            fallback_response = await self._execute_model_request(
                model_instance=model_instance,
                messages=execution_history,
                config=gen_config,
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
            if usage_obj.prompt_tokens > 0:
                global_estimator.calibrate(
                    usage_obj.prompt_tokens,
                    execution_history,
                    model_instance.model_name,
                )
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
