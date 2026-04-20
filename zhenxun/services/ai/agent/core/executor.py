import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.engine.token_estimator import (
    global_estimator,
    parse_usage_info,
)
from zhenxun.services.ai.events import (
    AgentEndEvent,
    AgentStartEvent,
    EventCenter,
    ModelEndEvent,
    ModelStartEvent,
    ToolStreamEvent,
)
from zhenxun.services.ai.llm.config.generation import LLMGenerationConfig
from zhenxun.services.ai.llm.hooks import _GLOBAL_AFTER_HOOKS, _GLOBAL_BEFORE_HOOKS
from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.engine.executor import ToolExecutor
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import (
    LLMMessage,
    AssistantMessage,
    LLMContentPart,
    TextPart,
    ToolCallPart,
)
from zhenxun.services.ai.types.tools import (
    ToolErrorResult,
    ToolResult,
    ToolResultChunk,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_validate

if TYPE_CHECKING:
    pass


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

    async def _execute_reflexion(
        self,
        original_call: ToolCallPart,
        error_result: ToolResult,
        history: list[LLMMessage],
        model_instance: Any,
        generation_config: LLMGenerationConfig,
        run_context: RunContext,
        tool_executor: ToolExecutor,
    ) -> list[LLMMessage]:
        """构造影子上下文，让 LLM 自我分析错误并尝试修复工具调用。"""
        shadow_history = list(history)
        shadow_history.append(
            AssistantMessage(content=[original_call])  # type: ignore
        )

        error_hint = error_result.display or str(error_result.output)
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
                parsed_error = model_validate(ToolErrorResult, error_payload)
                error_hint = parsed_error.message
            except Exception:
                pass

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

        hook_kwargs = {
            "model": model_instance.model_name,
            "config": generation_config,
            "tools": list(self.tools) if self.tools else None,
            "session_id": run_context.session_id,
        }
        for hook in _GLOBAL_BEFORE_HOOKS:
            shadow_history = await hook(shadow_history, hook_kwargs)

        try:
            response = await model_instance.generate_response(
                messages=shadow_history,
                config=generation_config,
                tools=list(self.tools) if self.tools else None,
            )
            for hook in _GLOBAL_AFTER_HOOKS:
                response = await hook(response, hook_kwargs)
            if response.tool_calls:
                new_results = await tool_executor.execute_batch(
                    response.tool_calls,
                    self.tools,
                    run_context,
                    model_name=model_instance.model_name,
                    history_messages=shadow_history,
                    retry_count=1,
                    max_retries=1,
                )
                assistant_content = cast(
                    list[LLMContentPart],
                    response.content_parts
                    if response.content_parts
                    else [TextPart(text=response.text or "")],
                )
                assistant_msg = AssistantMessage(content=assistant_content)  # type: ignore
                assistant_msg.content.extend(response.tool_calls)
                return [assistant_msg, *new_results]
            return [
                AssistantMessage(content=[TextPart(text=response.text or "")])  # type: ignore
            ]
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
        generation_config: LLMGenerationConfig | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any = None,
    ) -> tuple[
        list[LLMMessage],
        str | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """
        执行推理管线，包含工具循环与生命周期回调管理。
        返回: (最终历史消息列表, 移交目标Agent, 移交参数字典, 结构化数据, 移交Payload)
        """
        tool_executor = ToolExecutor()
        gen_config = generation_config or LLMGenerationConfig()

        execution_history = list(messages)
        start_time = time.monotonic()
        await EventCenter.publish(AgentStartEvent(messages=messages))

        try:
            for cycle_index in range(self.config.max_cycles):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                await EventCenter.publish(
                    ModelStartEvent(
                        model_name=model_instance.model_name, messages=execution_history
                    )
                )
                model_start = time.monotonic()

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

                hook_kwargs = {
                    "model": model_instance.model_name,
                    "config": gen_config,
                    "tools": list(self.tools) if self.tools else None,
                    "session_id": run_context.session_id,
                }
                for hook in _GLOBAL_BEFORE_HOOKS:
                    execution_history = await hook(execution_history, hook_kwargs)

                response = await model_instance.generate_response(
                    messages=execution_history,
                    config=gen_config,
                    tools=list(self.tools) if self.tools else None,
                    extra=extra,
                    cancellation_token=cancellation_token,
                )
                for hook in _GLOBAL_AFTER_HOOKS:
                    response = await hook(response, hook_kwargs)

                await EventCenter.publish(
                    ModelEndEvent(
                        response=response,
                        duration_ms=(time.monotonic() - model_start) * 1000,
                    )
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

                assistant_message = AssistantMessage(content=response.content_parts)  # type: ignore
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
                if usage_obj.completion_tokens > 0:
                    assistant_message.token_cost = usage_obj.completion_tokens

                execution_history.append(assistant_message)

                if not response.tool_calls:
                    logger.info("✅ AgentExecutor：模型未请求工具调用，推理循环结束。")
                    return execution_history, None, None, None, None

                tasks = [
                    tool_executor.execute_tool_call(
                        call,
                        self.tools,
                        run_context,
                        model_name=model_instance.model_name,
                        history_messages=execution_history,
                        retry_count=cycle_index,
                        max_retries=self.config.max_cycles,
                    )
                    for call in response.tool_calls
                ]
                tool_results = await asyncio.gather(*tasks, return_exceptions=True)

                handoff_target = None
                handoff_kwargs = None
                handoff_payload = None
                structured_result = None

                should_terminate = False
                system_prompts_to_append = []

                for i, res_or_exc in enumerate(tool_results):
                    original_call = response.tool_calls[i]
                    media_parts = []

                    if isinstance(res_or_exc, BaseException):
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

                        display_msg = getattr(tool_res, "display", None)

                        if display_msg:
                            asyncio.create_task(
                                EventCenter.publish(
                                    ToolStreamEvent(
                                        tool_call_id=original_call.id,
                                        tool_name=original_call.tool_name,
                                        chunk=ToolResultChunk(
                                            content="",
                                            metadata={"display": display_msg},
                                        ),
                                        session_id=run_context.session_id,
                                    )
                                )
                            )
                            logger.info(
                                f"📤 已通过 ToolStreamEvent 将工具 '{original_call.tool_name}' 的展示数据推入事件流。"
                            )

                        from zhenxun.services.ai.types.messages import (
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

                        if tool_res.session_state_updates:
                            run_context.extra.update(tool_res.session_state_updates)
                            logger.info(
                                f"🔄 [状态突变] 上下文已动态更新: {tool_res.session_state_updates}"
                            )
                        if tool_res.system_prompt_append:
                            system_prompts_to_append.append(
                                tool_res.system_prompt_append
                            )
                            logger.info(
                                f"📝 [提示词追加] 动态注入新系统指令: {tool_res.system_prompt_append[:20]}..."
                            )
                        if tool_res.terminate_run:
                            should_terminate = True
                            logger.info(
                                f"🛑 [中断执行] 工具 {original_call.tool_name} 触发了强制终止信号。"
                            )

                        if isinstance(final_content, str):
                            if '"__handoff__": true' in final_content:
                                try:
                                    data = json.loads(final_content)
                                    if data.get("__handoff__"):
                                        handoff_target = data.get("target_agent")
                                        handoff_kwargs = data.get("kwargs", {})
                                        handoff_payload = data.get("payload")
                                except Exception:
                                    pass
                            if '"__final_structured_result__": true' in final_content:
                                try:
                                    data = json.loads(final_content)
                                    if data.get("__final_structured_result__"):
                                        structured_result = data.get("data")
                                except Exception:
                                    pass

                    msg = LLMMessage.tool_response(
                        original_call.id, original_call.tool_name, final_content
                    )

                    if media_parts:
                        msg.content.extend(media_parts)

                    sig = (
                        original_call.metadata.get("thought_signature")
                        if original_call.metadata
                        else None
                    )
                    if sig:
                        msg.thought_signature = sig
                    execution_history.append(msg)

                    is_handoff = (
                        isinstance(final_content, str)
                        and '"__handoff__": true' in final_content
                    )
                    if is_handoff:
                        try:
                            data = json.loads(final_content)
                            if data.get("__handoff__"):
                                handoff_target = data.get("target_agent")
                                handoff_kwargs = data.get("kwargs", {})
                                handoff_payload = data.get("payload")
                        except Exception:
                            pass

                    is_final_structured = (
                        isinstance(final_content, str)
                        and '"__final_structured_result__": true' in final_content
                    )
                    if is_final_structured:
                        try:
                            data = json.loads(final_content)
                            if data.get("__final_structured_result__"):
                                structured_result = data.get("data")
                        except Exception:
                            pass

                if handoff_target:
                    return (
                        execution_history,
                        handoff_target,
                        handoff_kwargs,
                        None,
                        handoff_payload,
                    )

                if structured_result is not None:
                    logger.info("✅ AgentExecutor：拦截到结构化结果提交，结束循环。")
                    return execution_history, None, None, structured_result, None

                for sp in system_prompts_to_append:
                    execution_history.append(LLMMessage.system(sp))

                if should_terminate:
                    logger.debug(
                        "✅ AgentExecutor：捕获到工具发出的终止信号，提前结束推理循环。"
                    )
                    return execution_history, None, None, None, None

            raise LLMException(
                f"超过最大工具调用循环次数 ({self.config.max_cycles})。",
                code=LLMErrorCode.GENERATION_FAILED,
            )

        finally:
            duration = time.monotonic() - start_time
            await EventCenter.publish(
                AgentEndEvent(
                    final_history=execution_history, duration_ms=duration * 1000
                )
            )
