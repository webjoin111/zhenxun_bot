import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from jinja2 import Template
from nonebot.adapters import Bot, Event
from nonebot.internal.matcher import current_bot, current_event
from nonebot.matcher import Matcher
from pydantic import BaseModel

from zhenxun.services.agent.core.context import AgentContext
from zhenxun.services.llm import (
    LLMErrorCode,
    LLMException,
    LLMGenerationConfig,
    LLMMessage,
    LLMResponse,
    ModelName,
    get_model_instance,
)
from zhenxun.services.llm.session import AI
from zhenxun.services.llm.tools import RunContext
from zhenxun.services.llm.types import ToolExecutable
from zhenxun.services.llm.types.models import ToolDefinition
from zhenxun.services.llm.utils import extract_text_from_content
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy, model_dump_json

from .callbacks import BaseCallbackHandler, InteractiveCallbackHandler
from .executor import AgentExecutor
from .types import ExecutionConfig, MCPSource, ReviewerConfig, ToolFilter
from .utils import resolve_agent_tools


class Agent:
    """一个高级Agent的封装，持有配置并管理其生命周期内的资源。"""

    def __init__(
        self,
        name: str,
        instruction: str,
        model: ModelName | Callable[[], ModelName] = None,
        tools: list[str | MCPSource] | None = None,
        resources: list[str] | None = None,
        prompts: list[str] | None = None,
        config: LLMGenerationConfig | None = None,
        response_model: type[BaseModel] | None = None,
        reviewer: ReviewerConfig | None = None,
    ):
        self.name = name
        self.instruction = instruction
        self.model_name = model
        self.tool_definitions = tools or []
        self.tool_names = [t for t in (tools or []) if isinstance(t, str)]
        self.resources = resources or []
        self.response_model = response_model
        self.prompts = prompts or []
        self.reviewer = reviewer
        base_config = model_copy(config, deep=True) if config else LLMGenerationConfig()
        self.default_config = base_config
        self._resolved_tools: dict[str, ToolExecutable] | None = None

    async def _resolve_mcp_context(self) -> str:
        """
        根据配置的 MCP prompts/resources 构建额外的上下文。
        实际资源读取依赖 MCP Provider 的实现，这里提供接入点。
        """
        if not (self.prompts or self.resources):
            return ""

        from zhenxun.services.agent.providers.mcp import mcp_provider

        context_parts: list[str] = []

        if self.prompts:
            context_parts.append("--- Applied Prompts ---")
            for prompt_name in self.prompts:
                try:
                    fetch_prompt = getattr(mcp_provider, "get_prompt", None)
                    if callable(fetch_prompt):
                        fetch_func = cast(Callable[[str], Awaitable[Any]], fetch_prompt)
                        prompt_content = await fetch_func(prompt_name)
                        context_parts.append(
                            f"[Prompt {prompt_name}]\n{prompt_content}"
                        )
                    else:
                        context_parts.append(f"Applying prompt template: {prompt_name}")
                except Exception as exc:
                    logger.warning(
                        f"加载 MCP Prompt '{prompt_name}' 失败: {exc}", e=exc
                    )

        if self.resources:
            context_parts.append("--- Attached Resources ---")
            for resource_uri in self.resources:
                try:
                    read_resource = getattr(mcp_provider, "read_resource", None)
                    if callable(read_resource):
                        read_func = cast(Callable[[str], Awaitable[Any]], read_resource)
                        resource_content = await read_func(resource_uri)
                        context_parts.append(
                            f"[Resource {resource_uri}]\n{resource_content}"
                        )
                    else:
                        context_parts.append(f"Resource Attached: {resource_uri}")
                except Exception as exc:
                    logger.warning(
                        f"加载 MCP Resource '{resource_uri}' 失败: {exc}", e=exc
                    )

        return "\n".join(context_parts)

    async def _run_review_loop(
        self,
        initial_response: LLMResponse,
        context: AgentContext,
        executor: AgentExecutor,
        run_context: RunContext,
        messages_history: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """执行嵌套的审查-修正循环"""
        if not self.reviewer or not initial_response.text:
            return initial_response

        from zhenxun.services.agent import app

        reviewer_agent = app.get_agent(self.reviewer.agent_name)

        if not reviewer_agent:
            logger.warning(
                f"审查者 Agent '{self.reviewer.agent_name}' 未找到，跳过审查。"
            )
            return initial_response

        current_response = initial_response
        current_text = initial_response.text

        refinement_history = list(messages_history)
        refinement_history.append(LLMMessage(role="assistant", content=current_text))

        logger.info(
            f"🔄 [审查开始] Agent '{self.name}' 进入嵌套审查循环"
            f" (Reviewer: {self.reviewer.agent_name})"
        )

        for i in range(self.reviewer.max_turns):
            review_input = (
                f"这是原始问题：{context.user_input}\n"
                f"这是待审查的回答：\n{current_text}\n\n"
                f"{self.reviewer.prompt_template}"
            )

            review_context = AgentContext(
                session_id=f"{context.session_id}:review:{i}",
                user_input=review_input,
                scope=context.scope,
            )
            critique_result = await reviewer_agent.chat(
                context=review_context, **kwargs
            )
            if isinstance(critique_result, BaseModel):
                critique_text = model_dump_json(critique_result)
            else:
                critique_output = cast(LLMResponse, critique_result)
                critique_text = (critique_output.text or "").strip()

            if "PASS" in critique_text.upper() and len(critique_text) < 10:
                logger.info(f"✅ [审查通过] 在第 {i + 1} 轮通过审查。")
                return current_response

            logger.info(f"⚠️ [收到反馈] 第 {i + 1} 轮反馈: {critique_text[:50]}...")

            refinement_instruction = (
                f"收到审查反馈：{critique_text}\n请根据反馈修改你的回答。"
            )
            refinement_history.append(LLMMessage.user(refinement_instruction))

            refinement_msgs_result = await executor.run(
                refinement_history, context=run_context
            )

            final_msg = refinement_msgs_result[-1]
            if final_msg.role == "assistant":
                new_text = extract_text_from_content(final_msg.content)
                if new_text:
                    current_text = new_text
                    current_response = LLMResponse(
                        text=new_text, tool_calls=final_msg.tool_calls
                    )
                    refinement_history.append(final_msg)

        logger.warning(
            f"🛑 [审查结束] 达到最大修正轮数 ({self.reviewer.max_turns})，返回最终版本"
        )
        return current_response

    async def chat(
        self,
        context: AgentContext,
        matcher: Matcher | None = None,
        tool_filter: ToolFilter | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        config: ExecutionConfig | None = None,
        generation_config: LLMGenerationConfig | None = None,
    ) -> LLMResponse | BaseModel:
        """
        运行 Agent。自动从 context 中提取历史和变量进行 Prompt 渲染。
        """
        resolved_model_name = self.model_name
        if callable(resolved_model_name):
            resolved_model_name = resolved_model_name()

        if "bot" not in context.scope:
            try:
                bot: Bot | None = (
                    getattr(matcher, "bot", None) if matcher else current_bot.get()
                )
                context.scope["bot"] = bot
            except (LookupError, AttributeError):
                pass

        if "event" not in context.scope:
            try:
                event: Event | None = (
                    getattr(matcher, "event", None) if matcher else current_event.get()
                )
                context.scope["event"] = event
            except (LookupError, AttributeError):
                pass

        ai_session = AI(session_id=context.session_id)

        final_gen_config = model_copy(self.default_config, deep=True)
        if generation_config:
            final_gen_config = final_gen_config.merge_with(generation_config)

        resolved_tools_map = await resolve_agent_tools(self.tool_definitions)

        mcp_sources = [t for t in self.tool_definitions if isinstance(t, MCPSource)]
        if mcp_sources:
            for source in mcp_sources:
                has_tools_from_server = any(
                    getattr(tool, "server_name", None) == source.server_name
                    for tool in resolved_tools_map.values()
                    if hasattr(tool, "server_name")
                )
                if not has_tools_from_server:
                    error_msg = (
                        f"关键依赖缺失：无法加载 MCP 服务器 "
                        f"'{source.server_name}' 的工具。"
                        "Agent 执行已终止。"
                    )
                    logger.error(error_msg)
                    return LLMResponse(text=f"❌ {error_msg}")

        async with await get_model_instance(
            resolved_model_name,
            override_config=final_gen_config.to_dict(),
        ) as model_instance:
            effective_callbacks = callbacks.copy() if callbacks else []
            if matcher:
                effective_callbacks.append(InteractiveCallbackHandler(matcher))

            if tool_filter:
                pass

            temp_executor_for_discovery = AgentExecutor(
                model_instance, session=ai_session, tools_map=resolved_tools_map
            )
            effective_tools = await temp_executor_for_discovery._get_effective_tools()

            tool_descriptions = "\n--- 可用的工具 ---\n"

            definition_tasks = [
                tool.get_definition() for tool in effective_tools.values()
            ]
            definitions: list[ToolDefinition] = await asyncio.gather(*definition_tasks)

            for definition in definitions:
                tool_descriptions += (
                    f"- **{definition.name}**: {definition.description}\n"
                )

            resource_context = ""
            if self.prompts or self.resources:
                resource_context = await self._resolve_mcp_context()

            try:
                template = Template(self.instruction)
                render_context = {**context.scope}
                final_instruction = template.render(**render_context)
            except Exception as e:
                logger.warning(
                    f"Agent '{self.name}' Instruction 模板渲染失败: {e}", e=e
                )
                final_instruction = self.instruction
            if resource_context:
                final_instruction = f"{final_instruction}\n{resource_context}"
            final_instruction += tool_descriptions

            executor = AgentExecutor(
                model_instance,
                session=ai_session,
                tools_map=resolved_tools_map,
                callbacks=effective_callbacks,
            )

            run_context = RunContext(
                session_id=context.session_id,
                scope=context.scope,
                extra={"user_input": context.user_input},
            )

            history = await ai_session.memory.get_history(context.session_id)
            messages_for_run = [LLMMessage.system(final_instruction)]
            messages_for_run.extend(history)
            if context.user_input:
                messages_for_run.append(LLMMessage.user(context.user_input))

            if self.response_model:
                return await executor.run_structured(
                    messages_for_run,
                    response_model=self.response_model,
                    context=run_context,
                    config=config,
                    generation_config=final_gen_config,
                )

            final_history = await executor.run(
                messages_for_run, context=run_context, config=config
            )

            initial_response = None
            for msg in reversed(final_history):
                if msg.role == "assistant":
                    text = extract_text_from_content(msg.content)
                    initial_response = LLMResponse(text=text, tool_calls=msg.tool_calls)
                    break

            final_response = initial_response
            if self.reviewer and initial_response:
                final_response = await self._run_review_loop(
                    initial_response,
                    context,
                    executor,
                    run_context,
                    messages_for_run,
                    matcher=matcher,
                    tool_filter=tool_filter,
                    config=config,
                    generation_config=generation_config,
                )

            if context.user_input:
                await ai_session.add_user_message_to_history(context.user_input)

            if final_response and final_response.text:
                await ai_session.add_assistant_response_to_history(final_response.text)

            if final_response:
                return final_response

        raise LLMException(
            "Agent执行完毕但未产生任何回复。", code=LLMErrorCode.GENERATION_FAILED
        )
