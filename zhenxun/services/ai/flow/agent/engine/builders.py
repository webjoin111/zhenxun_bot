from contextlib import AsyncExitStack, asynccontextmanager
import copy
import inspect
from typing import Any

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.flow.agent.models import Persona
from zhenxun.services.ai.memory.interfaces import SessionMetadata
from zhenxun.services.ai.protocols.capabilities import CombinedCapability
from zhenxun.services.ai.run import DependencyInjector, RunContext, TemplateStr
from zhenxun.services.ai.tools.engine.registry import (
    ToolCollection,
    tool_provider_manager,
)
from zhenxun.services.ai.tools.models import GlobalToolFilter, ResolvedToolPayload


class ContextBuilder:
    """系统提示词与上下文记忆构建器"""

    @staticmethod
    async def build_system_prompt(
        instruction: str | TemplateStr,
        system_prompts: list[Any],
        run_context: RunContext,
        run_scoped_cap: CombinedCapability,
        persona: Persona | None = None,
    ) -> str:
        """解析系统提示词，结合动态函数、Jinja2模板与资源管理器"""
        dynamic_instructions = []
        sp_results = []

        for sp_func in system_prompts:
            sig = inspect.signature(sp_func)
            if len(sig.parameters) > 0:
                injected_kwargs = await DependencyInjector.resolve_all(
                    sig=sig,
                    call_kwargs={},
                    context=run_context,
                )
                res = (
                    (await sp_func(**injected_kwargs))
                    if is_coroutine_callable(sp_func)
                    else sp_func(**injected_kwargs)
                )
            else:
                res = (await sp_func()) if is_coroutine_callable(sp_func) else sp_func()
            if res:
                sp_results.append(str(res))

        if persona:
            persona_parts = [
                f"## 扮演角色 (Role)\n{persona.role}",
                f"## 核心目标 (Goal)\n{persona.goal}",
            ]
            if persona.backstory:
                persona_parts.append(f"## 角色背景 (Backstory)\n{persona.backstory}")
            dynamic_instructions.append("\n\n".join(persona_parts))

            if instruction:
                dynamic_instructions.append("## 本次任务指令 (Task)")

        if instruction:
            if isinstance(instruction, TemplateStr):
                dynamic_instructions.append(instruction.render(run_context))
            else:
                dynamic_instructions.append(str(instruction))

        dynamic_instructions.extend(sp_results)

        caps = (
            run_scoped_cap.capabilities
            if run_scoped_cap
            else getattr(run_context, "capabilities", [])
        )
        for cap in caps:
            cap_prompts = await cap.get_system_prompts(run_context)
            dynamic_instructions.extend(cap_prompts)

        final_instruction_text = "\n\n".join(dynamic_instructions)

        render_context = {
            "deps": run_context.deps,
            "bot": getattr(run_context.deps, "bot", None),
            "event": getattr(run_context.deps, "event", None),
            "matcher": getattr(run_context.deps, "matcher", None),
        }
        if run_context.state:
            render_context.update(run_context.state)

        final_instruction = PromptTemplate(final_instruction_text).render(
            **render_context
        )

        return final_instruction

    @staticmethod
    async def build_context_messages(
        model_name: str,
        user_input: Any | None,
        base_system_prompt: str,
        injected_prompts: list[str],
        session_metadata: SessionMetadata,
        memory_facade: Any,
        run_context: RunContext | None = None,
    ) -> list[LLMMessage]:
        """融合系统提示词、压缩后的历史记忆和用户输入，生成最终的消息数组"""
        system_prompt = base_system_prompt

        if injected_prompts:
            system_prompt += "\n\n--- 工具箱专属使用说明 ---\n\n"
            system_prompt += "\n\n".join(injected_prompts)

        messages_for_run: list[LLMMessage] = []
        if system_prompt:
            messages_for_run.append(LLMMessage.system(system_prompt))

        normalized_user_msg = None
        if user_input:
            from zhenxun.services.ai.message_builder import MessageBuilder

            bot_inst = run_context.get_bot() if run_context else None
            event_inst = run_context.get_event() if run_context else None

            msgs = await MessageBuilder.normalize_to_llm_messages(
                user_input, bot=bot_inst, event=event_inst
            )
            if msgs:
                normalized_user_msg = msgs[-1]

        current_history: list[LLMMessage] = []
        working_memory = memory_facade.working_memory if memory_facade else None

        if working_memory:
            current_history = await working_memory.get_history(session_metadata)

            from zhenxun.services.ai.config import get_llm_config
            from zhenxun.services.ai.memory.compression import CondenserPipeline
            from zhenxun.services.ai.memory.interfaces import BaseMemoryReducer

            config = get_llm_config().context_settings
            pipeline_reducers: list[BaseMemoryReducer] = []

            vw = config.vision_window_size
            if (
                memory_facade
                and getattr(memory_facade, "vision_window", None) is not None
            ):
                vw = memory_facade.vision_window
            if vw > 0:
                from zhenxun.services.ai.memory.compression import (
                    MultimodalPlaceholderReducer,
                )

                pipeline_reducers.append(MultimodalPlaceholderReducer(window_size=vw))

            policy = getattr(memory_facade, "policy", None) if memory_facade else None
            if policy is not None:
                pipeline_reducers.extend(policy)
            else:
                from zhenxun.services.ai.memory.policy import MemoryPolicy

                strategy = config.default_strategy
                s_kwargs = config.strategy_kwargs.get(strategy, {}).copy()

                threshold = config.trigger_threshold
                if (
                    memory_facade
                    and getattr(memory_facade, "context_threshold", None) is not None
                ):
                    threshold = memory_facade.context_threshold

                from zhenxun.services.ai.llm.capabilities import get_model_capabilities

                caps = get_model_capabilities(model_name)
                limit = (
                    int(caps.max_input_tokens * threshold)
                    if threshold <= 1.0
                    else int(threshold)
                )

                max_turns = config.max_history_turns
                if (
                    memory_facade
                    and getattr(memory_facade, "max_history_turns", None) is not None
                ):
                    max_turns = memory_facade.max_history_turns

                if strategy == "unlimited":
                    pipeline_reducers.extend(MemoryPolicy.unlimited())
                elif strategy == "llm_summary":
                    s_kwargs["trigger_tokens"] = limit
                    s_kwargs["max_turns"] = max_turns
                    pipeline_reducers.extend(MemoryPolicy.llm_summarize(**s_kwargs))
                elif strategy == "structured_summary":
                    s_kwargs["trigger_tokens"] = limit
                    s_kwargs["max_turns"] = max_turns
                    pipeline_reducers.extend(
                        MemoryPolicy.structured_summarize(**s_kwargs)
                    )
                else:
                    if max_turns is not None:
                        s_kwargs["max_turns"] = max_turns
                    pipeline_reducers.extend(MemoryPolicy.sliding_window(**s_kwargs))

            if pipeline_reducers:
                from zhenxun.services.log import logger

                pipeline = CondenserPipeline(pipeline_reducers)
                new_history, changed = await pipeline.run(
                    current_history,
                    model_name=model_name,
                    base_overhead=0,
                )

                if changed:
                    await working_memory.set_history(session_metadata, new_history)
                    logger.info(
                        f"💾 [上下文管线] 压缩/截断完毕，已同步覆写数据库。压缩后条数: {len(new_history)}"
                    )
                current_history = new_history

        messages_for_run.extend(current_history)

        if normalized_user_msg:
            messages_for_run.append(normalized_user_msg)

        return messages_for_run


class ToolBuilder:
    """系统工具集合解析与构建器"""

    @staticmethod
    async def resolve_tools(
        tool_definitions: list[Any],
        toolset_funcs: list[Any],
        system_tools: list[Any],
        namespace: str,
        tool_filter: GlobalToolFilter | None,
        run_context: RunContext,
        run_scoped_cap: CombinedCapability,
    ) -> ResolvedToolPayload:
        """解析、合并并过滤工具集"""
        defs_to_resolve = list(tool_definitions)

        for ts_func in toolset_funcs:
            sig = inspect.signature(ts_func)
            injected_kwargs = {}
            if len(sig.parameters) > 0:
                injected_kwargs = await DependencyInjector.resolve_all(
                    sig=sig,
                    call_kwargs={},
                    context=run_context,
                )

            res = (
                (await ts_func(**injected_kwargs))
                if is_coroutine_callable(ts_func)
                else ts_func(**injected_kwargs)
            )

            if res is not None:
                if isinstance(res, list):
                    defs_to_resolve.extend(res)
                else:
                    defs_to_resolve.append(res)

        if system_tools:
            for st in system_tools:
                if st not in defs_to_resolve:
                    defs_to_resolve.append(st)

        caps = (
            run_scoped_cap.capabilities
            if run_scoped_cap
            else getattr(run_context, "capabilities", [])
        )
        for cap in caps:
            cap_tools = await cap.get_tools(run_context)
            defs_to_resolve.extend(cap_tools)

        payload = await tool_provider_manager.resolve_tools(
            defs_to_resolve, namespace, context=run_context
        )

        return payload

    @staticmethod
    @asynccontextmanager
    async def mount_toolkits(toolkits: list[Any], session_id: str, context: RunContext):
        """安全挂载工具箱的生命周期，保障资源被正确回收"""
        async with AsyncExitStack() as stack:
            for tk in toolkits:
                if hasattr(tk, "enter_session"):
                    await tk.enter_session(session_id, context)
                    stack.push_async_callback(tk.exit_session, session_id)
            yield

    @staticmethod
    async def prepare_effective_tools(
        effective_tools: list[Any],
        context: RunContext,
        agent_prepare_tools: Any,
        run_scoped_cap: CombinedCapability,
    ) -> ToolCollection:
        """处理生命周期：在工具发往执行器前，进行最终的 Schema 拦截和清洗"""
        current_tool_defs = []
        for t_exec in effective_tools:
            if hasattr(t_exec, "get_definition"):
                t_def = await t_exec.get_definition(context)
                if t_def:
                    current_tool_defs.append(t_def)

        if agent_prepare_tools:
            _res = (
                await agent_prepare_tools(context, current_tool_defs)
                if is_coroutine_callable(agent_prepare_tools)
                else agent_prepare_tools(context, current_tool_defs)
            )
            if _res is not None:
                current_tool_defs = list(_res)

        _cap_res = await run_scoped_cap.prepare_tools(context, current_tool_defs)
        if _cap_res is not None:
            current_tool_defs = list(_cap_res)

        final_defs_map = {d.name.lower(): d for d in current_tool_defs if d}
        final_effective_tools = ToolCollection()
        for t_exec in effective_tools:
            t_name = getattr(t_exec, "name", "unknown")
            if t_name.lower() in final_defs_map:
                cloned_tool = copy.copy(t_exec)
                cloned_tool._dynamic_def = final_defs_map[t_name.lower()]
                final_effective_tools.append(cloned_tool)
        return final_effective_tools
