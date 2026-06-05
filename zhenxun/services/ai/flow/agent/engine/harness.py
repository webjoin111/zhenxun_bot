from typing import Any, cast
import uuid

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.flow.agent.engine.builders import ContextBuilder, ToolBuilder
from zhenxun.services.ai.flow.agent.models import AgentLoopConfig, AgentLoopContext
from zhenxun.services.ai.memory.engine import MemoryReader, MemoryWriter
from zhenxun.services.ai.memory.models import (
    MemoryConfig,
)
from zhenxun.services.ai.memory.types import MemoryIsolationLevel, SessionMetadata
from zhenxun.services.ai.memory.utils import generate_session_meta
from zhenxun.services.ai.protocols.capabilities import (
    AbstractCapability,
    CombinedCapability,
    DynamicCapability,
)
from zhenxun.services.ai.run import ExecutionConfig, RunContext
from zhenxun.services.ai.tools.engine.global_capabilities import GLOBAL_CAPABILITIES
from zhenxun.utils.pydantic_compat import model_copy


class AgentHarness:
    """
    Agent 装配线 (Harness)。
    负责将松散的配置、记忆、工具链组装成严谨的 AgentLoopContext 和 AgentLoopConfig。
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def normalize_session(self, context: RunContext) -> None:
        """处理上下文会话 ID 初始化与状态隔离"""
        if not context.session_id:
            context.session_id = f"ag-run-{uuid.uuid4()}"
            context.session.session_id = context.session_id

        is_stateless = getattr(self.agent.runtime_config, "stateless", True)
        if not is_stateless and getattr(context, "_is_auto_session_id", False):
            bot = context.get_bot()
            event = context.get_event()
            if bot and event:
                isolation_level = getattr(
                    self.agent.runtime_config, "isolation_level", None
                )
                if isolation_level is None:
                    if self.agent.memory_config:
                        isolation_level = (
                            self.agent.memory_config.short_term.isolation_level
                        )
                    else:
                        isolation_level = MemoryIsolationLevel.GROUP_USER

                _meta = generate_session_meta(
                    bot,
                    event,
                    isolation_level=isolation_level,
                    namespace=self.agent.namespace,
                    agent_name=self.agent.name,
                )
                context.session_id = _meta.session_id
                context.session.session_id = _meta.session_id

    async def prepare_loop(
        self,
        prompt: Any = None,
        context: RunContext | None = None,
        message_history: list[LLMMessage] | None = None,
        tool_filter: Any = None,
        config: ExecutionConfig | None = None,
        memory: bool | MemoryConfig | Any | None = None,
        generation_config: GenerationConfig | None = None,
        cancellation_token: Any = None,
        event_streamer: Any = None,
        capabilities: list[Any] | None = None,
    ) -> tuple[
        AgentLoopContext,
        AgentLoopConfig,
        MemoryWriter,
        list[Any],
        CombinedCapability,
        int,
    ]:
        """
        执行 Agent 启动前的全部装配工作。
        返回: loop_ctx, loop_config, writer, toolkits, run_scoped_cap, origin_msg_len
        """
        assert context is not None
        self.normalize_session(context)

        (
            task_obj,
            final_prompt_payload,
            extra_tools,
            run_output_type,
            task_guardrails,
        ) = self.agent._parse_task_prompt(prompt)

        dynamic_caps = []
        combined_guardrails = self.agent._guardrails + task_guardrails

        if run_output_type is not None and run_output_type is not str:
            from zhenxun.services.ai.flow.agent.capabilities import (
                OutputValidationCapability,
            )

            dynamic_caps.append(
                OutputValidationCapability(run_output_type, combined_guardrails)
            )
        elif combined_guardrails:
            from zhenxun.services.ai.flow.agent.capabilities import (
                OutputValidationCapability,
            )

            dynamic_caps.append(OutputValidationCapability(None, combined_guardrails))

        from zhenxun.services.ai.memory.builder import MemoryBuilder

        effective_memory = (
            MemoryBuilder.resolve(memory)
            if memory is not None
            else model_copy(self.agent.memory_config, deep=True)
        )

        session_metadata = SessionMetadata(
            session_id=context.session_id or "default_session",
            user_id=context.get_user_id(),
            group_id=context.get_group_id(),
            platform=context.get_platform(),
            namespace=self.agent.namespace,
            agent_name=self.agent.name,
        )

        reader = MemoryReader(
            session_meta=session_metadata, memory_config=effective_memory
        )
        writer = MemoryWriter(
            session_meta=session_metadata, memory_config=effective_memory
        )

        if task_obj:
            from zhenxun.services.ai.flow.agent.capabilities import (
                TaskTrackingCapability,
            )

            dynamic_caps.append(TaskTrackingCapability(task_obj, self.agent.name))

        run_level_caps = []
        if capabilities:
            for cap in capabilities:
                if isinstance(cap, AbstractCapability):
                    run_level_caps.append(cap)
                elif callable(cap):
                    run_level_caps.append(DynamicCapability(cap))

        base_caps = GLOBAL_CAPABILITIES.get("global", []).copy()
        if (
            self.agent.namespace != "global"
            and self.agent.namespace in GLOBAL_CAPABILITIES
        ):
            base_caps.extend(GLOBAL_CAPABILITIES[self.agent.namespace])

        combined_cap = CombinedCapability(
            base_caps
            + getattr(context, "capabilities", [])
            + self.agent.capabilities
            + run_level_caps
            + dynamic_caps
        )
        run_scoped_cap = cast(CombinedCapability, await combined_cap.for_run(context))

        if final_prompt_payload is not None:
            if isinstance(final_prompt_payload, str):
                context.run.user_input = final_prompt_payload
            elif hasattr(final_prompt_payload, "extract_plain_text"):
                context.run.user_input = final_prompt_payload.extract_plain_text()
            else:
                context.run.user_input = str(final_prompt_payload)
        context.run.agent_name = self.agent.name
        context.run.cancellation_token = cancellation_token
        context.run.streamer = event_streamer

        model_name_resolved = (
            self.agent.model_name()
            if callable(self.agent.model_name)
            else self.agent.model_name
        )
        if not model_name_resolved and context and context.run.current_model:
            model_name_resolved = context.run.current_model
        if not model_name_resolved:
            from zhenxun.services.ai.llm.manager import get_default_model

            model_name_resolved = get_default_model("chat")
        context.run.current_model = (
            str(model_name_resolved) if model_name_resolved else ""
        )

        long_term_fact = ""
        if context.run.user_input:
            long_term_fact = await reader.get_long_term_context(context.run.user_input)
        slots_fact = await reader.get_slots_context()

        static_prompt, dynamic_prompt = await ContextBuilder.build_prompts(
            instruction=self.agent.instruction,
            system_prompts=self.agent.dynamic_prompts,
            run_context=context,
            run_scoped_cap=run_scoped_cap,
            persona=self.agent.persona,
        )

        if long_term_fact:
            dynamic_prompt += f"\n\n{long_term_fact}"
        if slots_fact:
            dynamic_prompt += f"\n\n{slots_fact}"

        tool_payload = await ToolBuilder.resolve_tools(
            tool_definitions=self.agent.tool_definitions,
            toolset_funcs=getattr(self.agent, "toolset_funcs", []),
            system_tools=getattr(self.agent.default_config, "system_tools", [])
            if self.agent.default_config
            else [],
            namespace=self.agent.namespace or "unknown",
            tool_filter=tool_filter,
            run_context=context,
            run_scoped_cap=run_scoped_cap,
        )
        effective_tools = tool_payload.tools
        if extra_tools:
            effective_tools.extend(extra_tools)

        if effective_memory and effective_memory.long_term.enable:
            from zhenxun.services.ai.memory.manager import memory_manager

            ltm_scope = memory_manager.get_long_term_memory(effective_memory)
            if ltm_scope:
                from zhenxun.services.ai.tools.providers.builtin.memory import (
                    MemoryManagementToolkit,
                )

                mem_tk_payload = await MemoryManagementToolkit(
                    memory_scope=ltm_scope, session_meta=session_metadata
                ).resolve(context)
                effective_tools.extend(mem_tk_payload.tools)
                tool_payload.injected_prompts.extend(mem_tk_payload.injected_prompts)
                tool_payload.toolkits.extend(mem_tk_payload.toolkits)

        if effective_memory and effective_memory.slots.enable:
            from zhenxun.services.ai.tools.providers.builtin.slots import (
                MemorySlotToolkit,
            )

            slot_tk_payload = await MemorySlotToolkit(
                session_meta=session_metadata, memory_config=effective_memory
            ).resolve(context)
            effective_tools.extend(slot_tk_payload.tools)
            tool_payload.injected_prompts.extend(slot_tk_payload.injected_prompts)
            tool_payload.toolkits.extend(slot_tk_payload.toolkits)

        final_gen_config = model_copy(self.agent.default_config, deep=True)
        cap_dynamic_config = await run_scoped_cap.get_generation_config(context)
        if cap_dynamic_config:
            final_gen_config = final_gen_config.merge_with(cap_dynamic_config)
        if generation_config:
            final_gen_config = final_gen_config.merge_with(generation_config)

        exec_config = config or ExecutionConfig()

        if tool_payload.injected_prompts:
            static_prompt += "\n\n--- 工具箱专属使用说明 ---\n\n"
            static_prompt += "\n\n".join(tool_payload.injected_prompts)

        normalized_user_msg = None
        if final_prompt_payload is not None:
            from zhenxun.services.ai.message_builder import MessageBuilder

            bot_inst = context.get_bot()
            event_inst = context.get_event()
            msgs = await MessageBuilder.normalize_to_llm_messages(
                final_prompt_payload, bot=bot_inst, event=event_inst
            )
            if msgs:
                normalized_user_msg = msgs[-1]

        messages_for_run = await reader.get_short_term_context(
            model_name=context.run.current_model,
            override_history=message_history,
        )

        if normalized_user_msg:
            messages_for_run.append(normalized_user_msg)
            await writer.save_new_messages([normalized_user_msg])

        origin_msg_len = len(messages_for_run)

        final_tools = await ToolBuilder.prepare_effective_tools(
            effective_tools, context, self.agent.prepare_tools, run_scoped_cap
        )

        context.session.append_only_manager.build([static_prompt], final_tools)
        context.session.append_only_manager.sync_messages(messages_for_run)

        loop_ctx = AgentLoopContext(
            messages=messages_for_run,
            tools=final_tools,
            run_context=context,
            static_system_prompt=static_prompt,
            dynamic_system_prompt=dynamic_prompt,
        )

        loop_config = AgentLoopConfig(
            model_name=context.run.current_model,
            generation_config=final_gen_config,
            max_cycles=exec_config.max_cycles,
            reflexion_retries=exec_config.reflexion_retries,
            enable_fallback_summary=exec_config.enable_fallback_summary,
            cancellation_token=cancellation_token,
            event_streamer=event_streamer,
        )

        return (
            loop_ctx,
            loop_config,
            writer,
            tool_payload.toolkits,
            run_scoped_cap,
            origin_msg_len,
        )

    async def post_loop(
        self,
        loop_ctx: AgentLoopContext,
        raw_result: Any,
        writer: MemoryWriter,
        origin_msg_len: int,
    ) -> Any:
        """处理 Loop 结束后的扫尾工作，如提取文本、持久化消息"""
        from zhenxun.services.ai.core.messages import UsageInfo
        from zhenxun.services.ai.run.models import AgentRunResult
        from zhenxun.utils.pydantic_compat import model_construct

        final_messages = raw_result.messages
        new_msgs = final_messages[origin_msg_len:]
        await writer.save_new_messages(new_msgs)

        last_msg = final_messages[-1] if final_messages else None
        final_text = ""
        if last_msg:
            final_text = last_msg.extract_text

        early_output = getattr(raw_result, "output", None)
        final_output = early_output if early_output is not None else final_text

        return model_construct(
            AgentRunResult,
            output=final_output,
            messages=new_msgs,
            structured_data=getattr(raw_result, "structured_data", None),
            usage=getattr(raw_result, "usage", None) or UsageInfo(),
        )
