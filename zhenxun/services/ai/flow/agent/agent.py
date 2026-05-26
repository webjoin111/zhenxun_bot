import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, cast
from typing_extensions import Self

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.core.configs import (
    BaseOutputDefinition,
    GenerationConfig,
)
from zhenxun.services.ai.core.exceptions import (
    ControlFlowException,
)
from zhenxun.services.ai.core.messages import (
    LLMMessage,
    PromptInput,
    UsageInfo,
)
from zhenxun.services.ai.core.stream_events import EventStreamer
from zhenxun.services.ai.flow.agent.engine.builders import ContextBuilder, ToolBuilder
from zhenxun.services.ai.flow.agent.engine.executor import (
    AgentExecutor,
    AgentExecutorConfig,
)
from zhenxun.services.ai.flow.agent.models import (
    AgentRuntimeConfig,
    Persona,
)
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.knowledge.base import BaseKnowledge
from zhenxun.services.ai.llm.config.generation import IntentBuilder
from zhenxun.services.ai.llm.manager import get_model_instance
from zhenxun.services.ai.memory.builder import MemoryBuilder
from zhenxun.services.ai.memory.models import MemoryConfig, SessionMetadata
from zhenxun.services.ai.protocols.capabilities import (
    AbstractCapability,
    HitlCapability,
    SkillCapability,
)
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.run import (
    AgentDepsT,
    AgentRunResult,
    ExecutionConfig,
    OutputDataT,
    RunContext,
    Task,
    TemplateStr,
    ToolsPrepareFunc,
)
from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunError, AgentRunStart
from zhenxun.services.ai.tools.models import (
    GlobalToolFilter,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_construct, model_copy

if TYPE_CHECKING:
    from zhenxun.services.ai.run.models import StreamedRunResult


class Agent(
    BaseRunnable[AgentRunResult[OutputDataT]], Generic[AgentDepsT, OutputDataT]
):
    """
    Agent 运行时封装。
    负责组织模型、工具、记忆、护栏与能力插件，并驱动单轮或流式执行。
    """

    def __init__(
        self,
        name: str,
        instruction: str | TemplateStr = "",
        description: str | None = None,
        persona: Persona | dict | None = None,
        model: str | Callable[[], str] | None = None,
        tools: list | None = None,
        generation_config: GenerationConfig | IntentBuilder | dict | None = None,
        response_model: BaseOutputDefinition | type[OutputDataT] | None = None,
        dynamic_prompts: list[Callable] | None = None,
        memory: bool | MemoryConfig | MemoryBuilder = False,
        knowledge: BaseKnowledge | list[BaseKnowledge] | None = None,
        runtime_config: AgentRuntimeConfig | dict | None = None,
        prepare_tools: ToolsPrepareFunc | None = None,
        guardrails: list[Any] | None = None,
    ):
        """
        初始化 Agent。

        Args:
            name: Agent 名称，用于日志、事件和链路标识。
            instruction: 静态系统指令，可为普通字符串或模板字符串。
            persona: 可选人设配置；传入 dict 时会自动构造成 `Persona`。
            model: 默认模型名（如 `Provider/Model`）或返回模型名的回调。
            tools: 初始工具定义列表，可混用工具对象与字符串工具名。
            generation_config: 默认生成配置，支持 `GenerationConfig`、`IntentBuilder` 或 dict。
            response_model: 结构化输出模型；为空时按纯文本输出。
            dynamic_prompts: 动态系统提示词函数列表，运行时追加到系统提示。
            memory: 是否开启长期记忆与上下文压缩 (可传布尔值，或传入 AgentMemory)。
            knowledge: 挂载的知识库，支持单个或列表。底层会自动将其注册入工具链。
            runtime_config: 运行时行为配置；可传 `AgentRuntimeConfig` 或 dict。
            prepare_tools: 工具预处理钩子，在请求模型前可动态改写工具列表。
            guardrails: 护栏定义列表，支持可调用对象、规则字符串或护栏实例。
        """  # noqa: E501
        self.name = name

        if description:
            self.description = description
        elif persona:
            p_obj = persona if isinstance(persona, Persona) else Persona(**persona)
            self.description = f"角色：{p_obj.role}，目标：{p_obj.goal}"
        else:
            self.description = str(instruction)[:150] if instruction else "AI Agent"

        self.instruction = instruction

        if isinstance(persona, dict):
            self.persona = Persona(**persona)
        else:
            self.persona = persona
        self.model_name = model

        self.tool_definitions = tools or []

        if knowledge:
            if not isinstance(knowledge, list):
                knowledge = [knowledge]
            self.tool_definitions.extend(knowledge)

        from zhenxun.utils.utils import infer_plugin_namespace

        self.namespace = infer_plugin_namespace() or "unknown"

        self.tool_names = [t for t in (tools or []) if isinstance(t, str)]
        self.response_model = response_model
        if isinstance(generation_config, IntentBuilder):
            generation_config = generation_config.build()

        if isinstance(generation_config, dict):
            from zhenxun.utils.pydantic_compat import parse_as

            base_config = parse_as(GenerationConfig, generation_config)
        else:
            base_config = (
                model_copy(generation_config, deep=True)
                if generation_config
                else GenerationConfig()
            )
        self.default_config = base_config
        self._resolved_tools: dict[str, Any] | None = None

        self.dynamic_prompts = dynamic_prompts or []
        self.toolset_funcs = []
        from zhenxun.services.ai.core.guardrails import parse_guardrails

        self._guardrails = parse_guardrails(guardrails)
        self.prepare_tools = prepare_tools

        self.memory_config = MemoryBuilder.resolve(memory)

        if isinstance(runtime_config, dict):
            runtime_config = AgentRuntimeConfig(**runtime_config)
        self.runtime_config = runtime_config or AgentRuntimeConfig()

        self.runtime_config.stateless = not self.memory_config.short_term.enable

        self.capabilities: list[AbstractCapability] = []

        if self.runtime_config.enable_hitl:
            self.capabilities.append(HitlCapability())

        from zhenxun.services.ai.protocols.capabilities import ReflexionCapability

        self.capabilities.append(ReflexionCapability())

    def tool(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        settings: Any | None = None,
    ):
        """
        实例级工具注册装饰器。
        将普通函数绑定为该智能体的专属工具。
        """

        def decorator(f: Callable):
            from zhenxun.services.ai.tools.core.tool import FunctionTool
            from zhenxun.services.ai.tools.models import ToolOptions

            tool_name = name or f.__name__
            tool_desc = description or f.__doc__ or "未提供描述"
            base_settings = settings or getattr(f, "__tool_settings__", ToolOptions())

            func_tool = FunctionTool(
                func=f,
                name=tool_name,
                description=tool_desc,
                settings=base_settings,
            )
            if self.tool_definitions is None:
                self.tool_definitions = []
            self.tool_definitions.append(func_tool)
            return f

        return decorator if func is None else decorator(func)

    def system_prompt(self, func: Callable | None = None):
        """
        实例级动态系统提示词注册装饰器。支持依赖注入 (Inject.XXX)。
        被装饰函数可以接受 RunContext 及其他 Inject 依赖参数，返回字符串。
        """

        def decorator(f: Callable):
            if self.dynamic_prompts is None:
                self.dynamic_prompts = []
            self.dynamic_prompts.append(f)
            return f

        return decorator if func is None else decorator(func)

    def toolset(self, func: Callable | None = None):
        """
        实例级动态工具集注册装饰器。支持依赖注入 (Inject.XXX)。
        被装饰函数可以接受 RunContext 及其他 Inject 依赖参数，
        返回 BaseToolkit, list[BaseTool] 或 None。
        """

        def decorator(f: Callable):
            if getattr(self, "toolset_funcs", None) is None:
                self.toolset_funcs = []
            self.toolset_funcs.append(f)
            return f

        return decorator if func is None else decorator(func)

    def mount_private_skill(self, path: str | Path, as_catalog: bool = True) -> Self:
        """
        局部挂载私有技能。
        as_catalog=True: 作为元工具动态发现 (Meta 模式，大模型自主调用指令和脚本)
        as_catalog=False: 作为静态工具直接展开 (Static 模式，直接把脚本变成独立的工具)
        """
        from zhenxun.services.ai.tools.providers.skills.models import SkillMount

        mode = "meta" if as_catalog else "static"
        mount = SkillMount(path=Path(path), mode=mode)
        if self.tool_definitions is None:
            self.tool_definitions = []
        self.tool_definitions.append(mount)
        return self

    def load_skills(self, skills: list[str], as_catalog: bool = False) -> Self:
        """挂载底层技能栈"""
        cap = next(
            (c for c in self.capabilities if isinstance(c, SkillCapability)), None
        )
        if not cap:
            cap = SkillCapability()
            self.capabilities.append(cap)

        if as_catalog:
            cap.available_skills.extend(skills)
        else:
            cap.skills.extend(skills)
        return self

    def guardrail(self, func: Callable | str | Any | None = None):
        """护栏装饰器/注册器 (支持传入函数或自然语言风控规则字符串)"""
        if func is None:

            def decorator(f: Callable):
                from zhenxun.services.ai.core.guardrails import parse_guardrails

                self._guardrails.extend(parse_guardrails([f]))
                return f

            return decorator
        else:
            from zhenxun.services.ai.core.guardrails import parse_guardrails

            self._guardrails.extend(parse_guardrails([func]))
            return func

    async def __resolve_to_tools__(self) -> list[ToolExecutable]:
        """协议支持：将自身 Agent 转化为可被上级调用的工具"""
        from zhenxun.services.ai.tools.bridges.delegate import DelegateTool

        return [DelegateTool(self)]

    async def run(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        deps: AgentDepsT | None = None,
        context: RunContext[AgentDepsT] | None = None,
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: ExecutionConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        """
        智能体单次运行阻塞核心入口，内部使用上下文管理器静默消费事件流直至执行结束。

        参数:
            prompt: 用户输入的消息内容 or 标准数据契约任务对象 (Task)。
            deps: 强类型的外部依赖注入对象 (如 Bot, Event)。
            context: 显式传入的会话与运行上下文。
            message_history: 初始化的底层对话历史记录。
            tool_filter: 全局工具过滤器，用于限制本次运行可用的工具池。
            config: 核心执行引擎配置 (用于控制最大循环次数、并发调用等)。
            memory: 单次运行级别的记忆门面覆盖 (覆盖 __init__ 中的设定)。
            generation_config: 单次运行覆盖的大模型生成配置。
            kwargs: 透传的其他附加参数。

        返回:
            AgentRunResult[OutputDataT]: 包含最终输出数据、消息历史和用量统计的运行结果对象。
        """  # noqa: E501
        return await super().run(
            prompt=prompt,
            deps=deps,
            context=context,
            message_history=message_history,
            tool_filter=tool_filter,
            config=config,
            memory=memory,
            generation_config=generation_config,
            **kwargs,
        )

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        deps: AgentDepsT | None = None,
        context: RunContext[AgentDepsT] | None = None,
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: ExecutionConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        event_streamer: EventStreamer | None = None,
        **kwargs: Any,
    ) -> "AsyncIterator[StreamedRunResult[OutputDataT]]":
        """
        智能体流式运行入口。
        返回上下文管理器，可安全、解耦地获取底层事件或纯净文本结果。
        """
        from zhenxun.services.ai.run import StreamedRunResult

        streamer = event_streamer or EventStreamer()
        if context is None:
            explicit_session_id = kwargs.get("session_id")
            safe_context = RunContext[AgentDepsT](session_id=explicit_session_id)
            if deps is not None:
                safe_context.deps = cast(AgentDepsT, deps)
        else:
            safe_context = context
            if deps is not None and safe_context.deps is None:
                safe_context.deps = cast(AgentDepsT, deps)

        policy = getattr(self.runtime_config, "concurrency_policy", None)
        if policy is None:
            from zhenxun.services.ai.flow.base import ConcurrencyPolicy

            policy = (
                ConcurrencyPolicy.ALLOW
                if getattr(self.runtime_config, "stateless", True)
                else ConcurrencyPolicy.QUEUE
            )

        async def _execution_task():
            from zhenxun.services.ai.run.models import CancellationToken
            from zhenxun.services.ai.run.session_manager import session_manager

            cancel_token = safe_context.run.cancellation_token or CancellationToken()
            safe_context.run.cancellation_token = cancel_token

            try:
                async with session_manager.apply_concurrency_policy(
                    session_id=safe_context.session_id or "default_session",
                    policy=policy,
                    cancel_token=cancel_token,
                ):
                    await streamer.send(AgentRunStart(agent_name=self.name))
                    result = await self._run_step(
                        prompt=prompt,
                        context=safe_context,
                        message_history=message_history,
                        tool_filter=tool_filter,
                        config=config,
                        memory=memory,
                        generation_config=generation_config,
                        event_streamer=streamer,
                        **kwargs,
                    )
                    await streamer.send(AgentRunEnd(result=result))
            except ControlFlowException as e:
                await streamer.send(AgentRunError(error=e))
            except asyncio.CancelledError:
                from zhenxun.services.ai.core.exceptions import (
                    ConcurrencyInterruptException,
                )

                logger.debug(f"Agent {self.name} 执行被并发策略中断取消。")
                await streamer.send(
                    AgentRunError(
                        error=ConcurrencyInterruptException("任务已被新请求打断并接管")
                    )
                )
            except Exception as e:
                await streamer.send(AgentRunError(error=e))
            finally:
                await streamer.end()

        task = asyncio.create_task(_execution_task())
        result_obj = StreamedRunResult[OutputDataT](streamer)

        try:
            yield result_obj
        finally:
            if not task.done():
                task.cancel()

    def _parse_task_prompt(
        self, prompt: PromptInput | Task | None
    ) -> tuple[Task | None, Any | None, list[Any], Any, list[Any]]:
        """解析输入意图，提取数据契约 (Task)"""
        task_obj = None
        final_prompt_payload = None
        extra_tools = []
        run_output_type = self.response_model
        task_guardrails = []

        if isinstance(prompt, Task):
            task_obj = prompt
            if task_obj.response_model:
                run_output_type = task_obj.response_model
            if task_obj.tools:
                extra_tools.extend(task_obj.tools)
            if hasattr(task_obj, "_parsed_guardrails"):
                task_guardrails.extend(task_obj._parsed_guardrails)

            prompt_parts = [
                f"### 📋 [任务指令]\n{task_obj.description}",
                f"### 🎯 [预期产出要求]\n{task_obj.expected_output}",
            ]
            final_prompt_payload = "\n\n".join(prompt_parts)
        elif prompt is not None:
            final_prompt_payload = prompt

        return (
            task_obj,
            final_prompt_payload,
            extra_tools,
            run_output_type,
            task_guardrails,
        )

    async def _run_step(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        context: RunContext[AgentDepsT],
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: ExecutionConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        cancellation_token: Any = None,
        event_streamer: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        """执行原子步代理逻辑"""
        import uuid

        if not context.session_id:
            context.session_id = f"ag-run-{uuid.uuid4()}"
            context.session.session_id = context.session_id

        is_stateless = getattr(self.runtime_config, "stateless", True)
        if not is_stateless and getattr(context, "_is_auto_session_id", False):
            bot = context.get_bot()
            event = context.get_event()
            if bot and event:
                from zhenxun.services.ai.memory.utils import generate_session_meta

                isolation_level = getattr(self.runtime_config, "isolation_level", None)
                if isolation_level is None:
                    if self.memory_config:
                        isolation_level = self.memory_config.short_term.isolation_level
                    else:
                        from zhenxun.services.ai.memory.models import (
                            MemoryIsolationLevel,
                        )

                        isolation_level = MemoryIsolationLevel.GROUP_USER

                _meta = generate_session_meta(
                    bot,
                    event,
                    isolation_level=isolation_level,
                    namespace=self.namespace,
                    agent_name=self.name,
                )
                context.session_id = _meta.session_id
                context.session.session_id = _meta.session_id

        from zhenxun.services.ai.protocols.capabilities import CombinedCapability
        from zhenxun.services.ai.tools.engine.global_capabilities import (
            GLOBAL_CAPABILITIES,
        )

        (
            task_obj,
            final_prompt_payload,
            extra_tools,
            run_output_type,
            task_guardrails,
        ) = self._parse_task_prompt(prompt)

        dynamic_caps = []
        combined_guardrails = self._guardrails + task_guardrails

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

        from zhenxun.utils.pydantic_compat import model_copy

        effective_memory = model_copy(self.memory_config, deep=True)
        if memory is not None:
            if isinstance(memory, bool):
                effective_memory.short_term.enable = memory
            elif isinstance(memory, MemoryBuilder):
                effective_memory = memory.build()
            else:
                effective_memory = memory

        session_metadata = SessionMetadata(session_id=context.session_id)

        from zhenxun.services.ai.memory.engine import MemoryReader, MemoryWriter

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

            dynamic_caps.append(TaskTrackingCapability(task_obj, self.name))

        combined_cap = CombinedCapability(
            GLOBAL_CAPABILITIES
            + getattr(context, "capabilities", [])
            + self.capabilities
            + dynamic_caps
        )
        run_scoped_cap = cast(CombinedCapability, await combined_cap.for_run(context))

        await run_scoped_cap.before_run(context)

        original_capabilities = getattr(context, "capabilities", [])
        context.capabilities = run_scoped_cap.capabilities

        try:
            if final_prompt_payload is not None:
                if isinstance(final_prompt_payload, str):
                    context.run.user_input = final_prompt_payload
                elif hasattr(final_prompt_payload, "extract_plain_text"):
                    context.run.user_input = final_prompt_payload.extract_plain_text()
                else:
                    context.run.user_input = str(final_prompt_payload)
            context.run.agent_name = self.name

            long_term_fact = ""
            if context.run.user_input:
                long_term_fact = await reader.get_long_term_context(
                    context.run.user_input
                )

            system_prompt = await ContextBuilder.build_system_prompt(
                instruction=self.instruction,
                system_prompts=self.dynamic_prompts,
                run_context=context,
                run_scoped_cap=run_scoped_cap,
                persona=self.persona,
            )

            if long_term_fact:
                system_prompt += f"\n\n{long_term_fact}"

            tool_payload = await ToolBuilder.resolve_tools(
                tool_definitions=self.tool_definitions,
                toolset_funcs=getattr(self, "toolset_funcs", []),
                system_tools=(
                    getattr(self.default_config, "system_tools", [])
                    if self.default_config
                    else []
                ),
                namespace=self.namespace or "unknown",
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
                    tool_payload.injected_prompts.extend(
                        mem_tk_payload.injected_prompts
                    )
                    tool_payload.toolkits.extend(mem_tk_payload.toolkits)

            final_gen_config = model_copy(self.default_config, deep=True)

            cap_dynamic_config = await run_scoped_cap.get_generation_config(context)
            if cap_dynamic_config:
                final_gen_config = final_gen_config.merge_with(cap_dynamic_config)

            if generation_config:
                final_gen_config = final_gen_config.merge_with(generation_config)
            exec_config = config or ExecutionConfig()
            model_name_resolved = (
                self.model_name() if callable(self.model_name) else self.model_name
            )

            if not model_name_resolved and context and context.run.current_model:
                model_name_resolved = context.run.current_model

            if not model_name_resolved:
                from zhenxun.services.ai.llm.manager import get_default_model

                model_name_resolved = get_default_model("chat")

            context.run.current_model = (
                str(model_name_resolved) if model_name_resolved else ""
            )
            context.run.cancellation_token = cancellation_token
            context.run.streamer = event_streamer

            if tool_payload.injected_prompts:
                system_prompt += "\n\n--- 工具箱专属使用说明 ---\n\n"
                system_prompt += "\n\n".join(tool_payload.injected_prompts)

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
                model_name=str(model_name_resolved) if model_name_resolved else "",
                override_history=message_history,
            )

            if system_prompt:
                messages_for_run.insert(0, LLMMessage.system(system_prompt))
            if normalized_user_msg:
                messages_for_run.append(normalized_user_msg)
                await writer.save_new_messages([normalized_user_msg])

            async with ToolBuilder.mount_toolkits(
                tool_payload.toolkits, context.session_id, context
            ):
                for tk in tool_payload.toolkits:
                    if hasattr(tk, "before_llm_request"):
                        if is_coroutine_callable(tk.before_llm_request):
                            await tk.before_llm_request(context, messages_for_run)
                        else:
                            tk.before_llm_request(context, messages_for_run)

                final_tools = await ToolBuilder.prepare_effective_tools(
                    effective_tools, context, self.prepare_tools, run_scoped_cap
                )

                executor = AgentExecutor(
                    tools=final_tools,
                    config=AgentExecutorConfig(
                        max_cycles=exec_config.max_cycles,
                        reflexion_retries=exec_config.reflexion_retries,
                        enable_fallback_summary=exec_config.enable_fallback_summary,
                    ),
                )

                async with await get_model_instance(
                    str(model_name_resolved) if model_name_resolved else None,
                    override_config=None,
                ) as instance:
                    _run_result: Any = await executor.run(
                        messages=messages_for_run,
                        model_instance=instance,
                        run_context=context,
                        generation_config=final_gen_config,
                        cancellation_token=cancellation_token,
                        event_streamer=event_streamer,
                    )
                    final_messages = _run_result.messages
                    structured_data = _run_result.structured_data
                    final_usage = getattr(_run_result, "usage", None)
                    early_output = getattr(_run_result, "output", None)

            new_msgs = final_messages[len(messages_for_run) :]

            await writer.save_new_messages(new_msgs)

            last_msg = final_messages[-1]
            final_text = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else " ".join(
                    p.text for p in last_msg.content if p.type == "text" and p.text
                )
            )

            usage = final_usage or UsageInfo()

            final_output = early_output if early_output is not None else final_text

            raw_result = cast(
                AgentRunResult[OutputDataT],
                model_construct(
                    AgentRunResult,
                    output=final_output,
                    messages=new_msgs,
                    structured_data=structured_data,
                    usage=usage,
                ),
            )
            return await run_scoped_cap.after_run(context, raw_result)

        except ControlFlowException as e:
            raise e
        except Exception as e:
            logger.error(f"Agent '{self.name}' 运行失败: {e}", e=e)
            try:
                return await run_scoped_cap.on_run_error(context, e)
            except Exception as final_e:
                raise final_e
        finally:
            context.capabilities = original_capabilities
