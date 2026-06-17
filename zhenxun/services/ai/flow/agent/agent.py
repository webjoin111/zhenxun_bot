import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
import contextlib
from pathlib import Path
from typing import Any, Generic, cast

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.capabilities import (
    AbstractCapability,
    CombinedCapability,
    DynamicCapability,
)
from zhenxun.services.ai.context.knowledge.base import BaseKnowledge
from zhenxun.services.ai.context.memory.builder import MemoryBuilder
from zhenxun.services.ai.context.memory.models import MemoryConfig
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
)
from zhenxun.services.ai.core.messages import (
    PromptInput,
)
from zhenxun.services.ai.core.options import (
    BaseOutputDefinition,
    GenerationConfig,
)
from zhenxun.services.ai.core.protocols.tool import ToolExecutable, ToolResolvable
from zhenxun.services.ai.core.stream_events import EventStreamer
from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.flow.agent.capabilities import (
    OutputValidationCapability,
    TaskTrackingCapability,
)
from zhenxun.services.ai.flow.agent.engine.builders import ToolBuilder
from zhenxun.services.ai.flow.agent.models import (
    AgentRunProfile,
    AgentSettings,
    AgentState,
    Persona,
)
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.guardrails import GuardrailSource
from zhenxun.services.ai.llm.config.generation import IntentBuilder
from zhenxun.services.ai.run import (
    AgentRunResult,
    RunContext,
    Task,
)
from zhenxun.services.ai.run.context import AgentDepsT, ToolsPrepareFunc
from zhenxun.services.ai.run.models import (
    AgentRunEnd,
    AgentRunError,
    AgentRunStart,
    OutputDataT,
    StreamedRunResult,
)
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.providers.skills.models import Skill, SkillSource
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy
from zhenxun.utils.utils import infer_plugin_namespace

ToolSource = Callable | BaseTool | dict[str, Any] | str | BaseToolkit | ToolResolvable
"""任何可以作为工具提供给大模型的实体对象（函数、基础工具类、字典定义、工具名、工具箱）"""

CapabilitySource = Callable | AbstractCapability
"""能力/拦截器来源（函数或 AbstractCapability 实例）"""


class AgentBuilder(Generic[AgentDepsT, OutputDataT]):
    """
    Agent 链式构建器 (Fluent Builder)。
    """

    def __init__(self, name: str):
        self._name = name
        self._instruction: str | PromptTemplate = ""
        self._description: str | None = None
        self._persona: Persona | dict | None = None
        self._model: str | Callable[[], str] | None = None
        self._tools: list[ToolSource] = []
        self._skills: list[str | Path | Skill | SkillSource] = []
        self._generation_config: GenerationConfig | IntentBuilder | dict | None = None
        self._response_model: BaseOutputDefinition | type[OutputDataT] | None = None
        self._dynamic_prompts: list[Callable] = []
        self._memory: bool | MemoryConfig | MemoryBuilder = False
        self._knowledge: list[BaseKnowledge] = []
        self._settings: AgentSettings | dict | None = None
        self._guardrails: list[GuardrailSource] = []
        self._capabilities: list[CapabilitySource] = []
        self._executor: Any | None = None

    def with_instruction(
        self, instruction: str | PromptTemplate
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置静态系统指令。

        参数:
            instruction: 静态系统指令，可为普通字符串或模板字符串。
        """
        self._instruction = instruction
        return self

    def with_persona(
        self, role: str, goal: str, backstory: str | None = None
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置智能体人设与角色设定。

        参数:
            role: 扮演的角色身份。
            goal: 角色的核心目标。
            backstory: 角色背景故事或性格设定。
        """
        self._persona = Persona(role=role, goal=goal, backstory=backstory)
        return self

    def with_model(
        self, model: str | Callable[[], str]
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置默认调用的语言模型。

        参数:
            model: 默认模型名（如 `Provider/Model`）或返回模型名的回调。
        """
        self._model = model
        return self

    def with_tools(
        self, *tools: ToolSource | list[ToolSource]
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置可供智能体调用的工具列表。

        参数:
            tools: 初始工具定义，支持工具对象、函数、字典定义或工具名称。
        """
        for t in tools:
            if isinstance(t, list):
                self._tools.extend(t)
            else:
                self._tools.append(t)
        return self

    def with_skills(
        self, *skills: str | Path | Skill | SkillSource | Sequence
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置注入的领域知识技能。

        参数:
            skills: 注入的技能，支持 ID、目录 Path、Skill 对象或 SkillSource 动态源。
        """
        for s in skills:
            if isinstance(s, (list, tuple, set)):
                self._skills.extend(s)
            else:
                self._skills.append(cast(Any, s))
        return self

    def with_knowledge(
        self, *knowledge: BaseKnowledge | list[BaseKnowledge]
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置挂载的知识库。

        参数:
            knowledge: 挂载的知识库，支持单个或列表。底层会自动将其注册入工具链。
        """
        for k in knowledge:
            if isinstance(k, list):
                self._knowledge.extend(k)
            else:
                self._knowledge.append(k)
        return self

    def with_memory(
        self, memory: bool | MemoryConfig | MemoryBuilder
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置对话记忆与上下文管理策略。

        参数:
            memory: 是否开启长期记忆与上下文压缩，支持布尔值或显式配置对象。
        """
        self._memory = memory
        return self

    def with_generation_config(
        self, config: GenerationConfig | IntentBuilder | dict
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置大模型基础生成参数。

        参数:
            config: 默认生成配置，支持 `GenerationConfig`、`IntentBuilder` 或 dict。
        """
        self._generation_config = config
        return self

    def with_response_model(
        self, response_model: BaseOutputDefinition | type[Any]
    ) -> "AgentBuilder[AgentDepsT, Any]":
        """
        配置期望大模型输出的强类型结构化数据模型。

        参数:
            response_model: 结构化输出模型，传入 Pydantic 模型类或声明式输出对象。
        """
        self._response_model = response_model
        return cast(AgentBuilder[AgentDepsT, Any], self)

    def with_guardrails(
        self, *guardrails: GuardrailSource | list[GuardrailSource]
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置输入/输出安全合规护栏。

        参数:
            guardrails: 护栏定义，支持可调用对象、自然语言规则字符串或护栏实例。
        """
        for g in guardrails:
            if isinstance(g, list):
                self._guardrails.extend(g)
            else:
                self._guardrails.append(g)
        return self

    def with_capabilities(
        self, *capabilities: CapabilitySource | list[CapabilitySource]
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置智能体的高阶能力拦截器组件。

        参数:
            capabilities: 能力组件，可传入函数或 `AbstractCapability` 实例。
        """
        for c in capabilities:
            if isinstance(c, list):
                self._capabilities.extend(c)
            else:
                self._capabilities.append(c)
        return self

    def with_config(
        self, config: AgentSettings | dict | None = None, **kwargs
    ) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置智能体全局设置与宏观策略。

        参数:
            config: 宏观配置，合并了运行时策略与引擎执行策略，可传入 `AgentSettings` 或 dict。
            kwargs: 零散的配置参数，将自动覆盖或组装进配置对象中。
        """  # noqa: E501
        from zhenxun.utils.pydantic_compat import model_dump

        merged_kwargs = {}
        if config:
            merged_kwargs.update(
                config if isinstance(config, dict) else model_dump(config)
            )
        merged_kwargs.update(kwargs)

        self._settings = AgentSettings(**merged_kwargs)
        return self

    def with_executor(self, executor: Any) -> "AgentBuilder[AgentDepsT, OutputDataT]":
        """
        配置核心思考大循环的执行策略。

        参数:
            executor: 实现 BaseAgentExecutor 接口的实例。

        返回:
            AgentBuilder[AgentDepsT, OutputDataT]: 构建器自身。
        """
        self._executor = executor
        return self

    def build(self) -> "Agent[AgentDepsT, OutputDataT]":
        """
        构建并输出最终 of Agent 实例。
        """
        return Agent(
            name=self._name,
            instruction=self._instruction,
            description=self._description,
            persona=self._persona,
            model=self._model,
            tools=self._tools,
            skills=self._skills,
            generation_config=self._generation_config,
            response_model=self._response_model,
            dynamic_prompts=self._dynamic_prompts,
            memory=self._memory,
            knowledge=self._knowledge,
            settings=self._settings,
            guardrails=self._guardrails,
            capabilities=self._capabilities,
            executor=self._executor,
        )


class Agent(
    BaseRunnable[AgentRunResult[OutputDataT]], Generic[AgentDepsT, OutputDataT]
):
    """
    Agent 运行时封装。
    负责组织模型、工具、记忆、护栏与能力插件，并驱动单轮或流式执行。
    """

    @classmethod
    def builder(cls, name: str) -> AgentBuilder[Any, str]:
        """创建一个智能体链式构建器"""
        return AgentBuilder(name=name)

    def __init__(
        self,
        name: str,
        instruction: str | PromptTemplate = "",
        description: str | None = None,
        persona: Persona | dict | None = None,
        model: str | Callable[[], str] | None = None,
        tools: list[ToolSource] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        generation_config: GenerationConfig | IntentBuilder | dict | None = None,
        response_model: BaseOutputDefinition | type[OutputDataT] | None = None,
        dynamic_prompts: list[Callable] | None = None,
        memory: bool | MemoryConfig | MemoryBuilder = False,
        knowledge: BaseKnowledge | list[BaseKnowledge] | None = None,
        settings: AgentSettings | dict | None = None,
        prepare_tools: ToolsPrepareFunc | None = None,
        guardrails: list[GuardrailSource] | None = None,
        capabilities: list[CapabilitySource] | None = None,
        executor: Any | None = None,
    ):
        """
        初始化 Agent。

        参数:
            name: Agent 名称，用于日志、事件和链路标识。
            instruction: 静态系统指令，可为普通字符串或模板字符串。
            description: 智能体描述，用于外部路由节点决定是否调用。
            persona: 角色设定配置，传入 dict 会自动构造成 Persona。
            model: 默认模型名称 (如 Provider/Model) 或返回模型名的回调。
            tools: 初始工具定义列表，支持混合使用工具对象与字符串工具名。
            skills: 注入的领域知识技能，支持 ID、目录 Path、Skill 对象或动态源。
            generation_config: 默认生成配置，支持 GenerationConfig、IntentBuilder 或 dict。
            response_model: 结构化输出模型，若为空则按纯文本输出。
            dynamic_prompts: 动态系统提示词函数列表，运行时追加到系统提示。
            memory: 是否开启长期记忆与上下文压缩，支持布尔值或 MemoryBuilder/Config。
            knowledge: 挂载的知识库，支持单个或列表，底层自动将其注册入工具链。
            runtime_config: 运行时行为配置，控制是否无状态、HITL等，支持字典。
            engine_config: 核心执行引擎配置，控制最大循环次数、并发等，支持字典。
            prepare_tools: 工具预处理钩子，在请求模型前可动态改写发往 LLM 的工具列表。
            guardrails: 护栏定义列表，支持可调用对象、规则字符串或护栏实例。
            capabilities: 拦截器/能力插件列表，处理整个生命周期的切面逻辑。
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
        from zhenxun.services.ai.guardrails import parse_guardrails

        self._guardrails = parse_guardrails(guardrails)
        self.prepare_tools = prepare_tools

        self.memory_config = MemoryBuilder.resolve(memory)

        if isinstance(settings, dict):
            self.settings = AgentSettings(**settings)
        else:
            self.settings = settings or AgentSettings()

        self.runtime_config = self.settings
        self.engine_config = self.settings

        if self.settings.enable_hitl is None:
            from zhenxun.services.ai.config import get_llm_config

            self.settings.enable_hitl = get_llm_config().agent_settings.enable_hitl

        self.settings.stateless = not self.memory_config.short_term.enable

        self.capabilities: list[AbstractCapability] = []

        if capabilities:
            for cap in capabilities:
                if isinstance(cap, AbstractCapability):
                    self.capabilities.append(cap)
                elif callable(cap):
                    self.capabilities.append(DynamicCapability(cap))
        self.executor = executor
        if self.settings.enable_hitl:
            from zhenxun.services.ai.tools.providers.builtin.hitl import HITLToolkit

            self.tool_definitions.append(HITLToolkit())

        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            self.capabilities.append(
                SkillCapability(skills=skills, namespace=infer_plugin_namespace())
            )

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

    def guardrail(self, func: Callable | str | Any | None = None):
        """护栏装饰器/注册器 (支持传入函数或自然语言风控规则字符串)"""
        if func is None:

            def decorator(f: Callable):
                from zhenxun.services.ai.guardrails import parse_guardrails

                self._guardrails.extend(parse_guardrails([f]))
                return f

            return decorator
        else:
            from zhenxun.services.ai.guardrails import parse_guardrails

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
        profile: AgentRunProfile | None = None,
        deps: AgentDepsT | None = None,
        context: RunContext[AgentDepsT] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        """
        智能体单次运行阻塞核心入口，内部使用上下文管理器静默消费事件流直至执行结束。

        参数:
            prompt: 用户输入的消息内容或标准数据契约任务对象 (Task)。
            deps: 强类型的外部依赖注入对象 (例如 NoneBot 的 Bot, Event)。
            context: 显式传入的运行时与会话上下文 (RunContext)。
            profile: 单次运行时的动态配置覆盖，包含记忆、历史消息、中间件等。
            kwargs: 透传的其他附加参数。

        返回:
            AgentRunResult[OutputDataT]: 包含最终输出数据、消息历史和用量统计的运行结果对象。
        """  # noqa: E501
        return await super().run(
            prompt=prompt,
            profile=profile,
            deps=deps,
            context=context,
            **kwargs,
        )

    @contextlib.asynccontextmanager
    async def run_stream(
        self,
        prompt: PromptInput | Task | None = None,
        *,
        profile: AgentRunProfile | None = None,
        deps: AgentDepsT | None = None,
        context: RunContext[AgentDepsT] | None = None,
        event_streamer: EventStreamer | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult[OutputDataT]]:
        """
        智能体流式运行入口。
        返回上下文管理器，可安全、解耦地获取底层事件或纯净文本结果。
        """
        prof = profile or AgentRunProfile()

        merged_settings = model_copy(self.settings, deep=True)
        if prof.max_cycles is not None:
            merged_settings.max_cycles = prof.max_cycles

        if prof.skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            if prof.capabilities is None:
                prof.capabilities = []
            prof.capabilities.append(
                SkillCapability(skills=prof.skills, namespace=infer_plugin_namespace())
            )

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

        policy = getattr(self.settings, "concurrency_policy", None)
        if policy is None:
            from zhenxun.services.ai.flow.base import ConcurrencyPolicy

            policy = (
                ConcurrencyPolicy.ALLOW
                if getattr(self.settings, "stateless", True)
                else ConcurrencyPolicy.QUEUE
            )

        async def _execution_task():
            from zhenxun.services.ai.run.models import CancellationToken
            from zhenxun.services.ai.run.session import session_manager

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
                        profile=prof,
                        settings=merged_settings,
                        event_streamer=streamer,
                        **kwargs,
                    )
                    await streamer.send(AgentRunEnd(result=result))
            except ControlFlowExit as e:
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
        profile: AgentRunProfile,
        settings: AgentSettings,
        cancellation_token: Any = None,
        event_streamer: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        """执行原子步代理逻辑"""
        from zhenxun.services.ai.context.memory.engine import MemoryReader, MemoryWriter
        from zhenxun.services.ai.context.memory.types import SessionMetadata
        from zhenxun.services.ai.core.messages import UsageInfo
        from zhenxun.services.ai.flow.agent.engine.builders import ContextBuilder

        (
            task_obj,
            final_prompt_payload,
            extra_tools,
            run_output_type,
            task_guardrails,
        ) = self._parse_task_prompt(prompt)

        effective_memory = (
            MemoryBuilder.resolve(profile.memory)
            if profile.memory is not None
            else model_copy(self.memory_config, deep=True)
        )

        session_metadata = SessionMetadata(
            session_id=context.session_id or "default_session",
            user_id=context.get_user_id(),
            group_id=context.get_group_id(),
            platform=context.get_platform(),
            namespace=self.namespace,
            agent_name=self.name,
        )
        reader = MemoryReader(
            session_meta=session_metadata, memory_config=effective_memory
        )
        writer = MemoryWriter(
            session_meta=session_metadata,
            memory_config=effective_memory,
            context=context,
        )

        dynamic_caps = []
        combined_guardrails = self._guardrails + task_guardrails
        if run_output_type is not None and run_output_type is not str:
            dynamic_caps.append(
                OutputValidationCapability(run_output_type, combined_guardrails)
            )
        elif combined_guardrails:
            dynamic_caps.append(OutputValidationCapability(None, combined_guardrails))
        if task_obj:
            dynamic_caps.append(TaskTrackingCapability(task_obj, self.name))

        run_level_caps = []
        if profile.capabilities:
            for cap in profile.capabilities:
                if isinstance(cap, AbstractCapability):
                    run_level_caps.append(cap)
                elif callable(cap):
                    run_level_caps.append(DynamicCapability(cap))

        from zhenxun.services.ai.tools.engine.global_capabilities import (
            GLOBAL_CAPABILITIES,
        )

        base_caps = GLOBAL_CAPABILITIES.get("global", []).copy()
        if self.namespace != "global" and self.namespace in GLOBAL_CAPABILITIES:
            base_caps.extend(GLOBAL_CAPABILITIES[self.namespace])

        combined_cap = CombinedCapability(
            base_caps
            + getattr(context, "capabilities", [])
            + self.capabilities
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
        context.run.agent_name = self.name
        context.run.cancellation_token = cancellation_token
        context.run.streamer = event_streamer

        long_term_fact = await reader.get_long_term_context(
            context.run.user_input or ""
        )
        slots_fact = await reader.get_slots_context()
        static_prompt, dynamic_prompt = await ContextBuilder.build_prompts(
            instruction=self.instruction,
            system_prompts=self.dynamic_prompts,
            run_context=context,
            run_scoped_cap=run_scoped_cap,
            persona=cast(Persona | None, self.persona),
        )
        if long_term_fact:
            dynamic_prompt += f"\n\n{long_term_fact}"
        if slots_fact:
            dynamic_prompt += f"\n\n{slots_fact}"

        tool_payload = await ToolBuilder.resolve_tools(
            tool_definitions=self.tool_definitions,
            toolset_funcs=getattr(self, "toolset_funcs", []),
            system_tools=[],
            namespace=self.namespace or "unknown",
            tool_filter=profile.tool_filter,
            run_context=context,
            run_scoped_cap=run_scoped_cap,
        )
        effective_tools = tool_payload.tools
        if extra_tools:
            effective_tools.extend(extra_tools)

        final_gen_config = model_copy(self.default_config, deep=True)
        if cap_dynamic_config := await run_scoped_cap.get_generation_config(context):
            final_gen_config = final_gen_config.merge_with(cap_dynamic_config)
        if profile.generation_config:
            final_gen_config = final_gen_config.merge_with(profile.generation_config)

        static_prompts_list = [static_prompt]
        if tool_payload.injected_prompts:
            static_prompts_list.append(
                "--- 工具箱专属使用说明 ---\n\n"
                + "\n\n".join(tool_payload.injected_prompts)
            )

        messages_for_run = await reader.get_short_term_context(
            model_name=context.run.current_model or "",
            override_history=profile.message_history,
        )
        if final_prompt_payload is not None:
            from zhenxun.services.ai.message_builder import MessageBuilder

            if msgs := await MessageBuilder.normalize_to_llm_messages(
                final_prompt_payload, bot=context.get_bot(), event=context.get_event()
            ):
                messages_for_run.append(msgs[-1])
                await writer.save_new_messages([msgs[-1]])

        final_tools = await ToolBuilder.prepare_effective_tools(
            effective_tools, context, self.prepare_tools, run_scoped_cap
        )
        context.session.append_only_manager.build(static_prompts_list, final_tools)
        context.session.append_only_manager.sync_messages(messages_for_run)

        state = AgentState(
            messages=messages_for_run,
            tools=final_tools,
            run_context=context,
            static_system_prompt=static_prompts_list,
            dynamic_system_prompt=dynamic_prompt,
            usage=UsageInfo(),
            origin_msg_len=len(messages_for_run),
        )

        original_capabilities = getattr(context, "capabilities", [])
        context.capabilities = run_scoped_cap.capabilities

        async def inner_run_handler() -> AgentRunResult[OutputDataT]:
            for tk in tool_payload.toolkits:
                if hasattr(tk, "before_llm_request"):
                    if is_coroutine_callable(tk.before_llm_request):
                        await tk.before_llm_request(context, state.messages)
                    else:
                        tk.before_llm_request(context, state.messages)

            async with ToolBuilder.mount_toolkits(
                tool_payload.toolkits, context.session_id or "", context
            ):
                from zhenxun.services.ai.flow.agent.engine.executor import (
                    StandardAgentExecutor,
                )

                executor = profile.executor or self.executor or StandardAgentExecutor()

                from zhenxun.services.ai.llm.manager import get_model_instance

                async with await get_model_instance(
                    context.run.current_model, override_config=None
                ) as instance:
                    raw_result: Any = await executor.run(
                        state=state,
                        settings=settings,
                        generation_config=final_gen_config,
                        model_instance=instance,
                    )

            new_msgs = raw_result.messages[state.origin_msg_len :]
            await writer.save_new_messages(new_msgs)

            from zhenxun.utils.pydantic_compat import model_construct

            final_output = getattr(raw_result, "output", None) or (
                raw_result.messages[-1].extract_text if raw_result.messages else ""
            )

            return cast(
                AgentRunResult[OutputDataT],
                model_construct(
                    AgentRunResult,
                    output=final_output,
                    messages=new_msgs,
                    structured_data=getattr(raw_result, "structured_data", None),
                    usage=getattr(raw_result, "usage", None) or UsageInfo(),
                    handoff=getattr(raw_result, "handoff", None),
                ),
            )

        try:
            return await run_scoped_cap.wrap_run(context, inner_run_handler)
        except ControlFlowExit as e:
            raise e
        except Exception as e:
            raise e
        finally:
            context.capabilities = original_capabilities
