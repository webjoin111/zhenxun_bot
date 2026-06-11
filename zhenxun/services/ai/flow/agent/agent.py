import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
import contextlib
from pathlib import Path
from typing import Any, Generic, cast

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.core.configs import (
    BaseOutputDefinition,
    GenerationConfig,
)
from zhenxun.services.ai.core.exceptions import (
    ControlFlowExit,
)
from zhenxun.services.ai.core.guardrails import GuardrailSource
from zhenxun.services.ai.core.messages import (
    LLMMessage,
    PromptInput,
)
from zhenxun.services.ai.core.stream_events import EventStreamer
from zhenxun.services.ai.flow.agent.engine.builders import ToolBuilder
from zhenxun.services.ai.flow.agent.models import (
    AgentEngineConfig,
    AgentRuntimeConfig,
    Persona,
)
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.knowledge.base import BaseKnowledge
from zhenxun.services.ai.llm.config.generation import IntentBuilder
from zhenxun.services.ai.memory.builder import MemoryBuilder
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.protocols.capabilities import AbstractCapability
from zhenxun.services.ai.protocols.tool import ToolExecutable, ToolResolvable
from zhenxun.services.ai.run import (
    AgentRunResult,
    RunContext,
    Task,
    TemplateStr,
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
from zhenxun.services.ai.tools.models import (
    GlobalToolFilter,
)
from zhenxun.services.ai.tools.providers.skills.models import Skill, SkillSource
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy
from zhenxun.utils.utils import infer_plugin_namespace

ToolSource = Callable | BaseTool | dict[str, Any] | str | BaseToolkit | ToolResolvable
"""任何可以作为工具提供给大模型的实体对象（函数、基础工具类、字典定义、工具名、工具箱）"""

CapabilitySource = Callable | AbstractCapability
"""能力/拦截器来源（函数或 AbstractCapability 实例）"""


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
        tools: list[ToolSource] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        generation_config: GenerationConfig | IntentBuilder | dict | None = None,
        response_model: BaseOutputDefinition | type[OutputDataT] | None = None,
        dynamic_prompts: list[Callable] | None = None,
        memory: bool | MemoryConfig | MemoryBuilder = False,
        knowledge: BaseKnowledge | list[BaseKnowledge] | None = None,
        runtime_config: AgentRuntimeConfig | dict | None = None,
        prepare_tools: ToolsPrepareFunc | None = None,
        guardrails: list[GuardrailSource] | None = None,
        capabilities: list[CapabilitySource] | None = None,
    ):
        """
        初始化 Agent。

        Args:
            name: Agent 名称，用于日志、事件和链路标识。
            instruction: 静态系统指令，可为普通字符串或模板字符串。
            persona: 可选人设配置；传入 dict 时会自动构造成 `Persona`。
            model: 默认模型名（如 `Provider/Model`）或返回模型名的回调.
            tools: 初始工具定义列表，可混用工具对象与字符串工具名。
            skills: 注入的领域知识技能，支持 ID、目录 Path、Skill 对象或 SkillSource 动态源。
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

        if self.runtime_config.enable_hitl is None:
            from zhenxun.services.ai.config import get_llm_config

            self.runtime_config.enable_hitl = (
                get_llm_config().agent_settings.enable_hitl
            )

        self.runtime_config.stateless = not self.memory_config.short_term.enable

        self.capabilities: list[AbstractCapability] = []

        if capabilities:
            from zhenxun.services.ai.protocols.capabilities import DynamicCapability

            for cap in capabilities:
                if isinstance(cap, AbstractCapability):
                    self.capabilities.append(cap)
                elif callable(cap):
                    self.capabilities.append(DynamicCapability(cap))
        if self.runtime_config.enable_hitl:
            from zhenxun.services.ai.tools.providers.builtin.hitl import HITLToolkit

            self.tool_definitions.append(HITLToolkit())

        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            self.capabilities.append(SkillCapability(skills=skills))

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
        config: AgentEngineConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        capabilities: list[CapabilitySource] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
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
        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            capabilities = list(capabilities) if capabilities else []
            capabilities.append(SkillCapability(skills=skills))

        return await super().run(
            prompt=prompt,
            deps=deps,
            context=context,
            message_history=message_history,
            tool_filter=tool_filter,
            config=config,
            memory=memory,
            generation_config=generation_config,
            capabilities=capabilities,
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
        config: AgentEngineConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        event_streamer: EventStreamer | None = None,
        capabilities: list[CapabilitySource] | None = None,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult[OutputDataT]]:
        """
        智能体流式运行入口。
        返回上下文管理器，可安全、解耦地获取底层事件或纯净文本结果。
        """

        if skills:
            from zhenxun.services.ai.tools.providers.skills.capabilities import (
                SkillCapability,
            )

            capabilities = list(capabilities) if capabilities else []
            capabilities.append(SkillCapability(skills=skills))

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
                        capabilities=capabilities,
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
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: AgentEngineConfig | None = None,
        memory: bool | MemoryConfig | MemoryBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        cancellation_token: Any = None,
        event_streamer: Any = None,
        capabilities: list[Any] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[OutputDataT]:
        """执行原子步代理逻辑"""
        from zhenxun.services.ai.flow.agent.engine.harness import AgentHarness

        harness = AgentHarness(self)
        (
            loop_ctx,
            exec_config,
            final_gen_config,
            writer,
            toolkits,
            run_scoped_cap,
            origin_msg_len,
        ) = await harness.prepare_loop(
            prompt=prompt,
            context=context,
            message_history=message_history,
            tool_filter=tool_filter,
            config=config,
            memory=memory,
            generation_config=generation_config,
            cancellation_token=cancellation_token,
            event_streamer=event_streamer,
            capabilities=capabilities,
        )

        original_capabilities = getattr(context, "capabilities", [])
        context.capabilities = run_scoped_cap.capabilities

        async def inner_run_handler() -> AgentRunResult[OutputDataT]:
            for tk in toolkits:
                if hasattr(tk, "before_llm_request"):
                    if is_coroutine_callable(tk.before_llm_request):
                        await tk.before_llm_request(context, loop_ctx.messages)
                    else:
                        tk.before_llm_request(context, loop_ctx.messages)

            async with ToolBuilder.mount_toolkits(
                toolkits, context.session_id or "", context
            ):
                from zhenxun.services.ai.flow.agent.engine.executor import AgentExecutor

                executor = AgentExecutor()

                from zhenxun.services.ai.llm.manager import get_model_instance

                async with await get_model_instance(
                    context.run.current_model, override_config=None
                ) as instance:
                    raw_result: Any = await executor.run(
                        loop_ctx=loop_ctx,
                        exec_config=exec_config,
                        generation_config=final_gen_config,
                        model_instance=instance,
                    )

            return await harness.post_loop(loop_ctx, raw_result, writer, origin_msg_len)

        try:
            return await run_scoped_cap.wrap_run(context, inner_run_handler)
        except ControlFlowExit as e:
            raise e
        except Exception as e:
            logger.error(f"Agent '{self.name}' 运行失败: {e}", e=e)
            raise e
        finally:
            context.capabilities = original_capabilities
