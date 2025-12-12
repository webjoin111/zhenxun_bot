from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any, Literal

from nonebot.matcher import Matcher
from nonebot.params import Depends
from pydantic import BaseModel

from zhenxun.services.agent.core.context import AgentContext
from zhenxun.services.llm import (
    LLMGenerationConfig,
    LLMMessage,
    LLMResponse,
    ModelName,
)

from .core.agent import Agent
from .core.types import ExecutionConfig, MCPSource, ReviewerConfig, ToolFilter
from .workflows import (
    BaseWorkflow,
    ChainWorkflow,
    EvaluatorOptimizerWorkflow,
    OrchestratorWorkflow,
    ParallelWorkflow,
    RouterWorkflow,
)


@dataclass
class _AgentDefinition:
    name: str
    instruction: str
    model: ModelName | Callable[[], ModelName]
    tools: list[str | MCPSource] | None
    resources: list[str] | None
    prompts: list[str] | None
    config: LLMGenerationConfig | None
    func_handler: Callable
    reviewer: ReviewerConfig | None
    response_model: type[BaseModel] | None


@dataclass
class _WorkflowDefinition:
    name: str
    workflow_type: Literal[
        "chain", "router", "parallel", "orchestrator", "evaluator_optimizer"
    ]
    kwargs: dict[str, Any]
    func_handler: Callable


class AgentRegistryMixin(ABC):
    """
    Agent 注册基类，统一管理装饰器接口。
    子类需实现 _register_definition 以决定如何处理收集到的定义（存储或实例化）。
    """

    @abstractmethod
    def _register_definition(
        self, definition: _AgentDefinition | _WorkflowDefinition
    ) -> None:
        pass

    def agent(
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
        def decorator(func: Callable):
            definition = _AgentDefinition(
                name=name,
                instruction=instruction,
                model=model,
                tools=tools,
                resources=resources,
                prompts=prompts,
                config=config,
                func_handler=func,
                reviewer=reviewer,
                response_model=response_model,
            )
            self._register_definition(definition)
            return func

        return decorator

    def _register_workflow_decorator(
        self,
        wf_type: Literal[
            "chain", "router", "parallel", "orchestrator", "evaluator_optimizer"
        ],
        name: str,
        **kwargs,
    ):
        def decorator(func: Callable):
            definition = _WorkflowDefinition(
                name=name,
                workflow_type=wf_type,
                kwargs=kwargs,
                func_handler=func,
            )
            self._register_definition(definition)
            return func

        return decorator

    def chain(self, name: str, sequence: list[str]):
        return self._register_workflow_decorator("chain", name, sequence=sequence)

    def router(self, name: str, agents: list[str], model: ModelName = None):
        return self._register_workflow_decorator(
            "router", name, agents=agents, model=model
        )

    def parallel(self, name: str, fan_out: list[str], fan_in: str | None = None):
        return self._register_workflow_decorator(
            "parallel", name, fan_out=fan_out, fan_in=fan_in
        )

    def orchestrator(
        self, name: str, agents: list[str], planner_model: ModelName = None
    ):
        return self._register_workflow_decorator(
            "orchestrator", name, agents=agents, planner_model=planner_model
        )

    def evaluator_optimizer(
        self,
        name: str,
        generator: str,
        evaluator: str,
        judgement_func: Callable[[Any], bool] | None = None,
        evaluation_model: type[BaseModel] | None = None,
        prompt_template: str | None = None,
        max_refinements: int = 3,
        on_cycle: Callable[[int, Any, Any], Awaitable[None]] | None = None,
    ):
        return self._register_workflow_decorator(
            "evaluator_optimizer",
            name,
            generator=generator,
            evaluator=evaluator,
            judgement_func=judgement_func,
            evaluation_model=evaluation_model,
            prompt_template=prompt_template,
            max_refinements=max_refinements,
            on_cycle=on_cycle,
        )


class AgentRouter(AgentRegistryMixin):
    """Agent 路由器，用于组织命名空间。仅存储定义，不实例化。"""

    def __init__(self):
        self._definitions: list[_AgentDefinition] = []
        self._workflow_definitions: list[_WorkflowDefinition] = []

    def _register_definition(
        self, definition: _AgentDefinition | _WorkflowDefinition
    ) -> None:
        if isinstance(definition, _AgentDefinition):
            self._definitions.append(definition)
        elif isinstance(definition, _WorkflowDefinition):
            self._workflow_definitions.append(definition)


class AgentApp(AgentRegistryMixin):
    """受 fast-agent 启发的应用主类，提供声明式API。"""

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._workflows: dict[str, BaseWorkflow] = {}

    def _register_definition(
        self, definition: _AgentDefinition | _WorkflowDefinition
    ) -> None:
        """实现基类的注册接口：立即实例化并注册。"""
        if isinstance(definition, _AgentDefinition):
            self._register_agent(
                name=definition.name,
                instruction=definition.instruction,
                model=definition.model,
                tools=definition.tools,
                func_handler=definition.func_handler,
                resources=definition.resources,
                prompts=definition.prompts,
                config=definition.config,
                response_model=definition.response_model,
                reviewer=definition.reviewer,
            )
        elif isinstance(definition, _WorkflowDefinition):
            self._create_and_register_workflow(
                definition.workflow_type,
                definition.name,
                definition.kwargs,
                definition.func_handler,
            )

    def include_router(self, router: AgentRouter, prefix: str = ""):
        """
        将 Router 中的 Agent 注册到 App，可选命名空间前缀。
        """
        for definition in router._definitions:
            final_name = f"{prefix}_{definition.name}" if prefix else definition.name
            self._register_agent(
                name=final_name,
                instruction=definition.instruction,
                model=definition.model,
                tools=definition.tools,
                func_handler=definition.func_handler,
                resources=definition.resources,
                prompts=definition.prompts,
                config=definition.config,
                reviewer=definition.reviewer,
                response_model=definition.response_model,
            )

        for wf_def in router._workflow_definitions:
            final_name = f"{prefix}_{wf_def.name}" if prefix else wf_def.name
            final_kwargs = wf_def.kwargs.copy()
            if prefix:
                if "sequence" in final_kwargs:
                    final_kwargs["sequence"] = [
                        f"{prefix}_{item}" for item in final_kwargs["sequence"]
                    ]
                if "agents" in final_kwargs:
                    final_kwargs["agents"] = [
                        f"{prefix}_{item}" for item in final_kwargs["agents"]
                    ]
                if "fan_out" in final_kwargs:
                    final_kwargs["fan_out"] = [
                        f"{prefix}_{item}" for item in final_kwargs["fan_out"]
                    ]
                if final_kwargs.get("fan_in"):
                    final_kwargs["fan_in"] = f"{prefix}_{final_kwargs['fan_in']}"
                if "generator" in final_kwargs:
                    final_kwargs["generator"] = f"{prefix}_{final_kwargs['generator']}"
                if "evaluator" in final_kwargs:
                    final_kwargs["evaluator"] = f"{prefix}_{final_kwargs['evaluator']}"

            self._create_and_register_workflow(
                wf_def.workflow_type, final_name, final_kwargs, wf_def.func_handler
            )

    def _register_agent(
        self,
        name: str,
        instruction: str,
        model: ModelName | Callable[[], ModelName],
        tools: list[str | MCPSource] | None,
        func_handler: Callable,
        resources: list[str] | None = None,
        prompts: list[str] | None = None,
        config: LLMGenerationConfig | None = None,
        response_model: type[BaseModel] | None = None,
        reviewer: ReviewerConfig | None = None,
    ):
        """内部注册逻辑。"""
        if name in self._agents:
            pass
        if response_model is None:
            try:
                sig = inspect.signature(func_handler)
                return_type = sig.return_annotation
                if (
                    return_type is not inspect.Signature.empty
                    and isinstance(return_type, type)
                    and issubclass(return_type, BaseModel)
                    and return_type is not LLMResponse
                ):
                    response_model = return_type
            except (ValueError, TypeError):
                pass

        agent_instance = Agent(
            name,
            instruction,
            model,
            tools,
            resources=resources,
            prompts=prompts,
            config=config,
            response_model=response_model,
            reviewer=reviewer,
        )
        self._agents[name] = agent_instance

        async def agent_chat_wrapper(
            context: AgentContext | None = None,
            message: str | None = None,
            session_id: str | None = None,
            matcher: Matcher | None = None,
            tool_filter: ToolFilter | None = None,
            config: ExecutionConfig | None = None,
            generation_config: LLMGenerationConfig | None = None,
            **kwargs,
        ):
            """
            Agent 调用的统一入口。
            支持传入完整的 Context，也支持传入 message/session_id 自动构建 Context。
            kwargs 中的剩余参数会被放入 context.scope 中。
            """
            if context is None:
                if message is None:
                    raise ValueError(
                        "调用 Agent 时必须提供 'context' 对象或 'message' 字符串。"
                    )

                import uuid

                final_session_id = session_id or f"ag-{uuid.uuid4()}"

                context = AgentContext(
                    session_id=final_session_id,
                    user_input=message,
                    scope=kwargs,
                )

            return await agent_instance.chat(
                context=context,
                matcher=matcher,
                tool_filter=tool_filter,
                config=config,
                generation_config=generation_config,
            )

        setattr(self, name, agent_chat_wrapper)

    def _create_and_register_workflow(
        self,
        wf_type: Literal[
            "chain", "router", "parallel", "orchestrator", "evaluator_optimizer"
        ],
        name: str,
        kwargs: dict[str, Any],
        func_handler: Callable,
    ):
        """根据类型创建并注册工作流。"""
        if wf_type == "chain":
            workflow_instance = ChainWorkflow(app=self, name=name, **kwargs)
        elif wf_type == "router":
            workflow_instance = RouterWorkflow(app=self, name=name, **kwargs)
        elif wf_type == "parallel":
            workflow_instance = ParallelWorkflow(app=self, name=name, **kwargs)
        elif wf_type == "orchestrator":
            workflow_instance = OrchestratorWorkflow(app=self, name=name, **kwargs)
        elif wf_type == "evaluator_optimizer":
            workflow_instance = EvaluatorOptimizerWorkflow(
                app=self, name=name, **kwargs
            )
        else:
            raise ValueError(f"未知的工作流类型: {wf_type}")

        self.register_workflow(name, workflow_instance)

        async def workflow_run_wrapper(
            initial_input: str,
            matcher: Matcher = Depends(),
            session_id: str | None = None,
            history: list[LLMMessage] | None = None,
            **run_kwargs,
        ):
            return await workflow_instance.run(
                initial_input,
                matcher=matcher,
                session_id=session_id,
                history=history,
                **run_kwargs,
            )

        setattr(self, name, workflow_run_wrapper)

    def register_workflow(self, name: str, workflow: BaseWorkflow):
        """将工作流实例注册到应用中。"""
        if name in self._workflows or name in self._agents:
            raise ValueError(
                f"名称 '{name}' 已被注册为 Agent 或 Workflow，请使用其他名称。"
            )
        self._workflows[name] = workflow

    def get_agent(self, name: str) -> Agent | None:
        """根据名称获取已注册的 Agent 实例。"""
        return self._agents.get(name)
