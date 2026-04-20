from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from zhenxun.services.ai.agent.core.agent import Agent
from zhenxun.services.ai.llm import LLMMessage
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.types.agent import AgentRunResult
from zhenxun.services.log import logger


class BaseWorkflow(ABC):
    """
    所有工作流类的抽象基类。
    """

    def __init__(self, name: str):
        """
        初始化工作流。

        参数:
            name: 工作流的名称。
        """
        self._name = name

    async def __resolve_to_tools__(self) -> list[ToolExecutable]:
        """协议支持：将工作流转化为可被调用的工具"""
        from zhenxun.services.ai.tools.bridges.workflow import WorkflowTool

        return [WorkflowTool(self)]

    async def run(
        self,
        initial_input: str,
        deps: Any = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        执行工作流的模板方法（Template Method）。
        负责统一的生命周期日志记录和异常拦截。
        """
        workflow_type = self.__class__.__name__.replace("Workflow", "")
        logger.info(f"🚀 [工作流开始] {workflow_type}: '{self._name}'")

        try:
            result = await self._execute_workflow(
                initial_input=initial_input,
                deps=deps,
                history=history,
                **kwargs,
            )
            logger.info(f"🏁 [工作流结束] {workflow_type}: '{self._name}'")
            return result
        except Exception as e:
            logger.error(
                f"❌ [工作流异常] {workflow_type} '{self._name}' 发生错误: {e}"
            )
            raise

    @abstractmethod
    async def _execute_workflow(
        self,
        initial_input: str,
        deps: Any = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        子类需实现此方法以定义具体的工作流执行逻辑。
        """
        raise NotImplementedError


class SequenceWorkflow(BaseWorkflow):
    """
    一个按顺序执行 Agent 对象列表的工作流。
    前一个 Agent 的输出将作为后一个 Agent 的输入。
    """

    def __init__(self, name: str, sequence: list[Agent]):
        """
        初始化顺序工作流。

        参数:
            name: 工作流的名称。
            sequence: 按执行顺序列出的 Agent 对象列表。
        """
        super().__init__(name)
        self._sequence = sequence

    async def _execute_workflow(
        self,
        initial_input: str,
        deps: Any = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        按顺序执行链中的所有 Agent。
        """
        current_input = initial_input
        final_response = None
        current_history = history or []

        for i, agent in enumerate(self._sequence):
            logger.info(
                f"[步骤 {i + 1}/{len(self._sequence)}] 调用 Agent: '{agent.name}'"
            )

            final_response = await agent.run(
                prompt=current_input,
                deps=deps,
                message_history=current_history,
                **kwargs,
            )

            current_input = str(final_response.output)
            current_history.extend(final_response.messages)

        if final_response is None:
            raise RuntimeError(f"顺序工作流 '{self._name}'未能产生任何响应。")

        return final_response


class WorkflowRegistry:
    """
    工作流引擎注册表。支持动态扩展不同的工作流运行策略。
    """

    _registry: ClassVar[dict[str, type[BaseWorkflow]]] = {}

    @classmethod
    def register(cls, name: str, workflow_cls: type[BaseWorkflow]):
        cls._registry[name] = workflow_cls

    @classmethod
    def get(cls, name: str) -> type[BaseWorkflow]:
        if name not in cls._registry:
            raise ValueError(f"未注册的工作流策略: '{name}'")
        return cls._registry[name]


WorkflowRegistry.register("sequence", SequenceWorkflow)


__all__ = [
    "BaseWorkflow",
    "SequenceWorkflow",
    "WorkflowRegistry",
]
