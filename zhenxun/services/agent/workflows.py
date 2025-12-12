from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from jinja2 import Template
from nonebot.matcher import Matcher
from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel, Field

from zhenxun.services.agent.core.agent import Agent
from zhenxun.services.llm import LLMMessage, LLMResponse, ModelName
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump_json

from .core.context import AgentContext

if TYPE_CHECKING:
    from .app import AgentApp
    from .core.agent import Agent


class BaseWorkflow(ABC):
    """
    所有工作流类的抽象基类。
    """

    def __init__(self, app: "AgentApp", name: str):
        """
        初始化工作流。

        参数:
            app: AgentApp 的实例，用于在运行时查找和访问 Agent。
            name: 工作流的名称。
        """
        self._app = app
        self._name = name

    @abstractmethod
    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        执行工作流的入口方法。

        参数:
            initial_input: 工作流的初始输入，通常是用户的原始问题。
            matcher: 当前事件的 Matcher 实例，用于用户交互。
            session_id: 会话ID，用于状态管理。
            history: 历史消息列表，用于无状态调用。
            **kwargs: 可能需要的额外参数。

        返回:
            LLMResponse: 工作流执行完毕后的最终响应。
        """
        raise NotImplementedError

    def _get_runnable_agent(self, name: str) -> Callable[..., Awaitable[LLMResponse]]:
        """
        辅助方法：从 App 中获取可运行的 Agent 或 Workflow 包装函数。
        """
        method = getattr(self._app, name, None)
        if not callable(method):
            raise ValueError(
                f"工作流 '{self._name}' 无法在 App 中找到名为 '{name}' "
                "的可调用 Agent 或 Workflow。"
            )
        return cast(Callable[..., Awaitable[LLMResponse]], method)


class ChainWorkflow(BaseWorkflow):
    """
    一个按顺序执行 Agent 列表的工作流。
    前一个 Agent 的输出将作为后一个 Agent 的输入。
    """

    def __init__(self, app: "AgentApp", name: str, sequence: list[str]):
        """
        初始化链式工作流。

        参数:
            app: AgentApp 的实例。
            name: 工作流的名称。
            sequence: 按执行顺序列出的 Agent 名称列表。
        """
        super().__init__(app, name)
        self._sequence = sequence

    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        按顺序执行链中的所有 Agent。

        参数:
            initial_input: 链条的第一个输入。
            matcher: 当前事件的 Matcher 实例，用于用户交互。
            history: （可选）用于无状态调用的历史消息列表。

        返回:
            LLMResponse: 链条中最后一个 Agent 的响应。
        """
        current_input = initial_input
        final_response = None

        logger.info(f"🚀 [工作流开始] Chain: '{self._name}'")
        for i, agent_name in enumerate(self._sequence):
            logger.info(
                f"[步骤 {i + 1}/{len(self._sequence)}] 调用 Agent: '{agent_name}'"
            )

            agent_method = self._get_runnable_agent(agent_name)
            final_response = await agent_method(
                message=current_input, matcher=matcher, session_id=session_id, **kwargs
            )

            current_input = final_response.text

        logger.info(f"🏁 [工作流结束] Chain: '{self._name}'")

        if final_response is None:
            raise RuntimeError(f"链式工作流 '{self._name}'未能产生任何响应。")

        return final_response


class ParallelWorkflow(BaseWorkflow):
    """
    一个并行执行多个Agent（扇出），然后使用一个聚合Agent（扇入）来综合结果的工作流。
    """

    def __init__(self, app: "AgentApp", name: str, fan_out: list[str], fan_in: str):
        """
        初始化并行工作流。

        参数:
            app: AgentApp 的实例。
            name: 工作流的名称。
            fan_out: 并行执行的 Agent 名称列表。
            fan_in: 用于聚合结果的 Agent 名称。
        """
        super().__init__(app, name)
        self._fan_out_names = fan_out
        self._fan_in_name = fan_in

    def _build_synthesis_prompt(
        self, original_input: str, results: list[tuple[str, str]]
    ) -> str:
        """构建用于聚合结果的最终Prompt。"""
        prompt = (
            f"原始任务是：'{original_input}'\n\n"
            "为了完成这个任务，我咨询了多位专家，他们的反馈如下：\n\n"
        )
        for agent_name, response_text in results:
            if "执行时出错" in response_text:
                prompt += f"--- 专家 '{agent_name}' 在分析时遇到了问题 ---\n"
                prompt += f"错误信息: {response_text}\n\n"
            else:
                prompt += f"--- 来自专家 '{agent_name}' 的反馈 ---\n"
                prompt += f"{response_text}\n\n"

        prompt += (
            "请综合以上所有专家的反馈，形成一个全面、最终的回答。如果某些专家遇到了问题，"
            "请在最终回答中适当说明，并基于可用的反馈给出最佳建议。"
        )
        return prompt

    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行扇出和扇入操作。"""
        logger.info(f"🚀 [工作流开始] Parallel: '{self._name}'")

        fan_out_tasks = []
        for agent_name in self._fan_out_names:
            agent_method = self._get_runnable_agent(agent_name)
            fan_out_tasks.append(
                agent_method(
                    message=initial_input,
                    matcher=matcher,
                    session_id=session_id,
                    **kwargs,
                )
            )

        logger.info(f"[扇出] 并行调用 {len(fan_out_tasks)} 个 Agent...")
        results = await asyncio.gather(*fan_out_tasks, return_exceptions=True)

        collected_results = []
        for agent_name, result_or_exc in zip(self._fan_out_names, results):
            if isinstance(result_or_exc, Exception):
                error_text = f"Agent '{agent_name}' 执行时出错: {result_or_exc}"
                logger.error(f"[扇出结果] {error_text}")
                collected_results.append((agent_name, error_text))
            else:
                logger.info(f"[扇出结果] Agent '{agent_name}' 完成。")
                resp_obj = cast(LLMResponse, result_or_exc)
                collected_results.append((agent_name, resp_obj.text))

        fan_in_agent_method = self._get_runnable_agent(self._fan_in_name)

        synthesis_prompt = self._build_synthesis_prompt(
            initial_input, collected_results
        )

        logger.info(f"[扇入] 调用聚合 Agent '{self._fan_in_name}' 进行结果合成...")
        final_response = await fan_in_agent_method(
            message=synthesis_prompt, matcher=matcher, session_id=session_id, **kwargs
        )

        logger.info(f"🏁 [工作流结束] Parallel: '{self._name}'")
        return final_response


class RouterChoice(BaseModel):
    """用于规范路由器LLM响应的Pydantic模型。"""

    choice: str = Field(description="从提供的选项中选择的最合适的Agent名称。")
    reason: str = Field(description="做出这个选择的简要原因。")


DEFAULT_ROUTER_INSTRUCTION = """
你是一个智能任务路由器。你的任务是分析用户的请求，并从以下可用Agent中选择最合适的一个来处理该请求。

可用Agent列表:
{% for agent in agents %}
- **{{ agent.name }}**: {{ agent.instruction }}
{% endfor %}

请分析以下用户请求，并决定哪个Agent最适合处理它。
用户请求: "{{ initial_input }}"
"""


class RouterWorkflow(BaseWorkflow):
    """
    一个使用LLM来评估消息并将其路由到最合适Agent的工作流。
    """

    def __init__(
        self, app: "AgentApp", name: str, agents: list[str], model: str | None = None
    ):
        super().__init__(app, name)
        self._agent_names = agents
        self._router_model = model

        self._candidate_agents: list[Agent] = []
        for name in self._agent_names:
            agent = self._app.get_agent(name)
            if agent:
                self._candidate_agents.append(agent)
            else:
                logger.warning(
                    f"RouterWorkflow '{self._name}': Agent '{name}' 未找到，将被忽略。"
                )

        self._router_agent = Agent(
            name=f"{self._name}_internal_router",
            instruction=DEFAULT_ROUTER_INSTRUCTION,
            model=self._router_model,
            response_model=RouterChoice,
        )

    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行路由决策并调用选定的Agent。"""
        logger.info(f"🚀 [工作流开始] Router: '{self._name}'")

        context = AgentContext(
            session_id=session_id or f"router-{self._name}",
            user_input="",
            scope={
                "agents": self._candidate_agents,
                "initial_input": initial_input,
            },
        )

        logger.info(
            f"[路由决策] 使用模型 '{self._router_model or 'default'}' 进行决策..."
        )

        router_choice = cast(
            RouterChoice,
            await self._router_agent.chat(context=context, matcher=matcher),
        )

        chosen_agent_name = router_choice.choice
        logger.info(
            f"[路由决策] 模型选择: '{chosen_agent_name}'. 原因: {router_choice.reason}"
        )

        chosen_agent = self._app.get_agent(chosen_agent_name)
        if not chosen_agent:
            raise ValueError(
                f"路由模型返回了一个无效的Agent名称 '{chosen_agent_name}'，"
                f"它不在候选列表 {self._agent_names} 中。"
            )

        logger.info(f"[任务执行] 将任务转发给 Agent: '{chosen_agent_name}'")
        agent_method = self._get_runnable_agent(chosen_agent_name)
        final_response = await agent_method(
            message=initial_input, matcher=matcher, session_id=session_id, **kwargs
        )

        logger.info(f"🏁 [工作流结束] Router: '{self._name}'")
        return final_response


class PlanStep(BaseModel):
    agent_name: str = Field(description="应该执行此步骤的 Worker Agent 的名称。")
    task_description: str = Field(
        description="对此 Worker Agent 的具体、可执行的指令。"
    )


class Plan(BaseModel):
    thought: str = Field(description="你对当前状态的思考，以及为什么制定这个计划。")
    steps: list[PlanStep] = Field(
        default_factory=list, description="要执行的一系列步骤。"
    )
    final_answer: str | None = Field(
        default=None,
        description="如果任务已完成，这是给用户的最终答案。如果任务未完成，"
        "则此字段应为 null。",
    )


DEFAULT_ORCHESTRATOR_INSTRUCTION = """
你是一个专业的项目经理（Orchestrator）。你的职责是：
1. 理解用户的最终目标。
2. 将目标分解成一系列具体的、可执行的步骤。
3. 从可用的 Worker Agent 列表中，为每个步骤选择最合适的 Agent。
4. 严格按照指定的JSON格式输出你的思考和计划。

--- 可用的 Worker Agents ---
{% for agent in workers %}
- **{{ agent.name }}**: {{ agent.instruction }}
{% endfor %}
"""

DEFAULT_ORCHESTRATOR_INPUT = """
--- 任务历史 ---
{{ history_text }}

--- 当前目标 ---
{{ objective }}

请根据以上信息制定下一步计划。
"""


class OrchestratorWorkflow(BaseWorkflow):
    """
    一个使用\"规划者\"LLM来动态分解复杂任务并将其分派给多个\"工作者\"Agent的
    工作流。
    """

    def __init__(
        self,
        app: "AgentApp",
        name: str,
        agents: list[str],
        planner_model: ModelName = None,
    ):
        super().__init__(app, name)
        self._worker_agent_names = agents
        self._planner_model = planner_model
        self._max_iterations = 10

        self._workers: list[Agent] = []
        for name in self._worker_agent_names:
            agent = self._app.get_agent(name)
            if agent:
                self._workers.append(agent)
            else:
                logger.warning(f"Orchestrator '{self._name}': Worker '{name}' 未找到。")

        rendered_instruction = Template(DEFAULT_ORCHESTRATOR_INSTRUCTION).render(
            workers=self._workers
        )
        self._planner_agent = Agent(
            name=f"{self._name}_internal_planner",
            instruction=rendered_instruction,
            model=self._planner_model,
            response_model=Plan,
        )

    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行动态规划和任务分派循环。"""
        logger.info(f"🚀 [工作流开始] Orchestrator: '{self._name}'")

        execution_history: list[str] = []
        shared_context: list[str] = []

        for i in range(self._max_iterations):
            logger.info(f"[规划循环 {i + 1}/{self._max_iterations}] 开始规划...")

            history_text = (
                "\n".join(execution_history)
                if execution_history
                else "这是任务的第一步。\n"
            )
            input_prompt = Template(DEFAULT_ORCHESTRATOR_INPUT).render(
                history_text=history_text,
                objective=initial_input,
            )

            context = AgentContext(
                session_id=f"{session_id or 'orch'}-plan-{i}",
                user_input=input_prompt,
            )

            plan = cast(
                Plan,
                await self._planner_agent.chat(
                    context=context,
                    matcher=matcher,
                    **kwargs,
                ),
            )

            logger.info(f"[规划师思考]: {plan.thought}")

            if plan.final_answer:
                logger.info("[规划完成] 规划师给出了最终答案。")
                return LLMResponse(text=plan.final_answer)

            if not plan.steps:
                logger.warning(
                    "[规划警告] 规划师未给出下一步计划也未给出最终答案，工作流终止。"
                )
                return LLMResponse(text="任务已完成，但未能生成明确的最终答案。")

            for step in plan.steps:
                logger.info(
                    "  [执行步骤] Agent: "
                    f"'{step.agent_name}', "
                    f"任务: '{step.task_description[:50]}...'"
                )
                try:
                    worker_agent_method = self._get_runnable_agent(step.agent_name)
                except ValueError:
                    result_text = (
                        f"错误: 规划师指定了一个不存在的 Agent '{step.agent_name}'。"
                    )
                    logger.error(f"  {result_text}")
                else:
                    worker_prompt = step.task_description
                    if shared_context:
                        context_str = "\n".join(shared_context)
                        worker_prompt = (
                            f"--- 前序步骤的产出 (Context) ---\n{context_str}\n\n"
                            f"--- 你的任务 (Your Task) ---\n{step.task_description}"
                        )

                    response = await worker_agent_method(
                        message=worker_prompt,
                        matcher=matcher,
                        session_id=session_id,
                        **kwargs,
                    )
                    result_text = response.text

                    shared_context.append(
                        f"### {step.agent_name} 的输出:\n{result_text}\n"
                    )

                history_entry = (
                    f"步骤: 调用 Agent '{step.agent_name}' 执行 "
                    f"'{step.task_description}'.\n结果: {result_text}"
                )
                execution_history.append(history_entry)
                logger.info(f"  [步骤结果]: {result_text[:100]}...")

        final_text = "任务因达到最大迭代次数而终止。\n\n" + "\n".join(execution_history)
        logger.warning(
            f"🏁 [工作流结束] Orchestrator '{self._name}' 已达到最大迭代次数。"
        )
        return LLMResponse(text=final_text)


class DefaultEvaluationResult(BaseModel):
    """默认的评估结果模型，开发者可自定义替换"""

    score: int = Field(description="评分 (0-10)")
    feedback: str = Field(description="具体的、可操作的改进建议。")
    passed: bool = Field(description="是否达到通过标准。")


DEFAULT_REFINEMENT_TEMPLATE = (
    "这是你的原始任务：'{{ original_input }}'\n\n"
    "你之前的尝试未能达到质量标准。审阅反馈如下：\n"
    "--- 反馈意见 ---\n{{ feedback }}\n---\n\n"
    "这是你之前生成的版本：\n"
    "--- 之前的版本 ---\n{{ current_content }}\n---\n\n"
    "请根据反馈，生成一个全新的、经过改进的版本。"
)


class EvaluatorOptimizerWorkflow(BaseWorkflow):
    """
    一个结合\"生成器\"和\"评估器\"Agent，通过迭代反馈循环来提升输出质量的工作流。
    """

    def __init__(
        self,
        app: "AgentApp",
        name: str,
        generator: str | Any,
        evaluator: str | Any,
        judgement_func: Callable[[Any], bool] | None = None,
        evaluation_model: type[BaseModel] = DefaultEvaluationResult,
        prompt_template: str | None = None,
        max_refinements: int = 3,
        on_cycle: Callable[[int, Any, Any], Awaitable[None]] | None = None,
    ):
        super().__init__(app, name)
        self._generator = generator
        self._evaluator = evaluator
        self._judgement_func = judgement_func or (lambda r: getattr(r, "passed", False))
        self._evaluation_model = evaluation_model
        self._prompt_template = prompt_template or DEFAULT_REFINEMENT_TEMPLATE
        self._max_refinements = max_refinements
        self._on_cycle = on_cycle

    def _resolve_agent(self, agent_ref: str | Any) -> Any:
        """解析 Agent，支持字符串查找或直接对象"""
        if isinstance(agent_ref, str):
            agent = self._app.get_agent(agent_ref)
            if not agent:
                raise ValueError(
                    f"工作流 '{self._name}' 找不到名为 '{agent_ref}' 的 Agent。"
                )
            return agent
        return agent_ref

    async def run(
        self,
        initial_input: str,
        matcher: Matcher,
        session_id: str | None = None,
        history: list[LLMMessage] | None = None,
        on_cycle: Callable[[int, Any, Any], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行生成-评估-优化循环。"""
        logger.info(f"🚀 [工作流开始] Evaluator-Optimizer: '{self._name}'")

        generator_agent = self._resolve_agent(self._generator)
        evaluator_agent = self._resolve_agent(self._evaluator)
        current_on_cycle = on_cycle or self._on_cycle

        current_content = ""
        evaluation: Any | None = None
        session_marker = session_id or f"workflow:{self._name}"
        base_history = history.copy() if history else []

        def _response_text(resp: LLMResponse | BaseModel | str) -> str:
            if isinstance(resp, LLMResponse):
                return resp.text
            if isinstance(resp, BaseModel):
                try:
                    return model_dump_json(resp)
                except Exception:
                    return str(resp)
            return str(resp)

        if hasattr(evaluator_agent, "response_model"):
            evaluator_agent.response_model = self._evaluation_model

        for i in range(self._max_refinements + 1):
            if i == 0:
                logger.info("[迭代 1] 正在生成初始版本...")
                gen_context = AgentContext(
                    session_id=session_marker,
                    user_input=initial_input,
                    message_history=base_history,
                    scope=dict(kwargs),
                )
                response = await generator_agent.chat(
                    context=gen_context,
                    matcher=matcher,
                    **kwargs,
                )
                current_content = _response_text(response)
            else:
                assert evaluation is not None
                template = Template(self._prompt_template)
                refinement_prompt = template.render(
                    original_input=initial_input,
                    current_content=current_content,
                    feedback=getattr(evaluation, "feedback", str(evaluation)),
                )
                logger.info(f"[迭代 {i + 1}] 正在根据反馈进行优化...")

                if current_on_cycle:
                    if is_coroutine_callable(current_on_cycle):
                        await current_on_cycle(i, current_content, evaluation)
                    else:
                        current_on_cycle(i, current_content, evaluation)

                gen_context = AgentContext(
                    session_id=session_marker,
                    user_input=refinement_prompt,
                    message_history=base_history,
                    scope=dict(kwargs),
                )
                response = await generator_agent.chat(
                    context=gen_context,
                    matcher=matcher,
                    **kwargs,
                )
                current_content = _response_text(response)

            logger.info(f"[迭代 {i + 1}] 正在评估生成的内容...")

            eval_input = (
                "这是需要评估的内容：\n--- 内容开始 ---\n"
                f"{current_content}\n--- 内容结束 ---"
            )
            eval_context = AgentContext(
                session_id=f"{session_marker}:eval:{i}",
                user_input=eval_input,
                message_history=[],
                scope=dict(kwargs),
            )

            evaluation_response = await evaluator_agent.chat(
                context=eval_context,
                matcher=matcher,
                **kwargs,
            )

            if isinstance(evaluation_response, LLMResponse):
                logger.warning(
                    "Evaluator agent 返回了 LLMResponse 而非结构化对象，"
                    "请检查 Agent 配置。"
                )
                evaluation = evaluation_response
            else:
                evaluation = evaluation_response

            logger.debug(f"  [评估结果]: {evaluation}")

            is_pass = False
            try:
                is_pass = self._judgement_func(evaluation)
            except Exception as e:
                logger.error(f"判决函数执行出错: {e}")

            if is_pass:
                logger.info(f"[质量达标] 在第 {i + 1} 次迭代后达到质量标准。")
                break
        else:
            logger.warning(
                "[达到上限] 已达到最大优化次数 "
                f"({self._max_refinements})，返回当前最佳版本。"
            )

        logger.info(f"🏁 [工作流结束] Evaluator-Optimizer: '{self._name}'")
        return LLMResponse(text=current_content)


__all__ = [
    "DEFAULT_REFINEMENT_TEMPLATE",
    "BaseWorkflow",
    "ChainWorkflow",
    "DefaultEvaluationResult",
    "EvaluatorOptimizerWorkflow",
    "OrchestratorWorkflow",
    "ParallelWorkflow",
    "Plan",
    "PlanStep",
    "RouterChoice",
    "RouterWorkflow",
]
