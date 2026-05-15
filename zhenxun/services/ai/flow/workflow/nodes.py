import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import inspect
from typing import Any, cast

from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel, Field

from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    LoopExecutionCompletedEvent,
    LoopExecutionStartedEvent,
    LoopIterationCompletedEvent,
    LoopIterationStartedEvent,
    ParallelExecutionCompletedEvent,
    ParallelExecutionStartedEvent,
)
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.flow.workflow.base import BaseNode
from zhenxun.services.ai.flow.workflow.types import (
    BaseFailurePolicy,
    StepInput,
    StepOutput,
    StepType,
)
from zhenxun.services.ai.run import DependencyInjector, RunContext
from zhenxun.services.log import logger


class Step(BaseNode):
    """
    工作流中的最小执行单元门面 (Facade)。
    对外部隐藏了 AgentNode 和 FunctionNode 的具体实现。
    当实例化 Step 时，底层会自动根据 executor 的类型返回专属的节点对象。
    """

    def __new__(
        cls,
        name: str | None = None,
        executor: Any = None,
        prompt: Any = None,
        requires_confirmation: bool = False,
        confirmation_message: str | None = None,
        failure_policy: BaseFailurePolicy | None = None,
    ):
        if cls is Step:
            if isinstance(executor, BaseRunnable):
                return object.__new__(RunnableNode)
            if callable(executor):
                return object.__new__(FunctionNode)
            raise ValueError(f"Step '{name}' 的执行器类型不支持: {type(executor)}")
        return object.__new__(cls)

    def __init__(
        self,
        name: str | None = None,
        executor: Any = None,
        prompt: Any = None,
        requires_confirmation: bool = False,
        confirmation_message: str | None = None,
        failure_policy: BaseFailurePolicy | None = None,
    ):
        actual_name = name or getattr(
            executor, "name", getattr(executor, "__name__", "unnamed_step")
        )
        super().__init__(
            name=actual_name,
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message,
            failure_policy=failure_policy,
        )
        self.executor = executor
        self.prompt = prompt

    @property
    def node_type(self) -> StepType:
        return StepType.STEP

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        if False:
            yield None
        raise NotImplementedError(
            "This is a facade. Real execution happens in subclasses."
        )


class RunnableNode(Step):
    """专门处理 Agent/Team/Workflow 等 BaseRunnable 状态机执行的私有节点"""

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        import copy

        from zhenxun.services.ai.run import Task

        prompt_data = self.prompt if self.prompt is not None else step_input.input

        if isinstance(prompt_data, Task):
            prompt_data = copy.copy(prompt_data)
            if step_input.previous_step_content:
                prev_content = str(step_input.previous_step_content)
                prompt_data.description = f"### 🔙 [上游节点执行输出]\n{prev_content}\n\n### 🎯 [当前需执行的任务]\n{prompt_data.description}"
            context.run.user_input = prompt_data.description
        else:
            if step_input.previous_step_content:
                prompt_data = f"[上游节点执行输出]:\n{step_input.previous_step_content}\n\n[当前需执行的任务]:\n{prompt_data or ''}"
            context.run.user_input = str(prompt_data) if prompt_data else ""

        final_result = None
        sandbox_context = context.clone_for_member(self.name)

        async with self.executor.run_stream(
            prompt=prompt_data, context=sandbox_context
        ) as stream_result:
            async for event in stream_result.stream_events():
                from zhenxun.services.ai.run.models import AgentRunEnd

                if isinstance(event, AgentRunEnd):
                    final_result = event.result
                yield event

        context.state.update(sandbox_context.state)
        yield StepOutput(
            content=final_result.output if final_result else "无返回",
            success=True,
        )


class FunctionNode(Step):
    """专门处理 Python Callable 依赖注入与执行的私有节点"""

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        context.run.user_input = str(step_input.input) if step_input.input else ""

        sig = inspect.signature(self.executor)
        resolved_kwargs = await DependencyInjector.resolve_all(
            sig, call_kwargs={"step_input": step_input}, context=context
        )
        filtered_kwargs = {
            k: v for k, v in resolved_kwargs.items() if k in sig.parameters
        }

        if is_coroutine_callable(self.executor):
            func = cast(Callable[..., Awaitable[Any]], self.executor)
            res = await func(**filtered_kwargs)
        else:
            func = cast(Callable[..., Any], self.executor)
            res = func(**filtered_kwargs)

        if isinstance(res, StepOutput):
            yield res
        else:
            yield StepOutput(content=res, success=True)


class StepMeta(BaseModel):
    """承载工作流节点装饰器元数据的内部模型"""

    name: str | None = None
    requires_confirmation: bool = False
    confirmation_message: str | None = None
    failure_policy: Any = None


class ConditionMeta(BaseModel):
    name: str | None = None
    if_true: list[Any] = Field(default_factory=list)
    if_false: list[Any] = Field(default_factory=list)


class RouterMeta(BaseModel):
    name: str | None = None
    choices: list[Any] = Field(default_factory=list)


class Steps(BaseNode):
    """串行执行的工作流容器。按照列表顺序依次执行。"""

    def __init__(self, steps: list[Any], name: str = "StepsGroup"):
        super().__init__(name=name)
        self.steps = [NodeFactory.build(step) for step in steps]

    @property
    def node_type(self) -> StepType:
        return StepType.STEPS

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        current_input = StepInput(
            input=step_input.input,
            previous_step_content=step_input.previous_step_content,
            additional_data=step_input.additional_data.copy(),
        )

        all_outputs: list[StepOutput] = []
        for step_obj in self.steps:
            step_out: StepOutput | None = None
            async for event in step_obj.aexecute_stream(current_input, context):
                if isinstance(event, StepOutput):
                    step_out = event
                else:
                    yield event

            if step_out:
                all_outputs.append(step_out)
                current_input.previous_step_content = step_out.content
                if step_out.stop:
                    break

        yield StepOutput(
            content=all_outputs[-1].content if all_outputs else "No steps executed",
            success=all(o.success for o in all_outputs),
            is_paused=any(getattr(o, "is_paused", False) for o in all_outputs),
            steps=all_outputs,
        )


class Condition(BaseNode):
    """根据条件函数的返回结果，决定走向 steps 还是 else_steps"""

    def __init__(
        self,
        evaluator: Any,
        steps: list[Any],
        else_steps: list[Any] | None = None,
        name: str = "ConditionGroup",
    ):
        super().__init__(name=name)
        self.evaluator = evaluator
        self.steps = [NodeFactory.build(step) for step in steps]
        self.else_steps = [NodeFactory.build(step) for step in (else_steps or [])]

    @property
    def node_type(self) -> StepType:
        return StepType.CONDITION

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        if callable(self.evaluator):
            sig = inspect.signature(self.evaluator)
            resolved_kwargs = await DependencyInjector.resolve_all(
                sig, call_kwargs={"step_input": step_input}, context=context
            )
            filtered_kwargs = {
                k: v for k, v in resolved_kwargs.items() if k in sig.parameters
            }
            if is_coroutine_callable(self.evaluator):
                condition_result = await cast(
                    Callable[..., Awaitable[Any]], self.evaluator
                )(**filtered_kwargs)
            else:
                condition_result = cast(Callable[..., Any], self.evaluator)(
                    **filtered_kwargs
                )
        else:
            condition_result = bool(self.evaluator)

        target_steps = self.steps if condition_result else self.else_steps
        branch_name = "if" if condition_result else "else"

        if not target_steps:
            yield StepOutput(
                content=f"条件求值为 {condition_result}，无对应步骤需执行。",
                success=True,
            )
            return

        steps_container = Steps(
            steps=target_steps, name=f"{self.name}_{branch_name}_branch"
        )
        output = None
        async for event in steps_container.aexecute_stream(step_input, context):
            if isinstance(event, StepOutput):
                output = event
            else:
                yield event
        if output:
            yield output


class Router(BaseNode):
    """根据选择器函数的返回值(名称)，从候选项中挑选步骤执行"""

    def __init__(self, choices: list[Any], selector: Any, name: str = "RouterGroup"):
        super().__init__(name=name)
        self.choices = [NodeFactory.build(c) for c in choices]
        self.selector = selector
        self._choice_map = {}
        for c in self.choices:
            if c.name:
                self._choice_map[c.name] = c

    @property
    def node_type(self) -> StepType:
        return StepType.ROUTER

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        if callable(self.selector):
            sig = inspect.signature(self.selector)
            resolved_kwargs = await DependencyInjector.resolve_all(
                sig, call_kwargs={"step_input": step_input}, context=context
            )
            filtered_kwargs = {
                k: v for k, v in resolved_kwargs.items() if k in sig.parameters
            }
            if is_coroutine_callable(self.selector):
                func = cast(Callable[..., Awaitable[Any]], self.selector)
                selected = await func(**filtered_kwargs)
            else:
                func = cast(Callable[..., Any], self.selector)
                selected = func(**filtered_kwargs)
        else:
            selected = self.selector

        if not isinstance(selected, list):
            selected = [selected]

        target_steps = []
        for s in selected:
            if isinstance(s, str):
                if s in self._choice_map:
                    target_steps.append(self._choice_map[s])
                else:
                    logger.warning(f"Router '{self.name}' 选择了未知的步骤: '{s}'")
            else:
                target_steps.append(NodeFactory.build(s))

        if not target_steps:
            yield StepOutput(content="没有命中任何有效路由分支。", success=True)
            return

        steps_container = Steps(steps=target_steps, name=f"{self.name}_routed_steps")
        output = None
        async for event in steps_container.aexecute_stream(step_input, context):
            if isinstance(event, StepOutput):
                output = event
            else:
                yield event
        if output:
            yield output


class Loop(BaseNode):
    """循环执行工作流，直至达到最大次数或满足结束条件"""

    def __init__(
        self,
        steps: list[Any],
        max_iterations: int = 3,
        end_condition: Any = None,
        name: str = "LoopGroup",
    ):
        super().__init__(name=name)
        self.steps = [NodeFactory.build(step) for step in steps]
        self.max_iterations = max_iterations
        self.end_condition = end_condition

    @property
    def node_type(self) -> StepType:
        return StepType.LOOP

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        start_event = LoopExecutionStartedEvent(
            session_id=context.session_id,
            step_name=self.name,
            max_iterations=self.max_iterations,
        )
        await EventCenter.publish(start_event)
        yield start_event

        iteration = 0
        all_results: list[StepOutput] = []
        current_input = StepInput(
            input=step_input.input,
            previous_step_content=step_input.previous_step_content,
            additional_data=step_input.additional_data.copy(),
        )

        while iteration < self.max_iterations:
            iter_start_event = LoopIterationStartedEvent(
                session_id=context.session_id,
                step_name=self.name,
                iteration=iteration + 1,
            )
            await EventCenter.publish(iter_start_event)
            yield iter_start_event

            steps_container = Steps(
                steps=self.steps, name=f"{self.name}_iter_{iteration + 1}"
            )
            iter_output = None
            async for event in steps_container.aexecute_stream(current_input, context):
                if isinstance(event, StepOutput):
                    iter_output = event
                else:
                    yield event

            should_stop = False
            if iter_output:
                all_results.append(iter_output)
                if self.end_condition:
                    if callable(self.end_condition):
                        sig = inspect.signature(self.end_condition)
                        resolved_kwargs = await DependencyInjector.resolve_all(
                            sig,
                            call_kwargs={
                                "iteration_results": iter_output.steps or [iter_output]
                            },
                            context=context,
                        )
                        filtered_kwargs = {
                            k: v
                            for k, v in resolved_kwargs.items()
                            if k in sig.parameters
                        }
                        if is_coroutine_callable(self.end_condition):
                            func = cast(
                                Callable[..., Awaitable[Any]], self.end_condition
                            )
                            should_stop = await func(**filtered_kwargs)
                        else:
                            func = cast(Callable[..., Any], self.end_condition)
                            should_stop = func(**filtered_kwargs)
                    else:
                        should_stop = bool(self.end_condition)

                iter_comp_event = LoopIterationCompletedEvent(
                    session_id=context.session_id,
                    step_name=self.name,
                    iteration=iteration + 1,
                )
                await EventCenter.publish(iter_comp_event)
                yield iter_comp_event

                iteration += 1
                if should_stop or iter_output.stop:
                    break
                current_input.previous_step_content = iter_output.content
            else:
                iteration += 1
                break

        yield StepOutput(
            content=all_results[-1].content if all_results else "No iterations run",
            success=all(o.success for o in all_results),
            is_paused=any(getattr(o, "is_paused", False) for o in all_results),
            steps=all_results,
        )

        comp_event = LoopExecutionCompletedEvent(
            session_id=context.session_id,
            step_name=self.name,
            total_iterations=iteration,
        )
        await EventCenter.publish(comp_event)
        yield comp_event


class Parallel(BaseNode):
    """并发执行的工作流容器。无序地并发执行内部所有步骤，并最终聚合成一个输出。"""

    def __init__(self, *args: Any, name: str | None = None):
        super().__init__(name=name or "ParallelGroup")
        self.steps = []
        for arg in args:
            if isinstance(arg, str) and self.name == "ParallelGroup":
                self.name = arg
            else:
                self.steps.append(NodeFactory.build(arg))

    @property
    def node_type(self) -> StepType:
        return StepType.PARALLEL

    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        start_event = ParallelExecutionStartedEvent(
            session_id=context.session_id,
            step_name=self.name,
            parallel_step_count=len(self.steps),
        )
        await EventCenter.publish(start_event)
        yield start_event

        queue = asyncio.Queue()
        bg_tasks = []

        async def worker(idx: int, s_obj: Any, c_ctx: RunContext):
            try:
                async for evt in s_obj.aexecute_stream(step_input, c_ctx):
                    await queue.put(("event", evt))
            except asyncio.CancelledError:
                pass
            except Exception as e:
                await queue.put(("error", e, getattr(s_obj, "name", f"step_{idx}")))
            finally:
                await queue.put(
                    ("done", idx, c_ctx.state, getattr(c_ctx, "upstream_results", {}))
                )

        for i, step_obj in enumerate(self.steps):
            child_context = context.clone_for_execution()
            task = asyncio.create_task(worker(i, step_obj, child_context))
            bg_tasks.append(task)

        completed = 0
        all_outputs: list[StepOutput] = []
        aggregated_content_parts = [f"## 并发执行结果汇总 [{self.name}]\n"]
        has_any_failure = False
        early_stopped = False

        while completed < len(self.steps):
            msg_type, *data = await queue.get()
            if msg_type == "event":
                if isinstance(data[0], StepOutput):
                    out = cast(StepOutput, data[0])
                    all_outputs.append(out)
                    if not out.success:
                        has_any_failure = True
                    status_icon = "✅ 成功" if out.success else "❌ 失败"
                    aggregated_content_parts.append(
                        f"### {status_icon}: {out.step_name}\n{out.content}"
                    )
                    if out.stop and not early_stopped:
                        early_stopped = True
                        logger.info(
                            f"并行分支 '{out.step_name}' 请求终止，正在取消其他并发任务..."
                        )
                        for t in bg_tasks:
                            if not t.done():
                                t.cancel()
                else:
                    yield data[0]
            elif msg_type == "error":
                err, s_name = data
                logger.error(f"并发步骤 '{s_name}' 执行崩溃: {err}")
                out = StepOutput(
                    step_name=s_name,
                    step_type=StepType.STEP,
                    content=f"执行崩溃: {err}",
                    success=False,
                    error=str(err),
                )
                all_outputs.append(out)
                has_any_failure = True
                aggregated_content_parts.append(f"### ❌ 失败: {s_name}\n{err}")
            elif msg_type == "done":
                _, child_state, child_upstream_results = data
                context.state.update(child_state)
                if (
                    not hasattr(context, "upstream_results")
                    or context.upstream_results is None
                ):
                    context.upstream_results = {}
                context.upstream_results.update(child_upstream_results)
                completed += 1

        yield StepOutput(
            content="\n\n".join(aggregated_content_parts),
            success=not has_any_failure,
            is_paused=any(getattr(o, "is_paused", False) for o in all_outputs),
            steps=all_outputs,
            stop=any(getattr(o, "stop", False) for o in all_outputs),
        )

        comp_event = ParallelExecutionCompletedEvent(
            session_id=context.session_id,
            step_name=self.name,
            parallel_step_count=len(self.steps),
            step_results=all_outputs,
        )
        await EventCenter.publish(comp_event)
        yield comp_event


class NodeFactory:
    """统一节点装配工厂"""

    @staticmethod
    def build(item: Any, name: str | None = None) -> BaseNode:
        if isinstance(item, BaseNode):
            if name and item.name in (
                "unnamed_step",
                "StepsGroup",
                "ParallelGroup",
                "ConditionGroup",
                "RouterGroup",
                "LoopGroup",
            ):
                item.name = name
            return item

        if isinstance(item, BaseRunnable) or callable(item):
            cond_meta = getattr(item, "__workflow_condition_meta__", None)
            if cond_meta:
                final_name = name or cond_meta.name or "ConditionGroup"
                return Condition(
                    evaluator=item,
                    steps=cond_meta.if_true,
                    else_steps=cond_meta.if_false,
                    name=final_name,
                )

            router_meta = getattr(item, "__workflow_router_meta__", None)
            if router_meta:
                final_name = name or router_meta.name or "RouterGroup"
                return Router(
                    selector=item,
                    choices=router_meta.choices,
                    name=final_name,
                )

            step_meta = getattr(item, "__workflow_step_meta__", None)
            if step_meta:
                final_name = name or step_meta.name
                return Step(
                    executor=item,
                    name=final_name,
                    requires_confirmation=step_meta.requires_confirmation,
                    confirmation_message=step_meta.confirmation_message,
                    failure_policy=step_meta.failure_policy,
                )

            return Step(executor=item, name=name)

        raise ValueError(
            f"无法将类型 {type(item)} 装配为工作流节点。支持的类型：BaseRunnable, Callable 或 BaseNode。"
        )
