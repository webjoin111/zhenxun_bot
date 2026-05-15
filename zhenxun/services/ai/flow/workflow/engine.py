from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
import uuid

if TYPE_CHECKING:
    from zhenxun.services.ai.run import StreamedRunResult

from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    WorkflowCompletedEvent,
    WorkflowErrorEvent,
    WorkflowStartedEvent,
)
from zhenxun.services.ai.flow.base import BaseRunnable, BaseRuntimeConfig
from zhenxun.services.ai.flow.workflow.nodes import Steps
from zhenxun.services.ai.flow.workflow.types import (
    StepInput,
    StepOutput,
    WorkflowRunResult,
)
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.core.tool import FunctionTool
from zhenxun.services.log import logger


class Workflow(BaseRunnable[WorkflowRunResult]):
    """
    工作流顶层容器 (The Workflow Facade)。
    继承自 BaseRunnable，支持被作为节点嵌套在 Team 或 其他工作流中。
    """

    def __init__(self, name: str, steps: list[Any], description: str = ""):
        """
        静态图元工作流容器初始化。

        参数:
            name: 工作流的名称标识。
            steps: 工作流的节点列表（按列表顺序构成串行或嵌套结构）。
            description: 工作流的说明描述，用于被 Agent 调用时理解其功能。
        """
        self.name = name
        self.description = description
        self.id = uuid.uuid4().hex

        self.root_steps = Steps(steps=steps, name=f"{self.name}_Root")
        self.runtime_config = BaseRuntimeConfig(stateless=True)
        self.persona = None

    def _build_result(
        self,
        initial_input: StepInput,
        safe_context: RunContext,
        final_output: StepOutput,
    ) -> WorkflowRunResult:
        flat_outputs = {}

        def _extract(out: StepOutput):
            flat_outputs[out.step_name] = out
            if out.steps:
                for o in out.steps:
                    _extract(o)

        if final_output:
            _extract(final_output)

        paused_step = next(
            (
                v.step_name
                for v in reversed(list(flat_outputs.values()))
                if getattr(v, "is_paused", False) and v.step_name
            ),
            None,
        )
        status = (
            "paused"
            if paused_step
            else ("completed" if final_output and final_output.success else "error")
        )

        return WorkflowRunResult(
            workflow_id=self.id,
            workflow_name=self.name,
            status=status,
            original_input=initial_input.input,
            state=safe_context.state,
            step_outputs=flat_outputs,
            last_step_content=final_output.content if final_output else None,
            final_output=final_output,
            paused_step_name=paused_step,
        )

    def bind(self, **kwargs: Any) -> Any:
        """DI 注入语法糖"""
        from nonebot.params import Depends

        async def _dependency() -> "Workflow":
            return self

        return Depends(_dependency)

    async def reply(
        self, prompt: Any = None, reply_to: bool = False, **kwargs: Any
    ) -> WorkflowRunResult:
        """
        工作流交互执行语法糖，隐式提取上下文并自动将最终流水线产出发送回复给用户。

        参数:
            prompt: 传入工作流入口根节点的初始参数或指令。
            reply_to: 是否将结果作为回复消息发送 (at用户或引用原消息)。
            kwargs: 追加的工作流附带参数 (additional_data)。

        返回:
            WorkflowRunResult: 包含执行状态、断点快照、各节点产出的全量工作流结果对象。
        """
        from zhenxun.services.ai.run.context import RunContext
        from zhenxun.utils.message import MessageUtils

        ctx = RunContext()
        bot = getattr(ctx.deps, "bot", None)
        event = getattr(ctx.deps, "event", None)

        res = await self.run(prompt=prompt, context=ctx, **kwargs)

        if bot and event:
            if res.status == "completed" and res.final_output:
                msg = (
                    str(res.final_output.content)
                    if res.final_output.content
                    else "执行完毕"
                )
                await MessageUtils.build_message(msg).send(reply_to=reply_to)
            elif res.status == "paused":
                pause_msg = f"⏸️ 工作流执行已被挂起，停在步骤: {res.paused_step_name}。请提供授权或人工输入后继续。"
                await MessageUtils.build_message(pause_msg).send(reply_to=reply_to)
            elif res.status == "error":
                err_msg = res.final_output.error if res.final_output else "未知异常"
                await MessageUtils.build_message(
                    f"❌ 工作流执行发生错误: {err_msg}"
                ).send(reply_to=reply_to)

        return res

    async def run(
        self, prompt: Any = None, *, context: RunContext | None = None, **kwargs: Any
    ) -> WorkflowRunResult:
        """
        工作流单次运行阻塞核心入口，遍历所有图元节点直至终止。

        参数:
            prompt: 传入工作流入口根节点的初始参数或指令。
            context: 显式传入的会话与运行上下文。
            kwargs: 追加的工作流附带参数 (additional_data)。

        返回:
            WorkflowRunResult: 包含执行状态、断点快照、各节点产出的全量工作流结果对象。
        """
        session_id = (
            context.session_id if context and context.session_id else f"wf_{self.id}"
        )
        safe_context = context or RunContext(session_id=session_id)

        await EventCenter.publish(
            WorkflowStartedEvent(session_id=session_id, workflow_name=self.name)
        )

        initial_input = StepInput(input=prompt)
        if kwargs:
            initial_input.additional_data.update(kwargs)

        try:
            final_output = await self.root_steps.aexecute(initial_input, safe_context)

            await EventCenter.publish(
                WorkflowCompletedEvent(
                    session_id=session_id, workflow_name=self.name, result=final_output
                )
            )

            return self._build_result(initial_input, safe_context, final_output)

        except Exception as e:
            logger.error(f"Workflow '{self.name}' 执行崩溃: {e}", e=e)
            await EventCenter.publish(
                WorkflowErrorEvent(
                    session_id=session_id, workflow_name=self.name, error=e
                )
            )
            raise e

    import contextlib

    @contextlib.asynccontextmanager
    async def run_stream(
        self, prompt: Any = None, *, context: RunContext | None = None, **kwargs: Any
    ) -> AsyncIterator["StreamedRunResult[Any]"]:
        """对齐 BaseRunnable 接口的流式上下文管理器"""
        import asyncio

        from zhenxun.services.ai.core.stream_events import EventStreamer
        from zhenxun.services.ai.run import StreamedRunResult
        from zhenxun.services.ai.run.models import AgentRunError

        streamer = EventStreamer()
        if context:
            context.run.streamer = streamer

        async def _execution_task():
            try:
                async for event in self._internal_stream(prompt, context, **kwargs):
                    await streamer.send(event)
            except BaseException as e:
                await streamer.send(AgentRunError(error=e))
            finally:
                await streamer.end()

        task = asyncio.create_task(_execution_task())
        try:
            yield StreamedRunResult[Any](streamer)
        finally:
            if not task.done():
                task.cancel()

    async def _internal_stream(
        self, prompt: Any = None, context: RunContext | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """原 arun_stream 逻辑改名，供内部 _execution_task 调用"""
        session_id = (
            context.session_id if context and context.session_id else f"wf_{self.id}"
        )
        safe_context = context or RunContext(session_id=session_id)

        start_event = WorkflowStartedEvent(
            session_id=session_id, workflow_name=self.name
        )
        await EventCenter.publish(start_event)
        yield start_event

        initial_input = StepInput(input=prompt)
        if kwargs:
            initial_input.additional_data.update(kwargs)

        try:
            final_output = None
            async for event in self.root_steps.aexecute_stream(
                initial_input, safe_context
            ):
                if isinstance(event, StepOutput):
                    final_output = event
                else:
                    yield event

            if final_output:
                comp_event = WorkflowCompletedEvent(
                    session_id=session_id, workflow_name=self.name, result=final_output
                )
                await EventCenter.publish(comp_event)
                yield comp_event

                from zhenxun.services.ai.core.messages import UsageInfo
                from zhenxun.services.ai.run import AgentRunResult
                from zhenxun.services.ai.run.models import AgentRunEnd

                wf_result = self._build_result(
                    initial_input, safe_context, final_output
                )
                agent_res = AgentRunResult(
                    output=wf_result.last_step_content,
                    structured_data=wf_result,
                    usage=UsageInfo(),
                )
                yield AgentRunEnd(result=agent_res)
        except Exception as e:
            logger.error(f"Workflow '{self.name}' 流式执行崩溃: {e}", e=e)
            yield WorkflowErrorEvent(
                session_id=session_id, workflow_name=self.name, error=e
            )

    async def acontinue_run(
        self,
        run_result: WorkflowRunResult,
        user_auth_data: dict[str, Any] | None = None,
        context: RunContext | None = None,
    ) -> WorkflowRunResult:
        safe_context = context or RunContext(session_id=f"wf_{self.id}")
        safe_context.state.update(run_result.state)

        safe_context.state["__completed_steps__"] = run_result.step_outputs.copy()
        for step_name, out in run_result.step_outputs.items():
            safe_context.upstream_results[step_name] = out.content

        if run_result.paused_step_name:
            safe_context.state[f"__hitl_confirmed_{run_result.paused_step_name}"] = True
            if user_auth_data:
                safe_context.state[f"__hitl_input_{run_result.paused_step_name}"] = (
                    user_auth_data
                )

        resume_input = StepInput(
            input=run_result.original_input,
            previous_step_content=run_result.last_step_content,
        )

        logger.info(
            f"🚀 工作流 [{self.name}] 状态已恢复，正在快进到步骤: {run_result.paused_step_name}..."
        )

        final_output = await self.root_steps.aexecute(resume_input, safe_context)

        return self._build_result(resume_input, safe_context, final_output)

    def as_tool(self, tool_name: str | None = None) -> FunctionTool:
        async def _execute_workflow_tool(prompt: str, context: RunContext) -> str:
            run_result = await self.run(prompt=prompt, context=context)
            output = run_result.final_output

            if output and output.success:
                return (
                    f"工作流 [{self.name}] 执行完毕。最终流水线产出:\n{output.content}"
                )
            from zhenxun.services.ai.core.exceptions import ToolRetryError

            raise ToolRetryError(
                f"工作流执行失败: {output.error if output else 'unknown'}，请尝试换种方式处理。"
            )

        final_tool_name = tool_name or f"trigger_workflow_{self.id}"

        tool_desc = (
            f"触发执行专属流水线: {self.name}。\n"
            f"描述: {self.description}\n"
            f"注意：如果你认为该工作流能完全解决用户的问题，请立刻调用此工具，"
            f"并将用户的诉求提炼后作为 prompt 传入。"
        )

        return FunctionTool(
            func=_execute_workflow_tool, name=final_tool_name, description=tool_desc
        )
