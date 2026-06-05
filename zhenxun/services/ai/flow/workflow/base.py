from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from zhenxun.services.ai.core.exceptions import (
    AbortException,
    ControlFlowException,
    EndRunException,
    SubmitStructuredException,
    ToolFatalError,
)
from zhenxun.services.ai.flow.workflow.types import (
    AbortPolicy,
    BaseFailurePolicy,
    PolicyAction,
    StepInput,
    StepOutput,
    StepType,
)
from zhenxun.services.ai.run import RunContext
from zhenxun.services.log import logger


class BaseNode(ABC):
    """工作流节点统一抽象基类 (Template Method Pattern)"""

    def __init__(
        self,
        name: str,
        requires_confirmation: bool = False,
        confirmation_message: str | None = None,
        failure_policy: BaseFailurePolicy | None = None,
    ):
        self.name = name
        self.requires_confirmation = requires_confirmation
        self.confirmation_message = confirmation_message
        self.failure_policy = failure_policy or AbortPolicy()

    @property
    @abstractmethod
    def node_type(self) -> StepType:
        """节点类型标识 (供子类实现)"""
        pass

    @abstractmethod
    async def run_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        """子类必须实现的核心流式执行逻辑"""
        yield None

    async def aexecute(self, step_input: StepInput, context: RunContext) -> StepOutput:
        """非流式执行（聚合流并返回最终结果），子类无需重写"""
        output = None
        async for event in self.aexecute_stream(step_input, context):
            if isinstance(event, StepOutput):
                output = event

        if output is None:
            output = StepOutput(
                step_name=self.name,
                step_type=self.node_type,
                content="节点未产生有效输出",
                success=False,
            )
        return output

    async def aexecute_stream(
        self, step_input: StepInput, context: RunContext
    ) -> AsyncIterator[Any]:
        """标准化模板方法：处理缓存快进、授权挂起、异常熔断与生命周期事件分发"""
        logger.debug(f"  ⚙️ [节点] `{self.name}` 开始执行...")

        cached_out = context.state.get("__completed_steps__", {}).get(self.name)
        if (
            cached_out
            and cached_out.success
            and not getattr(cached_out, "is_paused", False)
        ):
            logger.debug(f"⏭️ 快进跳过已完成节点: {self.name}")

            yield cached_out
            return

        if self.requires_confirmation:
            if not context.state.get(f"__hitl_confirmed_{self.name}"):
                msg = (
                    self.confirmation_message
                    or f"⚠️ 工作流即将执行高危步骤：[{self.name}]，等待授权..."
                )
                logger.debug(f"  ⏸️ **[节点挂起]** `{self.name}`: {msg}")

                output = StepOutput(
                    step_name=self.name,
                    step_type=self.node_type,
                    content="[任务已挂起，等待人工授权/输入]",
                    success=True,
                    stop=True,
                    is_paused=True,
                    pause_reason=msg,
                )

                yield output
                return

        current_input = step_input
        attempt = 1

        while True:
            output = None
            try:
                async for event in self.run_stream(current_input, context):
                    if isinstance(event, StepOutput):
                        output = event
                        output.step_name = self.name
                        output.step_type = self.node_type
                    else:
                        yield event

                if output is None:
                    output = StepOutput(
                        step_name=self.name,
                        step_type=self.node_type,
                        content="执行完毕，无数据返回",
                        success=True,
                    )

                if (
                    not hasattr(context, "upstream_results")
                    or context.upstream_results is None
                ):
                    context.upstream_results = {}
                context.upstream_results[self.name] = output.content

                break

            except Exception as e:
                if isinstance(e, ControlFlowException):
                    logger.info(
                        f"⏭️ [控制流拦截] Node '{self.name}' 触发中断信号: {type(e).__name__} - {e}"
                    )

                    content = str(e)
                    if getattr(e, "display_content", None):
                        content = str(getattr(e, "display_content"))
                    elif getattr(e, "display", None):
                        content = str(getattr(e, "display"))
                    elif getattr(e, "result_output", None):
                        content = str(getattr(e, "result_output"))

                    output = StepOutput(
                        step_name=self.name,
                        step_type=self.node_type,
                        content=content,
                        success=isinstance(
                            e, (EndRunException, SubmitStructuredException)
                        ),
                        stop=True,
                        error=str(e)
                        if isinstance(e, (AbortException, ToolFatalError))
                        else None,
                    )
                    break
                else:
                    logger.warning(f"Node '{self.name}' 执行发生异常: {e}", e=e)

                    policy_result = await self.failure_policy.handle_failure(
                        self, e, current_input, context
                    )

                    if policy_result.action == PolicyAction.RETRY:
                        import asyncio

                        if policy_result.delay > 0:
                            await asyncio.sleep(policy_result.delay)

                        if policy_result.new_input:
                            current_input = policy_result.new_input
                            logger.debug(
                                f"  🩹 [自愈介入] 系统策略正在尝试修复 `{self.name}` 的参数输入错误..."
                            )

                        logger.debug(
                            f"  🔄 [节点重试] `{self.name}` 进行第 {attempt} 次重试..."
                        )

                        attempt += 1
                        continue

                    elif policy_result.action == PolicyAction.FALLBACK:
                        fallback_node = policy_result.fallback_node
                        fallback_name = getattr(fallback_node, "name", "FallbackNode")
                        logger.info(
                            f"🔀 节点 {self.name} 执行失败，触发降级路由至: {fallback_name}"
                        )

                        if fallback_node:
                            async for evt in fallback_node.aexecute_stream(
                                current_input, context
                            ):
                                if isinstance(evt, StepOutput):
                                    output = evt
                                else:
                                    yield evt
                        break

                    elif policy_result.action == PolicyAction.CONTINUE:
                        logger.warning(
                            f"Node '{self.name}' 执行异常，已被策略自动跳过: {e}"
                        )
                        output = StepOutput(
                            step_name=self.name,
                            step_type=self.node_type,
                            content=f"节点执行失败，已通过策略自动跳过: {e}",
                            success=False,
                            stop=False,
                            error=str(e),
                        )
                        break

                    else:
                        logger.error(
                            f"Node '{self.name}' 执行崩溃，已被策略中断执行流: {e}"
                        )
                        output = StepOutput(
                            step_name=self.name,
                            step_type=self.node_type,
                            content=f"执行崩溃: {e}",
                            success=False,
                            stop=True,
                            error=str(e),
                        )
                        break

        yield output
