from typing import Any, cast

from zhenxun.services.ai.core.configs import BaseOutputDefinition, ToolOutput
from zhenxun.services.ai.core.engine.structured_parser import (
    BaseOutputProcessor,
    SubmitFinalResultExecutable,
)
from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    TaskRunEndEvent,
    TaskRunErrorEvent,
    TaskRunStartEvent,
)
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.memory.long_term_memory import MemoryScope
from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.ai.protocols.capabilities import AbstractCapability
from zhenxun.services.ai.run import AgentRunResult, RunContext, Task
from zhenxun.services.log import logger


class OutputValidationCapability(AbstractCapability):
    """输出拦截与校验能力组件 (支持纯文本及结构化护栏)"""

    def __init__(
        self, output_type: Any | None = None, guardrails: list[Any] | None = None
    ):
        self.output_type = output_type

        from zhenxun.services.ai.core.guardrails import parse_guardrails

        self.guardrails = parse_guardrails(guardrails)
        self.processor = None
        self.submit_tool = None

        if self.output_type is not None:
            if isinstance(self.output_type, BaseOutputDefinition):
                out_type = self.output_type.type_
                tool_name_override = (
                    self.output_type.name
                    if isinstance(self.output_type, ToolOutput)
                    else None
                )
            else:
                out_type = cast(type[Any], self.output_type)
                tool_name_override = None

            self.processor = BaseOutputProcessor(
                response_model=out_type,
            )
            self.submit_tool = SubmitFinalResultExecutable(
                self.processor, self.guardrails
            )
            if tool_name_override:
                self.submit_tool.tool_name = tool_name_override
                self.submit_tool.name = tool_name_override

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        """动态注入结构化要求提示词"""
        if self.submit_tool:
            return [
                "### ⚠️ [核心任务：结构化输出要求]\n"
                "当前任务处于严格的 **结构化输出模式**。\n"
                "当你完成所有调查和思考后，必须且只能调用 "
                f"`{self.submit_tool.tool_name}` 工具来提交最终结果，"
                "禁止用纯文本直接作答。\n"
                "（📌 提示：最终需要返回的数据结构要求，请严格查阅并遵循 "
                f"`{self.submit_tool.tool_name}` 工具的参数 Schema 定义，"
                "将其视为唯一的数据约束）"
            ]
        return []

    async def get_tools(self, context: RunContext) -> list[Any]:
        """动态挂载提交最终结果的工具"""
        if self.submit_tool:
            return [self.submit_tool]
        return []

    async def before_model_request(self, context, llm_context):
        """将 Processor 和 Guardrails 传给底层的 IvrCapability"""
        llm_context.extra["output_processor"] = self.processor
        llm_context.extra["guardrails"] = self.guardrails
        return llm_context

    async def after_run(
        self, context: RunContext, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        """运行结束后，校验是否成功提取了结构化数据"""
        if self.output_type is not None:
            if result.structured_data is not None:
                result.output = result.structured_data
            else:
                tool_name = (
                    self.submit_tool.tool_name if self.submit_tool else "unknown"
                )
                logger.error(f"Agent 未能调用 {tool_name} 提交结构化数据。")
                raise LLMException(
                    "模型未能输出符合要求的结构化数据。",
                    code=LLMErrorCode.GENERATION_FAILED,
                )
        return result


class LongTermMemoryCapability(AbstractCapability):
    """长期记忆 (RAG) 检索增强能力组件"""

    def __init__(self, memory_scope: MemoryScope, session_meta: SessionMetadata):
        self.memory_scope = memory_scope
        self.session_meta = session_meta

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        """
        在系统构建 Prompt 时触发。
        利用当前用户的输入去长期记忆中进行语义检索 (Recall)，并将结果转化为系统补充设定。
        """
        user_input = context.run.user_input
        if not user_input:
            return []

        matches = await self.memory_scope.recall(
            session=self.session_meta, query=user_input
        )
        if not matches:
            return []

        fact_str = "\n".join(
            f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
        )

        logger.debug(
            f"🧠 [LTM Capability] 成功召回 {len(matches)} 条长期记忆并注入上下文。"
        )
        return [f"[系统补充：有关用户的长期记忆设定]\n{fact_str}"]


class TaskTrackingCapability(AbstractCapability):
    """数据契约任务状态追踪与事件遥测组件"""

    def __init__(self, task: Task, agent_name: str):
        self.task = task
        self.agent_name = agent_name

    async def before_run(self, context: RunContext) -> None:
        """任务开始时发布 Start 事件"""
        task_name = self.task.name or self.task.id[:8]
        await EventCenter.publish(
            TaskRunStartEvent(
                session_id=context.session_id or "unknown",
                task_id=self.task.id,
                task_name=task_name,
                agent_name=self.agent_name,
            )
        )

    async def after_run(
        self, context: RunContext, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        """任务成功结束时发布 End 事件"""
        task_name = self.task.name or self.task.id[:8]
        await EventCenter.publish(
            TaskRunEndEvent(
                session_id=context.session_id or "unknown",
                task_id=self.task.id,
                task_name=task_name,
            )
        )
        return result

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> AgentRunResult[Any]:
        """任务发生异常时发布 Error 事件"""
        task_name = self.task.name or self.task.id[:8]
        event_error = error if isinstance(error, Exception) else Exception(str(error))
        await EventCenter.publish(
            TaskRunErrorEvent(
                session_id=context.session_id or "unknown",
                task_id=self.task.id,
                task_name=task_name,
                error=event_error,
            )
        )
        raise error
