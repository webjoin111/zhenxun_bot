import json
from typing import Any
import uuid

from pydantic import BaseModel, Field

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.types.tools import ToolResult
from zhenxun.services.log import logger


class AgentToolArgs(BaseModel):
    task: str = Field(
        ..., description="指派给该智能体的具体任务描述、指令或需要回答的问题"
    )


class AgentTool(BaseTool):
    """
    将一个 Agent 包装为可被其他大模型调用的工具 (Agent as a Tool)。
    """

    def __init__(
        self, agent: Any, name: str | None = None, description: str | None = None
    ):
        resolved_name = name or getattr(agent, "name", "SubAgent")
        resolved_desc = description or getattr(
            agent, "instruction", f"调用 {resolved_name} 执行子任务"
        )
        super().__init__(name=resolved_name, description=resolved_desc)
        self.agent = agent
        self.args_schema = AgentToolArgs

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        task = kwargs.get("task", "")
        context = context or RunContext()

        depth = context.extra.get("agent_call_depth", 0)
        if depth >= 3:
            logger.warning(
                f"⚠️ [AgentAsTool] Agent 调用深度超限 ({depth})，强制阻断: {self.name}"
            )
            error_payload = {
                "error_type": "DepthLimitExceeded",
                "message": f"嵌套层级过深，系统已强制拒绝执行 {self.name}。",
                "is_retryable": False,
            }
            return ToolResult(
                output=error_payload,
                display=f"⚠️ {self.name} 嵌套层级过深",
                is_error=True,
                terminate_run=True,
            )

        base_session = context.session_id or "default"
        sub_session_id = f"{base_session}_sub_{self.name}_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"🔄 [AgentAsTool] 正在挂载子智能体 {self.name} (Task: {task[:30]}...)"
        )

        new_state = context.extra.copy()
        new_state["agent_call_depth"] = depth + 1

        try:
            response = await self.agent.run(
                prompt=task,
                session_id=sub_session_id,
                injected_state=new_state,
                bot=context.bot,
                event=context.event,
                matcher=context.matcher,
                deps=context.deps,
            )

            output_str = str(response.output)

            try:
                final_output = json.loads(output_str)
            except Exception:
                final_output = output_str

            is_fatal = (
                "DepthLimitExceeded" in output_str or "嵌套层级过深" in output_str
            )

            return ToolResult(
                output=final_output,
                display=f"🧠 子智能体 {self.name} 执行完毕",
                terminate_run=is_fatal,
            )
        except Exception as e:
            logger.error(f"子智能体 {self.name} 执行失败: {e}", e=e)
            return ToolResult(
                output=f"Agent Execution Error: {e}", is_error=True, terminate_run=True
            )

