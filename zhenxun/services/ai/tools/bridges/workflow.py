from typing import Any

from pydantic import BaseModel, Field

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.types.tools import ToolResult
from zhenxun.services.log import logger


class WorkflowToolArgs(BaseModel):
    initial_input: str = Field(
        ..., description="触发该工作流的初始输入、任务描述或核心数据"
    )


class WorkflowTool(BaseTool):
    """将 Workflow (如 SequenceWorkflow) 包装为大模型工具"""

    def __init__(
        self, workflow: Any, name: str | None = None, description: str | None = None
    ):
        resolved_name = name or getattr(workflow, "_name", "SubWorkflow")
        resolved_desc = description or f"调用 {resolved_name} 执行预设的流水线任务"
        super().__init__(name=resolved_name, description=resolved_desc)
        self.workflow = workflow
        self.args_schema = WorkflowToolArgs

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        initial_input = kwargs.get("initial_input", "")
        context = context or RunContext()

        logger.info(f"🌊 [WorkflowAsTool] 正在挂载并执行子工作流: {self.name}")

        try:
            response = await self.workflow.run(
                initial_input=initial_input,
                deps=context.deps,
                bot=context.bot,
                event=context.event,
                matcher=context.matcher,
            )
            return ToolResult(
                output=str(response.output),
                display=f"✅ 工作流 {self.name} 执行完毕",
            )
        except Exception as e:
            logger.error(f"工作流 {self.name} 执行失败: {e}", e=e)
            return ToolResult(output=f"Workflow Execution Error: {e}", is_error=True)

