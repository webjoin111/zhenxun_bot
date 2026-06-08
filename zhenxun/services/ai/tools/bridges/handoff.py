from typing import Any

from pydantic import BaseModel, Field, create_model

from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.models import ToolResult


class HandoffTool(BaseTool):
    """
    触发控制权移交 (Handoff) 信号的底层工具。
    当大模型调用此工具时，当前 Agent 执行流将被立即熔断并向外抛出 ToolHandoff 信号。
    """

    def __init__(
        self,
        target_name: str,
        target_description: str,
        input_schema: type[BaseModel] | Any | None = None,
    ):
        super().__init__(
            name=f"transfer_to_{target_name}",
            description=(
                f"将对话控制权移交给 {target_name}。专长/职责：{target_description}"
            ),
        )
        self.target_name = target_name

        actual_schema = None
        if input_schema:
            from zhenxun.services.ai.core.configs import BaseOutputDefinition

            if isinstance(input_schema, BaseOutputDefinition):
                actual_schema = input_schema.type_
            else:
                actual_schema = input_schema

        if actual_schema:
            fields: dict[str, Any] = {
                "reason": (str, Field(..., description="移交的原因或简要状态说明")),
            }
            schema_fields = getattr(
                actual_schema, "model_fields", getattr(actual_schema, "__fields__", {})
            )
            for k, v in schema_fields.items():
                fields[k] = (v.annotation, v)
            self.args_schema = create_model(
                f"DynamicHandoffArgs_{target_name}", **fields
            )
        else:

            class DefaultHandoffArgs(BaseModel):
                reason: str = Field(..., description="移交的原因或简要状态说明")
                context_data: Any = Field(
                    default="",
                    description="你需要传递给下一个负责人的所有核心数据",
                )

            self.args_schema = DefaultHandoffArgs

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        reason = kwargs.pop("reason", "")

        if "context_data" in kwargs and len(kwargs) == 1:
            context_data = kwargs["context_data"]
        else:
            context_data = kwargs

        from zhenxun.services.ai.tools.models import HandoffResult

        return HandoffResult(
            target=self.target_name,
            reason=reason,
            context_data=context_data,
            ui_display=f"🔄 正在将控制权移交给 {self.target_name}...",
        )
