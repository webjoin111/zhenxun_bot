from typing import Any

from pydantic import BaseModel, Field

from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.models import ToolResult


class HandoffArgs(BaseModel):
    reason: str = Field(
        ..., description="移交的原因、总结或需要传递给下一个负责人的上下文信息"
    )


class HandoffTool(BaseTool):
    """
    触发控制权移交 (Handoff) 信号的底层工具。
    当大模型调用此工具时，当前 Agent 执行流将被立即熔断并向外抛出 ToolHandoff 信号。
    """

    def __init__(self, target_name: str, target_description: str):
        super().__init__(
            name=f"transfer_to_{target_name}",
            description=f"将对话控制权移交给 {target_name}。专长/职责：{target_description}",
        )
        self.target_name = target_name
        self.args_schema = HandoffArgs

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        reason = kwargs.get("reason", "")
        from zhenxun.services.ai.core.exceptions import HandoffException

        raise HandoffException(
            target=self.target_name,
            payload={"reason": reason},
            display=f"🔄 正在将控制权移交给 {self.target_name}...",
        )
