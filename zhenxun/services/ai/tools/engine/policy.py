from typing import Any

from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.models import ToolOptions


class ToolExecutionPolicy:
    """
    工具执行策略 (Strategy Pattern)。
    负责解析工具私有配置与系统全局配置，决定最大重试次数、Fallback 路由目标等流转行为。
    """

    def __init__(self, tool: BaseTool, global_max_retries: int = 0):
        self.tool = tool
        self.settings: ToolOptions = getattr(tool, "settings", ToolOptions())
        self.metadata: dict[str, Any] = (
            self.settings.metadata if self.settings else getattr(tool, "metadata", {})
        )
        self.global_max_retries = global_max_retries

    @property
    def max_retries(self) -> int:
        """
        计算当前工具的绝对最大重试次数。
        优先使用工具级配置 (ToolOptions.max_retries)，如果未设置，则使用全局配置。
        由于重试机制是保证 Agent 稳定性的防线，即使全局为 0，底层默认也会给予至少 1 次的机会。
        """
        tool_retries = getattr(self.settings, "max_retries", None)
        if tool_retries is not None:
            return tool_retries
        return max(self.global_max_retries, 1)
