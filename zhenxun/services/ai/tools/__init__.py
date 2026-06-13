import nonebot

from zhenxun.services.ai.core.exceptions import ToolFinishException

from .bridges.matcher_bridge import MatcherTool, bind_matcher
from .core.decorators import Rules, tool
from .core.schema import FieldPermission, RequireAdminLevel, RequireSuperUser
from .core.toolkit import BaseToolkit
from .engine.global_capabilities import register_global_capability
from .engine.registry import tool_provider_manager
from .models import (
    ToolOptions,
    ToolResult,
)
from .providers.mcp.provider import mcp_provider
from .providers.builtin.native import (
    WebSearchTool,
    CodeExecutionTool,
    ComputerUseTool,
    FileSearchTool,
    GoogleMapsTool,
    UrlContextTool
)

tool_provider_manager.register(mcp_provider)


@nonebot.get_driver().on_shutdown
async def _shutdown_mcp_provider():
    """在服务关闭时停止所有 MCP 子进程服务器"""
    await mcp_provider.shutdown()


__all__ = [
    "BaseToolkit",
    "FieldPermission",
    "MatcherTool",
    "RequireAdminLevel",
    "RequireSuperUser",
    "Rules",
    "ToolFinishException",
    "ToolOptions",
    "ToolResult",
    "bind_matcher",
    "register_global_capability",
    "tool",
    "WebSearchTool",
    "CodeExecutionTool",
    "ComputerUseTool",
    "FileSearchTool",
    "GoogleMapsTool",
    "UrlContextTool"
]
