import nonebot

from zhenxun.services.ai.types.exceptions import ToolFinishException
from zhenxun.services.ai.types.tools import (
    ToolErrorResult,
    ToolErrorType,
    ToolOptions,
    ToolOverride,
    ToolResult,
)

from .core.context import (
    CurrentBot,
    CurrentEvent,
    CurrentGroupId,
    CurrentMatcher,
    CurrentPlatform,
    CurrentSession,
    CurrentUserId,
    Hidden,
    RunContext,
    emit,
    get_current_context,
    global_dependency_registry,
    set_current_context,
)
from .core.decorators import (
    direct_reply,
    require_admin_level,
    require_approval,
    require_config,
    require_group,
    require_minimum_gold,
    require_session_state,
    require_superuser,
    silent,
    tool,
    toolkit_tool,
    with_cache,
)
from .core.response import ToolResponse
from .core.tool import BaseTool, FunctionTool
from .core.toolkit import (
    ApiConnectToolkit,
    BaseToolkit,
    GroupSharedToolkit,
    UserPersonalToolkit,
)
from .engine.middlewares import UIStreamerContext, register_global_middleware
from .engine.registry import tool_provider_manager
from .providers.context_resource import context_resource_manager
from .providers.mcp.provider import mcp_provider

tool_provider_manager.register(mcp_provider)


@nonebot.get_driver().on_shutdown
async def _shutdown_mcp_provider():
    """在服务关闭时停止所有 MCP 子进程服务器"""
    await mcp_provider.shutdown()


__all__ = [
    "ApiConnectToolkit",
    "BaseTool",
    "BaseToolkit",
    "CurrentBot",
    "CurrentEvent",
    "CurrentGroupId",
    "CurrentMatcher",
    "CurrentPlatform",
    "CurrentSession",
    "CurrentUserId",
    "FunctionTool",
    "GroupSharedToolkit",
    "Hidden",
    "RunContext",
    "ToolErrorResult",
    "ToolErrorType",
    "ToolFinishException",
    "ToolOptions",
    "ToolOverride",
    "ToolResponse",
    "ToolResult",
    "UIStreamerContext",
    "UserPersonalToolkit",
    "context_resource_manager",
    "direct_reply",
    "emit",
    "get_current_context",
    "global_dependency_registry",
    "register_global_middleware",
    "require_admin_level",
    "require_approval",
    "require_config",
    "require_group",
    "require_minimum_gold",
    "require_session_state",
    "require_superuser",
    "set_current_context",
    "silent",
    "tool",
    "tool_provider_manager",
    "toolkit_tool",
    "with_cache",
]
