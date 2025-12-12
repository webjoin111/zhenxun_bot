from collections.abc import Iterable

from zhenxun.services.agent.core.types import MCPSource
from zhenxun.services.agent.providers.mcp import mcp_provider
from zhenxun.services.llm.tools import tool_provider_manager
from zhenxun.services.llm.types import ToolExecutable


async def resolve_agent_tools(
    tool_definitions: Iterable[str | MCPSource] | None,
) -> dict[str, ToolExecutable]:
    """
    统一解析 Agent 的工具配置，支持本地工具名和 MCPSource。
    """
    resolved_tools_map: dict[str, ToolExecutable] = {}
    if not tool_definitions:
        return resolved_tools_map

    local_tool_names = [t for t in tool_definitions if isinstance(t, str)]
    if local_tool_names:
        local_tools = await tool_provider_manager.resolve_specific_tools(
            local_tool_names
        )
        resolved_tools_map.update(local_tools)

    mcp_sources = [t for t in tool_definitions if isinstance(t, MCPSource)]
    if mcp_sources:
        for source in mcp_sources:
            server_tools = await mcp_provider.get_tools_for_server(source.server_name)

            if source.tool_whitelist:
                server_tools = {
                    k: v
                    for k, v in server_tools.items()
                    if any(k.endswith(allowed) for allowed in source.tool_whitelist)
                }

            resolved_tools_map.update(server_tools)

    return resolved_tools_map
