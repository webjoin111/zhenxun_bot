"""
MCP 子进程工具提供者。

负责从 mcp_tools.json 加载、实例化和执行通过子进程提供的 MCP 服务器工具。
"""

import asyncio
from contextlib import AsyncExitStack
import json
from typing import Any

from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.cache import Cache
from zhenxun.services.llm.types import ToolExecutable, ToolProvider
from zhenxun.services.llm.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.llm.types.models import ToolDefinition, ToolResult
from zhenxun.services.llm.utils import sanitize_schema_for_llm
from zhenxun.services.log import logger
from zhenxun.utils.decorator.retry import Retry
from zhenxun.utils.pydantic_compat import model_dump

from ..core.context import get_tool_trust_policy

MCP_TOOLS_CONFIG_PATH = DATA_PATH / "llm" / "mcp_tools.json"
MCP_TOOLS_CACHE_TYPE = "MCP_TOOLS"


class MCPServerConfig(BaseModel):
    """单个 MCP 服务器的配置模型"""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    description: str | None = None
    trust: bool = Field(
        False, description="是否信任此服务器。如果为False，执行其工具前需要用户确认。"
    )
    enabled: bool = Field(
        False,
        description="是否在全局工具发现中启用。显式指定的Agent(allowed_servers)可忽略此选项。",
    )


class MCPToolsConfig(BaseModel):
    """mcp_tools.json 文件的顶层模型"""

    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class SubprocessToolExecutable(ToolExecutable):
    """通过子进程执行的工具。"""

    def __init__(
        self,
        name: str,
        server_function_name: str,
        description: str,
        parameters: dict[str, Any],
        provider: "MCPSubprocessProvider",
        server_name: str,
    ):
        self._name = name
        self._server_function_name = server_function_name
        self._description = description
        self._parameters = parameters
        self._provider = provider
        self._server_name = server_name
        self._definition = ToolDefinition(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )

    @property
    def server_name(self) -> str:
        """获取所属服务器名称"""
        return self._server_name

    @property
    def provider(self) -> "MCPSubprocessProvider":
        """获取所属 Provider 实例"""
        return self._provider

    async def should_confirm(self, **kwargs: Any) -> str | None:
        """
        根据服务器配置和运行时信任策略决定是否需要用户确认。
        """
        runtime_policy = get_tool_trust_policy()
        if runtime_policy and runtime_policy.trusts_server(self._server_name):
            return None

        server_config = self._provider.get_server_config(self._server_name)
        if server_config and not server_config.trust:
            args_str = json.dumps(kwargs, ensure_ascii=False)
            return (
                f"即将执行来自【{self._server_name}】服务器的工具【{self._server_function_name}】"
                f"，参数为：\n`{args_str}`\n\n"
                "这可能会与外部服务交互或修改本地文件。是否确认执行？"
            )
        return None

    async def get_definition(self) -> ToolDefinition:
        """直接返回初始化时构建的工具定义。"""
        return self._definition

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        """通过已建立的 ClientSession 来执行工具。"""
        _ = context
        session = await self._provider.get_or_create_session(self._server_name)
        if not session:
            raise LLMException(
                f"无法找到或创建到 MCP 服务器 '{self._server_name}' 的活动会话。",
                LLMErrorCode.CONFIGURATION_ERROR,
            )

        try:
            mcp_result: mcp_types.CallToolResult = await session.call_tool(
                name=self._server_function_name,
                arguments=kwargs,
            )

            if mcp_result.isError:
                raise LLMException(
                    f"MCP 工具 '{self._server_function_name}' 执行返回错误: "
                    f"{mcp_result.content}",
                    LLMErrorCode.GENERATION_FAILED,
                )

            output_content = [model_dump(item) for item in mcp_result.content]

            return ToolResult(
                output=output_content,
                display_content=json.dumps(
                    output_content, ensure_ascii=False, indent=2
                ),
            )
        except Exception as e:
            msg = f"执行 MCP 工具 '{self._server_function_name}' 失败: {e}"
            raise LLMException(msg, cause=e)


class MCPSubprocessProvider(ToolProvider):
    """
    负责从 mcp_tools.json 加载所有基于子进程的 MCP 服务器工具。
    """

    def __init__(self):
        self._config: MCPToolsConfig | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()
        self._discovery_lock = asyncio.Lock()
        self._discovered_tools: dict[str, ToolExecutable] | None = None
        self._tool_definition_cache = Cache[list[dict]](MCP_TOOLS_CACHE_TYPE)

    def get_server_config(self, server_name: str) -> MCPServerConfig | None:
        """获取指定服务器的配置。"""
        if self._config:
            return self._config.mcpServers.get(server_name)
        return None

    async def initialize(self) -> None:
        """从 mcp_tools.json 加载配置。如果文件不存在，则创建默认配置。"""
        if self._config is not None:
            return

        if not MCP_TOOLS_CONFIG_PATH.exists():
            logger.info(
                f"未找到 MCP 工具配置文件，将在 '{MCP_TOOLS_CONFIG_PATH}' 创建一个"
            )
            MCP_TOOLS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            default_config_data = {
                "mcpServers": {
                    "baidu-map": {
                        "description": "百度地图工具，提供地理编码、路线规划等功能。",
                        "command": "npx",
                        "args": ["-y", "@baidumap/mcp-server-baidu-map"],
                        "env": {"BAIDU_MAP_API_KEY": "<YOUR_BAIDU_MAP_API_KEY>"},
                        "enabled": False,
                    },
                    "sequential-thinking": {
                        "description": "顺序思维工具，用于帮助模型进行多步骤推理。",
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-sequential-thinking",
                        ],
                        "enabled": False,
                    },
                }
            }
            self._config = MCPToolsConfig.model_validate(default_config_data)
            with MCP_TOOLS_CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(
                    model_dump(self._config),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            try:
                with MCP_TOOLS_CONFIG_PATH.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    self._config = MCPToolsConfig.model_validate(config_data)
            except Exception as e:
                logger.error(f"加载 MCP 工具配置文件失败: {e}", e=e)
                self._config = MCPToolsConfig()

    async def shutdown(self):
        """关闭所有与MCP服务器的连接。"""
        logger.info(f"正在关闭与 {len(self._sessions)} 个 MCP 服务器的连接...")
        await self._exit_stack.aclose()
        self._sessions.clear()
        logger.info("所有 MCP 连接已关闭。")

    async def get_or_create_session(self, server_name: str) -> ClientSession | None:
        """按需获取或创建到指定服务器的会话。"""
        if server_name in self._sessions:
            return self._sessions[server_name]

        if not self._config:
            await self.initialize()

        if self._config and (config := self._config.mcpServers.get(server_name)):
            try:
                params = StdioServerParameters(
                    command=config.command, args=config.args, env=config.env
                )
                transport = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._exit_stack.enter_async_context(
                    ClientSession(*transport)
                )
                await session.initialize()
                self._sessions[server_name] = session
                logger.info(f"懒加载：成功连接到 MCP 服务器: '{server_name}'")
                return session
            except Exception as e:
                logger.error(
                    f"懒加载：连接到 MCP 服务器 '{server_name}' 失败: {e}", e=e
                )
                return None
        return None

    async def get_tools_for_server(self, server_name: str) -> dict[str, ToolExecutable]:
        """
        [新增] 显式获取指定服务器的工具，如果尚未连接则建立连接。
        这是 Agent 按需加载的核心入口。
        """
        return await self.discover_tools(allowed_servers=[server_name])

    async def clear_discovery_cache(self, server_name: str | None = None):
        """
        手动清除工具发现的缓存。

        参数:
            server_name: (可选) 如果提供，则只清除指定服务器的缓存。
                         否则，清除所有已知服务器的缓存。
        """
        if not self._config:
            await self.initialize()

        if not self._config:
            return

        servers_to_clear = (
            [server_name] if server_name else self._config.mcpServers.keys()
        )
        cleared_count = 0
        for name in servers_to_clear:
            cache_key = f"definitions:{name}"
            if await self._tool_definition_cache.delete(cache_key):
                cleared_count += 1
                logger.info(f"已清除 MCP 工具缓存: {name}")

        self._discovered_tools = None
        logger.info(f"共清除了 {cleared_count} 个服务器的工具定义缓存。")

    async def discover_tools(
        self,
        allowed_servers: list[str] | None = None,
        excluded_servers: list[str] | None = None,
    ) -> dict[str, ToolExecutable]:
        """根据加载的配置发现并实例化所有 MCP 子进程工具。"""
        async with self._discovery_lock:
            if (
                self._discovered_tools is not None
                and allowed_servers is None
                and excluded_servers is None
            ):
                return self._discovered_tools

            if self._config is None:
                await self.initialize()

            if not self._config or not self._config.mcpServers:
                return {}

            all_servers = self._config.mcpServers
            servers_to_query: list[str] = []

            if allowed_servers is not None:
                servers_to_query = [
                    s for s in all_servers.keys() if s in allowed_servers
                ]
            else:
                servers_to_query = [
                    s for s, conf in all_servers.items() if conf.enabled
                ]

            if excluded_servers is not None:
                servers_to_query = [
                    s for s in servers_to_query if s not in excluded_servers
                ]

            if not servers_to_query:
                return {}

            all_tools: dict[str, ToolExecutable] = {}

            discovery_tasks = [
                self._discover_server_tools(server_name)
                for server_name in servers_to_query
            ]

            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    all_tools.update(result)
                elif isinstance(result, Exception):
                    logger.error(f"在并行发现工具时捕获到错误: {result}", e=result)

            if allowed_servers is None and excluded_servers is None:
                self._discovered_tools = all_tools

            return all_tools

    @Retry.simple(
        stop_max_attempt=3,
        wait_fixed_seconds=2,
        log_name="MCP Tool Discovery",
        return_on_failure={},
    )
    async def _discover_server_tools(
        self, server_name: str
    ) -> dict[str, ToolExecutable]:
        """
        发现单个服务器的工具，增加了缓存和重试机制。
        """
        cache_key = f"definitions:{server_name}"

        cached_definitions = await self._tool_definition_cache.get(cache_key)
        if cached_definitions is not None:
            logger.debug(f"缓存命中：从缓存加载服务器 '{server_name}' 的工具定义。")
            if not isinstance(cached_definitions, list):
                logger.warning(f"缓存中 '{server_name}' 的数据格式不正确，将忽略缓存。")
            else:
                return self._create_executables_from_definitions(
                    server_name, cached_definitions
                )

        logger.debug(f"缓存未命中：开始实时发现服务器 '{server_name}' 的工具。")

        tools: dict[str, ToolExecutable] = {}
        session = await self.get_or_create_session(server_name)

        if not session or not self._config:
            return {}

        try:
            mcp_tools_result = await session.list_tools()
            raw_definitions = [model_dump(t) for t in mcp_tools_result.tools]

            await self._tool_definition_cache.set(
                cache_key, raw_definitions, expire=3600
            )
            logger.debug(
                f"已将服务器 '{server_name}' 的工具定义存入缓存，有效期1小时。"
            )

            tools = self._create_executables_from_definitions(
                server_name, raw_definitions
            )
            logger.info(f"从 MCP 服务器 '{server_name}' 发现了 {len(tools)} 个工具。")
        except Exception as e:
            logger.error(f"发现 MCP 服务器 '{server_name}' 的工具时失败: {e}", e=e)
            raise

        return tools

    def _create_executables_from_definitions(
        self, server_name: str, definitions: list[dict]
    ) -> dict[str, ToolExecutable]:
        """从工具定义字典列表创建 ToolExecutable 实例。"""
        tools: dict[str, ToolExecutable] = {}
        if not self._config:
            return {}

        config = self._config.mcpServers.get(server_name)
        if not config:
            return {}

        for tool_decl in definitions:
            try:
                func_name = tool_decl.get("name")
                description = tool_decl.get("description")
                parameters = tool_decl.get("inputSchema") or tool_decl.get("parameters")

                if not func_name or not description or parameters is None:
                    logger.warning(
                        f"在 '{server_name}' 中发现不完整的工具定义,已跳过: {tool_decl}"
                    )
                    continue

                sanitized_parameters = sanitize_schema_for_llm(
                    parameters, api_type="gemini"
                )

                unique_tool_name = (
                    f"{server_name.replace('-', '_')}_{func_name.replace('-', '_')}"
                )

                tools[unique_tool_name] = SubprocessToolExecutable(
                    name=unique_tool_name,
                    server_function_name=func_name,
                    description=description
                    or config.description
                    or f"A tool from {server_name}",
                    parameters=sanitized_parameters,
                    provider=self,
                    server_name=server_name,
                )
            except Exception as e:
                logger.error(
                    f"从定义创建工具 '{tool_decl.get('name')}' 时出错: {e}", e=e
                )

        return tools

    async def get_tool_executable(
        self, name: str, config: dict[str, Any]
    ) -> ToolExecutable | None:
        _ = name, config
        return None


mcp_provider = MCPSubprocessProvider()

__all__ = ["MCPSubprocessProvider", "mcp_provider"]
