import asyncio
from collections.abc import Callable
import json
import re
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.protocols import ToolExecutable, ToolProvider
from zhenxun.services.ai.tools.providers.context_resource import (
    PromptProvider,
    ResourceProvider,
    context_resource_manager,
)
from zhenxun.services.ai.tools.providers.mcp.toolkit import MCPToolkit
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump, model_validate

MCP_PATH = DATA_PATH / "ai" / "mcp.json"


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "sse", "streamable-http", "sandbox_proxy"] = Field(
        default="stdio", description="传输协议"
    )
    url: str | None = Field(default=None, description="SSE/HTTP URL")
    timeout: int = Field(default=30, description="请求超时时间(秒)")
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = Field(default=None, description="进程的工作目录")
    install_command: str | None = Field(
        default=None, description="首次运行前的安装/预热命令"
    )
    description: str | None = None
    trust: bool = Field(default=True, description="是否信任此服务器")
    enabled: bool = Field(default=True, description="是否全局启用")
    admin_level: int = Field(default=0, description="执行该服务器工具所需的群管等级")
    tools_meta: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="特定工具的细粒度权限与经济配置"
    )


class MCPToolsConfig(BaseModel):
    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class GlobalMCPProvider(ToolProvider, PromptProvider, ResourceProvider):
    def __init__(self):
        self._config: MCPToolsConfig | None = None
        self._toolkits: dict[str, MCPToolkit] = {}
        self._discovery_lock = asyncio.Lock()
        self._discovered_tools: dict[str, ToolExecutable] | None = None
        self.env_provider: Callable[[Any], dict[str, str]] | None = None
        self.header_provider: Callable[[Any], dict[str, str]] | None = None
        self._code_registered_servers: dict[str, tuple[MCPServerConfig, bool]] = {}

    def register_server(
        self, name: str, config: MCPServerConfig, persist: bool = False
    ) -> None:
        """代码优先：动态注册 MCP 服务 (支持无痕内存级注册与热重载)"""
        self._code_registered_servers[name] = (config, persist)

        if self._config is not None:
            if persist:
                if name not in self._config.mcpServers:
                    self._config.mcpServers[name] = config
                    self._save_config()
                else:
                    json_conf = self._config.mcpServers[name]
                    code_dict = model_dump(config, exclude_unset=True)
                    json_dict = model_dump(json_conf, exclude_unset=True)
                    merged_dict = {**code_dict, **json_dict}
                    merged_conf = model_validate(MCPServerConfig, merged_dict)

                    if model_dump(merged_conf) != model_dump(json_conf):
                        self._config.mcpServers[name] = merged_conf
                        self._save_config()

            if name not in self._toolkits:
                self._setup_toolkit(name, config)
            logger.info(
                f"🔄 MCP 代理 '{name}' 已完成动态热重载注册 (持久化: {persist})"
            )

    async def unregister_server(self, name: str) -> None:
        """动态注销 MCP 服务，关闭底层进程并从配置文件中移除"""
        if name in self._code_registered_servers:
            self._code_registered_servers.pop(name)
        if self._config and name in self._config.mcpServers:
            self._config.mcpServers.pop(name)
            self._save_config()
        if tk := self._toolkits.pop(name, None):
            await tk.close()
        logger.info(f"🛑 MCP 代理 '{name}' 已注销并销毁所有连接")

    def _save_config(self) -> None:
        """将当前配置持久化到 JSON 文件"""
        if not self._config:
            return
        try:
            with MCP_PATH.open("w", encoding="utf-8") as f:
                data = model_dump(self._config, exclude_unset=True)
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存 MCP 配置失败: {e}", e=e)

    async def initialize(self) -> None:
        if self._config is not None:
            return

        if not MCP_PATH.exists():
            MCP_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._config = MCPToolsConfig(
                mcpServers={
                    "amap-maps": MCPServerConfig(
                        command="npx",
                        args=["-y", "@amap/amap-maps-mcp-server"],
                        env={"AMAP_MAPS_API_KEY": "Your_API_Key"},
                        enabled=False,
                    ),
                    "bilibili-search": MCPServerConfig(
                        command="npx",
                        args=["bilibili-mcp"],
                    ),
                    "fetch": MCPServerConfig(
                        command="uvx",
                        args=["mcp-server-fetch"],
                    ),
                    "bingcn": MCPServerConfig(
                        command="npx",
                        args=["bing-cn-mcp"],
                    ),
                }
            )
            with MCP_PATH.open("w", encoding="utf-8") as f:
                data = model_dump(self._config, exclude_unset=True)
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            try:
                with MCP_PATH.open("r", encoding="utf-8") as f:
                    self._config = model_validate(MCPToolsConfig, json.load(f))
            except Exception as e:
                logger.error(f"加载 MCP 工具配置文件失败: {e}", e=e)
                self._config = MCPToolsConfig()

        config_changed = False
        for name, (code_conf, persist) in self._code_registered_servers.items():
            if persist:
                if name not in self._config.mcpServers:
                    self._config.mcpServers[name] = code_conf
                    config_changed = True
                else:
                    json_conf = self._config.mcpServers[name]
                    code_dict = model_dump(code_conf, exclude_unset=True)
                    json_dict = model_dump(json_conf, exclude_unset=True)
                    merged_dict = {**code_dict, **json_dict}
                    merged_conf = model_validate(MCPServerConfig, merged_dict)

                    if model_dump(merged_conf) != model_dump(json_conf):
                        self._config.mcpServers[name] = merged_conf
                        config_changed = True

        if config_changed or not MCP_PATH.exists():
            self._save_config()

        for name, conf in self._config.mcpServers.items():
            self._setup_toolkit(name, conf)

        for name, (code_conf, persist) in self._code_registered_servers.items():
            if not persist and name not in self._toolkits:
                self._setup_toolkit(name, code_conf)

    def _setup_toolkit(self, name: str, conf: MCPServerConfig) -> None:
        """辅助方法：装载单个 MCP Toolkit"""
        metadata: dict[str, dict[str, Any]] = {}
        if conf.admin_level > 0:
            metadata["*"] = {"admin_level": conf.admin_level}
        if conf.tools_meta:
            for k, v in conf.tools_meta.items():
                metadata.setdefault(k, {}).update(v)
        self._toolkits[name] = MCPToolkit(
            server_name=name,
            transport=conf.transport,
            command=conf.command,
            args=conf.args,
            url=conf.url,
            env=conf.env,
            cwd=conf.cwd,
            install_command=conf.install_command,
            timeout=conf.timeout,
            trust=conf.trust,
            tool_metadata=metadata,
            header_provider=self.header_provider,
            env_provider=self.env_provider,
        )

    async def shutdown(self):
        for tk in self._toolkits.values():
            await tk.close()
        self._toolkits.clear()

    async def get_tools_for_server(self, server_name: str) -> dict[str, ToolExecutable]:
        return await self.discover_tools(allowed_servers=[server_name])

    async def discover_tools(
        self,
        allowed_servers: list[str] | None = None,
        excluded_servers: list[str] | None = None,
    ) -> dict[str, ToolExecutable]:
        async with self._discovery_lock:
            if not self._config:
                await self.initialize()

            if not self._config:
                return {}

            if (
                self._discovered_tools is not None
                and allowed_servers is None
                and excluded_servers is None
            ):
                return self._discovered_tools

            servers_to_query: list[str] = []
            for name, conf in self._config.mcpServers.items():
                if allowed_servers and name not in allowed_servers:
                    continue
                if excluded_servers and name in excluded_servers:
                    continue
                if not conf.enabled and not allowed_servers:
                    continue
                servers_to_query.append(name)

            for name, (conf, persist) in self._code_registered_servers.items():
                if persist:
                    continue
                if allowed_servers and name not in allowed_servers:
                    continue
                if excluded_servers and name in excluded_servers:
                    continue
                if not conf.enabled and not allowed_servers:
                    continue
                if name not in servers_to_query:
                    servers_to_query.append(name)

            all_tools: dict[str, ToolExecutable] = {}
            tasks = []
            for s_name in servers_to_query:
                if tk := self._toolkits.get(s_name):
                    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", s_name)
                    prefix = f"mcp_{clean_name}_"
                    tasks.append(tk.prefixed(prefix).get_tools())

            if tasks:
                import asyncio

                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, list):
                        for t in res:
                            all_tools[t.name] = t
                    elif isinstance(res, Exception):
                        logger.error(f"MCP 并发发现工具失败: {res}")

            if allowed_servers is None and excluded_servers is None:
                self._discovered_tools = all_tools
            return all_tools

    async def get_tool_executable(
        self, name: str, config: dict[str, Any]
    ) -> ToolExecutable | None:
        _ = name, config
        return None

    async def get_prompt(self, name: str, **kwargs: Any) -> str | None:
        """通过 MCP 协议向所有启用的服务器请求 Prompt"""
        if not self._config:
            await self.initialize()

        if not self._config:
            return None

        arguments = {k: str(v) for k, v in kwargs.items()} if kwargs else None
        for s_name, config in self._config.mcpServers.items():
            if not config.enabled:
                continue
            tk = self._toolkits.get(s_name)
            if not tk:
                continue
            session = await tk.get_session()
            if not session:
                continue
            try:
                result = await session.get_prompt(name, arguments=arguments)
                text_parts = []
                if getattr(result, "description", None):
                    text_parts.append(f"Description: {result.description}")

                for msg in getattr(result, "messages", []):
                    role_val = getattr(msg, "role", "unknown")
                    role = (
                        getattr(role_val, "value", str(role_val))
                        if not isinstance(role_val, str)
                        else role_val
                    )
                    content = getattr(msg, "content", msg)
                    text = getattr(content, "text", "")
                    if isinstance(content, dict):
                        text = content.get("text", text)
                    if text:
                        text_parts.append(f"[{role}]: {text}")
                return "\n".join(text_parts)
            except Exception:
                pass
        return None

    async def read_resource(self, uri: str, **kwargs: Any) -> str | None:
        """通过 MCP 协议向所有启用的服务器读取 Resource"""
        if not self._config:
            await self.initialize()

        if not self._config:
            return None

        for s_name, config in self._config.mcpServers.items():
            if not config.enabled:
                continue
            tk = self._toolkits.get(s_name)
            if not tk:
                continue
            session = await tk.get_session()
            if not session:
                continue
            try:
                result = await session.read_resource(AnyUrl(uri))
                text_parts = []
                for content in getattr(result, "contents", []):
                    text = getattr(content, "text", "")
                    if isinstance(content, dict):
                        text = content.get("text", text)
                    if text:
                        text_parts.append(text)
                return "\n\n".join(text_parts)
            except Exception:
                pass
        return None


mcp_provider = GlobalMCPProvider()
context_resource_manager.register_prompt_provider(mcp_provider)
context_resource_manager.register_resource_provider(mcp_provider)

__all__ = ["GlobalMCPProvider", "mcp_provider"]
