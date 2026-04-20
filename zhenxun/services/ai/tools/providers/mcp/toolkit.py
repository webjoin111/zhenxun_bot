import asyncio
import base64
from collections.abc import Callable
from contextlib import AsyncExitStack
import json
import re
from typing import Any, Literal, cast

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from zhenxun.services.ai.sandbox.extension import BaseMcpProxyPlugin
from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.types.sandbox import SandboxSecurityProfile
from zhenxun.services.ai.types.tools import ToolDefinition, ToolResult
from zhenxun.services.ai.utils.lifespan import ResourceLifespanMixin
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump


class MCPRemoteTool(BaseTool):
    """远端 MCP 工具在本地的代理对象（自动继承真寻生态）"""

    def __init__(
        self,
        name: str,
        original_tool_name: str,
        description: str,
        parameters: dict,
        toolkit: "MCPToolkit",
    ):
        super().__init__(name=name, description=description)
        self.original_tool_name = original_tool_name
        self.parameters = parameters
        self.toolkit = toolkit
        self.args_schema = None
        meta = toolkit.tool_metadata.get("*", {}).copy()
        meta.update(toolkit.tool_metadata.get(name, {}))
        self.metadata = meta

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            metadata=self.metadata or {},
        )
        if context and self.settings.prepare:
            from nonebot.utils import is_coroutine_callable

            if is_coroutine_callable(self.settings.prepare):
                tool_def = await self.settings.prepare(context, tool_def)
            else:
                tool_def = self.settings.prepare(context, tool_def)
        return tool_def

    async def should_confirm(self, **kwargs: Any) -> str | None:
        from zhenxun.services.ai.agent.core.context import get_tool_trust_policy

        policy = get_tool_trust_policy()
        if policy and policy.trusts_server(self.toolkit.server_name):
            return None
        if not self.toolkit.trust:
            args_str = json.dumps(kwargs, ensure_ascii=False)
            return (
                f"即将执行来自 [{self.toolkit.server_name}] 的远程工具 [{self.name}]\n"
                f"参数：{args_str}\n\n该服务器未完全受信任，是否确认执行？"
            )
        return await super().should_confirm(**kwargs)

    async def execute(self, context: RunContext | None = None, **kwargs) -> ToolResult:
        self.toolkit.touch(self.toolkit.server_name)
        session = await self.toolkit.get_session(context)
        if not session:
            return ToolResult(output="MCP Connection Error", is_error=True)
        try:
            result = await session.call_tool(self.original_tool_name, kwargs)
        except Exception as e:
            logger.warning(
                f"MCP Tool '{self.name}' execute failed "
                f"(possible connection loss): {e}. "
                "Attempting self-healing reconnect..."
            )
            await self.toolkit.close()
            session = await self.toolkit.get_session(context)
            if not session:
                return ToolResult(output="MCP Reconnection Failed", is_error=True)
            try:
                result = await session.call_tool(self.original_tool_name, kwargs)
            except Exception as retry_e:
                logger.error(
                    f"MCP Tool '{self.name}' execute failed after retry: {retry_e}"
                )
                return ToolResult(output=f"MCP Error: {retry_e}", is_error=True)

        if result.isError:
            return ToolResult(output=str(result.content), is_error=True)

        from nonebot_plugin_alconna import UniMessage, Image as AlcImage
        from zhenxun.services.ai.types.messages import ImagePart, TextPart
        
        output_content = []
        display_msg = UniMessage()
        has_display = False
        img_count = 0

        for item in result.content:
            item_type = getattr(item, "type", "text")

            if item_type == "image":
                b64_data = getattr(item, "data", "")
                mime_type = getattr(item, "mimeType", "image/png")
                if b64_data:
                    try:
                        img_bytes = base64.b64decode(b64_data)
                        display_msg += AlcImage(raw=img_bytes)
                        output_content.append(ImagePart(raw=img_bytes, mime_type=mime_type))
                        has_display = True
                        img_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"MCP 图片 Base64 解码失败: {e}")

                output_content.append(TextPart(text="[图片解码失败]"))

            elif item_type == "text":
                text = getattr(item, "text", str(item))
                output_content.append(TextPart(text=text))

                md_images = re.findall(r"!\[.*?\]\((https?://[^\)]+)\)", text)
                for img_url in md_images:
                    display_msg += AlcImage(url=img_url)
                    output_content.append(ImagePart(url=img_url))
                    has_display = True
                    img_count += 1
            else:
                dumped = model_dump(item) if hasattr(item, "model_dump") else str(item)
                output_content.append(TextPart(text=str(dumped)))

        return ToolResult(
            output=output_content,
            display=display_msg if has_display else None,
            log_content=f"获取到 {len(output_content)} 条返回数据，提取了 {img_count} 张图片",
        )


class MCPToolkit(BaseToolkit, ResourceLifespanMixin):
    """模型上下文协议 (MCP) 的工具箱封装 (支持声明式挂载与动态隔离)"""

    def __init__(
        self,
        server_name: str,
        transport: Literal[
            "stdio", "sse", "streamable-http", "sandbox_proxy"
        ] = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        install_command: str | None = None,
        timeout: int = 30,
        trust: bool = True,
        tool_metadata: dict[str, dict[str, Any]] | None = None,
        header_provider: Callable[[RunContext], dict[str, str]] | None = None,
        env_provider: Callable[[RunContext], dict[str, str]] | None = None,
        ttl: int = 600,
        sandbox_session_id: str | None = None,
        sandbox_profile: SandboxSecurityProfile | None = None,
    ):
        super().__init__()
        self.server_name = server_name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env or {}
        self.cwd = cwd
        self.install_command = install_command
        self.timeout = timeout
        self.trust = trust
        self.tool_metadata = tool_metadata or {}
        self.header_provider = header_provider
        self.env_provider = env_provider
        self.sandbox_session_id = sandbox_session_id
        self.sandbox_profile = sandbox_profile

        self._session: ClientSession | None = None
        self._tools: list[BaseTool] = []
        self._is_initialized = False
        self._bound_sandbox_session_id: str | None = None

        self.init_lifespan(ttl=ttl)

        self._server_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._init_exception: Exception | None = None

    async def _run_install_command(self, force: bool = False):
        """执行环境预安装（热加载依赖）"""
        if not self.install_command or not self.cwd:
            return

        from pathlib import Path

        marker_file = Path(self.cwd) / ".zx_installed"

        heuristic_missing = False
        if (
            "npm" in self.install_command
            and not (Path(self.cwd) / "node_modules").exists()
        ):
            heuristic_missing = True

        if not force and not heuristic_missing and marker_file.exists():
            return

        reason = "强制触发自愈重装" if force else "首次启动或检测到环境缺失"
        logger.info(
            f"🔧 [{self.server_name}] {reason}，正在执行: `{self.install_command}` ..."
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                self.install_command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err_msg = (
                    stderr.decode("utf-8", errors="ignore")
                    if stderr
                    else stdout.decode("utf-8", errors="ignore")
                )
                logger.error(
                    f"❌ [{self.server_name}] 环境安装失败!\n错误输出:\n{err_msg}"
                )
                raise RuntimeError(f"安装命令执行失败: {self.install_command}")
            else:
                marker_file.touch()
                logger.info(f"✅ [{self.server_name}] 环境装配成功！")
        except Exception as e:
            logger.error(f"执行预热安装异常: {e}")
            raise e

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        """会话隔离级别的生命周期入口"""
        await super().enter_session(session_id, context)
        if not self.sandbox_session_id:
            self._bound_sandbox_session_id = session_id

    async def get_session(
        self, context: RunContext | None = None
    ) -> ClientSession | None:
        self.touch(self.server_name)
        self._ensure_watchdog()
        if not self._is_initialized:
            await self.initialize(context)
        return self._session

    async def _server_loop(self, dynamic_headers: dict, dynamic_env: dict):
        """独立的后台协程：彻底隔离 AnyIO 的 AsyncExitStack，防止污染主任务上下文"""

        max_attempts = 2 if (self.transport == "stdio" and self.install_command) else 1

        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    if self.transport == "stdio":
                        await self._run_install_command(force=(attempt > 1))

                    async with AsyncExitStack() as stack:
                        read_stream: Any = None
                        write_stream: Any = None
                        if self.transport == "stdio":
                            if not self.command:
                                raise ValueError("stdio requires 'command'")
                            params = StdioServerParameters(
                                command=self.command,
                                args=self.args,
                                env=dynamic_env,
                                cwd=self.cwd,
                            )
                            transport_ctx = stdio_client(params)
                        elif self.transport == "sse":
                            if not self.url:
                                raise ValueError("sse requires 'url'")
                            transport_ctx = sse_client(
                                url=self.url,
                                headers=dynamic_headers,
                                timeout=self.timeout,
                            )
                        elif self.transport == "streamable-http":
                            if not self.url:
                                raise ValueError("streamable-http requires 'url'")

                            http_client = httpx.AsyncClient(
                                headers=dynamic_headers, timeout=self.timeout
                            )
                            await stack.enter_async_context(http_client)
                            transport_ctx = streamable_http_client(
                                url=self.url, http_client=http_client
                            )
                        elif self.transport == "sandbox_proxy":
                            from zhenxun.services.ai.sandbox.manager import (
                                sandbox_manager,
                            )
                            from zhenxun.services.ai.types.sandbox import (
                                SandboxSecurityProfile,
                            )

                            target_session_id = (
                                self.sandbox_session_id
                                or self._bound_sandbox_session_id
                                or "mcp_global_session"
                            )
                            req_profile = (
                                self.sandbox_profile
                                or SandboxSecurityProfile(enable_network=True)
                            )
                            driver = await sandbox_manager.get_or_create_session(
                                target_session_id,
                                profile=req_profile,
                            )

                            plugin_name = "universal_mcp"

                            await driver.mount_plugin(plugin_name)
                            mcp_plugin = cast(
                                BaseMcpProxyPlugin, driver.get_plugin(plugin_name)
                            )

                            if not self.command:
                                raise ValueError("sandbox_proxy requires 'command'")

                            streams = await stack.enter_async_context(
                                mcp_plugin.connect_mcp(
                                    self.command, self.args, dynamic_env
                                )
                            )
                            read_stream, write_stream = streams[0], streams[1]
                            transport_ctx = None
                        else:
                            raise ValueError(f"Unknown transport: {self.transport}")

                        if transport_ctx:
                            transport = await stack.enter_async_context(transport_ctx)
                            read_stream, write_stream = transport[0], transport[1]

                        self._session = await stack.enter_async_context(
                            ClientSession(read_stream, write_stream)
                        )
                        await self._session.initialize()

                        mcp_tools_res = await self._session.list_tools()
                        for t in mcp_tools_res.tools:
                            t_name = t.name.replace("-", "_")
                            self._tools.append(
                                MCPRemoteTool(
                                    name=t_name,
                                    original_tool_name=t.name,
                                    description=t.description or "",
                                    parameters=t.inputSchema,
                                    toolkit=self,
                                )
                            )

                        self._is_initialized = True
                        self._ready_event.set()
                        logger.info(
                            f"成功连接 MCP 服务器: {self.server_name}, "
                            f"获取了 {len(self._tools)} 个工具"
                        )

                        await self._stop_event.wait()
                        return

                except Exception as e:
                    if attempt < max_attempts:
                        logger.warning(
                            f"⚠️ [{self.server_name}] 进程启动或运行异常崩溃，疑似环境损坏。触发自愈机制 (准备重试)..."
                        )
                        self._session = None
                        self._is_initialized = False
                        continue

                    raise e

        except Exception as e:
            self._init_exception = e
            logger.error(
                f"MCP 服务器 '{self.server_name}' 初始化连接失败（模式: {self.transport}）。"
                f"错误原因: {e}"
            )
            if "Connection closed" in str(e):
                logger.error(
                    "提示：若是使用沙箱隧道，请进入 Docker Desktop 检查容器的日志。"
                    "多半是因为 npx 命令执行出错"
                    "（例如：镜像中缺少 Node 环境或国内网络无法连接 npm 等）。"
                )
        finally:
            self._is_initialized = False
            self._session = None
            self._ready_event.set()

    async def initialize(self, context: RunContext | None = None):
        if self._is_initialized:
            return

        self._tools.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._init_exception = None

        dynamic_headers = {}
        dynamic_env = self.env.copy()

        if context:
            if self.header_provider:
                try:
                    dynamic_headers.update(self.header_provider(context))
                except Exception as e:
                    logger.warning(f"Header provider failed: {e}")
            if self.env_provider:
                try:
                    dynamic_env.update(self.env_provider(context))
                except Exception as e:
                    logger.warning(f"Env provider failed: {e}")

        self._server_task = asyncio.create_task(
            self._server_loop(dynamic_headers, dynamic_env)
        )

        await self._ready_event.wait()

        if self._init_exception:
            raise self._init_exception

    async def get_tools(self) -> list[BaseTool]:
        self.touch(self.server_name)
        self._ensure_watchdog()
        if not self._is_initialized:
            await self.initialize()
        return self._tools

    async def release_resource(self, resource_id: str):
        await self.close()

    async def close(self):
        current_task = asyncio.current_task()
        if (
            self._watchdog_task
            and not self._watchdog_task.done()
            and self._watchdog_task is not current_task
        ):
            self._watchdog_task.cancel()
        self._watchdog_task = None

        self._stop_event.set()

        if (
            self._server_task
            and not self._server_task.done()
            and self._server_task is not current_task
        ):
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"MCP server task for {self.server_name} timed out, cancelling."
                )
                self._server_task.cancel()
        self._server_task = None

        self._session = None
        self._is_initialized = False
        self._tools.clear()

