import asyncio
import base64
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
import re
from typing import Any, Literal, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage
from pydantic import ValidationError

from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.sandbox.addons.base import BaseMcpProxyExtension
from zhenxun.services.ai.sandbox.models import SandboxBlueprint
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolDefinition, ToolkitConfig, ToolResult
from zhenxun.services.ai.utils.lifespan import ResourceLifespanMixin
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump

_MESSAGE_START_CHARS = {"{", "["}
_LITERAL_PREFIXES: tuple[str, ...] = ("true", "false", "null")


def _should_ignore_exception(exc: Exception) -> bool:
    """
    判断该异常是否是由非 JSON 的脏数据标准输出引起的。
    如果是脏数据，则可以安全忽略。
    """
    if not isinstance(exc, ValidationError):
        return False

    errors = exc.errors()
    first = next(iter(errors), None)
    if not first or first.get("type") != "json_invalid":
        return False

    input_value = first.get("input")
    if not isinstance(input_value, str):
        return False

    stripped = input_value.strip()
    if not stripped:
        return True

    first_char = stripped[0]
    lowered = stripped.lower()

    if first_char in _MESSAGE_START_CHARS or any(
        lowered.startswith(prefix) for prefix in _LITERAL_PREFIXES
    ):
        return False

    return True


@asynccontextmanager
async def filtered_stdio_client(
    server_name: str, server: StdioServerParameters
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    """
    包裹官方的 stdio_client，拦截并过滤掉非 JSON 格式的标准输出噪音。
    """
    async with stdio_client(server=server) as (read_stream, write_stream):
        filtered_send, filtered_recv = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)

        async def _forward_stdout() -> None:
            try:
                async with read_stream:
                    async for item in read_stream:
                        if isinstance(item, Exception) and _should_ignore_exception(
                            item
                        ):
                            if isinstance(item, ValidationError):
                                err_input = item.errors()[0].get("input", "")
                                logger.debug(
                                    f"🔇 [MCP Stdout Filter] {server_name} 忽略脏数据: {str(err_input)[:100].strip()}..."
                                )
                            continue
                        await filtered_send.send(item)
            except anyio.ClosedResourceError:
                pass
            finally:
                await filtered_send.aclose()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_forward_stdout)
            try:
                yield filtered_recv, write_stream
            finally:
                tg.cancel_scope.cancel()


class MCPRemoteTool(BaseTool):
    """远端 MCP 工具在本地的代理对象"""

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
        self.parent_toolkit = toolkit
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
        if context and self.settings.capabilities:
            from zhenxun.services.ai.protocols.capabilities import CombinedCapability

            combined_cap = CombinedCapability(self.settings.capabilities)
            defs = await combined_cap.prepare_tools(context, [tool_def])
            if not defs:
                return None
            tool_def = defs[0]
        return tool_def

    async def execute(self, context: RunContext | None = None, **kwargs) -> ToolResult:
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            if self.toolkit.isolation == "per_session" and context:
                sid = context.session_id or f"temp_{id(context)}"
                self.toolkit.touch(sid)
            else:
                self.toolkit.touch(self.toolkit.server_name)

            session = await self.toolkit.get_session(context)
            if not session:
                return ToolResult(output="MCP Connection Error").as_error()

            try:
                result = await session.call_tool(self.original_tool_name, kwargs)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"MCP Tool '{self.name}' execute failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}. "
                    "Attempting self-healing reconnect..."
                )
                if attempt < max_retries - 1:
                    await self.toolkit.close()
                    await asyncio.sleep(2**attempt)
        else:
            logger.error(
                f"MCP Tool '{self.name}' execute failed after {max_retries} retries: {last_error}"  # noqa: E501
            )
            return ToolResult(output=f"MCP Error: {last_error}").as_error()

        if result.isError:
            return ToolResult(output=str(result.content)).as_error()

        from zhenxun.services.ai.core.messages import ImagePart, TextPart

        output_content = []
        img_count = 0

        for item in result.content:
            item_type = getattr(item, "type", "text")

            if item_type == "image":
                b64_data = getattr(item, "data", "")
                mime_type = getattr(item, "mimeType", "image/png")
                if b64_data:
                    try:
                        img_bytes = base64.b64decode(b64_data)
                        output_content.append(
                            ImagePart(raw=img_bytes, mime_type=mime_type)
                        )
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
                    output_content.append(ImagePart(url=img_url))
                    img_count += 1
            else:
                dumped = model_dump(item) if hasattr(item, "model_dump") else str(item)
                output_content.append(TextPart(text=str(dumped)))

        tool_result = ToolResult(output=output_content).with_log(
            f"获取到 {len(output_content)} 条返回数据，提取了 {img_count} 张图片"
        )
        if img_count > 0:
            tool_result = tool_result.show_to_user(output_content)
        return tool_result


class MCPToolkit(BaseToolkit, ResourceLifespanMixin):
    """模型上下文协议 (MCP) 的工具箱封装 (支持声明式挂载与动态隔离)"""

    def __init__(
        self,
        server_name: str,
        prefix: str | None = None,
        transport: Literal[
            "stdio", "sse", "streamable-http", "sandbox_proxy"
        ] = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        install_command: str | None = None,
        isolation: Literal["shared", "per_session"] = "shared",
        timeout: int = 30,
        tool_metadata: dict[str, dict[str, Any]] | None = None,
        header_provider: Callable[[RunContext], dict[str, str]] | None = None,
        env_provider: Callable[[RunContext], dict[str, str]] | None = None,
        ttl: int = 600,
        sandbox_session_id: str | None = None,
        forward_bot_context: bool = False,
        sandbox_blueprint: SandboxBlueprint | None = None,
    ):
        super().__init__(
            config=ToolkitConfig(prefix=prefix) if prefix is not None else None
        )
        self.server_name = server_name
        self.transport = transport
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env or {}
        self.cwd = cwd
        self.install_command = install_command
        self.isolation = isolation
        self.timeout = timeout
        self.tool_metadata = tool_metadata or {}
        self.header_provider = header_provider
        self.env_provider = env_provider
        self.sandbox_session_id = sandbox_session_id
        self.forward_bot_context = forward_bot_context
        self.sandbox_blueprint = sandbox_blueprint

        self._shared_session: ClientSession | None = None
        self._session_pool: dict[str, ClientSession] = {}
        self._tools: list[BaseTool] = []
        self._is_initialized = False
        self._bound_sandbox_session_id: str | None = None

        self.init_lifespan(ttl=ttl)

        self._shared_task: asyncio.Task | None = None
        self._task_pool: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {"shared": asyncio.Event()}
        self._ready_events: dict[str, asyncio.Event] = {"shared": asyncio.Event()}
        self._init_exceptions: dict[str, Exception] = {}

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

    async def exit_session(self, session_id: str) -> None:
        """会话隔离级别的生命周期出口"""
        await super().exit_session(session_id)
        if self.isolation == "per_session" and self.ttl <= 0:
            await self.close_session(session_id)

    async def get_session(
        self, context: RunContext | None = None
    ) -> ClientSession | None:
        if self.isolation == "shared":
            self.touch(self.server_name)
            self._ensure_watchdog()
            if not self._is_initialized:
                await self.initialize(context)
            return self._shared_session
        else:
            if not context:
                if not self._is_initialized:
                    await self.initialize()
                return self._shared_session

            session_id = context.session_id or f"temp_{id(context)}"
            self.touch(session_id)
            self._ensure_watchdog()

            if not self._is_initialized:
                await self.initialize()

            if session_id in self._session_pool:
                return self._session_pool[session_id]

            dynamic_headers = {}
            dynamic_env = self.env.copy()
            if self.header_provider:
                try:
                    dynamic_headers.update(self.header_provider(context))
                except Exception as e:
                    logger.warning(f"Header provider failed for {session_id}: {e}")
            if self.forward_bot_context and context:
                uid = context.get_user_id()
                gid = context.get_group_id()
                plat = context.get_platform()
                if uid:
                    dynamic_headers["X-Zhenxun-User-Id"] = uid
                if gid:
                    dynamic_headers["X-Zhenxun-Group-Id"] = gid
                if plat:
                    dynamic_headers["X-Zhenxun-Platform"] = plat
                dynamic_headers["X-Zhenxun-Session-Id"] = session_id
            if self.env_provider:
                try:
                    dynamic_env.update(self.env_provider(context))
                except Exception as e:
                    logger.warning(f"Env provider failed for {session_id}: {e}")

            self._stop_events[session_id] = asyncio.Event()
            self._ready_events[session_id] = asyncio.Event()

            task = asyncio.create_task(
                self._spawn_session_task(session_id, dynamic_headers, dynamic_env)
            )
            self._task_pool[session_id] = task

            await self._ready_events[session_id].wait()

            if session_id in self._init_exceptions:
                raise self._init_exceptions[session_id]

            return self._session_pool.get(session_id)

    async def _spawn_session_task(
        self, session_key: str, dynamic_headers: dict, dynamic_env: dict
    ):
        """动态按需生成后台任务（兼容 shared 和 per_session 隔离模式）"""

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
                            transport_ctx = filtered_stdio_client(
                                self.server_name, params
                            )
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

                            target_session_id = (
                                self.sandbox_session_id
                                or self._bound_sandbox_session_id
                                or "mcp_global_session"
                            )
                            bp = self.sandbox_blueprint or SandboxBlueprint()
                            bp.enable_network = True

                            driver = await sandbox_manager.get_or_create_session(
                                target_session_id,
                                blueprint=bp,
                            )

                            plugin_name = "universal_mcp"

                            await driver.mount_extension(plugin_name)
                            mcp_plugin = cast(
                                BaseMcpProxyExtension, driver.get_extension(plugin_name)
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

                        client_session = await stack.enter_async_context(
                            ClientSession(read_stream, write_stream)
                        )
                        await client_session.initialize()

                        if session_key == "shared":
                            self._shared_session = client_session
                        else:
                            self._session_pool[session_key] = client_session

                        if not self._is_initialized:
                            mcp_tools_res = await client_session.list_tools()
                            tools_list = list(mcp_tools_res.tools)
                            cursor = getattr(mcp_tools_res, "nextCursor", None)

                            while cursor:
                                mcp_tools_res = await client_session.list_tools(
                                    cursor=cursor
                                )
                                tools_list.extend(mcp_tools_res.tools)
                                cursor = getattr(mcp_tools_res, "nextCursor", None)

                            for t in tools_list:
                                t_name = t.name.replace("-", "_")
                                final_name = (
                                    f"{self.config.prefix}{t_name}"
                                    if self.config.prefix
                                    else t_name
                                )
                                self._tools.append(
                                    MCPRemoteTool(
                                        name=final_name,
                                        original_tool_name=t.name,
                                        description=t.description or "",
                                        parameters=t.inputSchema,
                                        toolkit=self,
                                    )
                                )
                            self._is_initialized = True

                        self._ready_events[session_key].set()
                        logger.info(
                            f"成功连接 MCP 服务器: {self.server_name} [{session_key}], "
                            f"获取了 {len(self._tools)} 个工具"
                        )

                        await self._stop_events[session_key].wait()
                        return

                except Exception as e:
                    if attempt < max_attempts:
                        logger.warning(
                            f"⚠️ [{self.server_name}] 进程启动或运行异常崩溃，疑似环境损坏。触发自愈机制 (准备重试)..."  # noqa: E501
                        )
                        if session_key == "shared":
                            self._shared_session = None
                        else:
                            self._session_pool.pop(session_key, None)
                        if not self._is_initialized:
                            self._is_initialized = False
                        continue

                    raise e

        except Exception as e:
            self._init_exceptions[session_key] = e
            logger.error(
                f"MCP 服务器 '{self.server_name}' [{session_key}] 初始化连接失败（模式: {self.transport}）。"  # noqa: E501
                f"错误原因: {e}"
            )
            if "Connection closed" in str(e):
                logger.error(
                    "提示：若是使用沙箱隧道，请进入 Docker Desktop 检查容器的日志。"
                    "多半是因为 npx 命令执行出错"
                    "（例如：镜像中缺少 Node 环境或国内网络无法连接 npm 等）。"
                )
        finally:
            if session_key == "shared" and not self._is_initialized:
                self._is_initialized = False
            if session_key == "shared":
                self._shared_session = None
            else:
                self._session_pool.pop(session_key, None)
            self._ready_events[session_key].set()

    async def initialize(self, context: RunContext | None = None):
        if self._is_initialized:
            return

        self._tools.clear()
        self._stop_events["shared"].clear()
        self._ready_events["shared"].clear()
        self._init_exceptions.pop("shared", None)

        dynamic_headers = {}
        dynamic_env = self.env.copy()

        self._shared_task = asyncio.create_task(
            self._spawn_session_task("shared", dynamic_headers, dynamic_env)
        )

        await self._ready_events["shared"].wait()

        if "shared" in self._init_exceptions:
            raise self._init_exceptions["shared"]

    async def get_tools(self, context: RunContext | None = None) -> dict[str, BaseTool]:
        self.touch(self.server_name)
        self._ensure_watchdog()
        if not self._is_initialized:
            await self.initialize(context)

        tools_dict = {}
        for t in self._tools:
            tools_dict[t.name] = t
        return tools_dict

    async def release_resource(self, resource_id: str):
        if self.isolation == "per_session":
            await self.close_session(resource_id)
        else:
            await self.close()

    async def close_session(self, session_id: str):
        """清理单个隔离会话的进程和任务资源"""
        if session_id in self._stop_events:
            self._stop_events[session_id].set()

        task = self._task_pool.pop(session_id, None)
        if task and not task.done() and task is not asyncio.current_task():
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"MCP server task for {self.server_name} ({session_id}) timed out, cancelling."
                )
                task.cancel()

        self._session_pool.pop(session_id, None)
        self._stop_events.pop(session_id, None)
        self._ready_events.pop(session_id, None)
        self._init_exceptions.pop(session_id, None)

    async def close(self):
        current_task = asyncio.current_task()
        if (
            self._watchdog_task
            and not self._watchdog_task.done()
            and self._watchdog_task is not current_task
        ):
            self._watchdog_task.cancel()
        self._watchdog_task = None

        self._stop_events["shared"].set()

        if (
            self._shared_task
            and not self._shared_task.done()
            and self._shared_task is not current_task
        ):
            try:
                await asyncio.wait_for(self._shared_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"MCP server task for {self.server_name} timed out, cancelling."
                )
                self._shared_task.cancel()
        self._shared_task = None

        self._shared_session = None
        self._is_initialized = False
        self._tools.clear()

        for sid in list(self._session_pool.keys()):
            await self.close_session(sid)
