import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
from typing import Any, cast

import anyio
from anyio import create_memory_object_stream, create_task_group
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
import websockets

from zhenxun.services.ai.sandbox.extension import (
    BaseMcpProxyPlugin,
    SupportsCommandExecution,
    SupportsPortMapping,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump_json, model_validate


class UniversalMcpPlugin(BaseMcpProxyPlugin):
    @property
    def plugin_name(self) -> str:
        return "universal_mcp"

    @asynccontextmanager
    async def connect_mcp(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> AsyncGenerator[tuple[Any, Any], None]:
        if isinstance(self.channel, SupportsPortMapping) and isinstance(
            self.channel, SupportsCommandExecution
        ):
            async with self._connect_docker(command, args, env) as streams:
                yield streams
        else:
            raise RuntimeError("当前沙箱底座不具备运行 MCP Proxy 的能力")

    @asynccontextmanager
    async def _connect_docker(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ):
        from zhenxun.services.ai.sandbox.drivers.docker import DockerDriver
        driver = cast(DockerDriver, self.channel)
        logger.info(
            f"[UniversalMcpPlugin] 正在 Docker 沙箱内启动 MCP 服务器: {command} {' '.join(args)}"
        )

        payload = {"command": command, "args": args, "env": env or {}}
        resp = await driver._ipc_request("POST", "/mcp/start", json=payload, timeout=30)
        if resp.status_code != 200:
            raise ValueError(f"Failed to start MCP in Docker: {resp.text}")

        host_port = resp.json()["port"]
        async with await anyio.connect_tcp("127.0.0.1", host_port) as stream:
            read_prod, read_cons = create_memory_object_stream(10)
            write_prod, write_cons = create_memory_object_stream(10)

            async def tcp_reader():
                buffer = b""
                try:
                    while True:
                        data = await stream.receive()
                        buffer += data
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            if not line.strip():
                                continue
                            try:
                                msg = model_validate(JSONRPCMessage, json.loads(line))
                                await read_prod.send(SessionMessage(message=msg))
                            except Exception as exc:
                                await read_prod.send(exc)
                except Exception:
                    pass
                finally:
                    await read_prod.aclose()

            async def tcp_writer():
                try:
                    async for msg in write_cons:
                        data = (
                            model_dump_json(
                                msg.message, by_alias=True, exclude_none=True
                            ).encode("utf-8")
                            + b"\n"
                        )
                        await stream.send(data)
                except Exception:
                    pass
                finally:
                    await stream.aclose()

            async with create_task_group() as tg:
                tg.start_soon(tcp_reader)
                tg.start_soon(tcp_writer)
                yield read_cons, write_prod
