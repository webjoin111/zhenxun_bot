from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
from typing import Any

from anyio import create_memory_object_stream, create_task_group
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage


from zhenxun.services.ai.sandbox.addons.base import BaseMcpProxyExtension
from zhenxun.services.ai.sandbox.registry import SandboxRegistry

from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump_json, model_validate


class UniversalMcpExtension(BaseMcpProxyExtension):
    @property
    def extension_name(self) -> str:
        return "universal_mcp"

    @asynccontextmanager
    async def connect_mcp(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> AsyncGenerator[tuple[Any, Any], None]:
        # 验证底层是否是 Docker 实例
        if not getattr(self.session, "container", None):
             raise RuntimeError("目前 MCP 代理仅支持基于 Docker 的沙箱底座。")
        async with self._connect_docker(command, args, env) as streams:
            yield streams

    @asynccontextmanager
    async def _connect_docker(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ):
        logger.info(
            "[UniversalMcpExtension] 正在 Docker 沙箱内原生启动 MCP 服务器: "
            f"{command} {' '.join(args)}"
        )

        cmd_list = [command, *args]
        env_list = [f"{k}={v}" for k, v in env.items()] if env else None

        session = self.session
        exec_inst = await session.container.exec(
            cmd=cmd_list,
            stdin=True,
            stdout=True,
            stderr=False,
            environment=env_list,
            workdir="/workspace",
        )

        async with exec_inst.start(detach=False) as raw_stream:
            read_prod, read_cons = create_memory_object_stream(10)
            write_prod, write_cons = create_memory_object_stream(10)

            async def stream_reader():
                buffer = b""
                try:
                    while True:
                        msg = await raw_stream.read_out()
                        if msg is None:
                            break
                        if msg.stream == 1:
                            buffer += msg.data
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                if not line.strip():
                                    continue
                                try:
                                    msg = model_validate(
                                        JSONRPCMessage, json.loads(line)
                                    )
                                    await read_prod.send(SessionMessage(message=msg))
                                except Exception as exc:
                                    await read_prod.send(exc)
                except Exception:
                    pass
                finally:
                    await read_prod.aclose()

            async def stream_writer():
                try:
                    async for msg in write_cons:
                        data = (
                            model_dump_json(
                                msg.message, by_alias=True, exclude_none=True
                            ).encode("utf-8")
                            + b"\n"
                        )
                        await raw_stream.write_in(data)
                except Exception:
                    pass

            async with create_task_group() as tg:
                tg.start_soon(stream_reader)
                tg.start_soon(stream_writer)
                yield read_cons, write_prod
                tg.cancel_scope.cancel()


SandboxRegistry.register_extension(UniversalMcpExtension)

__all__ = [
    "UniversalMcpExtension",
]
