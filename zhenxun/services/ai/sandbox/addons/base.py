from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxSession


class BaseSandboxExtension(ABC):
    def __init__(self, session: "BaseSandboxSession"):
        self.session = session

    @property
    @abstractmethod
    def extension_name(self) -> str:
        pass

    async def on_mount(self) -> None:
        logger.debug(f"[SandboxExtension] 扩展 '{self.extension_name}' 已挂载。")

    async def on_unmount(self) -> None:
        logger.debug(f"[SandboxExtension] 扩展 '{self.extension_name}' 已卸载。")


class BaseMcpProxyExtension(BaseSandboxExtension):
    @abstractmethod
    @asynccontextmanager
    async def connect_mcp(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> AsyncGenerator[tuple[Any, Any], None]:
        yield None, None
