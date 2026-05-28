from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from zhenxun.services.ai.sandbox.models import SandboxExecutionResult
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxDriver


class InteractiveTerminalSession(Protocol):
    @abstractmethod
    async def start(self, cmd: str, env: dict[str, str] | None = None) -> None: ...

    @abstractmethod
    async def send_input(self, text: str) -> None: ...

    @abstractmethod
    async def read_output(self, timeout: int = 5) -> str: ...

    @abstractmethod
    async def interrupt(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


@runtime_checkable
class SupportsCommandExecution(Protocol):
    async def execute_raw_command(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int = 30,
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult: ...


@runtime_checkable
class SupportsInteractivePTY(Protocol):
    async def create_pty_session(self) -> InteractiveTerminalSession: ...


@runtime_checkable
class SupportsFileSystem(Protocol):
    async def write_raw_file(self, path: str, content: str) -> bool: ...
    async def read_raw_file(self, path: str) -> str: ...
    async def delete_raw_file(self, path: str) -> bool: ...
    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool: ...


@runtime_checkable
class SupportsPortMapping(Protocol):
    def get_meta(self, key: str, default: Any = None) -> Any: ...


class SandboxChannel(ABC):
    """基础沙箱通道（已被剥离具体执行方法，退化为标识与元数据基类）"""

    @abstractmethod
    def get_meta(self, key: str, default: Any = None) -> Any: ...


class BaseSandboxExtension(ABC):
    def __init__(self, channel: SandboxChannel):
        self.channel = channel

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


class SandboxRegistry:
    _drivers: ClassVar[dict[str, type["BaseSandboxDriver"]]] = {}
    _extensions: ClassVar[dict[str, type[BaseSandboxExtension]]] = {}

    @classmethod
    def register(cls, name: str, driver_cls: type["BaseSandboxDriver"]) -> None:
        if name in cls._drivers:
            logger.warning(f"[SandboxRegistry] 覆盖已存在的沙箱驱动: {name}")
        cls._drivers[name] = driver_cls
        logger.info(f"[SandboxRegistry] 成功注册沙箱驱动: {name}")

    @classmethod
    def get_driver_cls(cls, name: str) -> type["BaseSandboxDriver"]:
        if name not in cls._drivers:
            raise ValueError(f"未找到名为 '{name}' 的沙箱驱动。")
        return cls._drivers[name]

    @classmethod
    def get_all_drivers(cls) -> dict[str, type["BaseSandboxDriver"]]:
        return cls._drivers.copy()

    @classmethod
    def register_extension(cls, extension_cls: type[BaseSandboxExtension]) -> None:
        if (
            isinstance(extension_cls.extension_name, property)
            and extension_cls.extension_name.fget
        ):
            name = extension_cls.extension_name.fget(None)
        else:
            name = getattr(extension_cls, "extension_name", str(extension_cls))
        cls._extensions[name] = extension_cls
        logger.debug(f"[SandboxRegistry] 成功注册沙箱扩展: {name}")

    @classmethod
    def get_extension_cls(cls, name: str) -> type[BaseSandboxExtension] | None:
        return cls._extensions.get(name)

