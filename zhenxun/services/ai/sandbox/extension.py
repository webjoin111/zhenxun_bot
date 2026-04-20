from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from zhenxun.services.ai.types.sandbox import SandboxExecutionResult
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxDriver
    from zhenxun.services.ai.sandbox.providers.base import BaseSandboxProvider


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


class BaseSandboxPlugin(ABC):
    def __init__(self, channel: SandboxChannel):
        self.channel = channel

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        pass

    async def on_mount(self) -> None:
        logger.debug(f"[SandboxPlugin] 插件 '{self.plugin_name}' 已挂载。")

    async def on_unmount(self) -> None:
        logger.debug(f"[SandboxPlugin] 插件 '{self.plugin_name}' 已卸载。")


class BaseMcpProxyPlugin(BaseSandboxPlugin):
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
    _drivers: dict[str, type["BaseSandboxDriver"]] = {}
    _providers: dict[str, "BaseSandboxProvider"] = {}
    _plugins: dict[str, type[BaseSandboxPlugin]] = {}

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
    def register_provider(cls, provider: "BaseSandboxProvider") -> None:
        name = provider.get_name()
        if name in cls._providers:
            logger.warning(f"[SandboxRegistry] 覆盖已存在的沙箱提供者: {name}")
        cls._providers[name] = provider
        logger.info(f"[SandboxRegistry] 成功注册沙箱提供者: {name}")

    @classmethod
    def get_all_providers(cls) -> dict[str, "BaseSandboxProvider"]:
        return cls._providers.copy()

    @classmethod
    def register_plugin(cls, plugin_cls: type[BaseSandboxPlugin]) -> None:
        if isinstance(plugin_cls.plugin_name, property) and plugin_cls.plugin_name.fget:
            name = plugin_cls.plugin_name.fget(None)
        else:
            name = getattr(plugin_cls, "plugin_name", str(plugin_cls))
        cls._plugins[name] = plugin_cls
        logger.debug(f"[SandboxRegistry] 成功注册沙箱插件: {name}")

    @classmethod
    def get_plugin_cls(cls, name: str) -> type[BaseSandboxPlugin] | None:
        return cls._plugins.get(name)
