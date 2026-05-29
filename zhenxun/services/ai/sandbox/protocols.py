from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from zhenxun.services.ai.sandbox.models import SandboxExecutionResult

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
        on_output: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> SandboxExecutionResult: ...

    async def exec(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_output: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> SandboxExecutionResult: ...

@runtime_checkable
class SupportsInteractivePTY(Protocol):
    async def create_pty_session(self) -> InteractiveTerminalSession: ...

@runtime_checkable
class SupportsFileSystem(Protocol):
    async def write_raw_file(self, path: str | Path, content: str) -> bool: ...
    async def read_raw_file(self, path: str | Path) -> str: ...
    async def delete_raw_file(self, path: str | Path) -> bool: ...
    async def upload_raw_dir(
        self, local_dir_path: str | Path, sandbox_target_path: str | Path
    ) -> bool: ...

    async def write(self, path: str | Path, data: bytes) -> bool: ...
    async def read(self, path: str | Path) -> bytes: ...
    async def rm(self, path: str | Path, recursive: bool = False) -> bool: ...
    async def mkdir(self, path: str | Path, parents: bool = False) -> bool: ...

@runtime_checkable
class SupportsPortMapping(Protocol):
    def get_meta(self, key: str, default: Any = None) -> Any: ...

class SandboxChannel(ABC):
    @abstractmethod
    def get_meta(self, key: str, default: Any = None) -> Any: ...
