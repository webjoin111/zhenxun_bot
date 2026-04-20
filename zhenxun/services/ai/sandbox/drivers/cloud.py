import asyncio
from pathlib import Path
import re
from typing import TYPE_CHECKING

from zhenxun.configs.config import BotConfig
from zhenxun.services.ai.sandbox.extension import (
    InteractiveTerminalSession,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
)
from zhenxun.services.ai.types.sandbox import (
    SandboxExecutionResult,
    SandboxSecurityProfile,
)
from zhenxun.services.log import logger

from .base import BaseSandboxDriver

try:
    from e2b_code_interpreter import AsyncSandbox

    E2B_AVAILABLE = True
except ImportError:
    AsyncSandbox = None
    E2B_AVAILABLE = False

if TYPE_CHECKING:
    from e2b_code_interpreter import AsyncSandbox as AsyncSandboxType


def _strip_ansi(text: str) -> str:
    """去除 PTY 终端返回的 ANSI 颜色控制字符"""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class E2BInteractiveTerminalSession(InteractiveTerminalSession):
    """E2B 云端驱动的真实 PTY 交互终端实现"""

    def __init__(self, sandbox: "AsyncSandboxType"):
        self.sandbox = sandbox
        self.terminal = None
        self.output_buffer = ""
        self._lock = asyncio.Lock()

    def _on_data(self, data: bytes):
        """E2B 原生回调：接收终端流式输出"""
        text = _strip_ansi(data.decode("utf-8", errors="replace"))
        self.output_buffer += text

    async def setup(self):
        from e2b_code_interpreter import PtySize

        self.terminal = await self.sandbox.pty.create(
            size=PtySize(cols=120, rows=40), on_data=self._on_data, timeout=0
        )
        await asyncio.sleep(1.5)
        self.output_buffer = ""

    async def start(self, cmd: str, env: dict[str, str] | None = None) -> None:
        async with self._lock:
            if not self.terminal:
                raise RuntimeError("Terminal not initialized")

            self.output_buffer = ""
            payload = f"{cmd}\n".encode()

            await self.sandbox.pty.send_stdin(self.terminal.pid, payload)

    async def send_input(self, text: str) -> None:
        async with self._lock:
            if not self.terminal:
                return
            await self.sandbox.pty.send_stdin(self.terminal.pid, text.encode("utf-8"))

    async def read_output(self, timeout: int = 5) -> str:
        await asyncio.sleep(1)
        async with self._lock:
            return self.output_buffer.strip()

    async def interrupt(self) -> None:
        async with self._lock:
            if not self.terminal:
                return
            await self.sandbox.pty.send_stdin(self.terminal.pid, b"\x03")

    async def close(self) -> None:
        if self.terminal:
            await self.sandbox.pty.kill(self.terminal.pid)


class E2BCloudDriver(
    BaseSandboxDriver,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
):
    """
    E2B 云端驱动：通过 API 将代码发送到 Firecracker 微型虚拟机中执行。
    - 绝对安全，完美隔离。
    - 原生支持 Jupyter 协议，自动捕获 matplotlib 图表为 Base64。
    - 无需本地安装 Docker，极其适合 Windows 小白用户。
    """

    @property
    def supports_state(self) -> bool:
        return True

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def requires_local_docker(self) -> bool:
        return False

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.sandbox: "AsyncSandboxType | None" = None

    async def execute_raw_command(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int = 30,
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        self.touch()
        if not self.sandbox:
            return SandboxExecutionResult(exit_code=-1, error="Sandbox not started.")
        cmd_str = command if isinstance(command, str) else " ".join(command)
        try:
            result = await self.sandbox.commands.run(
                cmd_str, cwd=cwd or "", timeout=timeout, envs=env
            )
            return SandboxExecutionResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_code or 0,
                error=getattr(result.error, "value", str(result.error))
                if result.error
                else None,
            )
        except Exception as e:
            return SandboxExecutionResult(exit_code=-1, error=str(e))

    async def create_pty_session(self) -> InteractiveTerminalSession:
        self.touch()
        if not self.sandbox:
            raise RuntimeError("Sandbox not started.")
        session = E2BInteractiveTerminalSession(self.sandbox)
        await session.setup()
        return session

    async def write_raw_file(self, path: str, content: str) -> bool:
        self.touch()
        if not self.sandbox:
            return False
        await self.sandbox.files.write(path, content)
        return True

    async def read_raw_file(self, path: str) -> str:
        self.touch()
        if not self.sandbox:
            return "Error: Sandbox not started."
        try:
            return await self.sandbox.files.read(path)
        except Exception as e:
            return f"Failed to read file: {e}"

    async def delete_raw_file(self, path: str) -> bool:
        res = await self.execute_raw_command(f"rm -f {path}")
        return res.exit_code == 0

    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool:
        self.touch()
        if not self.sandbox:
            return False

        local_path = Path(local_dir_path)
        if not local_path.is_dir():
            logger.error(f"[E2BCloudDriver] 待上传的本地目录不存在: {local_dir_path}")
            return False

        tasks = []
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_path)
                target_file_path = (
                    f"{sandbox_target_path.rstrip('/')}/{rel_path.as_posix()}"
                )
                try:
                    content = file_path.read_bytes()
                    tasks.append(self.sandbox.files.write(target_file_path, content))
                except Exception as e:
                    logger.error(f"[E2BCloudDriver] 读取本地文件失败 {file_path}: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return True

    async def is_alive(self) -> bool:
        """E2B 主动探活（基于 sandbox 对象是否存在且运行中）"""
        if not self.sandbox:
            return False
        try:
            return getattr(self.sandbox, "is_running", True)
        except Exception:
            return False

    async def start(
        self, session_id: str, profile: SandboxSecurityProfile | None = None
    ) -> None:
        self.session_id = session_id
        if not E2B_AVAILABLE:
            raise RuntimeError("e2b_code_interpreter library is not installed.")

        logger.info(f"[E2BCloudDriver] 正在启动云端虚拟机 (Session: {session_id})")
        from e2b_code_interpreter import AsyncSandbox as _AsyncSandbox

        kwargs = {}
        if BotConfig.system_proxy:
            kwargs["proxy"] = BotConfig.system_proxy

        kwargs["network"] = {"allow_public_traffic": False}

        self.sandbox = await _AsyncSandbox.create(api_key=self.api_key, **kwargs)

        self._meta["traffic_access_token"] = self.sandbox.traffic_access_token

        await self.sandbox.commands.run(
            "sudo mkdir -p /workspace && sudo chown -R user:user /workspace"
        )

        self.touch()

    async def close(self) -> None:
        if self.sandbox:
            logger.info(
                f"[E2BCloudDriver] 正在销毁云端虚拟机 (Session: {self.session_id})"
            )
            await self.sandbox.kill()
            self.sandbox = None
