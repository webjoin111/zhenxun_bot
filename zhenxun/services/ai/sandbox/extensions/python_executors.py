import asyncio
import time
import typing

import aiohttp

from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    SandboxExecutionCompletedEvent,
    SandboxExecutionStartedEvent,
)
from zhenxun.services.ai.sandbox.extension import (
    BaseSandboxExtension,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsPortMapping,
)
from zhenxun.services.ai.sandbox.models import SandboxExecutionResult
from zhenxun.services.ai.sandbox.utils import JupyterKernelClient


class UniversalPythonExtension(BaseSandboxExtension):
    """大一统 Python 执行扩展，根据底层驱动能力自动路由集"""

    @property
    def extension_name(self) -> str:
        return "universal_python"

    @property
    def supports_state(self) -> bool:
        if getattr(self, "_use_jupyter", False):
            return True
        return False

    @property
    def supports_plot(self) -> bool:
        return self.supports_state

    def __init__(self, channel):
        super().__init__(channel)
        self._use_jupyter = False
        self.jupyter_client: JupyterKernelClient | None = None
        self._http_session = None

    async def on_mount(self) -> None:
        await super().on_mount()
        if hasattr(self.channel, "sandbox") or hasattr(self.channel, "workspace"):
            return

        if isinstance(self.channel, SupportsCommandExecution) and isinstance(
            self.channel, SupportsPortMapping
        ):
            base_url = self.channel.get_meta("base_url")
            ws_url = self.channel.get_meta("ws_url")
            if base_url and ws_url:
                probe = await self.channel.execute_raw_command(
                    "command -v jupyter-server"
                )
                if probe.exit_code == 0:
                    self._http_session = aiohttp.ClientSession()
                    for _ in range(15):
                        try:
                            async with self._http_session.get(
                                f"{base_url}/api/kernels"
                            ) as resp:
                                if resp.status == 200:
                                    break
                        except Exception:
                            pass
                        await asyncio.sleep(1)
                    else:
                        raise RuntimeError("Jupyter KernelGateway 启动超时！")

                    async with self._http_session.post(
                        f"{base_url}/api/kernels", json={"name": "python3"}
                    ) as resp:
                        kernel_id = (await resp.json()).get("id")

                    self.jupyter_client = JupyterKernelClient(
                        self._http_session, base_url, ws_url, kernel_id
                    )
                    self._use_jupyter = True

    async def on_unmount(self) -> None:
        if self.jupyter_client:
            await self.jupyter_client.close()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def execute(self, code: str, timeout: int = 30) -> SandboxExecutionResult:
        session_id = self.channel.get_meta("session_id", "unknown")
        start_t = time.monotonic()
        await EventCenter.publish(
            SandboxExecutionStartedEvent(session_id=session_id, code=code)
        )

        if hasattr(self.channel, "workspace"):
            result = await self._execute_wasm(code, timeout)
        elif getattr(self, "_use_jupyter", False):
            result = await self._execute_jupyter(code, timeout)
        elif isinstance(self.channel, SupportsCommandExecution) and isinstance(
            self.channel, SupportsFileSystem
        ):
            result = await self._execute_basic(code, timeout)
        else:
            result = SandboxExecutionResult(
                exit_code=-1, error="当前沙箱底座不具备执行 Python 的能力"
            )

        await EventCenter.publish(
            SandboxExecutionCompletedEvent(
                session_id=session_id,
                exit_code=result.exit_code,
                duration_ms=(time.monotonic() - start_t) * 1000,
            )
        )
        return result

    async def _execute_basic(self, code: str, timeout: int) -> SandboxExecutionResult:
        script_path = "/tmp/basic_exec_script.py"
        fs_channel = typing.cast(SupportsFileSystem, self.channel)
        cmd_channel = typing.cast(SupportsCommandExecution, self.channel)
        await fs_channel.write_raw_file(script_path, code)
        return await cmd_channel.execute_raw_command(
            f"python3 {script_path}", timeout=timeout
        )

    async def _execute_jupyter(self, code: str, timeout: int) -> SandboxExecutionResult:
        if not self.jupyter_client:
            raise RuntimeError("JupyterClient 尚未就绪。")
        return await self.jupyter_client.execute(code, timeout)

    async def _execute_wasm(self, code: str, timeout: int) -> SandboxExecutionResult:
        from zhenxun.services.ai.sandbox.drivers.wasm import (
            WasmDriver,
            WasmtimeCoreEngine,
        )

        driver = typing.cast(WasmDriver, self.channel)
        res = await WasmtimeCoreEngine.run_code(
            code, fuel=2_000_000_000, workspace_dir=driver.workspace
        )
        return SandboxExecutionResult(
            stdout=res["stdout"],
            stderr=res["stderr"],
            exit_code=res["exit_code"],
            error="Wasm Execution Trap / Fuel Exhausted"
            if res["exit_code"] != 0 and "Trap" in res["stderr"]
            else None,
        )
