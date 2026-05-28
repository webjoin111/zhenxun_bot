import asyncio
import typing

import aiohttp

from zhenxun.services.ai.sandbox.extension import (
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsPortMapping,
)
from zhenxun.services.ai.sandbox.models import CodeBlock, SandboxExecutionResult
from zhenxun.services.ai.sandbox.utils import JupyterKernelClient
from zhenxun.services.log import logger

from .base import BaseCodeExecutor


class PythonJupyterExecutor(BaseCodeExecutor):
    """Python 有状态多模态执行器。动态在沙箱内按需启动 Jupyter Server。"""

    def __init__(self, driver):
        super().__init__(driver)
        self.jupyter_client = None
        self._http_session = None

    async def _ensure_jupyter_started(self):
        if self.jupyter_client:
            return

        cmd_channel = typing.cast(SupportsCommandExecution, self.driver)
        port_channel = typing.cast(SupportsPortMapping, self.driver)

        jupyter_port = port_channel.get_meta("jupyter_port")
        if not jupyter_port:
            raise RuntimeError("沙箱未映射 Jupyter 端口")

        check_jupyter = await cmd_channel.execute_raw_command(
            "command -v jupyter-server"
        )
        if check_jupyter.exit_code != 0:
            raise RuntimeError("沙箱内未安装 jupyter-server，无法启动高级环境")

        rpc_env = await self._prepare_rpc_env()
        env_str = " ".join([f"{k}={v}" for k, v in rpc_env.items()])

        start_cmd = (
            f"nohup env {env_str} jupyter-server "
            "--ServerApp.ip=0.0.0.0 --ServerApp.port=8888 "
            "--ServerApp.token='' --ServerApp.password='' "
            "--ServerApp.disable_check_xsrf=True "
            "--ServerApp.allow_origin='*' --ServerApp.allow_root=True "
            "> /workspace/jupyter.log 2>&1 &"
        )
        await cmd_channel.execute_raw_command(start_cmd)

        base_url = f"http://127.0.0.1:{jupyter_port}"
        ws_url = f"ws://127.0.0.1:{jupyter_port}"

        self._http_session = aiohttp.ClientSession()
        for _ in range(15):
            try:
                async with self._http_session.get(f"{base_url}/api/kernels") as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Jupyter 服务动态拉起超时")

        async with self._http_session.post(
            f"{base_url}/api/kernels", json={"name": "python3"}
        ) as resp:
            kernel_id = (await resp.json()).get("id")

        self.jupyter_client = JupyterKernelClient(
            self._http_session, base_url, ws_url, kernel_id
        )
        logger.info("[PythonJupyterExecutor] Jupyter 内核动态拉起成功！")

    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None,
    ) -> SandboxExecutionResult:
        if not isinstance(self.driver, SupportsCommandExecution) or not isinstance(
            self.driver, SupportsFileSystem
        ):
            return SandboxExecutionResult(exit_code=-1, error="当前沙箱底座能力不足")

        await self._ensure_jupyter_started()

        fs_channel = typing.cast(SupportsFileSystem, self.driver)
        if injected_code:
            await fs_channel.write_raw_file("/workspace/zhenxun_host.py", injected_code)

        combined_code = "\n".join([b.code for b in code_blocks])
        result = await self.jupyter_client.execute(combined_code, timeout=timeout)
        return result

    async def close(self):
        if self.jupyter_client:
            await self.jupyter_client.close()
            self.jupyter_client = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
