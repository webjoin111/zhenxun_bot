import typing

from zhenxun.services.ai.sandbox.extension import (
    SupportsCommandExecution,
    SupportsFileSystem,
)
from zhenxun.services.ai.sandbox.models import CodeBlock, SandboxExecutionResult

from .base import BaseCodeExecutor


class PythonCLIExecutor(BaseCodeExecutor):
    """Python 命令行轻量执行器。将代码写入文件后直接拉起解释器。"""

    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None,
    ) -> SandboxExecutionResult:
        if not isinstance(self.driver, SupportsCommandExecution) or not isinstance(
            self.driver, SupportsFileSystem
        ):
            return SandboxExecutionResult(
                exit_code=-1, error="当前沙箱底座不具备命令行或文件系统能力"
            )

        cmd_channel = typing.cast(SupportsCommandExecution, self.driver)
        fs_channel = typing.cast(SupportsFileSystem, self.driver)

        if injected_code:
            await fs_channel.write_raw_file("/workspace/zhenxun_host.py", injected_code)

        rpc_env = await self._prepare_rpc_env()

        combined_code = "\n".join([b.code for b in code_blocks])
        script_path = "/workspace/tmp_exec.py"
        await fs_channel.write_raw_file(script_path, combined_code)

        result = await cmd_channel.execute_raw_command(
            f"python3 {script_path}", timeout=timeout, env=rpc_env
        )
        return result
