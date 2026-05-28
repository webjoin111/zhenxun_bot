from abc import ABC, abstractmethod

from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxDriver
from zhenxun.services.ai.sandbox.extension import SupportsFileSystem
from zhenxun.services.ai.sandbox.models import CodeBlock, SandboxExecutionResult
from zhenxun.services.ai.sandbox.rpc import STUB_TEMPLATE, sandbox_rpc_server


class BaseCodeExecutor(ABC):
    """
    代码执行器抽象基类 (Autogen 范式)
    分离沙箱引擎和具体语言的执行逻辑。
    """
    def __init__(self, driver: BaseSandboxDriver):
        self.driver = driver

    async def _prepare_rpc_env(self) -> dict[str, str]:
        """将 stub 写入工作区，并返回需要注入的环境变量"""
        if isinstance(self.driver, SupportsFileSystem):
            await self.driver.write_raw_file(
                "/workspace/zhenxun_stub.py", STUB_TEMPLATE
            )

        is_docker = self.driver.__class__.__name__ == "DockerDriver"
        host_ip = "host.docker.internal" if is_docker else "127.0.0.1"

        base_env = self.driver.get_meta("env", {})
        base_env.update({
            "ZHENXUN_RPC_URL": f"http://{host_ip}:{sandbox_rpc_server.port}/rpc",
            "ZHENXUN_SESSION_ID": self.driver.session_id or "default",
        })
        return base_env

    @abstractmethod
    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None
    ) -> SandboxExecutionResult:
        pass
