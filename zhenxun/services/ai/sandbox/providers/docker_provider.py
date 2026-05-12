from zhenxun.services.ai.sandbox.models import (
    SandboxCapabilities,
    SandboxRequirements,
    SandboxSecurityProfile,
)

from ..drivers.base import BaseSandboxDriver
from ..drivers.docker import DOCKER_AVAILABLE, DockerDriver
from ..extension import SandboxRegistry
from .base import BaseSandboxProvider


class DockerSandboxProvider(BaseSandboxProvider):
    """本地 Docker 容器沙箱提供者"""

    _engine_available: bool = False

    def set_engine_status(self, status: bool) -> None:
        """由框架启动钩子注入引擎存活状态"""
        self._engine_available = status

    def get_name(self) -> str:
        return "docker"

    def get_capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            supports_state=True,
            supported_capabilities=[
                "PythonExecutionCapability",
                "FileSystemCapability",
                "TerminalCapability",
                "SkillEnvironmentCapability",
            ],
            isolation_level=8,
            startup_latency=500,
        )

    def is_available(self) -> bool:
        return DOCKER_AVAILABLE and self._engine_available

    def score(
        self, profile: SandboxSecurityProfile, requirements: SandboxRequirements | None
    ) -> int:
        if profile.sandbox_type == self.get_name():
            return 100
        if profile.sandbox_type != "auto":
            return -1

        return 80

    def create_driver(self, session_id: str) -> BaseSandboxDriver:
        return DockerDriver()


SandboxRegistry.register_provider(DockerSandboxProvider())

