from zhenxun.configs.config import Config
from zhenxun.services.ai.types.sandbox import (
    SandboxCapabilities,
    SandboxRequirements,
    SandboxSecurityProfile,
)

from ..drivers.base import BaseSandboxDriver
from ..drivers.cloud import E2B_AVAILABLE, E2BCloudDriver
from ..extension import SandboxRegistry
from .base import BaseSandboxProvider


class E2BSandboxProvider(BaseSandboxProvider):
    """E2B 云端微型虚拟机沙箱提供者"""

    def get_name(self) -> str:
        return "e2b"

    def get_capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            supports_state=True,
            supported_capabilities=[
                "PythonExecutionCapability",
                "FileSystemCapability",
                "TerminalCapability",
                "SkillEnvironmentCapability",
            ],
            isolation_level=10,
            startup_latency=1500,
        )

    def _get_api_key(self) -> str:
        api_keys = Config.get_config("sandbox", "SANDBOX_API_KEYS", {})
        return api_keys.get("E2B", "")

    def is_available(self) -> bool:
        if not E2B_AVAILABLE:
            return False
        key = self._get_api_key()
        if not key or key == "YOUR_API_KEY":
            return False
        return True

    def score(
        self, profile: SandboxSecurityProfile, requirements: SandboxRequirements | None
    ) -> int:
        if profile.sandbox_type == self.get_name():
            return 100
        if profile.sandbox_type != "auto":
            return -1

        if profile.require_gpu:
            return -1

        return 100

    def create_driver(self, session_id: str) -> BaseSandboxDriver:
        return E2BCloudDriver(api_key=self._get_api_key())


SandboxRegistry.register_provider(E2BSandboxProvider())
