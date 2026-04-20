from zhenxun.services.ai.types.sandbox import (
    SandboxCapabilities,
    SandboxRequirements,
    SandboxSecurityProfile,
    SandboxTier,
)

from ..drivers.base import BaseSandboxDriver
from ..drivers.wasm import WASMTIME_AVAILABLE, WasmDriver, WasmtimeCoreEngine
from ..extension import SandboxRegistry
from .base import BaseSandboxProvider


class WasmSandboxProvider(BaseSandboxProvider):
    """Wasmtime 极速轻量沙箱提供者"""

    def get_name(self) -> str:
        return "wasm"

    def get_capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            supports_state=False,
            supported_capabilities=["PythonExecutionCapability"],
            isolation_level=10,
            startup_latency=5,
        )

    def is_available(self) -> bool:
        return WASMTIME_AVAILABLE and WasmtimeCoreEngine.check_wasm_file()

    def score(
        self, profile: SandboxSecurityProfile, requirements: SandboxRequirements | None
    ) -> int:
        if profile.sandbox_type == self.get_name():
            return 200
        if profile.sandbox_type != "auto":
            return -1

        if profile.require_gpu or profile.enable_network or profile.needs_state:
            return -1

        implied_tier = requirements.tier if requirements else SandboxTier.LIGHTWEIGHT

        if implied_tier == SandboxTier.LIGHTWEIGHT:
            return 200

        return -1

    def create_driver(self, session_id: str) -> BaseSandboxDriver:
        return WasmDriver()


SandboxRegistry.register_provider(WasmSandboxProvider())
