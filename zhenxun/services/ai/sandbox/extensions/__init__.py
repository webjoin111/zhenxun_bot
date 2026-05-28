from zhenxun.services.ai.sandbox.extension import SandboxRegistry

from .mcp_proxies import UniversalMcpExtension

SandboxRegistry.register_extension(UniversalMcpExtension)

__all__ = [
    "UniversalMcpExtension",
]
