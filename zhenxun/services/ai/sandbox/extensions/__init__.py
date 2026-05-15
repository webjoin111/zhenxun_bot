from zhenxun.services.ai.sandbox.extension import SandboxRegistry

from .mcp_proxies import UniversalMcpExtension
from .python_executors import UniversalPythonExtension

SandboxRegistry.register_extension(UniversalPythonExtension)
SandboxRegistry.register_extension(UniversalMcpExtension)

__all__ = [
    "UniversalMcpExtension",
    "UniversalPythonExtension",
]
