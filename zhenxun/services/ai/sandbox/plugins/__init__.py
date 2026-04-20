from zhenxun.services.ai.sandbox.extension import SandboxRegistry

from .browser import PlaywrightBrowserPlugin
from .mcp_proxies import UniversalMcpPlugin
from .python_executors import UniversalPythonPlugin

SandboxRegistry.register_plugin(UniversalPythonPlugin)
SandboxRegistry.register_plugin(PlaywrightBrowserPlugin)
SandboxRegistry.register_plugin(UniversalMcpPlugin)

__all__ = [
    "PlaywrightBrowserPlugin",
    "UniversalMcpPlugin",
    "UniversalPythonPlugin",
]
