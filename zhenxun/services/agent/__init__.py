import nonebot

from zhenxun.services.llm.tools import tool_provider_manager

from .app import AgentApp, AgentRouter
from .core.types import MCPSource, ToolFilter
from .providers.mcp import mcp_provider
from .workflows import (
    BaseWorkflow,
    ChainWorkflow,
    EvaluatorOptimizerWorkflow,
    OrchestratorWorkflow,
    ParallelWorkflow,
    RouterWorkflow,
)

tool_provider_manager.register(mcp_provider)


@nonebot.get_driver().on_shutdown
async def _shutdown_mcp_provider():
    await mcp_provider.shutdown()


app = AgentApp()

__all__ = [
    "AgentRouter",
    "BaseWorkflow",
    "ChainWorkflow",
    "EvaluatorOptimizerWorkflow",
    "MCPSource",
    "OrchestratorWorkflow",
    "ParallelWorkflow",
    "RouterWorkflow",
    "ToolFilter",
    "app",
]
