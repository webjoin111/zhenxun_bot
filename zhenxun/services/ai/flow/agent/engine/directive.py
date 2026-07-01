from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from zhenxun.services.ai.flow.agent.models import AgentRunResources, AgentState
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.utils import infer_plugin_namespace

DirectiveHandlerFunc = Callable[
    [AgentState, AgentRunResources, ToolResult], Awaitable[tuple[Any, str, bool]]
]


class DirectiveManager:
    """工具指令路由注册中心"""

    def __init__(self):
        self._handlers: dict[str, dict[str, DirectiveHandlerFunc]] = defaultdict(dict)

    def register(
        self, name: str, handler: DirectiveHandlerFunc, namespace: str = "global"
    ) -> None:
        self._handlers[namespace][name] = handler
        logger.debug(f"已注册工具副作用指令: '{name}' -> Namespace: '{namespace}'")

    def get_handler(
        self, name: str, namespace: str = "global"
    ) -> DirectiveHandlerFunc | None:
        """优先从指定 namespace 找，找不到回退到 global"""
        ns_dict = self._handlers.get(namespace, {})
        if name in ns_dict:
            return ns_dict[name]
        return self._handlers.get("global", {}).get(name)


directive_manager = DirectiveManager()


def directive(name: str | None = None, namespace: str | None = None):
    """
    注册一个自定义工具副作用指令处理器的装饰器。

    参数:
        name: 指令的名称，如果不填则默认使用被装饰的函数名。
        namespace: 插件命名空间，如果不填则基于代码调用栈自动推断。
    """

    def decorator(func: DirectiveHandlerFunc):
        dir_name = name or func.__name__
        ns = namespace if namespace is not None else infer_plugin_namespace()
        directive_manager.register(dir_name, func, ns)
        return func

    return decorator


@directive("submit_structured", namespace="global")
async def handle_submit_structured(
    state: AgentState, resources: AgentRunResources, tool_res: ToolResult
) -> tuple[Any, str, bool]:
    state.structured_result = tool_res.output
    return tool_res.ui_display, "✅ 结构化结果处理完毕。", True


@directive("end_run", namespace="global")
async def handle_end_run(
    state: AgentState, resources: AgentRunResources, tool_res: ToolResult
) -> tuple[Any, str, bool]:
    state.should_terminate = True
    state.early_result_output = tool_res.output
    return tool_res.ui_display, "✅ 已获取最终结果，结束当前任务。", True


@directive("handoff", namespace="global")
async def handle_handoff(
    state: AgentState, resources: AgentRunResources, tool_res: ToolResult
) -> tuple[Any, str, bool]:
    state.should_terminate = True
    state.early_result_output = tool_res.output
    state.handoff_triggered = tool_res
    target = getattr(tool_res, "target", "unknown")
    return tool_res.ui_display, f"✅ 已决定移交控制权至 {target}。", True
