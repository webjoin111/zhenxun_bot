from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.protocols.capabilities import AbstractCapability
from zhenxun.services.ai.tools.models import ToolOptions

if TYPE_CHECKING:
    from zhenxun.services.ai.run import RunContext


def _update_settings(func: Callable, **kwargs) -> Callable:
    """叠加协议底层：创建或更新函数的 __tool_settings__"""
    if not hasattr(func, "__tool_settings__"):
        setattr(func, "__tool_settings__", ToolOptions())
    settings: ToolOptions = getattr(func, "__tool_settings__")
    for k, v in kwargs.items():
        if k == "capabilities":
            settings.capabilities.extend(v)
        elif k == "metadata":
            settings.metadata.update(v)
        else:
            setattr(settings, k, v)
    return func


def require_sandbox(
    python_packages: list[str] | None = None,
    node_packages: list[str] | None = None,
    system_packages: list[str] | None = None,
):
    """显式声明该工具所需安装的沙箱环境依赖"""

    def decorator(func):
        setattr(
            func,
            "__sandbox_requirements__",
            {
                "python": python_packages or [],
                "node": node_packages or [],
                "system": system_packages or [],
            },
        )
        return func

    return decorator


def tool(
    name: str | None = None,
    description: str | None = None,
    settings: ToolOptions | None = None,
    tags: list[str] | None = None,
):
    """
    将普通函数或类方法注册为 LLM 工具的统一装饰器大一统。

    参数:
        name: 工具的名称(英文字母及下划线)。
            大模型将看到此名称。如果为空则默认使用函数名。
        description: 工具描述。
            大模型将基于此决定何时、如何使用该工具。如果为空，将读取函数 docstring。
        settings: 工具的高阶配置对象 (ToolOptions)。
            用于控制极速缓存、人工审批拦截、静默执行等扩展能力。
        tags: 工具的标签列表。
            用于被 Agent 的智能字符串路由识别并进行能力注入。
    返回:
        Callable | FunctionTool: 包装后的函数或方法。
    """

    def decorator(func: Callable):
        base_settings = settings or ToolOptions()
        if tags:
            base_settings.tags = list(set(base_settings.tags + tags))
        existing_settings = getattr(func, "__tool_settings__", None)
        if existing_settings and isinstance(existing_settings, ToolOptions):
            base_settings = existing_settings.merge(base_settings)

        reqs = getattr(func, "__sandbox_requirements__", None)
        if reqs:
            base_settings.sandbox_requirements = reqs

        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "未提供描述"

        import inspect

        is_method = False
        if (
            hasattr(func, "__qualname__")
            and "." in func.__qualname__
            and "<locals>" not in func.__qualname__
        ):
            is_method = True
        else:
            try:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if params and params[0] in ("self", "cls"):
                    is_method = True
            except Exception:
                pass

        if is_method:
            setattr(func, "__toolkit_tool__", True)
            setattr(func, "__tool_name__", tool_name)
            setattr(func, "__tool_desc__", tool_desc)
            setattr(func, "__tool_settings__", base_settings)
            return func
        else:
            from zhenxun.services.ai.tools.core.tool import FunctionTool
            from zhenxun.services.ai.tools.engine.registry import tool_provider_manager
            from zhenxun.services.log import logger

            func_tool = FunctionTool(
                func=func,
                name=tool_name,
                description=tool_desc,
                settings=base_settings,
            )
            tool_provider_manager.register_tool(func_tool)
            logger.info(f"已注册全局工具(Callable): '{tool_name}'")

            setattr(func_tool, "__tool_settings__", base_settings)
            return func_tool

    return decorator


def with_cache(ttl: int = 3600, cache_function: Callable | None = None):
    """开启极速缓存，阻止参数相同的重复请求发往底层。"""

    from zhenxun.services.ai.tools.core.capabilities import CacheCapability

    def decorator(func: Callable):
        return _update_settings(
            func,
            capabilities=[CacheCapability(ttl=ttl, cache_function=cache_function)],
        )

    return decorator


def silent():
    """静默执行。工具执行过程与结果不会作为界面流渲染给用户，仅作大模型内部参考。"""

    class SilentCapability(AbstractCapability):
        async def after_tool_execute(self, context, tool_name, arguments, result):
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(result, ToolResult):
                result.ui_display = None
            return result

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[SilentCapability()])

    return decorator


def direct_reply():
    """直出模式。工具执行完毕后强制中断大模型的思考循环，将工具输出作为最终回答返回。"""

    class DirectReplyCapability(AbstractCapability):
        async def after_tool_execute(self, context, tool_name, arguments, result):
            from zhenxun.services.ai.tools.models import ToolResult

            if isinstance(result, ToolResult):
                if not result.ui_display:
                    result.ui_display = result.output
                from zhenxun.services.ai.core.exceptions import EndRunException

                raise EndRunException(
                    result_output=result.output, display=result.ui_display
                )
            return result

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[DirectReplyCapability()])

    return decorator


def interactive():
    """开启交互式参数补全。如果参数缺失或校验失败，
    会主动通过 Bot 向用户提问要求补全。"""
    from zhenxun.services.ai.tools.core.capabilities import InteractiveCapability

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[InteractiveCapability()])

    return decorator


def require_approval():
    """高危操作标记。调用前将拦截并发送至群组要求超管人工审核。"""
    from zhenxun.services.ai.tools.core.capabilities import ApprovalCapability

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[ApprovalCapability()])

    return decorator


def fallback(tool_name: str):
    """降级备用工具。主工具执行失败时自动路由至备用工具。"""
    from zhenxun.services.ai.tools.core.capabilities import FallbackCapability

    def decorator(func: Callable):
        return _update_settings(
            func,
            capabilities=[FallbackCapability(fallback_tool_name=tool_name)],
        )

    return decorator


def before_execute(hook_func: Callable):
    """生命周期拦截：在工具即将执行前触发，可在此篡改全局状态或通过依赖注入访问当前参数。"""
    from zhenxun.services.ai.tools.core.capabilities import LifecycleCapability

    def decorator(func: Callable):
        return _update_settings(
            func, capabilities=[LifecycleCapability(before_execute=hook_func)]
        )

    return decorator


def after_execute(hook_func: Callable):
    """生命周期拦截：在工具执行完毕后触发，可在此修改将发往大模型的返回结果。"""
    from zhenxun.services.ai.tools.core.capabilities import LifecycleCapability

    def decorator(func: Callable):
        return _update_settings(
            func, capabilities=[LifecycleCapability(after_execute=hook_func)]
        )

    return decorator


def validate_args(hook_func: Callable):
    """生命周期拦截：在工具反序列化校验前触发。如果校验失败直接抛出异常，将触发大模型重试机制。"""
    from zhenxun.services.ai.tools.core.capabilities import LifecycleCapability

    def decorator(func: Callable):
        return _update_settings(
            func, capabilities=[LifecycleCapability(validate_args=hook_func)]
        )

    return decorator


def prepare_tool(hook_func: Callable):
    """生命周期拦截：在向大模型渲染 JSON Schema 前触发，
    可在此动态修改该工具的定义或返回 None 对大模型隐藏。"""
    from zhenxun.services.ai.tools.core.capabilities import LifecycleCapability

    def decorator(func: Callable):
        return _update_settings(
            func, capabilities=[LifecycleCapability(prepare_tool=hook_func)]
        )

    return decorator


def require_superuser():
    """仅限超级管理员使用。若调用者无权限，该工具将直接对大模型隐藏。"""
    from zhenxun.services.ai.tools.core.capabilities import SuperuserCapability

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[SuperuserCapability()])

    return decorator


def require_admin_level(min_level: int = 1):
    """仅限满足指定群聊权限等级的用户使用。"""
    from zhenxun.services.ai.tools.core.capabilities import AdminLevelCapability

    def decorator(func: Callable):
        return _update_settings(
            func,
            capabilities=[AdminLevelCapability(min_level)],
            metadata={"admin_level": min_level},
        )

    return decorator


def require_group():
    """限制该工具仅能在群聊环境中被大模型调用。"""
    from zhenxun.services.ai.tools.core.capabilities import GroupOnlyCapability

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[GroupOnlyCapability()])

    return decorator


def require_minimum_gold(amount: int):
    """需满足一定的金币余额才可调用，同时执行后会自动扣除该金币。"""

    def decorator(func: Callable):
        return _update_settings(func, metadata={"cost_gold": amount})

    return decorator


def require_session_state(key: str, expected_value: Any = None):
    """需要当前执行上下文中包含指定状态变量，否则隐藏工具"""

    class StateDependencyCapability(AbstractCapability):
        async def prepare_tools(
            self, context: RunContext, tool_defs: list[Any]
        ) -> list[Any]:
            val = context.state.get(key)
            if expected_value is not None:
                if val == expected_value:
                    return tool_defs
            elif val is not None and val is not False:
                return tool_defs
            return []

    def decorator(func: Callable):
        return _update_settings(func, capabilities=[StateDependencyCapability()])

    return decorator


class Rules:
    """
    [命名空间] 大模型工具规则与权限装饰器聚合。
    用于控制工具的极速缓存、沙箱要求、前端静默以及人机交互审批等。
    """

    sandbox = staticmethod(require_sandbox)
    """环境：声明执行该工具所需的物理沙箱及 pip/npm 包依赖"""

    approval = staticmethod(require_approval)
    """高危：拦截执行，并在群内发起人工审批 (HITL) 确认"""

    superuser = staticmethod(require_superuser)
    """权限：仅允许被超级管理员触发时，该工具才对大模型可见"""

    admin_level = staticmethod(require_admin_level)
    """权限：要求触发用户的群等级达到指定级别"""

    group_only = staticmethod(require_group)
    """环境：限制该工具只有在群聊场景下才可被调用"""

    minimum_gold = staticmethod(require_minimum_gold)
    """经济：要求用户满足金币余额，且执行后将自动扣除该金币"""

    session_state = staticmethod(require_session_state)
    """状态：要求当前会话上下文中包含指定的状态变量时才可用"""

    before_execute = staticmethod(before_execute)
    """拦截器：在工具正式执行前触发，可动态修改传入参数 (利用 DI 注入)"""

    after_execute = staticmethod(after_execute)
    """拦截器：在工具执行完毕后触发，可动态修改返回结果 (利用 DI 注入)"""

    validate_args = staticmethod(validate_args)
    """拦截器：在工具参数校验前触发，校验失败抛出异常直接由大模型重试"""

    prepare_tool = staticmethod(prepare_tool)
    """拦截器：向大模型渲染 JSON Schema 前触发，可篡改工具定义对象"""

    cache = staticmethod(with_cache)
    """性能：开启极速缓存，参数相同时直接返回上次结果，免去重复计算"""

    silent = staticmethod(silent)
    """UI：静默执行，执行过程与结果不以气泡形式发送给用户，仅供大模型参考"""

    interactive = staticmethod(interactive)
    """交互：开启交互式参数补全，必填参数缺失时自动向用户提问补全"""

    fallback = staticmethod(fallback)
    """容错：执行失败时透明降级重定向至备用工具"""

    direct_reply = staticmethod(direct_reply)
    """流控：直出模式，执行后直接将结果发给用户并强制中断大模型后续思考"""


__all__ = [
    "Rules",
    "tool",
]
