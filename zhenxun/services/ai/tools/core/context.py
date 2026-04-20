from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses
import inspect
from typing import (
    Annotated,
    Any,
    Generic,
    get_origin,
)
from typing_extensions import TypeVar

from nonebot.adapters import Bot, Event
from nonebot.matcher import Matcher
from nonebot_plugin_session import EventSession


@dataclasses.dataclass
class ModelExecutionInfo:
    """大模型执行底层的内部元数据，向普通业务开发者隐藏"""

    model_name: str | None = None
    history_messages: list[Any] = dataclasses.field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 0
    tool_name: str | None = None
    tool_call_id: str | None = None


RunContextAgentDepsT = TypeVar("RunContextAgentDepsT", default=Any)


@dataclasses.dataclass(kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    """
    依赖注入容器（DI Container），保留原有上下文信息的同时提升获取类型的能力。
    """

    session_id: str | None = None
    """当前运行所在的会话ID，用于区分不同用户的独立上下文"""

    bot: Bot | None = None
    """当前触发事件的 Bot 实例"""

    event: Event | None = None
    """当前触发的事件实例"""

    matcher: Matcher | None = None
    """当前处理该事件的 Matcher 实例"""

    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    """运行时透传的额外变量集合，供中间件和 Toolkit 存取状态使用"""

    deps: RunContextAgentDepsT | None = None
    """强类型的外部依赖注入对象，用于跨工具共享业务状态或配置"""

    cancellation_token: Any = None
    """全局取消令牌，用于在异步链路中传递中止信号"""

    _model_execution_info: ModelExecutionInfo = dataclasses.field(
        default_factory=ModelExecutionInfo
    )
    """隐藏的大模型执行级元数据"""

    def clone_for_execution(self, **kwargs) -> "RunContext":
        """
        基于浅拷贝克隆上下文，防止深拷贝引发异步锁或连接池崩溃。
        保障并发执行环境下的状态安全。
        """
        exec_info_kwargs = {}
        for k in [
            "model_name",
            "history_messages",
            "retry_count",
            "max_retries",
            "tool_name",
            "tool_call_id",
        ]:
            if k in kwargs:
                exec_info_kwargs[k] = kwargs.pop(k)

        new_extra = self.extra.copy()
        changes = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        if "extra" not in changes:
            changes["extra"] = new_extra

        new_ctx = dataclasses.replace(self, **changes)
        new_ctx._model_execution_info = dataclasses.replace(
            self._model_execution_info, **exec_info_kwargs
        )
        return new_ctx

    def get_user_id(self) -> str | None:
        """安全提取当前触发用户的 ID"""
        if self.deps and hasattr(self.deps, "user_id"):
            return str(getattr(self.deps, "user_id"))
        if self.event:
            try:
                return str(self.event.get_user_id())
            except Exception:
                return str(getattr(self.event, "user_id", "")) or None
        return None

    def get_group_id(self) -> str | None:
        """安全提取当前群组 ID"""
        if self.deps and hasattr(self.deps, "group_id"):
            return str(getattr(self.deps, "group_id"))
        if self.event:
            return str(getattr(self.event, "group_id", "")) or None
        return None

    def get_platform(self) -> str:
        """安全提取平台名称"""
        if self.deps and hasattr(self.deps, "platform"):
            return str(getattr(self.deps, "platform"))
        if self.bot:
            from zhenxun.utils.platform import PlatformUtils

            return PlatformUtils.get_platform(self.bot)
        return "unknown"

    async def emit(
        self,
        message: str,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """主动向外界报告工具执行进度，直接触发 EventCenter"""
        from zhenxun.services.ai.events import EventCenter, ToolStreamEvent
        from zhenxun.services.ai.types.tools import ToolResultChunk

        chunk = ToolResultChunk(content=message, status=status, metadata=metadata)
        await EventCenter.publish(
            ToolStreamEvent(
                tool_call_id=self._model_execution_info.tool_call_id or "manual_emit",
                tool_name=self._model_execution_info.tool_name or "unknown",
                chunk=chunk,
                session_id=self.session_id,
            )
        )


"""隐式上下文管理 (ContextVar)"""
_CURRENT_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "zhenxun.current_run_context", default=None
)


def get_current_context() -> RunContext:
    """获取当前运行环境下的上下文，若不存在则抛出异常"""
    ctx = _CURRENT_RUN_CONTEXT.get()
    if ctx is None:
        raise RuntimeError(
            "无法获取 RunContext，请确保在 set_current_context 上下文管理器内调用。"
        )
    return ctx


@contextmanager
def set_current_context(ctx: RunContext):
    """设置当前异步/同步执行作用域下的 RunContext"""
    token = _CURRENT_RUN_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _CURRENT_RUN_CONTEXT.reset(token)


async def emit(
    message: str, status: str = "running", metadata: dict[str, Any] | None = None
) -> None:
    """全局免参进度反馈方法，内部自动获取 ContextVar 并发起推送"""
    ctx = get_current_context()
    await ctx.emit(message, status, metadata)


def _is_run_context_type(annotation: Any) -> bool:
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is RunContext:
        return True
    if inspect.isclass(origin) and issubclass(origin, RunContext):
        return True
    if inspect.isclass(annotation) and issubclass(annotation, RunContext):
        return True
    if "RunContext" in str(annotation):
        return True
    return False


class Hidden:
    """
    标记一个参数为隐藏参数。
    在生成大模型工具 Schema 时，带有该标记的 Annotated 参数将被剔除。
    """

    pass


class _InjectMarker:
    """内部魔术标记，用于识别 DI 类型糖"""

    def __init__(self, key: str):
        self.key = key


CurrentUserId = Annotated[str | None, Hidden(), _InjectMarker("user_id")]
CurrentGroupId = Annotated[str | None, Hidden(), _InjectMarker("group_id")]
CurrentPlatform = Annotated[str, Hidden(), _InjectMarker("platform")]
CurrentBot = Annotated[Bot | None, Hidden(), _InjectMarker("bot")]
CurrentEvent = Annotated[Event | None, Hidden(), _InjectMarker("event")]
CurrentMatcher = Annotated[Matcher | None, Hidden(), _InjectMarker("matcher")]
CurrentSession = Annotated[EventSession | None, Hidden(), _InjectMarker("session")]


class GlobalDependencyRegistry:
    """全局依赖注册表，用于提供跨工具和跨中间件的对象注入支持"""

    _providers: dict[type, Callable[[], Any]] = {}

    @classmethod
    def register(cls, type_: type, provider: Callable[[], Any]) -> None:
        """注册一个特定类型的提供者工厂"""
        cls._providers[type_] = provider

    @classmethod
    def get(cls, type_: type) -> Any | None:
        """获取并实例化该类型，若未注册返回 None"""
        if type_ in cls._providers:
            return cls._providers[type_]()
        return None

    @classmethod
    def has_provider(cls, type_: type) -> bool:
        """判断是否注册了该类型的提供者"""
        return type_ in cls._providers


global_dependency_registry = GlobalDependencyRegistry()
