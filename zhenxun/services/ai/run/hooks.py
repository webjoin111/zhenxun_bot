from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import anyio
from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from zhenxun.services.ai.core.messages import LLMResponse
from zhenxun.services.ai.core.protocols.middleware import LLMContext

from .context import RunContext
from .models import AgentRunResult

_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])


class HookTimeoutError(TimeoutError):
    """当 Hook 函数执行超过配置的时间时抛出此异常。"""

    def __init__(self, hook_name: str, func_name: str, timeout: float):
        self.hook_name = hook_name
        self.func_name = func_name
        self.timeout = timeout
        super().__init__(
            f"Hook {hook_name!r} 中的函数 {func_name!r} 执行超时 ({timeout}s)"
        )


@dataclass
class _HookEntry(Generic[_FuncT]):
    """基础 Hook 注册实体，支持超时配置"""

    func: _FuncT
    timeout: float | None = None


@dataclass
class _ToolHookEntry(_HookEntry[_FuncT]):
    """工具层 Hook 注册实体，支持工具过滤器"""

    tools: frozenset[str] | None = None


class BeforeRunHookFunc(Protocol):
    def __call__(self, ctx: RunContext[Any], /) -> None | Awaitable[None]: ...


class AfterRunHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...


class WrapRunHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, handler: WrapRunHandler
    ) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...


class OnRunErrorHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, error: BaseException
    ) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...


class BeforeModelRequestHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], request_context: LLMContext, /
    ) -> LLMContext | Awaitable[LLMContext]: ...


class AfterModelRequestHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        request_context: LLMContext,
        response: LLMResponse,
    ) -> LLMResponse | Awaitable[LLMResponse]: ...


class WrapModelRequestHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        request_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse | Awaitable[LLMResponse]: ...


class OnModelRequestErrorHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, request_context: LLMContext, error: Exception
    ) -> LLMResponse | Awaitable[LLMResponse]: ...


class PrepareToolsHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], tool_defs: list[Any], /
    ) -> list[Any] | Awaitable[list[Any]]: ...


class BeforeToolValidateHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any] | Awaitable[str | dict[str, Any]]: ...


class AfterToolValidateHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class WrapToolValidateHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class OnToolValidateErrorHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        tool_name: str,
        args: str | dict[str, Any],
        error: Exception,
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class BeforeToolExecuteHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | Awaitable[dict[str, Any]]: ...


class AfterToolExecuteHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any | Awaitable[Any]: ...


class WrapToolExecuteHookFunc(Protocol):
    def __call__(
        self,
        ctx: RunContext[Any],
        /,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any | Awaitable[Any]: ...


class OnToolExecuteErrorHookFunc(Protocol):
    def __call__(
        self, ctx: RunContext[Any], /, *, tool_name: str, error: Exception
    ) -> Any | Awaitable[Any]: ...


async def _call_func(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if is_coroutine_callable(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


async def _call_entry(
    entry: _HookEntry[Any], hook_name: str, *args: Any, **kwargs: Any
) -> Any:
    """调用 Hook 函数实体，并自动应用熔断超时保护"""
    func = entry.func
    if entry.timeout is not None:
        try:
            with anyio.fail_after(entry.timeout):
                return await _call_func(func, *args, **kwargs)
        except TimeoutError:
            raise HookTimeoutError(
                hook_name=hook_name,
                func_name=getattr(func, "__name__", repr(func)),
                timeout=entry.timeout,
            ) from None
    return await _call_func(func, *args, **kwargs)


def _filter_tool_entries(
    entries: list[_HookEntry[Any]], *, tool_name: str
) -> list[_HookEntry[Any]]:
    """按工具名称过滤 Hook 实体"""
    return [
        entry
        for entry in entries
        if not (
            isinstance(entry, _ToolHookEntry)
            and entry.tools is not None
            and tool_name not in entry.tools
        )
    ]


def _bare_or_parameterized(
    registry: dict[str, list[_HookEntry[Any]]],
    key: str,
    func: _FuncT | None,
    *,
    timeout: float | None = None,
) -> _FuncT | Callable[[_FuncT], _FuncT]:
    """处理无参数钩子的带参/不带参装饰器逻辑"""
    if func is not None:
        registry.setdefault(key, []).append(_HookEntry(func, timeout=timeout))
        return func

    def decorator(f: _FuncT) -> _FuncT:
        registry.setdefault(key, []).append(_HookEntry(f, timeout=timeout))
        return f

    return decorator


def _tool_bare_or_parameterized(
    registry: dict[str, list[_HookEntry[Any]]],
    key: str,
    func: _FuncT | None,
    *,
    tools: list[str] | None = None,
    timeout: float | None = None,
) -> _FuncT | Callable[[_FuncT], _FuncT]:
    """处理工具钩子的带参/不带参装饰器逻辑"""
    frozen_tools = frozenset(tools) if tools is not None else None
    if func is not None:
        registry.setdefault(key, []).append(
            _ToolHookEntry(func, timeout=timeout, tools=frozen_tools)
        )
        return func

    def decorator(f: _FuncT) -> _FuncT:
        registry.setdefault(key, []).append(
            _ToolHookEntry(f, timeout=timeout, tools=frozen_tools)
        )
        return f

    return decorator


class _HookRegistration:
    """
    Hooks 的装饰器命名空间。
    利用 @overload 提供完美的 IDE 强类型补全和文档提示。
    """

    def __init__(self, hooks: "Hooks"):
        self._hooks = hooks

    @property
    def _r(self) -> dict[str, list[_HookEntry[Any]]]:
        return self._hooks._registry

    from typing import overload

    @overload
    def before_run(self, func: BeforeRunHookFunc, /) -> BeforeRunHookFunc: ...
    @overload
    def before_run(
        self, *, timeout: float | None = None
    ) -> Callable[[BeforeRunHookFunc], BeforeRunHookFunc]: ...
    def before_run(
        self, func: BeforeRunHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        """注册运行前钩子。在 Agent 启动任何流转前触发。"""
        return _bare_or_parameterized(self._r, "before_run", func, timeout=timeout)

    @overload
    def after_run(self, func: AfterRunHookFunc, /) -> AfterRunHookFunc: ...
    @overload
    def after_run(
        self, *, timeout: float | None = None
    ) -> Callable[[AfterRunHookFunc], AfterRunHookFunc]: ...
    def after_run(
        self, func: AfterRunHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        """注册运行后钩子。在 Agent 获取最终结果后触发，可修改结果。"""
        return _bare_or_parameterized(self._r, "after_run", func, timeout=timeout)

    @overload
    def wrap_run(self, func: WrapRunHookFunc, /) -> WrapRunHookFunc: ...
    @overload
    def wrap_run(
        self, *, timeout: float | None = None
    ) -> Callable[[WrapRunHookFunc], WrapRunHookFunc]: ...
    def wrap_run(
        self, func: WrapRunHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        """注册运行包裹钩子。以洋葱模型接管整个 Agent 运行过程。"""
        return _bare_or_parameterized(self._r, "wrap_run", func, timeout=timeout)

    @overload
    def on_run_error(self, func: OnRunErrorHookFunc, /) -> OnRunErrorHookFunc: ...
    @overload
    def on_run_error(
        self, *, timeout: float | None = None
    ) -> Callable[[OnRunErrorHookFunc], OnRunErrorHookFunc]: ...
    def on_run_error(
        self, func: OnRunErrorHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        """注册运行异常钩子。捕获 Agent 级别的致命错误。"""
        return _bare_or_parameterized(self._r, "on_run_error", func, timeout=timeout)

    @overload
    def before_model_request(
        self, func: BeforeModelRequestHookFunc, /
    ) -> BeforeModelRequestHookFunc: ...
    @overload
    def before_model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[BeforeModelRequestHookFunc], BeforeModelRequestHookFunc]: ...
    def before_model_request(
        self,
        func: BeforeModelRequestHookFunc | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """注册大模型请求前钩子。可在此修改发送给 LLM 的 Messages 等上下文。"""
        return _bare_or_parameterized(
            self._r, "before_model_request", func, timeout=timeout
        )

    @overload
    def after_model_request(
        self, func: AfterModelRequestHookFunc, /
    ) -> AfterModelRequestHookFunc: ...
    @overload
    def after_model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[AfterModelRequestHookFunc], AfterModelRequestHookFunc]: ...
    def after_model_request(
        self,
        func: AfterModelRequestHookFunc | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """注册大模型请求后钩子。可在此验证或修改 LLM 的原始 Response。"""
        return _bare_or_parameterized(
            self._r, "after_model_request", func, timeout=timeout
        )

    @overload
    def wrap_model_request(
        self, func: WrapModelRequestHookFunc, /
    ) -> WrapModelRequestHookFunc: ...
    @overload
    def wrap_model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[WrapModelRequestHookFunc], WrapModelRequestHookFunc]: ...
    def wrap_model_request(
        self,
        func: WrapModelRequestHookFunc | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """注册大模型请求包裹钩子。以洋葱模型接管 LLM 的网络请求过程。"""
        return _bare_or_parameterized(
            self._r, "wrap_model_request", func, timeout=timeout
        )

    @overload
    def on_model_request_error(
        self, func: OnModelRequestErrorHookFunc, /
    ) -> OnModelRequestErrorHookFunc: ...
    @overload
    def on_model_request_error(
        self, *, timeout: float | None = None
    ) -> Callable[[OnModelRequestErrorHookFunc], OnModelRequestErrorHookFunc]: ...
    def on_model_request_error(
        self,
        func: OnModelRequestErrorHookFunc | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """注册大模型请求异常钩子。捕获超时或网络等异常。"""
        return _bare_or_parameterized(
            self._r, "on_model_request_error", func, timeout=timeout
        )

    @overload
    def before_tool_execute(
        self, func: BeforeToolExecuteHookFunc, /
    ) -> BeforeToolExecuteHookFunc: ...
    @overload
    def before_tool_execute(
        self, *, tools: list[str] | None = None, timeout: float | None = None
    ) -> Callable[[BeforeToolExecuteHookFunc], BeforeToolExecuteHookFunc]: ...
    def before_tool_execute(
        self,
        func: BeforeToolExecuteHookFunc | None = None,
        *,
        tools: list[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """注册工具执行前钩子。可通过 tools 参数指定拦截特定工具，可篡改传入参数。"""
        return _tool_bare_or_parameterized(
            self._r, "before_tool_execute", func, tools=tools, timeout=timeout
        )

    @overload
    def after_tool_execute(
        self, func: AfterToolExecuteHookFunc, /
    ) -> AfterToolExecuteHookFunc: ...
    @overload
    def after_tool_execute(
        self, *, tools: list[str] | None = None, timeout: float | None = None
    ) -> Callable[[AfterToolExecuteHookFunc], AfterToolExecuteHookFunc]: ...
    def after_tool_execute(
        self,
        func: AfterToolExecuteHookFunc | None = None,
        *,
        tools: list[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """注册工具执行后钩子。可通过 tools 参数指定拦截特定工具，可篡改返回结果。"""
        return _tool_bare_or_parameterized(
            self._r, "after_tool_execute", func, tools=tools, timeout=timeout
        )

    @overload
    def wrap_tool_execute(
        self, func: WrapToolExecuteHookFunc, /
    ) -> WrapToolExecuteHookFunc: ...
    @overload
    def wrap_tool_execute(
        self, *, tools: list[str] | None = None, timeout: float | None = None
    ) -> Callable[[WrapToolExecuteHookFunc], WrapToolExecuteHookFunc]: ...
    def wrap_tool_execute(
        self,
        func: WrapToolExecuteHookFunc | None = None,
        *,
        tools: list[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """注册工具执行包裹钩子。以洋葱模型接管特定工具的执行逻辑。"""
        return _tool_bare_or_parameterized(
            self._r, "wrap_tool_execute", func, tools=tools, timeout=timeout
        )

    @overload
    def on_tool_execute_error(
        self, func: OnToolExecuteErrorHookFunc, /
    ) -> OnToolExecuteErrorHookFunc: ...
    @overload
    def on_tool_execute_error(
        self, *, tools: list[str] | None = None, timeout: float | None = None
    ) -> Callable[[OnToolExecuteErrorHookFunc], OnToolExecuteErrorHookFunc]: ...
    def on_tool_execute_error(
        self,
        func: OnToolExecuteErrorHookFunc | None = None,
        *,
        tools: list[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """注册工具执行异常钩子。捕获特定工具的崩溃异常，可用于自愈重试。"""
        return _tool_bare_or_parameterized(
            self._r, "on_tool_execute_error", func, tools=tools, timeout=timeout
        )


class Hooks(AbstractCapability):
    """
    面向开发者的极简拦截器语法糖。
    允许通过 `@hooks.on.xxx` 装饰器快速介入大模型及工具生命周期的各个阶段。
    """

    def __init__(self):
        self._registry: dict[str, list[_HookEntry[Any]]] = {}
        self.on = _HookRegistration(self)

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """派发：洋葱模型组装与全周期执行接管"""
        for entry in self._registry.get("before_run", []):
            await _call_entry(entry, "before_run", context)

        entries = self._registry.get("wrap_run", [])
        chain = handler
        if entries:
            for entry in reversed(entries):

                def _wrap(
                    e: _HookEntry[Any], h: Callable[..., Any]
                ) -> Callable[..., Any]:
                    async def _wrapped() -> Any:
                        return await _call_entry(e, "wrap_run", context, h)

                    return _wrapped

                chain = _wrap(entry, chain)

        try:
            result = await chain()
        except BaseException as error:
            for err_entry in reversed(self._registry.get("on_run_error", [])):
                try:
                    return await _call_entry(err_entry, "on_run_error", context, error)
                except BaseException as new_err:
                    error = new_err
            raise error

        for after_entry in reversed(self._registry.get("after_run", [])):
            result = await _call_entry(after_entry, "after_run", context, result)
        return result

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        """派发：洋葱模型接管网络请求及前后生命周期"""
        for entry in self._registry.get("before_model_request", []):
            llm_context = await _call_entry(
                entry, "before_model_request", context, llm_context
            )

        entries = self._registry.get("wrap_model_request", [])
        chain = handler
        if entries:
            for entry in reversed(entries):

                def _wrap(
                    e: _HookEntry[Any], h: Callable[..., Any]
                ) -> Callable[..., Any]:
                    async def _wrapped(ctx_inner: LLMContext) -> Any:
                        return await _call_entry(
                            e, "wrap_model_request", context, ctx_inner, h
                        )

                    return _wrapped

                chain = _wrap(entry, chain)

        try:
            response = await chain(llm_context)
        except Exception as error:
            for err_entry in reversed(self._registry.get("on_model_request_error", [])):
                try:
                    return await _call_entry(
                        err_entry, "on_model_request_error", context, llm_context, error
                    )
                except Exception as new_err:
                    error = new_err
            raise error

        for after_entry in reversed(self._registry.get("after_model_request", [])):
            response = await _call_entry(
                after_entry, "after_model_request", context, llm_context, response
            )
        return response

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        """派发：洋葱模型接管工具参数校验及生命周期"""
        for entry in self._registry.get("before_tool_validate", []):
            args = await _call_entry(
                entry, "before_tool_validate", context, tool_name, args
            )

        entries = self._registry.get("wrap_tool_validate", [])
        chain = handler
        if entries:
            for entry in reversed(entries):

                def _wrap(
                    e: _HookEntry[Any], h: Callable[..., Any]
                ) -> Callable[..., Any]:
                    async def _wrapped(args_inner: str | dict[str, Any]) -> Any:
                        return await _call_entry(
                            e, "wrap_tool_validate", context, tool_name, args_inner, h
                        )

                    return _wrapped

                chain = _wrap(entry, chain)

        try:
            validated_args = await chain(args)
        except Exception as error:
            for err_entry in reversed(self._registry.get("on_tool_validate_error", [])):
                try:
                    return await _call_entry(
                        err_entry,
                        "on_tool_validate_error",
                        context,
                        tool_name,
                        args,
                        error,
                    )
                except Exception as new_err:
                    error = new_err
            raise error

        for after_entry in reversed(self._registry.get("after_tool_validate", [])):
            validated_args = await _call_entry(
                after_entry, "after_tool_validate", context, tool_name, validated_args
            )
        return validated_args

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """派发：洋葱模型接管工具执行（支持匹配名称）及生命周期"""
        for entry in _filter_tool_entries(
            self._registry.get("before_tool_execute", []), tool_name=tool_name
        ):
            arguments = await _call_entry(
                entry, "before_tool_execute", context, tool_name, arguments
            )

        entries = _filter_tool_entries(
            self._registry.get("wrap_tool_execute", []), tool_name=tool_name
        )
        chain = handler
        if entries:
            for entry in reversed(entries):

                def _wrap(
                    e: _HookEntry[Any], h: Callable[..., Any]
                ) -> Callable[..., Any]:
                    async def _wrapped(args_inner: dict[str, Any]) -> Any:
                        return await _call_entry(
                            e, "wrap_tool_execute", context, tool_name, args_inner, h
                        )

                    return _wrapped

                chain = _wrap(entry, chain)

        try:
            result = await chain(arguments)
        except Exception as error:
            for err_entry in reversed(
                _filter_tool_entries(
                    self._registry.get("on_tool_execute_error", []), tool_name=tool_name
                )
            ):
                try:
                    return await _call_entry(
                        err_entry, "on_tool_execute_error", context, tool_name, error
                    )
                except Exception as new_err:
                    error = new_err
            raise error

        for after_entry in reversed(
            _filter_tool_entries(
                self._registry.get("after_tool_execute", []), tool_name=tool_name
            )
        ):
            result = await _call_entry(
                after_entry, "after_tool_execute", context, tool_name, arguments, result
            )
        return result
