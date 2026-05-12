from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.core.messages import LLMResponse
from zhenxun.services.ai.protocols.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from zhenxun.services.ai.protocols.middleware import LLMContext
from zhenxun.services.ai.run import AgentRunResult, RunContext

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import GenerationConfig

_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])


class _HookEntry(Generic[_FuncT]):
    def __init__(self, func: _FuncT, tools: list[str] | None = None):
        self.func = func
        self.tools = tools


async def _call_func(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if is_coroutine_callable(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


class _HookRegistration:
    def __init__(self, hooks: "Hooks"):
        self._hooks = hooks

    def _register(
        self, key: str, func: Callable[..., Any] | None, tools: list[str] | None = None
    ) -> Any:
        """底层注册逻辑，支持带参数或不带参数的装饰器调用。"""
        if func is not None:
            self._hooks._registry.setdefault(key, []).append(_HookEntry(func, tools))
            return func

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            self._hooks._registry.setdefault(key, []).append(_HookEntry(f, tools))
            return f

        return decorator

    def before_run(self, func: Callable[..., Any] | None = None) -> Any:
        """注册运行前钩子。在 Agent 启动任何流转前触发。"""
        return self._register("before_run", func)

    def get_generation_config(self, func: Callable[..., Any] | None = None) -> Any:
        """注册配置动态生成钩子。可在此根据上下文动态返回 GenerationConfig。"""
        return self._register("get_generation_config", func)

    def after_run(self, func: Callable[..., Any] | None = None) -> Any:
        """注册运行后钩子。在 Agent 获取最终结果后触发，可修改结果。"""
        return self._register("after_run", func)

    def wrap_run(self, func: Callable[..., Any] | None = None) -> Any:
        """注册运行包裹钩子。以洋葱模型接管整个 Agent 运行过程。"""
        return self._register("wrap_run", func)

    def on_run_error(self, func: Callable[..., Any] | None = None) -> Any:
        """注册运行异常钩子。捕获 Agent 级别的致命错误。"""
        return self._register("on_run_error", func)

    def before_model_request(self, func: Callable[..., Any] | None = None) -> Any:
        """注册大模型请求前钩子。可在此修改发送给 LLM 的 Messages 等上下文。"""
        return self._register("before_model_request", func)

    def after_model_request(self, func: Callable[..., Any] | None = None) -> Any:
        """注册大模型请求后钩子。可在此验证或修改 LLM 的原始 Response。"""
        return self._register("after_model_request", func)

    def wrap_model_request(self, func: Callable[..., Any] | None = None) -> Any:
        """注册大模型请求包裹钩子。以洋葱模型接管 LLM 的网络请求过程。"""
        return self._register("wrap_model_request", func)

    def on_model_request_error(self, func: Callable[..., Any] | None = None) -> Any:
        """注册大模型请求异常钩子。捕获超时或网络等异常。"""
        return self._register("on_model_request_error", func)

    def before_tool_execute(
        self, func: Callable[..., Any] | None = None, *, tools: list[str] | None = None
    ) -> Any:
        """注册工具执行前钩子。可通过 tools 参数指定拦截特定工具，可篡改传入参数。"""
        return self._register("before_tool_execute", func, tools)

    def after_tool_execute(
        self, func: Callable[..., Any] | None = None, *, tools: list[str] | None = None
    ) -> Any:
        """注册工具执行后钩子。可通过 tools 参数指定拦截特定工具，可篡改返回结果。"""
        return self._register("after_tool_execute", func, tools)

    def wrap_tool_execute(
        self, func: Callable[..., Any] | None = None, *, tools: list[str] | None = None
    ) -> Any:
        """注册工具执行包裹钩子。以洋葱模型接管特定工具的执行逻辑。"""
        return self._register("wrap_tool_execute", func, tools)

    def on_tool_execute_error(
        self, func: Callable[..., Any] | None = None, *, tools: list[str] | None = None
    ) -> Any:
        """注册工具执行异常钩子。捕获特定工具的崩溃异常，可用于自愈重试。"""
        return self._register("on_tool_execute_error", func, tools)


class Hooks(AbstractCapability):
    """
    面向开发者的极简拦截器语法糖。
    允许通过 `@hooks.on.xxx` 装饰器快速介入大模型及工具生命周期的各个阶段。
    所有业务逻辑拦截（如打印、参数篡改、临时授权等）均可使用此类。
    """

    def __init__(self):
        self._registry: dict[str, list[_HookEntry[Any]]] = {}
        self.on = _HookRegistration(self)

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        """派发：获取动态生成的 GenerationConfig"""
        final_config = None
        for entry in self._registry.get("get_generation_config", []):
            cap_config = await _call_func(entry.func, context)
            if cap_config:
                if final_config is None:
                    final_config = cap_config
                else:
                    final_config = final_config.merge_with(cap_config)
        return final_config

    async def before_run(self, context: RunContext) -> None:
        """派发：运行前拦截"""
        for entry in self._registry.get("before_run", []):
            await _call_func(entry.func, context)

    async def after_run(
        self, context: RunContext, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        """派发：运行后拦截"""
        for entry in reversed(self._registry.get("after_run", [])):
            result = await _call_func(entry.func, context, result)
        return result

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """派发：洋葱模型组装与执行"""
        entries = self._registry.get("wrap_run", [])
        if not entries:
            return await handler()
        chain = handler
        for entry in reversed(entries):

            def _wrap(e, h):
                async def _wrapped():
                    return await _call_func(e.func, context, h)

                return _wrapped

            chain = _wrap(entry, chain)
        return await chain()

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> AgentRunResult[Any]:
        """派发：运行时异常捕获"""
        for entry in reversed(self._registry.get("on_run_error", [])):
            try:
                return await _call_func(entry.func, context, error)
            except BaseException as new_err:
                error = new_err
        raise error

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        """派发：模型请求前拦截"""
        for entry in self._registry.get("before_model_request", []):
            llm_context = await _call_func(entry.func, context, llm_context)
        return llm_context

    async def after_model_request(
        self, context: RunContext, llm_context: LLMContext, response: LLMResponse
    ) -> LLMResponse:
        """派发：模型返回后拦截"""
        for entry in reversed(self._registry.get("after_model_request", [])):
            response = await _call_func(entry.func, context, llm_context, response)
        return response

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        """派发：洋葱模型接管网络请求"""
        entries = self._registry.get("wrap_model_request", [])
        if not entries:
            return await handler(llm_context)
        chain = handler
        for entry in reversed(entries):

            def _wrap(e, h):
                async def _wrapped(ctx_inner):
                    return await _call_func(e.func, context, ctx_inner, h)

                return _wrapped

            chain = _wrap(entry, chain)
        return await chain(llm_context)

    async def on_model_request_error(
        self, context: RunContext, llm_context: LLMContext, error: Exception
    ) -> LLMResponse:
        """派发：网络请求异常捕获"""
        for entry in reversed(self._registry.get("on_model_request_error", [])):
            try:
                return await _call_func(entry.func, context, llm_context, error)
            except Exception as new_err:
                error = new_err
        raise error

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """派发：工具执行前拦截（支持匹配名称）"""
        for entry in self._registry.get("before_tool_execute", []):
            if entry.tools is None or tool_name in entry.tools:
                arguments = await _call_func(entry.func, context, tool_name, arguments)
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        """派发：工具执行后拦截（支持匹配名称）"""
        for entry in reversed(self._registry.get("after_tool_execute", [])):
            if entry.tools is None or tool_name in entry.tools:
                result = await _call_func(
                    entry.func, context, tool_name, arguments, result
                )
        return result

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """派发：洋葱模型接管工具执行（支持匹配名称）"""
        entries = [
            e
            for e in self._registry.get("wrap_tool_execute", [])
            if e.tools is None or tool_name in e.tools
        ]
        if not entries:
            return await handler(arguments)
        chain = handler
        for entry in reversed(entries):

            def _wrap(e, h):
                async def _wrapped(args_inner):
                    return await _call_func(e.func, context, tool_name, args_inner, h)

                return _wrapped

            chain = _wrap(entry, chain)
        return await chain(arguments)

    async def on_tool_execute_error(
        self, context: RunContext, tool_name: str, error: Exception
    ) -> Any:
        """派发：工具执行异常捕获（支持匹配名称）"""
        for entry in reversed(self._registry.get("on_tool_execute_error", [])):
            if entry.tools is None or tool_name in entry.tools:
                try:
                    return await _call_func(entry.func, context, tool_name, error)
                except Exception as new_err:
                    error = new_err
        raise error
