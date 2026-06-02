from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import GenerationConfig
    from zhenxun.services.ai.core.messages import LLMResponse
    from zhenxun.services.ai.flow.agent.models import CapabilitySpec
    from zhenxun.services.ai.protocols.middleware import LLMContext
    from zhenxun.services.ai.run import AgentRunResult, RunContext

WrapRunHandler = Callable[[], Awaitable["AgentRunResult[Any]"]]
WrapModelRequestHandler = Callable[["LLMContext"], Awaitable["LLMResponse"]]
WrapToolValidateHandler = Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]
WrapToolExecuteHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class AbstractCapability:
    """
    Agent 能力组件基类协议。
    所有业务逻辑拦截（限流、权限、动态 Prompt）请在此实现。
    底层网络重试、并发控制等请勿在此处理。
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """用于 YAML/JSON 反序列化的注册标识符"""
        return cls.__name__

    @classmethod
    def from_spec(cls, **kwargs) -> "AbstractCapability":
        """从 Spec 的 kwargs 中实例化对象"""
        return cls(**kwargs)

    def __init_subclass__(cls, **kwargs):
        """自动将继承此类的所有拦截器注册到中心表"""
        super().__init_subclass__(**kwargs)
        CapabilityRegistry.register(cls)

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        """获取专用于单次运行的实例。
        默认返回自身(无状态)。
        若需要记录单次运行的上下文状态，请返回深/浅拷贝(如 return copy.copy(self))。
        """
        return self

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        """运行开始前触发。允许动态下发大模型配置（覆盖或合并 Agent 的默认配置）。"""
        return None

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        return []

    async def get_tools(self, context: RunContext) -> list[Any]:
        return []

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        """运行开始前/装配工具时触发。允许动态增删改当前将发往大模型的工具列表。
        默认实现：无操作，直接返回传入的工具列表。"""
        return tool_defs

    async def before_run(self, context: RunContext) -> None:
        """运行开始前触发。仅用于观察或初始化状态。"""
        pass

    async def after_run(
        self, context: RunContext, result: "AgentRunResult[Any]"
    ) -> "AgentRunResult[Any]":
        """运行成功结束后触发。可修改最终的运行结果。"""
        return result

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        """包裹整个 Agent 运行过程 (洋葱模型)。"""
        return await handler()

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> "AgentRunResult[Any]":
        """运行发生致命异常时触发。若不处理，必须重新抛出 error。
        可返回 AgentRunResult实现自愈。"""
        raise error

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        """大模型发起请求前触发。可动态修改 Prompt、工具列表或生成配置。"""
        return llm_context

    async def after_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        response: LLMResponse,
    ) -> LLMResponse:
        """大模型成功返回后触发。可修改或验证大模型的原始返回对象。"""
        return response

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        """包裹单次大模型 API 请求 (洋葱模型)。"""
        return await handler(llm_context)

    async def on_model_request_error(
        self, context: RunContext, llm_context: LLMContext, error: Exception
    ) -> LLMResponse:
        """大模型请求失败（如网络超时）时触发。
        可调用备用模型实现故障转移，若不处理需抛出 error。"""
        raise error

    async def before_tool_validate(
        self, context: RunContext, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        """工具参数校验前触发。可清洗、修改原始参数字符串或字典。"""
        return args

    async def after_tool_validate(
        self, context: RunContext, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """工具参数校验通过后触发。接收的是反序列化后的标准字典。"""
        return args

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        """包裹工具的参数校验过程 (洋葱模型)。"""
        return await handler(args)

    async def on_tool_validate_error(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        """参数校验失败（如 Schema 不匹配）时触发。可用于交互式参数补全或自愈。"""
        raise error

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """工具实际执行前触发。可校验或篡改传入参数。"""
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        """工具成功执行后触发。可加工或过滤工具的输出结果。"""
        return result

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """包裹单一工具的执行 (洋葱模型)。"""
        return await handler(arguments)

    async def on_tool_execute_error(
        self, context: RunContext, tool_name: str, error: Exception
    ) -> Any:
        """工具执行发生异常时触发。
        可返回特定提示信息引导大模型自我反思 (Reflexion)，若不处理需抛出 error。"""
        raise error


class CapabilityRegistry:
    """Capability 序列化注册表"""

    _registry: ClassVar[dict[str, type[AbstractCapability]]] = {}

    @classmethod
    def register(cls, cap_cls: type[AbstractCapability]):
        name = cap_cls.get_serialization_name()
        if name:
            cls._registry[name] = cap_cls

    @classmethod
    def get(cls, name: str) -> type[AbstractCapability] | None:
        return cls._registry.get(name)

    @classmethod
    def create_from_spec(cls, spec: "CapabilitySpec") -> AbstractCapability:
        cap_cls = cls.get(spec.name)
        if not cap_cls:
            raise ValueError(f"未知的 Capability 插件标识符: {spec.name}")
        extra_kwargs = cast(dict[str, Any], spec.model_extra or {})
        return cap_cls.from_spec(**extra_kwargs)


class CombinedCapability(AbstractCapability):
    """
    组合能力容器。
    将多个 Capability 按顺序融合成一个复合的洋葱模型，
    处理生命周期的正序/倒序和链式调用。
    """

    def __init__(self, capabilities: list[AbstractCapability]):
        deduped = []
        seen = set()
        for c in capabilities:
            if id(c) not in seen:
                seen.add(id(c))
                deduped.append(c)
        self.capabilities = deduped

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        new_caps = []
        changed = False
        for cap in self.capabilities:
            new_cap = await cap.for_run(context)
            new_caps.append(new_cap)
            if new_cap is not cap:
                changed = True

        if changed:
            return CombinedCapability(new_caps)
        return self

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        final_config = None
        for cap in self.capabilities:
            cap_config = await cap.get_generation_config(context)
            if cap_config:
                if final_config is None:
                    final_config = cap_config
                else:
                    final_config = final_config.merge_with(cap_config)
        return final_config

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        prompts = []
        for cap in self.capabilities:
            prompts.extend(await cap.get_system_prompts(context))
        return prompts

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []
        for cap in self.capabilities:
            tools.extend(await cap.get_tools(context))
        return tools

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        current_defs = list(tool_defs)
        for cap in self.capabilities:
            res = await cap.prepare_tools(context, current_defs)
            if res is not None:
                current_defs = res
        return current_defs

    async def before_run(self, context: RunContext) -> None:
        for cap in self.capabilities:
            await cap.before_run(context)

    async def after_run(
        self, context: RunContext, result: "AgentRunResult[Any]"
    ) -> "AgentRunResult[Any]":
        for cap in reversed(self.capabilities):
            result = await cap.after_run(context, result)
        return result

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped():
                    return await c.wrap_run(context, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain()

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> "AgentRunResult[Any]":
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_run_error(context, error)
            except BaseException as new_error:
                error = new_error
        raise error

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        for cap in self.capabilities:
            llm_context = await cap.before_model_request(context, llm_context)
        return llm_context

    async def after_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        response: LLMResponse,
    ) -> LLMResponse:
        for cap in reversed(self.capabilities):
            response = await cap.after_model_request(context, llm_context, response)
        return response

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(ctx_inner):
                    return await c.wrap_model_request(context, ctx_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(llm_context)

    async def on_model_request_error(
        self, context: RunContext, llm_context: LLMContext, error: Exception
    ) -> LLMResponse:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_model_request_error(context, llm_context, error)
            except Exception as new_error:
                error = new_error
        raise error

    async def before_tool_validate(
        self, context: RunContext, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        for cap in self.capabilities:
            args = await cap.before_tool_validate(context, tool_name, args)
        return args

    async def after_tool_validate(
        self, context: RunContext, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        for cap in reversed(self.capabilities):
            args = await cap.after_tool_validate(context, tool_name, args)
        return args

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(args_inner):
                    return await c.wrap_tool_validate(context, tool_name, args_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(args)

    async def on_tool_validate_error(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_tool_validate_error(context, tool_name, args, error)
            except Exception as new_error:
                error = new_error
        raise error

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        for cap in self.capabilities:
            arguments = await cap.before_tool_execute(context, tool_name, arguments)
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        for cap in reversed(self.capabilities):
            result = await cap.after_tool_execute(context, tool_name, arguments, result)
        return result

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(args_inner):
                    return await c.wrap_tool_execute(context, tool_name, args_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(arguments)

    async def on_tool_execute_error(
        self, context: RunContext, tool_name: str, error: Exception
    ) -> Any:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_tool_execute_error(context, tool_name, error)
            except Exception as new_error:
                error = new_error
        raise error

