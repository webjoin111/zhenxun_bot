from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import graphlib
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Union, cast

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import GenerationConfig
    from zhenxun.services.ai.core.messages import LLMResponse
    from zhenxun.services.ai.flow.agent.models import CapabilitySpec
    from zhenxun.services.ai.protocols.middleware import LLMContext
    from zhenxun.services.ai.run import AgentRunResult, RunContext

WrapRunHandler = Callable[[], Awaitable["AgentRunResult[Any]"]]
"""整个 Agent 运行过程包裹的处理函数类型"""

WrapModelRequestHandler = Callable[["LLMContext"], Awaitable["LLMResponse"]]
"""单次大模型 API 请求包裹的处理函数类型"""

WrapToolValidateHandler = Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]
"""工具参数校验过程包裹的处理函数类型"""

WrapToolExecuteHandler = Callable[[dict[str, Any]], Awaitable[Any]]
"""单一工具执行过程包裹的处理函数类型"""


CapabilityPosition = Literal["outermost", "innermost"]
"""Capability 在洋葱模型中的固定执行位置（最外层或最内层）"""

CapabilityRef = Union[type["AbstractCapability"], "AbstractCapability"]
"""对 Capability 的引用，可以是 Capability 实例或类类型"""



@dataclass
class CapabilityOrdering:
    """定义拦截器 (Capability) 的拓扑排序约束。
    采用洋葱模型语义：排在列表前面的拦截器在最外层执行。
    """

    position: CapabilityPosition | None = None
    """固定位置：outermost (最外层) 或 innermost (最内层)"""
    wraps: Sequence[CapabilityRef] = ()
    """当前拦截器必须包裹（即在...之前执行）目标拦截器"""
    wrapped_by: Sequence[CapabilityRef] = ()
    """当前拦截器必须被包裹（即在...之后执行）目标拦截器"""
    requires: Sequence[type["AbstractCapability"]] = ()
    """当前拦截器依赖的其他拦截器类型，若缺失则报错"""


def sort_capabilities(caps: list["AbstractCapability"]) -> list["AbstractCapability"]:
    """使用标准库 graphlib.TopologicalSorter 实现拦截器拓扑排序，解决执行顺序冲突"""
    if len(caps) <= 1:
        return caps

    ts = graphlib.TopologicalSorter()
    n = len(caps)
    for i in range(n):
        ts.add(i)

    orderings = [c.get_ordering() for c in caps]
    leaf_types = [{type(c)} for c in caps]

    def _ref_matches(
        ref: CapabilityRef, types: set[type], inst: AbstractCapability
    ) -> bool:
        if isinstance(ref, type):
            return any(issubclass(t, ref) for t in types)
        return inst is ref

    all_types = set().union(*leaf_types)
    for i, o in enumerate(orderings):
        if o and o.requires:
            for req in o.requires:
                if not any(issubclass(t, req) for t in all_types):
                    raise ValueError(
                        f"Capability '{type(caps[i]).__name__}' 依赖 '{req.__name__}' 但未在管线中找到该组件。"
                    )

    outermost = {i for i, o in enumerate(orderings) if o and o.position == "outermost"}
    innermost = {i for i, o in enumerate(orderings) if o and o.position == "innermost"}

    for oi in outermost:
        for j in range(n):
            if j != oi and j not in outermost:
                ts.add(j, oi)

    for ii in innermost:
        for j in range(n):
            if j != ii and j not in innermost:
                ts.add(ii, j)

    for i, o in enumerate(orderings):
        if not o:
            continue
        for ref in o.wraps:
            for j in range(n):
                if i != j and _ref_matches(ref, leaf_types[j], caps[j]):
                    ts.add(j, i)
        for ref in o.wrapped_by:
            for j in range(n):
                if i != j and _ref_matches(ref, leaf_types[j], caps[j]):
                    ts.add(i, j)

    try:
        order = list(ts.static_order())
    except graphlib.CycleError:
        raise ValueError(
            "Capability 拓扑排序失败，存在循环依赖约束。请检查 wraps 或 wrapped_by 的配置。"
        )

    return [caps[i] for i in order]


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

    def get_ordering(self) -> CapabilityOrdering | None:
        """获取该拦截器的拓扑排序约束。子类可重写此方法以锁定执行顺序。"""
        return None

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

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        """包裹整个 Agent 运行过程 (洋葱模型)。"""
        return await handler()

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        """包裹单次大模型 API 请求 (洋葱模型)。"""
        return await handler(llm_context)

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        """包裹工具的参数校验过程 (洋葱模型)。"""
        return await handler(args)

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """包裹单一工具的执行 (洋葱模型)。"""
        return await handler(arguments)




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
        flat = []
        for c in capabilities:
            if isinstance(c, CombinedCapability):
                flat.extend(c.capabilities)
            else:
                flat.append(c)

        deduped = []
        seen = set()
        for c in flat:
            if id(c) not in seen:
                seen.add(id(c))
                deduped.append(c)
        self.capabilities = sort_capabilities(deduped)

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

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_wrap_link(cap, "wrap_run", context, {}, chain, None)
        return await chain()

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_wrap_link(
                cap, "wrap_model_request", context, {}, chain, "llm_context"
            )
        return await chain(llm_context)

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_wrap_link(
                cap,
                "wrap_tool_validate",
                context,
                {"tool_name": tool_name},
                chain,
                "args",
            )
        return await chain(args)

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):
            chain = _make_wrap_link(
                cap,
                "wrap_tool_execute",
                context,
                {"tool_name": tool_name},
                chain,
                "arguments",
            )
        return await chain(arguments)


def _make_wrap_link(
    cap: AbstractCapability,
    hook_name: str,
    ctx: RunContext,
    static_kwargs: dict[str, Any],
    inner_handler: Callable[..., Any],
    handler_arg: str | None,
) -> Callable[..., Any]:
    """构建洋葱模型中间件链的单一闭包节点。"""
    frozen_kwargs = dict(static_kwargs)

    if handler_arg:

        async def wrapper(value: Any) -> Any:
            kw = dict(frozen_kwargs)
            kw[handler_arg] = value
            hook_method = getattr(cap, hook_name)
            return await hook_method(ctx, handler=inner_handler, **kw)

        return wrapper

    async def wrapper_no_arg() -> Any:
        hook_method = getattr(cap, hook_name)
        return await hook_method(ctx, handler=inner_handler, **frozen_kwargs)

    return wrapper_no_arg


class DynamicCapability(AbstractCapability):
    """动态能力注入：允许在运行时基于上下文生成真正的 Capability"""

    def __init__(self, capability_func: Callable):
        self.capability_func = capability_func

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        from nonebot.utils import is_coroutine_callable

        if is_coroutine_callable(self.capability_func):
            cap = await self.capability_func(context)
        else:
            cap = self.capability_func(context)
        if cap is None:
            return self
        return await cap.for_run(context)


class WrapperCapability(AbstractCapability):
    """
    代理包装能力基类 (Decorator Pattern)。
    默认将所有生命周期钩子透明透传给内部包裹的 (wrapped) 实例。
    """

    def __init__(self, wrapped: AbstractCapability):
        self.wrapped = wrapped

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        new_wrapped = await self.wrapped.for_run(context)
        if new_wrapped is self.wrapped:
            return self
        import copy

        new_self = copy.copy(self)
        new_self.wrapped = new_wrapped
        return new_self

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        return await self.wrapped.get_generation_config(context)

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        return await self.wrapped.get_system_prompts(context)

    async def get_tools(self, context: RunContext) -> list[Any]:
        return await self.wrapped.get_tools(context)

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        return await self.wrapped.prepare_tools(context, tool_defs)

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        return await self.wrapped.wrap_run(context, handler)

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        return await self.wrapped.wrap_model_request(context, llm_context, handler)

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        return await self.wrapped.wrap_tool_validate(context, tool_name, args, handler)

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        return await self.wrapped.wrap_tool_execute(
            context, tool_name, arguments, handler
        )
