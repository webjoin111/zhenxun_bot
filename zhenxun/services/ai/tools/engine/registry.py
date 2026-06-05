from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import inspect
from typing import Any, Generic, SupportsIndex, TypeVar, cast, overload
from typing_extensions import Self

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.protocols.tool import (
    ToolExecutable,
    ToolProvider,
)
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import Query, ResolvedToolPayload
from zhenxun.services.log import logger

T = TypeVar("T", bound=ToolExecutable)


class ToolCollection(list[T], Generic[T]):
    """支持按索引和按名称获取的工具集合 (List + Dict)"""

    def __init__(self, tools: Iterable[T] | None = None):
        super().__init__(tools or [])
        self._name_cache: dict[str, T] = {}
        self._build_name_cache()

    def _build_name_cache(self) -> None:
        self._name_cache = {}
        for tool in self:
            name = getattr(tool, "name", None)
            if name:
                self._name_cache[name.lower()] = tool

    @overload
    def __getitem__(self, key: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> list[T]: ...

    @overload
    def __getitem__(self, key: str) -> T: ...

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._name_cache[key.lower()]
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: SupportsIndex, value: T) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]) -> None: ...

    @overload
    def __setitem__(self, key: str, value: T) -> None: ...

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            name = key.lower()
            if not hasattr(value, "name"):
                setattr(value, "name", key)
            if name in self._name_cache:
                old_tool = self._name_cache[name]
                try:
                    idx = super().index(old_tool)
                    super().__setitem__(idx, value)
                except ValueError:
                    super().append(value)
            else:
                super().append(value)
            self._name_cache[name] = value
        else:
            super().__setitem__(key, value)
            name = getattr(value, "name", None)
            if name:
                self._name_cache[name.lower()] = value

    def get(self, key: str, default: Any = None) -> T | Any:
        return self._name_cache.get(key.lower(), default)

    def append(self, tool: T) -> None:
        name = getattr(tool, "name", None)
        if name and name.lower() in self._name_cache:
            old_tool = self._name_cache[name.lower()]
            try:
                idx = super().index(old_tool)
                super().__setitem__(idx, tool)
            except ValueError:
                super().append(tool)
        else:
            super().append(tool)
        if name:
            self._name_cache[name.lower()] = tool

    def extend(self, tools: Iterable[T]) -> None:
        for t in tools:
            self.append(t)

    def remove(self, tool: T) -> None:
        super().remove(tool)
        name = getattr(tool, "name", None)
        if name and name.lower() in self._name_cache:
            del self._name_cache[name.lower()]

    def pop(self, index: Any = -1) -> T:
        tool = super().pop(index)
        name = getattr(tool, "name", None)
        if name and name.lower() in self._name_cache:
            del self._name_cache[name.lower()]
        return tool

    def filter_by_names(self, names: list[str] | None = None) -> "ToolCollection[T]":
        if names is None:
            return self
        return ToolCollection(
            [
                tool
                for name in names
                if (tool := self._name_cache.get(name.lower())) is not None
            ]
        )

    def clear(self) -> None:
        super().clear()
        self._name_cache.clear()

    def keys(self):
        return self._name_cache.keys()

    def values(self):
        return self._name_cache.values()

    def items(self):
        return self._name_cache.items()


class _StringResolver:
    def __init__(
        self, name: str, manager: "ToolProviderManager", default_namespace: str
    ):
        self.name = name
        self.manager = manager
        self.default_namespace = default_namespace

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        if self.name in self.manager._macro_resolvers:
            resolver = self.manager._macro_resolvers[self.name]
            resolved = (
                await resolver() if is_coroutine_callable(resolver) else resolver()
            )
            return await self.manager._normalize_to_resolver(
                resolved, self.default_namespace
            ).resolve(context)

        s = self.name

        if "." in s:
            ns, target = s.split(".", 1)
        else:
            ns = self.default_namespace
            target = s

        from zhenxun.services.ai.tools.models import Query

        if target == "*":
            query = Query(namespace=ns)
        elif target.startswith("#"):
            tags = [t for t in target.split("#") if t]
            query = Query(tags=tags, namespace=ns)
        else:
            query = Query(name=target, namespace=ns)

        logger.debug(f"🔍 [StringRouter] 语法解析: '{self.name}' -> {query}")
        return await _QueryResolver(query, self.manager).resolve(context)


class _QueryResolver:
    def __init__(self, query: Query, manager: "ToolProviderManager"):
        self.query = query
        self.manager = manager

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        payload = ResolvedToolPayload()
        namespaces_to_search = []

        if self.query.namespace == "global":
            namespaces_to_search = list(self.manager._namespaced_tools.keys())
        elif self.query.namespace:
            namespaces_to_search = [self.query.namespace]
        else:
            raise ValueError(f"Query 对象必须显式指定 namespace 作用域: {self.query}")

        for ns in namespaces_to_search:
            if ns in self.manager._namespaced_tools:
                for tool in self.manager._namespaced_tools[ns]:
                    if self.query.match(tool):
                        if hasattr(tool, "resolve"):
                            p = await tool.resolve(context)
                            if p:
                                payload.tools.extend(p.tools)
                                payload.injected_prompts.extend(p.injected_prompts)
                                payload.toolkits.extend(p.toolkits)
                        else:
                            payload.tools.append(tool)

        if self.query.name and not payload.tools and not self.query.tags:
            specific = await self.manager.resolve_specific_tools([self.query.name])
            for t in specific:
                if self.query.match(t):
                    payload.tools.append(t)

        return payload


class _CallableResolver:
    def __init__(self, func: Callable, manager: "ToolProviderManager"):
        self.func = func
        self.manager = manager

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        for candidate in (
            getattr(self.func, "__tool_name__", None),
            getattr(self.func, "__name__", None),
        ):
            if candidate:
                for ns_tools in self.manager._namespaced_tools.values():
                    if t := ns_tools.get(candidate):
                        if hasattr(t, "resolve"):
                            return await t.resolve(context)
                        return ResolvedToolPayload(tools=[t])
        from zhenxun.services.ai.tools.core.tool import FunctionTool

        t = FunctionTool(func=self.func)
        return await t.resolve(context)


class _TypeAdapterResolver:
    def __init__(
        self,
        item: Any,
        resolver_func: Callable,
        manager: "ToolProviderManager",
        default_namespace: str,
    ):
        self.item = item
        self.resolver_func = resolver_func
        self.manager = manager
        self.default_namespace = default_namespace

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        resolved = (
            await self.resolver_func(self.item)
            if is_coroutine_callable(self.resolver_func)
            else self.resolver_func(self.item)
        )
        return await self.manager._normalize_to_resolver(
            resolved, self.default_namespace
        ).resolve(context)


class _LegacyExecutableResolver:
    def __init__(self, executable: Any):
        self.executable = executable

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        import copy

        if hasattr(self.executable, "get_definition"):
            run_scoped_tool = copy.copy(self.executable)
            if not hasattr(run_scoped_tool, "name"):
                setattr(run_scoped_tool, "name", f"instance_{id(run_scoped_tool)}")
            return ResolvedToolPayload(tools=[run_scoped_tool])
        return ResolvedToolPayload(tools=[self.executable])


class ToolProviderManager:
    """工具提供者的中心化管理器，采用单例模式。"""

    _instance: "ToolProviderManager | None" = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Self, cls._instance)

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._providers: list[ToolProvider] = []
        self._namespaced_tools: dict[str, ToolCollection] = {}
        self._resolved_tools: ToolCollection | None = None
        self._init_lock = asyncio.Lock()
        self._init_promise: asyncio.Task | None = None
        self._initialized = True

        self._macro_resolvers: dict[str, Callable] = {}
        self._type_resolvers: dict[type, Callable] = {}

    def register_macro_resolver(self, macro_str: str, resolver_func: Callable) -> None:
        self._macro_resolvers[macro_str] = resolver_func

    def register_type_resolver(
        self, target_type: type, resolver_func: Callable
    ) -> None:
        self._type_resolvers[target_type] = resolver_func

    def register(self, provider: ToolProvider):
        """注册一个新的 ToolProvider。"""
        if provider not in self._providers:
            self._providers.append(provider)
            logger.info(f"已注册工具提供者: {provider.__class__.__name__}")

    def register_tool(self, tool: ToolExecutable):
        """注册由 @tool 生成的单一工具"""
        if not hasattr(tool, "name"):
            setattr(tool, "name", str(id(tool)))
        from zhenxun.utils.utils import infer_plugin_namespace

        ns = infer_plugin_namespace()
        if ns not in self._namespaced_tools:
            self._namespaced_tools[ns] = ToolCollection()
        self._namespaced_tools[ns].append(tool)
        self._resolved_tools = None

    def register_toolkit(self, toolkit: Any) -> None:
        """
        注册一个完整的 Toolkit 实例，使其可通过智能字符串路由（Tag或Name）被动态发现。
        """
        from zhenxun.utils.utils import infer_plugin_namespace

        ns = infer_plugin_namespace()
        if ns not in self._namespaced_tools:
            self._namespaced_tools[ns] = ToolCollection()
        self._namespaced_tools[ns].append(toolkit)
        self._resolved_tools = None

    async def initialize(self) -> None:
        """懒加载初始化所有已注册的 ToolProvider。"""
        if not self._init_promise:
            async with self._init_lock:
                if not self._init_promise:
                    self._init_promise = asyncio.create_task(
                        self._initialize_providers()
                    )
        await self._init_promise

    async def _initialize_providers(self) -> None:
        """内部初始化逻辑。"""
        logger.info(f"开始初始化 {len(self._providers)} 个工具提供者...")
        init_tasks = [provider.initialize() for provider in self._providers]
        await asyncio.gather(*init_tasks, return_exceptions=True)
        logger.info("所有工具提供者初始化完成。")

    async def get_resolved_tools(
        self,
        allowed_servers: list[str] | None = None,
        excluded_servers: list[str] | None = None,
        namespaces: list[str] | None = None,
    ) -> ToolCollection:
        """
        获取所有已发现和解析的工具。
        此方法会触发懒加载初始化，并根据是否传入过滤器来决定是否使用全局缓存。
        """
        await self.initialize()

        has_filters = (
            allowed_servers is not None
            or excluded_servers is not None
            or namespaces is not None
        )

        if not has_filters and self._resolved_tools is not None:
            logger.debug("使用全局工具缓存。")
            return self._resolved_tools

        if has_filters:
            logger.info("检测到过滤器，执行临时工具发现 (不使用缓存)。")
            logger.debug(
                f"过滤器详情: allowed_servers={allowed_servers}, "
                f"excluded_servers={excluded_servers}, "
                f"namespaces={namespaces}"
            )
        else:
            logger.info("未应用过滤器，开始全局工具发现...")

        all_tools = ToolCollection()

        discover_tasks = []
        provider_indices = []
        for i, provider in enumerate(self._providers):
            sig = inspect.signature(provider.discover_tools)
            params_to_pass = {}
            if "allowed_servers" in sig.parameters:
                params_to_pass["allowed_servers"] = allowed_servers
            if "excluded_servers" in sig.parameters:
                params_to_pass["excluded_servers"] = excluded_servers

            discover_tasks.append(provider.discover_tools(**params_to_pass))
            provider_indices.append(i)

        results = await asyncio.gather(*discover_tasks, return_exceptions=True)

        for result_idx, provider_result in enumerate(results):
            provider = self._providers[provider_indices[result_idx]]
            provider_name = provider.__class__.__name__

            if isinstance(provider_result, dict):
                logger.debug(
                    f"提供者 '{provider_name}' 发现了 {len(provider_result)} 个工具。"
                )
                for name, executable in provider_result.items():
                    if not hasattr(executable, "name"):
                        setattr(executable, "name", name)
                    if all_tools.get(name):
                        logger.warning(
                            f"发现重复的工具名称 '{name}'，后发现的将覆盖前者。"
                        )
                    all_tools.append(executable)
            elif isinstance(provider_result, Exception):
                logger.error(
                    f"提供者 '{provider_name}' 在发现工具时出错: {provider_result}"
                )

        for ns_tools in self._namespaced_tools.values():
            for t in ns_tools:
                all_tools.append(t)

        if not has_filters:
            self._resolved_tools = all_tools
            logger.info(f"全局工具发现完成，共找到并缓存了 {len(all_tools)} 个工具。")
        else:
            logger.info(f"带过滤器的工具发现完成，共找到 {len(all_tools)} 个工具。")

        return all_tools

    async def resolve_specific_tools(self, tool_names: list[str]) -> ToolCollection:
        """
        仅解析指定名称的工具，避免触发全量工具发现。
        优先从全局游离工具中查找，再回退到 Provider。
        """
        resolved = ToolCollection()
        if not tool_names:
            return resolved

        await self.initialize()

        for name in tool_names:
            found_in_global = False
            for ns_tools in self._namespaced_tools.values():
                if t := ns_tools.get(name):
                    resolved.append(t)
                    found_in_global = True
                    break
            if found_in_global:
                continue

            config: dict[str, Any] = {"name": name}
            for provider in self._providers:
                try:
                    executable = await provider.get_tool_executable(name, config)
                except Exception as exc:
                    logger.error(
                        f"provider '{provider.__class__.__name__}' 在解析工具 '{name}'"
                        f"时出错: {exc}",
                        e=exc,
                    )
                    continue

                if executable:
                    if not hasattr(executable, "name"):
                        setattr(executable, "name", name)
                    resolved.append(executable)
                    break
            else:
                if not resolved.get(name):
                    logger.warning(f"没有找到名为 '{name}' 的工具，已跳过。")

        return resolved

    async def get_function_tools(
        self, names: list[str] | None = None
    ) -> ToolCollection:
        """
        仅从直接注册的游离工具中解析指定的工具。
        """
        all_tools_list = []
        for ns_tools in self._namespaced_tools.values():
            all_tools_list.extend(list(ns_tools))
        all_function_tools = ToolCollection(all_tools_list)
        if names is None:
            return all_function_tools

        resolved_tools = ToolCollection()
        for name in names:
            if t := all_function_tools.get(name):
                resolved_tools.append(t)
            else:
                logger.warning(f"全局工具 '{name}' 未通过 @tool 注册，将被忽略。")
        return resolved_tools

    def _normalize_to_resolver(self, item: Any, default_ns: str) -> Any:
        """将任意类型包装为含有 resolve() 方法的解析器对象"""
        if hasattr(item, "resolve"):
            return item
        if isinstance(item, Query):
            return _QueryResolver(item, self)
        if isinstance(item, str):
            return _StringResolver(item, self, default_ns)
        if type(item) in self._type_resolvers:
            return _TypeAdapterResolver(
                item, self._type_resolvers[type(item)], self, default_ns
            )
        if callable(item):
            return _CallableResolver(item, self)
        return _LegacyExecutableResolver(item)

    async def resolve_tools(
        self,
        tool_definitions: Iterable[Any] | None,
        namespace: str | None = None,
        context: RunContext | None = None,
    ) -> ResolvedToolPayload:
        """
        统一解析工具配置，全面采用多态解析器与并发聚合管线。
        """
        if not tool_definitions:
            return ResolvedToolPayload()

        if not namespace:
            from zhenxun.utils.utils import infer_plugin_namespace

            namespace = infer_plugin_namespace()
            logger.debug(
                f"🔍 [StringRouter] 自动推断当前调用者所在插件为: '{namespace}'"
            )

        defs = []

        def _flatten(items):
            for item in items:
                if isinstance(item, list):
                    _flatten(item)
                else:
                    defs.append(item)

        _flatten(tool_definitions)

        resolvers = [self._normalize_to_resolver(t, namespace) for t in defs]

        for i, r in enumerate(resolvers):
            if asyncio.iscoroutine(r):
                r = await r
                resolvers[i] = self._normalize_to_resolver(r, namespace)

        tasks = [r.resolve(context) for r in resolvers]
        payloads = await asyncio.gather(*tasks, return_exceptions=True)

        final_payload = ResolvedToolPayload()
        global_toolkit = BaseToolkit(prefix="")

        for p in payloads:
            if isinstance(p, BaseException):
                logger.error(f"工具解析器流水线内部错误: {p}")
                continue
            if not p:
                continue
            p = cast(ResolvedToolPayload, p)

            for t in p.tools:
                if not getattr(t, "parent_toolkit", None):
                    t.parent_toolkit = global_toolkit
                final_payload.tools.append(t)

            final_payload.injected_prompts.extend(p.injected_prompts)
            final_payload.toolkits.extend(p.toolkits)

        final_payload.tools = ToolCollection(final_payload.tools)
        return final_payload


tool_provider_manager = ToolProviderManager()


from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException


async def _dict_ad_hoc_resolver(config: dict):
    name = config.get("name")
    if not name:
        raise LLMException(
            "工具配置字典必须包含 'name' 字段。",
            code=LLMErrorCode.CONFIGURATION_ERROR,
        )

    for provider in tool_provider_manager._providers:
        executable = await provider.get_tool_executable(name, config)
        if executable:
            return executable

    raise LLMException(
        f"没有为 ad-hoc 工具 '{name}' 找到合适的提供者。",
        code=LLMErrorCode.CONFIGURATION_ERROR,
    )


tool_provider_manager.register_type_resolver(dict, _dict_ad_hoc_resolver)
