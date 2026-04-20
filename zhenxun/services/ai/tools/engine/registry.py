import asyncio
from collections.abc import Callable, Iterable
import inspect
from typing import Any, Generic, SupportsIndex, TypeVar, cast, overload
from typing_extensions import Self

from nonebot.utils import is_coroutine_callable

from zhenxun.services.ai.protocols.tool import (
    ToolExecutable,
    ToolProvider,
    ToolResolvable,
)
from zhenxun.services.ai.types.tools import MCPSource, ResolvedToolPayload
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
        self._global_tools: ToolCollection = ToolCollection()
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
        self._global_tools.append(tool)
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
    ) -> Any:
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

        for t in self._global_tools:
            all_tools.append(t)

        if not has_filters:
            self._resolved_tools = all_tools
            logger.info(f"全局工具发现完成，共找到并缓存了 {len(all_tools)} 个工具。")
        else:
            logger.info(f"带过滤器的工具发现完成，共找到 {len(all_tools)} 个工具。")

        return all_tools

    async def resolve_specific_tools(self, tool_names: list[str]) -> Any:
        """
        仅解析指定名称的工具，避免触发全量工具发现。
        优先从全局游离工具中查找，再回退到 Provider。
        """
        resolved = ToolCollection()
        if not tool_names:
            return resolved

        await self.initialize()

        for name in tool_names:
            if t := self._global_tools.get(name):
                resolved.append(t)
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

    async def get_function_tools(self, names: list[str] | None = None) -> Any:
        """
        仅从直接注册的游离工具中解析指定的工具。
        """
        all_function_tools = ToolCollection(list(self._global_tools))
        if names is None:
            return all_function_tools

        resolved_tools = ToolCollection()
        for name in names:
            if t := all_function_tools.get(name):
                resolved_tools.append(t)
            else:
                logger.warning(f"全局工具 '{name}' 未通过 @tool 注册，将被忽略。")
        return resolved_tools

    async def resolve_tools(
        self,
        tool_definitions: Iterable[Any] | None,
        namespace: str | None = None,
        context: Any | None = None,
    ) -> ResolvedToolPayload:
        """
        统一解析工具配置，全面采用 ToolResolvable + 注册制解析器。
        """
        _ = namespace
        resolved_tools_map = ToolCollection()
        injected_prompts: list[str] = []
        resolved_toolkits: list[Any] = []

        if tool_definitions is None:
            return ResolvedToolPayload(
                tools=resolved_tools_map,
                injected_prompts=injected_prompts,
                toolkits=resolved_toolkits,
            )

        if not tool_definitions:
            return ResolvedToolPayload(
                tools=resolved_tools_map,
                injected_prompts=injected_prompts,
                toolkits=resolved_toolkits,
            )

        from zhenxun.services.ai.types.tools import ToolOverride

        async def _process_resolved(res_obj: Any) -> None:
            """通用提取并入库函数。"""
            if res_obj is None:
                return

            if isinstance(res_obj, list):
                for item in res_obj:
                    await _process_resolved(item)
                return
            if hasattr(res_obj, "get_tool_declaration"):
                decl = res_obj.get_tool_declaration()
                if decl:
                    name = next(iter(decl.keys()), f"platform_tool_{id(res_obj)}")
                    try:
                        if not hasattr(res_obj, "name"):
                            setattr(res_obj, "name", name)
                    except ValueError:
                        pass
                    resolved_tools_map.append(res_obj)
                return
            if hasattr(res_obj, "get_definition"):
                import copy
                try:
                    run_scoped_tool = copy.copy(res_obj)
                    definition = await run_scoped_tool.get_definition(context)
                    if definition is None:
                        logger.debug(f"工具 {getattr(run_scoped_tool, 'name', 'unknown')} 被 prepare 钩子隐藏")
                        return
                    run_scoped_tool._dynamic_def = definition
                    if not hasattr(run_scoped_tool, "name"):
                        setattr(run_scoped_tool, "name", definition.name)
                    resolved_tools_map.append(run_scoped_tool)
                except Exception:
                    run_scoped_tool = copy.copy(res_obj)
                    if not hasattr(run_scoped_tool, "name"):
                        setattr(run_scoped_tool, "name", f"instance_{id(run_scoped_tool)}")
                    resolved_tools_map.append(run_scoped_tool)
                return
            if callable(res_obj):
                for candidate in (
                    getattr(res_obj, "__tool_name__", None),
                    getattr(res_obj, "__name__", None),
                ):
                    if candidate and (t := self._global_tools.get(candidate)):
                        await _process_resolved(t)
                        return

        local_tool_names = []

        for t in tool_definitions:
            if isinstance(t, ToolOverride):
                found_tools = await self.resolve_specific_tools([t.name])
                if found_tools:
                    base_tool = found_tools[0]
                    if hasattr(base_tool, "clone_with_options"):
                        cloned_tool = base_tool.clone_with_options(t)
                        await _process_resolved(cloned_tool)
                    else:
                        logger.warning(f"工具 {t.name} 不支持动态覆盖，将原样装配。")
                        await _process_resolved(base_tool)
                else:
                    logger.warning(f"ToolOverride 找不到目标基础工具: {t.name}")
            elif isinstance(t, MCPSource):
                from zhenxun.services.ai.tools.providers.mcp.provider import mcp_provider

                server_tools = await mcp_provider.get_tools_for_server(t.server_name)
                if t.tool_whitelist:
                    server_tools = {
                        k: v
                        for k, v in server_tools.items()
                        if any(k.endswith(allowed) for allowed in t.tool_whitelist)
                    }
                await _process_resolved(list(server_tools.values()))
            elif isinstance(t, ToolResolvable) or hasattr(t, "__resolve_to_tools__"):
                tools_list = await t.__resolve_to_tools__()
                await _process_resolved(tools_list)

                if hasattr(t, "enter_session") and hasattr(t, "exit_session"):
                    resolved_toolkits.append(t)
                if hasattr(t, "get_instructions") and (
                    instructions := getattr(t, "get_instructions")()
                ):
                    injected_prompts.append(instructions)
            elif isinstance(t, str):
                if t == "__all__":
                    pass
                elif t in self._macro_resolvers:
                    resolver = self._macro_resolvers[t]
                    resolved = (
                        await resolver()
                        if is_coroutine_callable(resolver)
                        else resolver()
                    )
                    await _process_resolved(resolved)
                else:
                    local_tool_names.append(t)
            elif type(t) in self._type_resolvers:
                resolver = self._type_resolvers[type(t)]
                resolved = (
                    await resolver(t)
                    if is_coroutine_callable(resolver)
                    else resolver(t)
                )
                await _process_resolved(resolved)
            else:
                await _process_resolved(t)

        if "__all__" in (tool_definitions or []):
            await _process_resolved(list(self._global_tools))
            for macro_name, resolver in self._macro_resolvers.items():
                resolved = (
                    await resolver() if is_coroutine_callable(resolver) else resolver()
                )
                await _process_resolved(resolved)

        if local_tool_names:
            local_tools = await self.resolve_specific_tools(local_tool_names)
            await _process_resolved(list(local_tools))

        return ResolvedToolPayload(
            tools=resolved_tools_map,
            injected_prompts=injected_prompts,
            toolkits=resolved_toolkits,
        )


tool_provider_manager = ToolProviderManager()


from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.tools import (
    GeminiCodeExecution,
    GeminiGoogleMaps,
    GeminiGoogleSearch,
    GeminiUrlContext,
)

tool_provider_manager.register_macro_resolver(
    "google_search", lambda: GeminiGoogleSearch()
)
tool_provider_manager.register_macro_resolver(
    "code_execution", lambda: GeminiCodeExecution()
)
tool_provider_manager.register_macro_resolver("google_map", lambda: GeminiGoogleMaps())
tool_provider_manager.register_macro_resolver("url_context", lambda: GeminiUrlContext())


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
