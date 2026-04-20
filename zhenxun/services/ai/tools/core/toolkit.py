from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import json
from typing import Any, Generic, TypeVar

import aiofiles
from pydantic import BaseModel

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.types.exceptions import NeedsAuthException
from zhenxun.services.ai.types.tools import ToolOptions, ToolResult
from zhenxun.services.ai.utils.lifespan import ResourceLifespanMixin
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import dump_json_safely, parse_as

from .context import RunContext
from .tool import BaseTool, FunctionTool


class BaseToolkit:
    """
    状态工具箱基类。继承此类的对象可以维持自身状态，
    其内部被 @toolkit_tool 标记的方法将被自动解析为工具。

    参数:
        prefix: 工具名称前缀，默认为空。如果不为空，将自动添加到其所有工具名称的前面。
        include: 如果提供，仅注册列表中的方法作为工具。
        exclude: 如果提供，将排除列表中的方法。
        tools: 除了带有 @toolkit_tool 装饰器的方法外，要注入的其他独立工具。
        prepare: 该 Toolkit 中所有工具共享的前置准备钩子。
        global_cache: 是否对 Toolkit 中所有工具开启全局缓存。
        global_cache_ttl: 全局缓存的过期时间（秒）。
        global_require_approval: 是否对所有工具开启全局人工审批。
        global_middlewares: 所有工具共享的中间件列表。
        require_approval_patterns: 匹配该通配符模式的工具将自动开启人工审批。
        cache_patterns: 匹配该通配符模式的工具将自动开启缓存。
        auto_register: 是否将未标记 @toolkit_tool 但拥有独立 Docstring 的方法自动注册为工具。
    """

    default_instructions: str = ""
    """类级别的默认工具箱使用说明，大模型会读取此说明。"""

    prefix: str
    include: list[str] | None
    exclude: list[str] | None

    def __init__(
        self,
        prefix: str = "",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        tools: list[Callable | BaseTool] | None = None,
        prepare: Callable | None = None,
        global_cache: bool | None = None,
        global_cache_ttl: int | None = None,
        global_require_approval: bool | None = None,
        global_middlewares: list[Any] | None = None,
        require_approval_patterns: list[str] | None = None,
        cache_patterns: list[str] | None = None,
        auto_register: bool = False,
        **kwargs: Any,
    ):

        from zhenxun.utils.utils import infer_plugin_namespace

        self._inferred_namespace = infer_plugin_namespace()

        if not prefix and self._inferred_namespace != "unknown":
            self.prefix = f"{self._inferred_namespace}_"
        else:
            self.prefix = prefix

        self.auto_register = auto_register
        self.include = include
        self.exclude = exclude
        self.prepare = prepare
        self.global_cache = global_cache
        self.global_cache_ttl = global_cache_ttl
        self.global_require_approval = global_require_approval
        self.global_middlewares = global_middlewares or []
        self.require_approval_patterns = require_approval_patterns or []
        self.cache_patterns = cache_patterns or []
        self._injected_tools: list[BaseTool] = []

        if tools:
            for t in tools:
                if isinstance(t, BaseTool):
                    if self.prepare:
                        old_prepare = t.settings.prepare
                        if old_prepare:

                            def make_chained(p1, p2):
                                async def chained(ctx, tdef):
                                    from nonebot.utils import is_coroutine_callable

                                    res1 = (
                                        await p1(ctx, tdef)
                                        if is_coroutine_callable(p1)
                                        else p1(ctx, tdef)
                                    )
                                    if res1 is None:
                                        return None
                                    res2 = (
                                        await p2(ctx, res1)
                                        if is_coroutine_callable(p2)
                                        else p2(ctx, res1)
                                    )
                                    return res2

                                return chained

                            t.settings.prepare = make_chained(self.prepare, old_prepare)
                        else:
                            t.settings.prepare = self.prepare
                    self._injected_tools.append(t)
                elif callable(t):
                    tool_name = getattr(
                        t, "__tool_name__", getattr(t, "__name__", "unnamed_tool")
                    )
                    tool_desc = getattr(t, "__tool_desc__", "未提供描述")
                    settings = getattr(
                        t, "__tool_settings__", ToolOptions()
                    ).model_copy()
                    if self.prepare:
                        old_prepare = settings.prepare
                        if old_prepare:

                            def make_chained(p1, p2):
                                async def chained(ctx, tdef):
                                    from nonebot.utils import is_coroutine_callable

                                    res1 = (
                                        await p1(ctx, tdef)
                                        if is_coroutine_callable(p1)
                                        else p1(ctx, tdef)
                                    )
                                    if res1 is None:
                                        return None
                                    res2 = (
                                        await p2(ctx, res1)
                                        if is_coroutine_callable(p2)
                                        else p2(ctx, res1)
                                    )
                                    return res2

                                return chained

                            settings.prepare = make_chained(self.prepare, old_prepare)
                        else:
                            settings.prepare = self.prepare

                    func_tool = FunctionTool(
                        func=t,
                        name=f"{self.prefix}{tool_name}",
                        description=tool_desc,
                        settings=settings,
                    )
                    self._injected_tools.append(func_tool)

    async def enter_session(self, session_id: str, context: Any) -> None:
        """会话隔离级别的生命周期入口，供 Agent 在具体会话开始时预热专属资源"""
        pass

    async def before_llm_request(
        self, context: RunContext, messages: list[Any]
    ) -> None:
        """LLM 请求发起前的拦截钩子，允许动态修改消息列表（如注入实时状态）"""
        pass

    async def exit_session(self, session_id: str) -> None:
        """会话隔离级别的生命周期出口，供 Agent 在具体会话结束后释放专属资源"""
        pass

    async def __resolve_to_tools__(self) -> list[ToolExecutable]:
        """协议支持：将当前工具箱解析为工具列表"""
        return list(await self.get_tools())

    async def get_tools(self) -> Sequence[BaseTool]:
        tools: list[BaseTool] = list(self._injected_tools)
        import fnmatch
        import inspect

        base_methods = set(dir(BaseToolkit))

        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            is_explicit = callable(attr) and getattr(attr, "__toolkit_tool__", False)

            is_auto = False
            if (
                not is_explicit
                and getattr(self, "auto_register", False)
                and callable(attr)
            ):
                if not attr_name.startswith("_") and attr_name not in base_methods:
                    if getattr(attr, "__doc__", None):
                        is_auto = True

            if not (is_explicit or is_auto):
                continue

            if is_explicit:
                original_tool_name = getattr(attr, "__tool_name__")
                original_desc = getattr(attr, "__tool_desc__")
                settings = getattr(
                    attr, "__tool_settings__", ToolOptions()
                ).model_copy()
            else:
                original_tool_name = attr_name
                original_desc = inspect.cleandoc(getattr(attr, "__doc__"))
                settings = ToolOptions()

            if self.include is not None and original_tool_name not in self.include:
                continue
            if self.exclude is not None and original_tool_name in self.exclude:
                continue

            final_tool_name = f"{self.prefix}{original_tool_name}"

            final_desc = original_desc
            if (
                getattr(self, "_inferred_namespace", None)
                and self._inferred_namespace != "unknown"
            ):
                ns_prefix = f"[所属插件: {self._inferred_namespace}] "
                if not final_desc.startswith(ns_prefix):
                    final_desc = f"{ns_prefix}{final_desc}"

            if self.global_cache is not None and not settings.cache:
                settings.cache = self.global_cache

            for pat in self.cache_patterns:
                if fnmatch.fnmatch(original_tool_name, pat):
                    settings.cache = True
                    break

            if self.global_cache_ttl is not None and settings.cache_ttl == 3600:
                settings.cache_ttl = self.global_cache_ttl

            if (
                self.global_require_approval is not None
                and not settings.require_approval
            ):
                settings.require_approval = self.global_require_approval

            for pat in self.require_approval_patterns:
                if fnmatch.fnmatch(original_tool_name, pat):
                    settings.require_approval = True
                    break

            if self.global_middlewares:
                settings.middlewares = self.global_middlewares + settings.middlewares

            if self.prepare:
                old_prepare = settings.prepare
                if old_prepare:

                    def make_chained(p1, p2):
                        async def chained(ctx, tdef):
                            from nonebot.utils import is_coroutine_callable

                            res1 = (
                                await p1(ctx, tdef)
                                if is_coroutine_callable(p1)
                                else p1(ctx, tdef)
                            )
                            if res1 is None:
                                return None
                            res2 = (
                                await p2(ctx, res1)
                                if is_coroutine_callable(p2)
                                else p2(ctx, res1)
                            )
                            return res2

                        return chained

                    settings.prepare = make_chained(self.prepare, old_prepare)
                else:
                    settings.prepare = self.prepare

            from zhenxun.services.ai.tools.core.tool import LazyToolProxy

            def make_tool_factory(
                attr_func=attr,
                t_name=final_tool_name,
                t_desc=final_desc,
                t_settings=settings,
            ):
                def factory():
                    from zhenxun.services.ai.tools.core.tool import FunctionTool

                    return FunctionTool(
                        func=attr_func,
                        name=t_name,
                        description=t_desc,
                        settings=t_settings,
                    )

                return factory

            from typing import cast

            proxy = LazyToolProxy(
                name=final_tool_name,
                description=final_desc,
                factory=make_tool_factory(),
            )
            tools.append(cast(BaseTool, proxy))
        return tools

    def get_instructions(self) -> str | None:
        """
        获取工具箱的系统提示词说明。
        """
        text = getattr(self, "default_instructions", "")
        if not text:
            return None
        class_name = self.__class__.__name__
        tag_name = (
            f"{self.prefix}{class_name}_Instructions"
            if self.prefix
            else f"{class_name}_Instructions"
        )
        return f"<{tag_name}>\n{text}\n</{tag_name}>"

    def prefixed(self, prefix: str) -> "PrefixedToolkit":
        return PrefixedToolkit(self, prefix)

    def filtered(self, filter_func: Callable[[BaseTool], bool]) -> "FilteredToolkit":
        return FilteredToolkit(self, filter_func)

    def prepared(self, prepare_func: Callable) -> "PreparedToolkit":
        return PreparedToolkit(self, prepare_func)


class WrapperToolkit(BaseToolkit):
    """高阶工具箱包装器基类 (装饰器模式)"""

    def __init__(self, wrapped: BaseToolkit):
        super().__init__()
        self.wrapped = wrapped
        self._injected_tools = []

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        await self.wrapped.enter_session(session_id, context)

    async def before_llm_request(
        self, context: RunContext, messages: list[Any]
    ) -> None:
        await self.wrapped.before_llm_request(context, messages)

    async def exit_session(self, session_id: str) -> None:
        await self.wrapped.exit_session(session_id)

    async def get_tools(self) -> Sequence[BaseTool]:
        return await self.wrapped.get_tools()


class PrefixedToolkit(WrapperToolkit):
    """为所有内部工具添加名称前缀的包装器"""

    def __init__(self, wrapped: BaseToolkit, prefix: str):
        super().__init__(wrapped)
        self.prefix_str = prefix

    async def get_tools(self) -> Sequence[BaseTool]:
        from zhenxun.services.ai.types.tools import ToolOverride

        tools = await self.wrapped.get_tools()
        new_tools = []
        for t in tools:
            new_tools.append(
                t.clone_with_options(
                    ToolOverride(name=t.name, new_name=f"{self.prefix_str}{t.name}")
                )
            )
        return new_tools


class FilteredToolkit(WrapperToolkit):
    """过滤内部工具的包装器"""

    def __init__(self, wrapped: BaseToolkit, filter_func: Callable[[BaseTool], bool]):
        super().__init__(wrapped)
        self.filter_func = filter_func

    async def get_tools(self) -> Sequence[BaseTool]:
        tools = await self.wrapped.get_tools()
        return [t for t in tools if self.filter_func(t)]


class PreparedToolkit(WrapperToolkit):
    """为所有内部工具批量附加前置拦截钩子的包装器"""

    def __init__(self, wrapped: BaseToolkit, prepare_func: Callable):
        super().__init__(wrapped)
        self.prepare_func = prepare_func

    async def get_tools(self) -> Sequence[BaseTool]:
        from zhenxun.services.ai.types.tools import ToolOverride

        tools = await self.wrapped.get_tools()
        new_tools = []
        for t in tools:
            cloned = t.clone_with_options(
                ToolOverride(name=t.name, prepare=self.prepare_func)
            )
            new_tools.append(cloned)
        return new_tools


class ApiConnectToolkit(BaseToolkit, ResourceLifespanMixin, ABC):
    """
    托管连接池的 Toolkit 基类。
    自动处理第三方凭证的获取、客户端的懒加载初始化，并在会话结束时安全释放连接池资源。

    参数:
        ttl: 资源存活时长（秒）。会话结束或超出此时间无调用后，连接将被自动释放。默认 600 秒。
        kwargs: 传递给 BaseToolkit 的其他参数。
    """

    required_auth_providers: tuple[str, ...] = ()

    def __init__(self, ttl: int = 600, **kwargs: Any):
        super().__init__(**kwargs)
        self.init_lifespan(ttl=ttl)
        self._clients: dict[str, Any] = {}

    async def get_client(self, context: RunContext) -> Any:
        """获取当前会话的客户端。如果未初始化，则检查凭证并自动初始化。"""
        session_id = context.session_id or "default_session"
        self.touch(session_id)
        self._ensure_watchdog()

        if session_id in self._clients:
            return self._clients[session_id]

        tokens = context.extra.get("auth_tokens", {})
        for provider in self.required_auth_providers:
            if provider not in tokens:
                raise NeedsAuthException(
                    provider, f"需要 {provider} 的授权凭证以初始化客户端"
                )

        client = await self.create_client(tokens)
        self._clients[session_id] = client
        logger.debug(f"[ApiConnectToolkit] 客户端已成功初始化 (Session: {session_id})")
        return client

    @abstractmethod
    async def create_client(self, auth_tokens: dict[str, str]) -> Any:
        """根据提供的凭证实例化外部 SDK 客户端"""
        pass

    @abstractmethod
    async def close_client(self, client: Any) -> None:
        """安全关闭外部 SDK 客户端，释放网络连接等资源"""
        pass

    async def release_resource(self, resource_id: str):
        if resource_id in self._clients:
            client = self._clients.pop(resource_id)
            try:
                await self.close_client(client)
                logger.debug(
                    f"[ApiConnectToolkit] 客户端已安全销毁 (Session: {resource_id})"
                )
            except Exception as e:
                logger.error(
                    f"[ApiConnectToolkit] 销毁客户端失败 (Session: {resource_id}): {e}"
                )

    async def exit_session(self, session_id: str) -> None:
        """会话结束时框架调用。如果 TTL <= 0 则立刻回收，否则交由看门狗回收"""
        await super().exit_session(session_id)
        if self.ttl <= 0:
            async with self._lifespan_lock:
                await self.release_resource(session_id)
                self._last_active_times.pop(session_id, None)


StateT = TypeVar("StateT", bound=BaseModel)


class GroupSharedToolkit(BaseToolkit, Generic[StateT]):
    """
    群组共享工具箱基类 (Group Shared Toolkit)。

    提供群聊级别的物理状态隔离。同群内触发的调用将共享同一份状态，私聊则退化为用户隔离。
    开发者可通过重写 `load_state` 和 `save_state` 实现自定义的持久化逻辑。

    参数:
        default_state_factory: 状态模型初始化工厂函数，用于创建新的状态实例。
        persist_to_disk: 是否默认将状态持久化落盘至本地文件系统。
        prefix: 注册工具时的名称前缀。
        include: 允许注册的工具名称白名单。
        exclude: 排除注册的工具名称黑名单。
        kwargs: 传递给 BaseToolkit 的其他参数。
    """

    def __init__(
        self,
        default_state_factory: Callable[[], StateT],
        persist_to_disk: bool = False,
        prefix: str = "",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(prefix=prefix, include=include, exclude=exclude, **kwargs)
        self.default_state_factory = default_state_factory
        self.persist_to_disk = persist_to_disk
        self._memory_states: dict[str, StateT] = {}
        self._active_states: dict[str, StateT] = {}
        self._session_to_state_key: dict[str, str] = {}

        if self.persist_to_disk:
            self._disk_dir = DATA_PATH / "ai" / "agent_states" / self.__class__.__name__
            self._disk_dir.mkdir(parents=True, exist_ok=True)

    async def load_state(self, state_key: str) -> StateT:
        """从外部存储加载状态。完全交由第三方开发者决定存储介质。"""
        if self.persist_to_disk:
            file_path = self._disk_dir / f"{state_key}.json"
            if file_path.exists():
                try:
                    async with aiofiles.open(file_path, encoding="utf-8") as f:
                        data = await f.read()
                        state_type = type(self.default_state_factory())
                        return parse_as(state_type, json.loads(data))
                except Exception as e:
                    logger.warning(f"从磁盘恢复状态失败 {file_path}: {e}")

        if state_key in self._memory_states:
            return self._memory_states[state_key].model_copy(deep=True)
        return self.default_state_factory()

    async def save_state(self, state_key: str, state: StateT) -> None:
        """将状态保存到外部存储。完全交由第三方开发者决定存储介质。"""
        self._memory_states[state_key] = state.model_copy(deep=True)

        if self.persist_to_disk:
            file_path = self._disk_dir / f"{state_key}.json"
            try:
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(dump_json_safely(state, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"保存状态到磁盘失败 {file_path}: {e}")

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        """框架级 Hook：提取群组上下文，隔离装载群聊共享状态"""
        await super().enter_session(session_id, context)

        group_id = context.get_group_id()
        if group_id:
            state_key = f"group_{group_id}"
        else:
            user_id = context.get_user_id()
            state_key = f"user_{user_id}" if user_id else session_id

        self._session_to_state_key[session_id] = state_key
        state = await self.load_state(state_key)
        self._active_states[session_id] = state
        context.extra[f"toolkit_state_{self.__class__.__name__}"] = state
        context.extra[f"di_type_{type(state).__name__}"] = state

    async def exit_session(self, session_id: str) -> None:
        """框架级 Hook：每次 Agent 运行结束时，抽取并保存最新状态"""
        await super().exit_session(session_id)
        if session_id in self._active_states:
            state = self._active_states.pop(session_id)
            state_key = self._session_to_state_key.pop(session_id, session_id)
            await self.save_state(state_key, state)

    def get_state(self, context: RunContext) -> StateT:
        """快捷方法：在 @toolkit_tool 中调用以获取强类型的状态实例"""
        key = f"toolkit_state_{self.__class__.__name__}"
        if key not in context.extra:
            raise RuntimeError(
                f"State 未初始化。请确认 {self.__class__.__name__} "
                "被正确注册到 Agent 的 tools 中。"
            )
        return context.extra[key]

    def state_aware_result(
        self, output: Any, prompt_injection: str, display: Any = None
    ) -> ToolResult:
        """
        阶段二能力：生成具有大模型状态感知的执行结果。
        向 LLM 追加最新的状态系统提示词，防止其产生信息幻觉。
        """
        return ToolResult(
            output=output,
            display=display,
            system_prompt_append=f"[系统通知(状态同步)]：{prompt_injection}",
        )


class UserPersonalToolkit(GroupSharedToolkit[StateT]):
    """
    用户私有工具箱基类 (User Personal Toolkit)。

    与 GroupSharedToolkit 的区别在于：本类强制基于 user_id 进行物理状态隔离，
    哪怕在群内调用，A 用户的状态也绝对不会泄漏给 B 用户。适用于私人的待办清单、个人积分等。
    """

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        await BaseToolkit.enter_session(self, session_id, context)

        user_id = context.get_user_id()
        state_key = f"user_{user_id}" if user_id else session_id

        self._session_to_state_key[session_id] = state_key
        state = await self.load_state(state_key)
        self._active_states[session_id] = state
        context.extra[f"toolkit_state_{self.__class__.__name__}"] = state
        context.extra[f"di_type_{type(state).__name__}"] = state
