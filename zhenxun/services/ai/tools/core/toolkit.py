from abc import ABC, abstractmethod
from collections.abc import Callable
import json
from typing import Any, ClassVar, Generic, TypeVar, cast

import aiofiles
from pydantic import BaseModel

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.core.exceptions import NeedsAuthException
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.models import (
    ResolvedToolPayload,
    ToolkitConfig,
    ToolOptions,
)
from zhenxun.services.log import logger
from zhenxun.utils.lifespan import LifespanManager
from zhenxun.utils.pydantic_compat import dump_json_safely, model_copy, parse_as

from .tool import BaseTool, FunctionTool


class BaseToolkit:
    """
    状态工具箱基类。继承此类的对象可以维持自身状态，
    其内部被 @tool 标记的方法将被自动解析为工具。

    参数:
        config: 工具箱的全局配置对象 (ToolkitConfig)。
            如果不传，则自动读取类内部的 Config 定义。
        tools: 除了带有 @tool 装饰器的方法外，要注入的其他独立工具。
        instructions: 当前工具箱实例的补充提示词；未传时使用 default_instructions。
        **kwargs: 预留扩展参数，供子类构造函数透传。
    """

    default_instructions: str = ""
    config: ToolkitConfig
    _class_tools_meta: ClassVar[dict[str, dict[str, Any]]] = {}
    _default_config: ClassVar[ToolkitConfig] = ToolkitConfig()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._class_tools_meta = {}
        for base in reversed(cls.__mro__):
            for attr_name, attr_value in base.__dict__.items():
                if callable(attr_value) and getattr(
                    attr_value, "__toolkit_tool__", False
                ):
                    cls._class_tools_meta[attr_name] = {
                        "original_tool_name": getattr(attr_value, "__tool_name__"),
                        "original_desc": getattr(attr_value, "__tool_desc__"),
                        "settings": getattr(attr_value, "__tool_settings__"),
                    }

        config_dict = {}
        if hasattr(cls, "Config"):
            config_cls = getattr(cls, "Config")
            for k in dir(config_cls):
                if not k.startswith("_"):
                    config_dict[k] = getattr(config_cls, k)
        cls._default_config = ToolkitConfig(**config_dict)

    def __init__(
        self,
        config: ToolkitConfig | None = None,
        tools: list[Callable | BaseTool | ToolExecutable] | None = None,
        instructions: str | None = None,
        **kwargs: Any,
    ):

        from zhenxun.utils.utils import infer_plugin_namespace

        raw_namespace = infer_plugin_namespace()

        if raw_namespace in ("asyncio", "zhenxun", "zhenxun_bot", "unknown"):
            self._inferred_namespace = "unknown"
        else:
            self._inferred_namespace = raw_namespace

        if config is not None:
            merged_dict = {
                **self._default_config.model_dump(exclude_unset=True),
                **config.model_dump(exclude_unset=True),
            }
            self.config = ToolkitConfig(**merged_dict)
        else:
            self.config = model_copy(self._default_config, deep=True)

        if self.config.prefix is None:
            if self._inferred_namespace != "unknown":
                self.config.prefix = f"{self._inferred_namespace}_"
            else:
                self.config.prefix = ""
        else:
            if self.config.prefix == "":
                self._inferred_namespace = "unknown"

        self._instance_filter: Callable[[BaseTool], bool] | None = None
        self._injected_tools: list[BaseTool] = []
        self._cached_tools: dict[str, BaseTool] | None = None
        self._instance_instructions = (
            instructions
            if instructions is not None
            else getattr(self, "default_instructions", "")
        )

        if tools:
            for t in tools:
                if isinstance(t, BaseTool):
                    self._injected_tools.append(t)
                elif callable(t):
                    tool_name = getattr(
                        t, "__tool_name__", getattr(t, "__name__", "unnamed_tool")
                    )
                    tool_desc = getattr(t, "__tool_desc__", "未提供描述")
                    settings = model_copy(
                        getattr(t, "__tool_settings__", ToolOptions())
                    )

                    func_tool = FunctionTool(
                        func=t,
                        name=f"{self.config.prefix}{tool_name}",
                        description=tool_desc,
                        settings=settings,
                    )
                    self._injected_tools.append(func_tool)

    def _apply_global_settings(
        self, original_name: str, settings: ToolOptions
    ) -> ToolOptions:
        if self.config.global_capabilities:
            settings.capabilities = (
                self.config.global_capabilities + settings.capabilities
            )

        if self.config.global_tags:
            settings.tags = list(set(settings.tags + self.config.global_tags))

        return settings

    async def enter_session(self, session_id: str, context: RunContext) -> None:
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

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        """实现 ToolResolvable 协议，递归解析内部工具并返回标准 Payload"""
        tools_dict = await self.get_tools(context)
        resolved_tools = []

        for t in tools_dict.values():
            if hasattr(t, "resolve"):
                payload = await t.resolve(context)
                if payload and payload.tools:
                    for rt in payload.tools:
                        rt.parent_toolkit = self
                    resolved_tools.extend(payload.tools)
            else:
                t.parent_toolkit = self
                resolved_tools.append(t)

        injected_prompts = []
        if instructions := self.get_instructions():
            injected_prompts.append(instructions)

        return ResolvedToolPayload(
            tools=resolved_tools, injected_prompts=injected_prompts, toolkits=[self]
        )

    async def get_tools(self, context: RunContext | None = None) -> dict[str, BaseTool]:
        if self._cached_tools is not None:
            return self._cached_tools

        tools_dict: dict[str, BaseTool] = {}
        for t in self._injected_tools:
            t.parent_toolkit = self
            tools_dict[t.name] = t

        for attr_name, meta in self._class_tools_meta.items():
            attr = getattr(self, attr_name)
            original_tool_name = meta["original_tool_name"]
            original_desc = meta["original_desc"]
            settings = model_copy(meta["settings"])

            if (
                self.config.include is not None
                and original_tool_name not in self.config.include
            ):
                continue
            if (
                self.config.exclude is not None
                and original_tool_name in self.config.exclude
            ):
                continue

            final_tool_name = f"{self.config.prefix}{original_tool_name}"

            final_desc = original_desc

            settings = self._apply_global_settings(original_tool_name, settings)

            from zhenxun.services.ai.tools.core.tool import FunctionTool

            func_tool = FunctionTool(
                func=attr,
                name=final_tool_name,
                description=final_desc,
                settings=settings,
            )
            func_tool.parent_toolkit = self
            tools_dict[final_tool_name] = cast(BaseTool, func_tool)

        if self._instance_filter:
            tools_dict = {
                name: t for name, t in tools_dict.items() if self._instance_filter(t)
            }
        self._cached_tools = tools_dict
        return tools_dict

    def get_instructions(self) -> str | None:
        """
        获取工具箱的系统提示词说明。
        """
        text = self._instance_instructions
        if not text:
            return None
        class_name = self.__class__.__name__
        tag_name = (
            f"{self.config.prefix}{class_name}_Instructions"
            if self.config.prefix
            else f"{class_name}_Instructions"
        )
        return f"<{tag_name}>\n{text}\n</{tag_name}>"

    def prefixed(self, prefix: str) -> "BaseToolkit":
        import copy

        new_tk = copy.copy(self)
        new_tk.config = model_copy(self.config, deep=True)
        current_prefix = new_tk.config.prefix or ""
        new_tk.config.prefix = f"{current_prefix}{prefix}"
        new_tk._cached_tools = None
        return new_tk

    def filtered(self, filter_func: Callable[[BaseTool], bool]) -> "BaseToolkit":
        import copy

        new_tk = copy.copy(self)
        new_tk.config = model_copy(self.config, deep=True)
        old_filter = self._instance_filter
        if old_filter:
            new_tk._instance_filter = lambda t: old_filter(t) and filter_func(t)
        else:
            new_tk._instance_filter = filter_func
        new_tk._cached_tools = None
        return new_tk


class CompositeToolkit(BaseToolkit):
    """
    聚合工具箱 (Composite Toolkit)。
    将一组独立的 Toolkit 组合在一起，统一其生命周期并合并其工具。
    可以为其指定统一的 prefix、全局标签或拦截器，底层会自动向内部的所有 Toolkit 透传。
    """

    def __init__(
        self,
        toolkits: list[BaseToolkit],
        config: ToolkitConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self.toolkits = toolkits

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        for tk in self.toolkits:
            await tk.enter_session(session_id, context)

    async def exit_session(self, session_id: str) -> None:
        for tk in self.toolkits:
            await tk.exit_session(session_id)

    async def before_llm_request(
        self, context: RunContext, messages: list[Any]
    ) -> None:
        for tk in self.toolkits:
            await tk.before_llm_request(context, messages)

    async def resolve(self, context: RunContext | None = None) -> ResolvedToolPayload:
        payload = ResolvedToolPayload()
        prefix = self.config.prefix or ""

        for tk in self.toolkits:
            active_tk = tk
            if prefix:
                active_tk = active_tk.prefixed(prefix)
            if self._instance_filter:
                active_tk = active_tk.filtered(self._instance_filter)

            child_payload = await active_tk.resolve(context)

            for tool in child_payload.tools:
                if self.config.global_tags:
                    tool.settings.tags = list(
                        set(tool.settings.tags + self.config.global_tags)
                    )
                if self.config.global_capabilities:
                    tool.settings.capabilities = (
                        self.config.global_capabilities + tool.settings.capabilities
                    )

            payload.tools.extend(child_payload.tools)
            payload.injected_prompts.extend(child_payload.injected_prompts)
            payload.toolkits.extend(child_payload.toolkits)

        payload.toolkits.append(self)
        if instructions := self.get_instructions():
            payload.injected_prompts.append(instructions)

        return payload


class ApiConnectToolkit(BaseToolkit, ABC):
    """
    托管连接池的 Toolkit 基类。
    自动处理第三方凭证的获取、客户端的懒加载初始化，并在会话结束时安全释放连接池资源。

    参数:
        ttl: 资源存活时长（秒）。
            会话结束或超出此时间无调用后，连接将被自动释放。默认 600 秒。
        kwargs: 传递给 BaseToolkit 的其他参数。
    """

    required_auth_providers: tuple[str, ...] = ()

    def __init__(self, ttl: int = 600, **kwargs: Any):
        super().__init__(**kwargs)
        self.ttl = ttl
        self._clients: dict[str, Any] = {}
        self.lifespan_manager = LifespanManager()

    async def get_client(self, context: RunContext) -> Any:
        """获取当前会话的客户端。如果未初始化，则检查凭证并自动初始化。"""
        session_id = context.session_id or "default_session"
        await self.lifespan_manager.register(
            session_id, ttl=float(self.ttl), cleanup_callback=self.release_resource
        )

        if session_id in self._clients:
            return self._clients[session_id]

        tokens = context.session.auth_tokens
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
            await self.lifespan_manager.unregister(session_id)
            await self.release_resource(session_id)


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
        config: ToolkitConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
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
            return model_copy(self._memory_states[state_key], deep=True)
        return self.default_state_factory()

    async def save_state(self, state_key: str, state: StateT) -> None:
        """将状态保存到外部存储。完全交由第三方开发者决定存储介质。"""
        self._memory_states[state_key] = model_copy(state, deep=True)

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

    async def exit_session(self, session_id: str) -> None:
        """框架级 Hook：每次 Agent 运行结束时，抽取并保存最新状态"""
        await super().exit_session(session_id)
        if session_id in self._active_states:
            state = self._active_states.pop(session_id)
            state_key = self._session_to_state_key.pop(session_id, session_id)
            await self.save_state(state_key, state)

    def get_active_state(self, session_id: str | None) -> StateT | None:
        """获取当前活跃会话的状态。

        (专供 before_llm_request 等内部框架生命周期钩子使用)
        """
        return self._active_states.get(session_id or "default_session")


class UserPersonalToolkit(GroupSharedToolkit[StateT]):
    """
    用户私有工具箱基类 (User Personal Toolkit)。

    与 GroupSharedToolkit 的区别在于：本类强制基于 user_id 进行物理状态隔离，
    哪怕在群内调用，A 用户的状态也绝对不会泄漏给 B 用户。
    适用于私人的待办清单、个人积分等。
    """

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        await BaseToolkit.enter_session(self, session_id, context)

        user_id = context.get_user_id()
        state_key = f"user_{user_id}" if user_id else session_id

        self._session_to_state_key[session_id] = state_key
        state = await self.load_state(state_key)
        self._active_states[session_id] = state
