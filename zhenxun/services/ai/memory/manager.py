from collections.abc import Callable

from zhenxun.services.ai.memory.interfaces import (
    BaseChatContext,
)
from zhenxun.services.ai.rag.consolidation import Consolidator as MemoryConsolidator, NullConsolidator
from zhenxun.services.ai.memory.long_term_memory import MemoryScope
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.memory.short_term_memory import InMemoryChatContext
from zhenxun.services.ai.rag import Embedder, StorageBackend
from zhenxun.services.ai.rag.backends import RagRegistry
from zhenxun.services.log import logger


class GlobalMemoryManager:
    """
    全局记忆大管家 (IoC 容器)。
    负责管理短/长期记忆引擎的注册与分配，解耦 Agent 配置与底层数据库实例。
    """

    def __init__(self):
        self._chat_backends: dict[str, BaseChatContext] = {}
        self._storage_engines: dict[str, Callable[[], StorageBackend]] = {}
        self._consolidator_engines: dict[str, Callable[[], MemoryConsolidator]] = {}

        self._default_chat_backend_name: str = "default"
        self._default_ltm_backend_name: str = "default"

        self.register_chat_backend("default", InMemoryChatContext())
        self.register_consolidator_engine("default", lambda: NullConsolidator())

    def register_chat_backend(self, name: str, backend: BaseChatContext) -> None:
        """注册一个短期对话历史存储引擎"""
        self._chat_backends[name] = backend
        logger.debug(f"已注册短期记忆存储后端: '{name}'")

    def register_storage_engine(
        self, name: str, factory: Callable[[], StorageBackend]
    ) -> None:
        """注册一个长期向量存储引擎工厂"""
        self._storage_engines[name] = factory
        logger.debug(f"已注册长期记忆存储工厂: '{name}'")

    def register_consolidator_engine(
        self, name: str, factory: Callable[[], MemoryConsolidator]
    ) -> None:
        """注册一个记忆融合反思引擎工厂"""
        self._consolidator_engines[name] = factory


    def set_default_chat_backend(self, name: str) -> None:
        """设置全局默认的短期记忆引擎标识符"""
        self._default_chat_backend_name = name

    def set_default_ltm_backend(self, name: str) -> None:
        """设置全局默认的长期记忆引擎标识符"""
        self._default_ltm_backend_name = name

    def get_embedder(self, embedder_str: str | None) -> Embedder | None:
        """通过前缀路由解析并获取向量化引擎实例。格式: provider/model_name"""
        if not embedder_str:
            return None

        parts = embedder_str.split("/", 1)
        if len(parts) == 2:
            provider, model = parts[0].lower(), parts[1]
            factory = RagRegistry.get_embedder(provider)
            if factory:
                return factory(model)

        fallback_factory = RagRegistry.get_embedder("default")
        return fallback_factory(embedder_str) if fallback_factory else None

    def get_chat_context(self, config: MemoryConfig | None) -> BaseChatContext | None:
        """根据配置分配对应的短期对话历史实例"""
        if not config or not config.enable_short_term:
            return None

        backend_name = config.chat_backend or self._default_chat_backend_name
        backend = self._chat_backends.get(backend_name)

        if backend is None:
            logger.warning(
                f"请求的短期记忆后端 '{backend_name}' 未注册，将回退到默认引擎。"
            )
            return self._chat_backends.get(self._default_chat_backend_name)
        return backend

    def get_long_term_memory(self, config: MemoryConfig | None) -> MemoryScope | None:
        """根据声明式配置，JIT 即时拉取工厂组件并动态组装长期向量记忆实例"""
        if not config or not getattr(config, "enable_ltm", False):
            return None

        backend_name = config.ltm_backend or self._default_ltm_backend_name
        storage_factory = self._storage_engines.get(backend_name)
        if not storage_factory:
            logger.warning(f"长期记忆 Storage 引擎 '{backend_name}' 未注册。")
            return None

        if getattr(config, "ltm_auto_consolidate", True):
            consolidator_factory = self._consolidator_engines.get(
                config.ltm_consolidator or "default"
            )
        else:
            consolidator_factory = lambda: NullConsolidator()

        embedder = self.get_embedder(config.ltm_embedder)

        return MemoryScope(
            storage=storage_factory(),
            consolidator=consolidator_factory() if consolidator_factory else None,
            embedder=embedder,
            async_write=config.ltm_async_write,
        )


memory_manager = GlobalMemoryManager()
