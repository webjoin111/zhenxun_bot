from collections.abc import Callable
from typing import Any, cast

from zhenxun.services.ai.memory.interfaces import (
    BaseChatContext,
    BaseSlotContext,
)
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.memory.storage import (
    InMemoryChatContext,
    InMemorySlotContext,
    MemoryScope,
)
from zhenxun.services.ai.rag.backends import Embedder, StorageBackend
from zhenxun.services.ai.rag.consolidation import Consolidator as MemoryConsolidator
from zhenxun.services.ai.rag.consolidation import NullConsolidator
from zhenxun.utils.utils import infer_plugin_namespace


class GlobalMemoryManager:
    """
    全局记忆大管家 (IoC 容器)。
    使用现代化依赖注入机制管理短/长期记忆引擎的默认实例。
    """

    def __init__(self):
        self._chat_backends: dict[str, BaseChatContext] = {
            "global": InMemoryChatContext()
        }
        self._slot_backends: dict[str, BaseSlotContext] = {
            "global": InMemorySlotContext()
        }

        from zhenxun.services.ai.rag.backends import DictStorageBackend

        self._storage_factories: dict[str, Callable[[], StorageBackend]] = {
            "global": lambda: DictStorageBackend()
        }
        self._consolidator_factories: dict[str, Callable[[], MemoryConsolidator]] = {
            "global": lambda: NullConsolidator()
        }

    def register_chat_backend(
        self, backend: BaseChatContext, scope: str | None = None
    ) -> None:
        """注册特定命名空间的短期记忆存储后端。"""
        ns = scope if scope is not None else infer_plugin_namespace()
        self._chat_backends[ns] = backend

    def register_slot_backend(
        self, backend: BaseSlotContext, scope: str | None = None
    ) -> None:
        """注册特定命名空间的中期记忆槽存储后端。"""
        ns = scope if scope is not None else infer_plugin_namespace()
        self._slot_backends[ns] = backend

    def register_storage_factory(
        self, factory: Callable[[], StorageBackend], scope: str | None = None
    ) -> None:
        """注册特定命名空间的长期记忆向量存储工厂。"""
        ns = scope if scope is not None else infer_plugin_namespace()
        self._storage_factories[ns] = factory

    def register_consolidator_factory(
        self, factory: Callable[[], MemoryConsolidator], scope: str | None = None
    ) -> None:
        """注册特定命名空间的长期记忆融合引擎工厂。"""
        ns = scope if scope is not None else infer_plugin_namespace()
        self._consolidator_factories[ns] = factory

    def get_embedder(self, embedder_val: "Embedder | str | None") -> Embedder | None:
        """获取向量化引擎实例。如果传入的是字符串，则视为 API 模型名称。"""
        if not embedder_val:
            return None

        if isinstance(embedder_val, str):
            from zhenxun.services.ai.rag.backends.embedders import DefaultEmbedder

            return DefaultEmbedder(model_name=embedder_val)

        return embedder_val

    def get_chat_context(
        self, config: MemoryConfig | None, namespace: str = "global"
    ) -> BaseChatContext | None:
        """根据配置分配对应的短期对话历史实例"""
        if not config or not config.short_term.enable:
            return None

        backend_cfg = config.short_term.backend
        if backend_cfg is not None:
            return cast(BaseChatContext, backend_cfg)

        return self._chat_backends.get(namespace) or self._chat_backends["global"]

    def get_slot_context(
        self, config: MemoryConfig | None, namespace: str = "global"
    ) -> BaseSlotContext | None:
        """根据配置分配对应的槽位记忆实例"""
        if not config or not config.slots.enable:
            return None

        backend_cfg = config.slots.backend
        if backend_cfg is not None:
            return cast(BaseSlotContext, backend_cfg)

        return self._slot_backends.get(namespace) or self._slot_backends["global"]

    def get_long_term_memory(
        self, config: MemoryConfig | None, namespace: str = "global"
    ) -> MemoryScope | None:
        """根据声明式配置动态组装长期向量记忆实例"""
        if not config or not config.long_term.enable:
            return None

        storage_instance = None
        backend_cfg = config.long_term.backend
        if backend_cfg is not None:
            storage_instance = cast(StorageBackend, backend_cfg)
        else:
            factory = (
                self._storage_factories.get(namespace)
                or self._storage_factories["global"]
            )
            storage_instance = factory()

        consolidator_instance = None

        if config.long_term.auto_consolidate:
            consolidator_cfg = config.long_term.consolidator
            if consolidator_cfg is not None:
                consolidator_instance = cast(MemoryConsolidator, consolidator_cfg)
            else:
                factory = (
                    self._consolidator_factories.get(namespace)
                    or self._consolidator_factories["global"]
                )
                consolidator_instance = factory()

        embedder = self.get_embedder(config.long_term.embedder)

        from zhenxun.services.ai.rag.builder import RAGBuilder

        builder = RAGBuilder(storage_instance).with_scope("/")
        if embedder:
            builder.with_embedder(embedder)

        from zhenxun.services.ai.memory.models import MemoryScoringConfig

        scoring_cfg = MemoryScoringConfig()

        if config.long_term.auto_consolidate and consolidator_instance:
            builder.enable_consolidation(
                consolidator=consolidator_instance,
                threshold=scoring_cfg.consolidation_threshold,
            )

        builder.enable_lifecycle_scoring(
            half_life_days=scoring_cfg.recency_half_life_days,
            decay_weight=scoring_cfg.recency_weight,
            semantic_weight=scoring_cfg.semantic_weight,
            importance_weight=scoring_cfg.importance_weight,
            reinforcement_weight=scoring_cfg.reinforcement_weight,
        )

        client = builder.build()

        return MemoryScope(
            rag_client=client,
            async_write=config.long_term.async_write,
            capacity_limit=scoring_cfg.capacity_limit,
            evict_ratio=scoring_cfg.evict_ratio,
        )


memory_manager = GlobalMemoryManager()
