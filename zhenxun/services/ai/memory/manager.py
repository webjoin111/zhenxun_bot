from collections.abc import Callable
from typing import Any, cast

from zhenxun.services.ai.memory.interfaces import (
    BaseChatContext,
)
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.memory.storage import InMemoryChatContext, MemoryScope
from zhenxun.services.ai.rag import Embedder, StorageBackend
from zhenxun.services.ai.rag.consolidation import Consolidator as MemoryConsolidator
from zhenxun.services.ai.rag.consolidation import NullConsolidator
from zhenxun.services.log import logger


class GlobalMemoryManager:
    """
    全局记忆大管家 (IoC 容器)。
    使用现代化依赖注入机制管理短/长期记忆引擎的默认实例。
    """

    def __init__(self):
        self.default_chat_backend: BaseChatContext = InMemoryChatContext()

        from zhenxun.services.ai.rag.backends import DictStorageBackend

        self.default_storage_factory: Callable[[], StorageBackend] = lambda: (
            DictStorageBackend()
        )
        self.default_consolidator_factory: Callable[[], MemoryConsolidator] = lambda: (
            NullConsolidator()
        )

    def set_default_chat_backend(self, backend: BaseChatContext) -> None:
        """设置全局默认的短期记忆引擎实例"""
        self.default_chat_backend = backend
        logger.debug(f"已设置全局默认短期记忆存储后端: {backend.__class__.__name__}")

    def set_default_storage_factory(
        self, factory: Callable[[], StorageBackend]
    ) -> None:
        """设置全局默认的长期向量存储引擎工厂"""
        self.default_storage_factory = factory

    def set_default_consolidator_factory(
        self, factory: Callable[[], MemoryConsolidator]
    ) -> None:
        """设置全局默认的记忆融合引擎工厂"""
        self.default_consolidator_factory = factory

    def get_embedder(self, embedder_val: Any | None) -> Embedder | None:
        """获取向量化引擎实例。如果传入的是字符串，则视为 API 模型名称。"""
        if not embedder_val:
            return None

        if isinstance(embedder_val, str):
            from zhenxun.services.ai.rag.backends.embedders import DefaultEmbedder

            return DefaultEmbedder(model_name=embedder_val)

        return embedder_val

    def get_chat_context(self, config: MemoryConfig | None) -> BaseChatContext | None:
        """根据配置分配对应的短期对话历史实例"""
        if not config or not config.short_term.enable:
            return None

        backend_cfg = config.short_term.backend
        if backend_cfg is not None:
            return cast(BaseChatContext, backend_cfg)

        return self.default_chat_backend

    def get_long_term_memory(self, config: MemoryConfig | None) -> MemoryScope | None:
        """根据声明式配置动态组装长期向量记忆实例"""
        if not config or not config.long_term.enable:
            return None

        storage_instance = None
        backend_cfg = config.long_term.backend
        if backend_cfg is not None:
            storage_instance = cast(StorageBackend, backend_cfg)
        else:
            storage_instance = self.default_storage_factory()

        consolidator_instance = None

        if config.long_term.auto_consolidate:
            consolidator_cfg = config.long_term.consolidator
            if consolidator_cfg is not None:
                consolidator_instance = cast(MemoryConsolidator, consolidator_cfg)
            else:
                consolidator_instance = self.default_consolidator_factory()

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

        builder.enable_time_decay(
            half_life_days=scoring_cfg.recency_half_life_days,
            decay_weight=scoring_cfg.recency_weight,
            semantic_weight=scoring_cfg.semantic_weight,
            importance_weight=scoring_cfg.importance_weight,
        )

        client = builder.build()

        return MemoryScope(
            rag_client=client,
            async_write=config.long_term.async_write,
        )


memory_manager = GlobalMemoryManager()
