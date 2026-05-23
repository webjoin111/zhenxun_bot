import asyncio
import time
from typing import Any
import uuid

from zhenxun.services.ai.llm.api import generate_structured
from zhenxun.services.ai.rag.consolidation import (
    Consolidator as MemoryConsolidator,
    NullConsolidator,
    LLMConsolidator as LLMMemoryConsolidator,
)
from zhenxun.services.ai.memory.models import (
    MemoryMatch,
    MemoryRecord,
    MemoryScoringConfig,
    SessionMetadata,
)
from zhenxun.services.ai.rag import BaseRecord, Embedder, StorageBackend
from zhenxun.services.log import logger


class MemoryScope:
    """长期记忆的作用域视图与 RAG 管线。"""

    def __init__(
        self,
        storage: StorageBackend,
        embedding_model: str | None = None,
        consolidator: MemoryConsolidator | None = None,
        rerank_model: str | None = None,
        config: MemoryScoringConfig | None = None,
        embedder: Embedder | None = None,
        retriever: Any | None = None,
        async_write: bool = True,
    ):
        self.storage = storage
        self.embedding_model = embedding_model
        self.consolidator = consolidator or NullConsolidator()
        self.rerank_model = rerank_model
        self.config = config or MemoryScoringConfig()
        self.async_write = async_write

        resolved_embedder: Embedder
        if embedder:
            resolved_embedder = embedder
        else:
            from zhenxun.services.ai.memory.manager import memory_manager

            routed_embedder = memory_manager.get_embedder(self.embedding_model)
            if not routed_embedder:
                from zhenxun.services.ai.rag import DefaultEmbedder

                routed_embedder = DefaultEmbedder(model_name=self.embedding_model)
            resolved_embedder = routed_embedder

        self.embedder = resolved_embedder

        if retriever:
            self.retriever = retriever
        else:
            from zhenxun.services.ai.rag import RAGManager
            from zhenxun.services.ai.rag.models import RAGConfig

            self.retriever = RAGManager.build_retriever(
                RAGConfig(
                    use_rerank=bool(self.rerank_model),
                    rerank_model=self.rerank_model,
                    use_time_decay=True,
                    half_life_days=self.config.recency_half_life_days,
                ),
                storage=self.storage,
                embedder=self.embedder,
            )

        from zhenxun.services.ai.rag.ingestion import (
            ChunkingNode,
            ConsolidationNode,
            DocumentChunking,
            EmbeddingNode,
            IngestionPipeline,
            StorageWriteNode,
        )

        self.pipeline = IngestionPipeline(
            nodes=[
                ChunkingNode(DocumentChunking(chunk_size=1000)),
                EmbeddingNode(self.embedder),
                ConsolidationNode(
                    self.storage,
                    self.consolidator,
                    self.embedder,
                    self.config.consolidation_threshold,
                ),
                StorageWriteNode(self.storage),
            ]
        )

    async def remember(
        self,
        session: SessionMetadata,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """通过 RAG Ingestion Pipeline 完成记忆落盘"""
        meta = metadata.copy() if metadata else {}
        meta.update(
            {
                "scope": session.scope_prefix,
                "importance": importance,
                "created_at": time.time(),
            }
        )
        record = BaseRecord(content=content, metadata=meta)

        if self.async_write:
            asyncio.create_task(self.pipeline.run([record]))
        else:
            await self.pipeline.run([record])

    async def recall(
        self,
        session: SessionMetadata,
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryMatch]:
        """委托至 Retriever 检索与重排"""
        results = await self.retriever.retrieve(
            query=query,
            limit=limit,
            metadata_filters=metadata_filter,
            scope_prefix=session.scope_prefix,
        )
        matches = []
        import time

        for r in results:
            meta = r.record.metadata.copy()
            mem_record = MemoryRecord(
                id=r.record.id,
                content=r.record.content,
                embedding=r.record.embedding,
                scope=meta.pop("scope", "/"),
                importance=meta.pop("importance", 0.5),
                created_at=meta.pop("created_at", time.time()),
                metadata=meta,
            )
            matches.append(
                MemoryMatch(
                    record=mem_record, score=r.score, match_reasons=["semantic"]
                )
            )
        return matches

    async def forget(
        self, session: SessionMetadata, record_ids: list[str] | None = None
    ) -> int:
        return await self.storage.delete(
            record_ids=record_ids, scope_prefix=session.scope_prefix
        )


def get_plugin_memory_scope(
    storage: StorageBackend,
    plugin_name: str,
    group_id: str | None = None,
    user_id: str | None = None,
    embedding_model: str | None = None,
    consolidator: MemoryConsolidator | None = None,
    rerank_model: str | None = None,
) -> MemoryScope:
    """(遗留辅助函数：已简化)"""
    return MemoryScope(
        storage=storage,
        embedding_model=embedding_model,
        consolidator=consolidator,
        rerank_model=rerank_model,
    )
