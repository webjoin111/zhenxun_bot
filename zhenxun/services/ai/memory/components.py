from typing import Any

from zhenxun.services.ai.memory.interfaces import (
    MemoryRetriever,
)
from zhenxun.services.ai.memory.models import (
    MemoryMatch,
    MemoryRecord,
    MemoryScoringConfig,
    SessionMetadata,
)
from zhenxun.services.ai.memory.utils import compute_composite_score
from zhenxun.services.ai.rag import Embedder, QueryRequest, StorageBackend
from zhenxun.services.log import logger


class StandardRetriever(MemoryRetriever):
    """标准的记忆检索器。支持向量搜索、过滤与复合分数重排。"""

    def __init__(
        self,
        storage: StorageBackend,
        embedder: Embedder,
        config: MemoryScoringConfig | None = None,
        rerank_model: str | None = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.config = config or MemoryScoringConfig()
        self.rerank_model = rerank_model

    async def retrieve(
        self,
        session: SessionMetadata,
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryMatch]:
        query_vectors = await self.embedder([query], task="query")
        query_vector = query_vectors[0] if query_vectors and query_vectors[0] else None

        rag_query = QueryRequest(
            text=query,
            embedding=query_vector,
            metadata_filters=metadata_filter,
            limit=limit * 2,
        )

        rag_results = await self.storage.search(query=rag_query, scope_prefix=session.scope_prefix)

        raw_results = []
        import time
        for res in rag_results:
            b = res.record
            meta = b.metadata.copy()
            rec = MemoryRecord(
                id=b.id, content=b.content, embedding=b.embedding,
                scope=meta.pop("scope", "/"),
                importance=meta.pop("importance", 0.5),
                created_at=meta.pop("created_at", time.time()),
                metadata=meta
            )
            raw_results.append((rec, res.score))

        if self.rerank_model and raw_results:
            from zhenxun.services.ai.llm.api import rerank

            documents_to_rank: list[str | dict[str, str]] = [
                r.content for r, _ in raw_results
            ]
            try:
                reranked = await rerank(
                    query=query,
                    documents=documents_to_rank,
                    top_n=limit * 2,
                    model=self.rerank_model,
                )
                rerank_score_map = {res.index: res.relevance_score for res in reranked}
                new_raw_results = []
                for idx, (record, _) in enumerate(raw_results):
                    if idx in rerank_score_map:
                        new_raw_results.append((record, rerank_score_map[idx]))
                raw_results = new_raw_results
            except Exception as e:
                logger.warning(f"Rerank 重排失败，降级使用原向量分数: {e}")

        matches = []
        for record, similarity in raw_results:
            composite_score, reasons = compute_composite_score(
                record, similarity, self.config
            )
            matches.append(
                MemoryMatch(record=record, score=composite_score, match_reasons=reasons)
            )

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:limit]
