import asyncio
from typing import Any, cast

from zhenxun.services.ai.rag.models import SearchResult
from zhenxun.services.ai.rag.retrieval import BaseRetriever
from zhenxun.services.log import logger


class HybridRetriever(BaseRetriever):
    """
    双轨混合检索器 (Hybrid Search Engine)。
    并发调用 Dense (VectorDB) 和 Sparse (BM25)，并使用倒数秩融合 (RRF) 算法合并结果。
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        oversample_limit = max(limit * 2, 20)

        results = await asyncio.gather(
            self.dense_retriever.retrieve(query, limit=oversample_limit, **kwargs),
            self.sparse_retriever.retrieve(query, limit=oversample_limit, **kwargs),
            return_exceptions=True,
        )

        dense_res = (
            cast(list[SearchResult], results[0])
            if not isinstance(results[0], BaseException)
            else []
        )
        sparse_res = (
            cast(list[SearchResult], results[1])
            if not isinstance(results[1], BaseException)
            else []
        )

        if isinstance(results[0], BaseException):
            logger.error(f"[HybridSearch] 向量检索异常: {results[0]}")
        if isinstance(results[1], BaseException):
            logger.error(f"[HybridSearch] BM25 检索异常: {results[1]}")

        rrf_scores: dict[str, float] = {}
        merged_records = {}

        for rank, res in enumerate(dense_res):
            record_id = res.record.id
            merged_records[record_id] = res.record
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[record_id] = rrf_scores.get(record_id, 0.0) + (
                self.dense_weight * rrf_score
            )

        for rank, res in enumerate(sparse_res):
            record_id = res.record.id
            merged_records[record_id] = res.record
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[record_id] = rrf_scores.get(record_id, 0.0) + (
                self.sparse_weight * rrf_score
            )

        final_results = []
        for record_id, score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            final_results.append(
                SearchResult(record=merged_records[record_id], score=score)
            )

        logger.debug(
            f"⚖️ [HybridSearch] 融合完成: "
            f"Dense({len(dense_res)}) + Sparse({len(sparse_res)}) "
            f"-> Merged({len(final_results)}), 截取 Top {limit}"
        )
        return final_results[:limit]
