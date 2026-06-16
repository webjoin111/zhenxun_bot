import asyncio
from typing import Any

from zhenxun.services.ai.context.rag.backends import (
    StorageBackend,
)
from zhenxun.services.ai.context.rag.configs import RAGConfig
from zhenxun.services.ai.context.rag.indexing import (
    IndexPipeline,
)
from zhenxun.services.ai.context.rag.models import (
    BaseRecord,
    SearchResult,
)
from zhenxun.services.ai.context.rag.retrieval import (
    BaseRetriever,
)
from zhenxun.services.ai.utils.scope import normalize_scope_path
from zhenxun.services.log import logger


class ScopedRAGClient:
    """
    RAG 基础设施门面。
    封装了存储后端、检索器和写入管线，将所有操作透明地限定在指定的作用域前缀下。
    """

    def __init__(
        self,
        storage: StorageBackend,
        retriever: BaseRetriever,
        pipeline: IndexPipeline,
        scopes: str | list[str] = "/",
        config: RAGConfig | None = None,
    ):
        self.storage = storage
        self.retriever = retriever
        self.pipeline = pipeline
        self.config = config or RAGConfig()
        self._background_tasks = set()
        if isinstance(scopes, str):
            self.scopes = [normalize_scope_path(scopes)]
        else:
            self.scopes = [normalize_scope_path(s) for s in scopes]

        self.scope_prefix = self.scopes[0] if self.scopes else "/"

    async def ingest(self, records: list[BaseRecord], async_write: bool = False) -> int:
        """通过 RAG Ingestion Pipeline 处理并入库数据"""
        if async_write:
            import asyncio

            task = asyncio.create_task(self.pipeline.run(records))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return len(records)
        else:
            res = await self.pipeline.run(records)
            return len(res)

    async def search(
        self,
        query: Any,
        limit: int = 10,
        scopes: str | list[str] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        多作用域联合切片视图检索 (Union Search)。
        并发向多个独立的作用域发起检索，并对结果进行合并、去重和重排。
        """
        target_scopes = self.scopes
        if scopes is not None:
            target_scopes = (
                [normalize_scope_path(scopes)]
                if isinstance(scopes, str)
                else [normalize_scope_path(s) for s in scopes]
            )

        if not target_scopes:
            return []

        if len(target_scopes) == 1:
            kwargs["scope_prefix"] = target_scopes[0]
            return await self.retriever.retrieve(query, limit=limit, **kwargs)

        oversample_limit = limit * 2
        tasks = []

        for scope in target_scopes:
            call_kwargs = kwargs.copy()
            call_kwargs["scope_prefix"] = scope
            tasks.append(
                self.retriever.retrieve(query, limit=oversample_limit, **call_kwargs)
            )

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for res_list in results_lists:
            if isinstance(res_list, BaseException):
                logger.error(f"[ScopedRAGClient] 并发检索子任务失败: {res_list}")
                continue

            for res in res_list:
                if res.record.id not in seen_ids:
                    seen_ids.add(res.record.id)
                    all_results.append(res)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]

    async def update(self, record: BaseRecord) -> None:
        record.metadata["scope"] = self.scope_prefix
        await self.storage.update(record)

    async def delete(self, record_ids: list[str] | None = None, **kwargs: Any) -> int:
        kwargs["scope_prefix"] = self.scope_prefix
        return await self.storage.delete(record_ids=record_ids, **kwargs)
