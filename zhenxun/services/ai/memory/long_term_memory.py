import time
from typing import Any
import uuid

from tortoise import fields

from zhenxun.services.ai.memory.interfaces import StorageBackend
from zhenxun.services.ai.memory.models import (
    MemoryConfig,
    MemoryMatch,
    MemoryQuery,
    MemoryRecord,
)
from zhenxun.services.ai.memory.utils import (
    compute_composite_score,
    cosine_similarity,
    join_scope_paths,
)
from zhenxun.services.db_context import Model
from zhenxun.services.log import logger


class AbstractVectorRecord(Model):
    """
    Tortoise ORM 向量记忆基类 (Mixin)。
    """

    id = fields.CharField(pk=True, max_length=64, description="记忆主键")
    scope = fields.CharField(max_length=255, index=True, description="作用域前缀")
    content = fields.TextField(description="记忆内容")
    importance = fields.FloatField(default=0.5, description="重要性分数")
    embedding = fields.JSONField(null=True, description="向量数组 (list[float])")
    meta_data = fields.JSONField(null=True, description="额外元数据")
    created_at = fields.FloatField(description="创建时间的时间戳")

    class Meta:  # type: ignore
        abstract = True


class TortoiseStorageBackend(StorageBackend):
    """基于 Tortoise ORM 的混合存储后端。"""

    def __init__(self, model_class: type[AbstractVectorRecord]):
        self.model_class = model_class

    def _to_memory_record(self, row: AbstractVectorRecord) -> MemoryRecord:
        return MemoryRecord(
            id=row.id,
            content=row.content,
            scope=row.scope,
            importance=row.importance,
            embedding=row.embedding if isinstance(row.embedding, list) else None,
            metadata=row.meta_data if isinstance(row.meta_data, dict) else {},
            created_at=row.created_at,
        )

    async def save(self, records: list[MemoryRecord]) -> None:
        for r in records:
            await self.model_class.update_or_create(
                id=r.id,
                defaults={
                    "content": r.content,
                    "scope": r.scope,
                    "importance": r.importance,
                    "embedding": r.embedding,
                    "meta_data": r.metadata,
                    "created_at": r.created_at,
                },
            )

    async def search(
        self,
        query: MemoryQuery,
        scope_prefix: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        query_orm = self.model_class.all()
        if scope_prefix:
            query_orm = query_orm.filter(scope__startswith=scope_prefix)
        if not query.embedding and query.text:
            query_orm = query_orm.filter(content__icontains=query.text)
        rows = await query_orm
        results = []
        for row in rows:
            if query.metadata_filters:
                row_meta = row.meta_data if isinstance(row.meta_data, dict) else {}
                if not all(
                    row_meta.get(k) == v for k, v in query.metadata_filters.items()
                ):
                    continue
            if not isinstance(row.embedding, list) or not query.embedding:
                results.append((self._to_memory_record(row), 0.1))
            else:
                sim = cosine_similarity(query.embedding, row.embedding)
                results.append((self._to_memory_record(row), sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[: query.limit]

    async def delete(
        self, scope_prefix: str | None = None, record_ids: list[str] | None = None
    ) -> int:
        query = self.model_class.all()
        if scope_prefix:
            query = query.filter(scope__startswith=scope_prefix)
        if record_ids is not None:
            if not record_ids:
                return 0
            query = query.filter(id__in=record_ids)
        return await query.delete()


class MemoryScope:
    """长期记忆的作用域视图与 RAG 管线。"""

    def __init__(
        self,
        storage: StorageBackend,
        root_path: str = "/",
        embedding_model: str | None = None,
        rerank_model: str | None = None,
        config: MemoryConfig | None = None,
    ):
        self.storage = storage
        self.root_path = root_path
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.config = config or MemoryConfig()

    async def _get_embedding(self, text: str) -> list[float]:
        if not self.embedding_model or not text.strip():
            return []
        from zhenxun.services.ai.llm.api import embed

        try:
            response = await embed([text], model=self.embedding_model)
            return response.vector
        except Exception as e:
            logger.warning(f"获取记忆向量失败，将降级处理: {e}")
            return []

    async def remember(
        self,
        content: str,
        importance: float = 0.5,
        inner_scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        final_scope = join_scope_paths(self.root_path, inner_scope)
        vector = await self._get_embedding(content)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            scope=final_scope,
            importance=importance,
            embedding=vector,
            metadata=metadata or {},
            created_at=time.time(),
        )
        await self.storage.save([record])
        return record

    async def recall(
        self,
        query: str,
        limit: int = 10,
        inner_scope: str = "",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryMatch]:
        final_scope = join_scope_paths(self.root_path, inner_scope)
        query_vector = await self._get_embedding(query)
        fetch_limit = limit * 3 if self.rerank_model else limit * 2

        memory_query = MemoryQuery(
            text=query,
            embedding=query_vector,
            metadata_filters=metadata_filter,
            limit=fetch_limit,
        )

        raw_results = await self.storage.search(
            memory_query,
            scope_prefix=final_scope,
        )

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
                for idx, (record, orig_score) in enumerate(raw_results):
                    if idx in rerank_score_map:
                        new_raw_results.append((record, rerank_score_map[idx]))
                raw_results = new_raw_results
            except Exception as e:
                logger.warning(f"Rerank 失败，降级使用原向量分数: {e}")

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

    async def forget(
        self, record_ids: list[str] | None = None, inner_scope: str = ""
    ) -> int:
        final_scope = join_scope_paths(self.root_path, inner_scope)
        return await self.storage.delete(
            scope_prefix=final_scope, record_ids=record_ids
        )


def get_plugin_memory_scope(
    storage: StorageBackend,
    plugin_name: str,
    group_id: str | None = None,
    user_id: str | None = None,
    embedding_model: str | None = None,
    rerank_model: str | None = None,
) -> MemoryScope:
    root_path = f"/{plugin_name}"
    if group_id:
        root_path = join_scope_paths(root_path, f"group/{group_id}")
    if user_id:
        root_path = join_scope_paths(root_path, f"user/{user_id}")
    return MemoryScope(
        storage=storage,
        root_path=root_path,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
    )


class DictStorageBackend(StorageBackend):
    """
    基于内存字典的轻量级长期记忆实现。
    供测试使用，实现了全新的 StorageBackend 协议。
    """

    def __init__(self):
        self._records: dict[str, MemoryRecord] = {}

    async def save(self, records: list[MemoryRecord]) -> None:
        for r in records:
            self._records[r.id] = r

    async def search(
        self,
        query: MemoryQuery,
        scope_prefix: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        results = []
        for record in self._records.values():
            if scope_prefix and not record.scope.startswith(scope_prefix):
                continue

            if query.metadata_filters:
                if not record.metadata or not all(
                    record.metadata.get(k) == v
                    for k, v in query.metadata_filters.items()
                ):
                    continue
            if not query.embedding and query.text and query.text not in record.content:
                continue

            if not record.embedding or not query.embedding:
                results.append((record, 0.1))
            else:
                sim = cosine_similarity(query.embedding, record.embedding)
                results.append((record, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: query.limit]

    async def delete(
        self,
        scope_prefix: str | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        to_delete = []
        for r_id, r in self._records.items():
            if scope_prefix and not r.scope.startswith(scope_prefix):
                continue
            if record_ids and r_id not in record_ids:
                continue
            to_delete.append(r_id)

        for r_id in to_delete:
            del self._records[r_id]

        return len(to_delete)
