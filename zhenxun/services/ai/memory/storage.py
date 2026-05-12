from typing import Any

from tortoise import fields

from zhenxun.services.ai.memory.utils import cosine_similarity
from zhenxun.services.ai.protocols.memory import StorageBackend
from zhenxun.services.ai.memory.models import MemoryRecord
from zhenxun.services.db_context import Model


class AbstractVectorRecord(Model):
    """
    Tortoise ORM 向量记忆基类 (Mixin)。
    为第三方插件提供关系型数据库的记忆持久化能力，并在应用层实现余弦降级搜索。
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
    """
    基于 Tortoise ORM 的混合存储后端。
    利用 SQL `LIKE` 进行前缀过滤，利用纯 Python `cosine_similarity` 进行向量计算。
    完美兼顾了轻量级用户的无门槛部署和高级数据隔离。
    """

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
        query_embedding: list[float],
        scope_prefix: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        query = self.model_class.all()
        if scope_prefix:
            query = query.filter(scope__startswith=scope_prefix)

        rows = await query
        results = []
        for row in rows:
            if metadata_filter:
                row_meta = row.meta_data if isinstance(row.meta_data, dict) else {}
                if not all(row_meta.get(k) == v for k, v in metadata_filter.items()):
                    continue

            if not isinstance(row.embedding, list) or not query_embedding:
                results.append((self._to_memory_record(row), 0.1))
            else:
                sim = cosine_similarity(query_embedding, row.embedding)
                results.append((self._to_memory_record(row), sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def delete(
        self,
        scope_prefix: str | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        query = self.model_class.all()
        if scope_prefix:
            query = query.filter(scope__startswith=scope_prefix)
        if record_ids is not None:
            if not record_ids:
                return 0
            query = query.filter(id__in=record_ids)

        return await query.delete()

