from typing import Any

from zhenxun.services.ai.memory.utils import cosine_similarity
from zhenxun.services.ai.protocols.memory import StorageBackend
from zhenxun.services.ai.types.memory import MemoryRecord


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
        query_embedding: list[float],
        scope_prefix: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        results = []
        for record in self._records.values():
            if scope_prefix and not record.scope.startswith(scope_prefix):
                continue

            if metadata_filter:
                if not record.metadata or not all(
                    record.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue

            if not record.embedding or not query_embedding:
                results.append((record, 0.1))
            else:
                sim = cosine_similarity(query_embedding, record.embedding)
                results.append((record, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

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
