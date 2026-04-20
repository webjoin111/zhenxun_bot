import time
from typing import Any
import uuid

from zhenxun.services.ai.memory.utils import compute_composite_score, join_scope_paths
from zhenxun.services.ai.protocols.memory import StorageBackend
from zhenxun.services.ai.types.memory import MemoryConfig, MemoryMatch, MemoryRecord
from zhenxun.services.log import logger


class MemoryScope:
    """
    长期记忆的作用域视图与 RAG 管线。
    将 StorageBackend(存储) 与 Embedding(模型) 结合，对外提供优雅的面向对象接口。
    """

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
        """内部方法：获取文本的向量。如果未配置模型，则返回空列表触发存储后端的降级机制"""
        if not self.embedding_model or not text.strip():
            return []

        from zhenxun.services.ai.llm.api import embed

        try:
            vectors = await embed([text], model=self.embedding_model)
            return vectors[0] if vectors else []
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
        """将事实存入长期记忆"""
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
        """检索相关的长期记忆（RAG Pipeline）"""
        final_scope = join_scope_paths(self.root_path, inner_scope)
        query_vector = await self._get_embedding(query)

        fetch_limit = limit * 3 if self.rerank_model else limit * 2
        raw_results = await self.storage.search(
            query_vector,
            scope_prefix=final_scope,
            metadata_filter=metadata_filter,
            limit=fetch_limit,
        )

        if self.rerank_model and raw_results:
            from zhenxun.services.ai.llm.api import rerank

            documents_to_rank: list[str | dict[str, str]] = [r.content for r, _ in raw_results]
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
        """遗忘/删除长期记忆"""
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
    """为第三方插件获取标准化的、经过路径隔离的 MemoryScope。"""

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
