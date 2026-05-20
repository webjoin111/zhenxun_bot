from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.models import (
    ConsolidationPlan,
    MemoryQuery,
    MemoryRecord,
    SessionMetadata,
)


class MemoryConsolidator(Protocol):
    """记忆整合器协议。决定新记忆与相似旧记忆之间的合并、覆盖或删除关系。"""

    async def consolidate(
        self, new_content: str, existing_records: list[MemoryRecord]
    ) -> ConsolidationPlan:
        """分析并返回整合计划"""
        ...


class BaseChatContext(ABC):
    """短期对话历史记忆接口 (取代原 WorkingMemory 和 MessageStore)"""

    @abstractmethod
    async def get_messages(self, session: SessionMetadata) -> list[LLMMessage]: ...

    @abstractmethod
    async def search(self, query: str, session: SessionMetadata, limit: int = 10) -> list[LLMMessage]: ...

    @abstractmethod
    async def add_messages(self, session: SessionMetadata, messages: list[LLMMessage]) -> None: ...

    @abstractmethod
    async def set_messages(self, session: SessionMetadata, messages: list[LLMMessage]) -> None: ...

    @abstractmethod
    async def clear(self, session: SessionMetadata) -> None: ...


@runtime_checkable
class StorageBackend(Protocol):
    """向量与事实存储后端协议。
    允许第三方插件自由决定将记忆存入 SQLite(基于内存降级检索)
    还是专业的 Milvus/ChromaDB。
    """

    async def save(self, records: list[MemoryRecord]) -> None:
        """保存或更新记忆片段集。"""
        ...

    async def search(
        self,
        query: MemoryQuery,
        scope_prefix: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """在指定的前缀作用域内，检索与 query_embedding 相似的记忆。
        返回元组列表：(记忆记录, 相似度得分0-1)
        """
        ...

    async def update(self, record: MemoryRecord) -> None:
        """更新现有的记忆实体"""
        ...

    async def delete(
        self,
        scope_prefix: str | None = None,
        record_ids: list[str] | None = None,
    ) -> int:
        """删除满足条件的记忆，返回被删除的条数。"""
        ...


class BaseMemoryReducer(ABC):
    """记忆压缩器基类"""

    @abstractmethod
    async def reduce(
        self,
        messages: list[LLMMessage],
        current_tokens: int,
        model_name: str,
        base_overhead: int = 0,
    ) -> tuple[list[LLMMessage], bool, int]: ...
