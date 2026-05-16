from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.models import MemoryRecord, SessionMetadata


class BaseMessageStore(ABC):
    """底层存储接口"""

    @abstractmethod
    async def get_messages(self, session: SessionMetadata) -> list[LLMMessage]: ...

    @abstractmethod
    async def search(
        self, query: str, session: SessionMetadata, limit: int = 10
    ) -> list[LLMMessage]: ...

    @abstractmethod
    async def add_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None: ...

    @abstractmethod
    async def set_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None: ...

    @abstractmethod
    async def clear(self, session: SessionMetadata) -> None: ...


class BaseWorkingMemory(ABC):
    """短期工作记忆系统逻辑基类"""

    @abstractmethod
    async def get_history(self, session: SessionMetadata) -> list[LLMMessage]: ...

    @abstractmethod
    async def add_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None: ...

    @abstractmethod
    async def clear_history(self, session: SessionMetadata) -> None: ...

    @abstractmethod
    async def set_history(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None: ...


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
        query_embedding: list[float],
        scope_prefix: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoryRecord, float]]:
        """在指定的前缀作用域内，检索与 query_embedding 相似的记忆。
        返回元组列表：(记忆记录, 相似度得分0-1)
        """
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
        target_tokens: int,
        current_tokens: int,
        model_name: str,
        base_overhead: int = 0,
    ) -> tuple[list[LLMMessage], bool, int]: ...
