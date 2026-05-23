from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.models import (
    MemoryMatch,
    MemoryRecord,
    SessionMetadata,
)


class BaseChatContext(ABC):
    """短期对话历史记忆接口 (取代原 WorkingMemory 和 MessageStore)"""

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



class MemoryRetriever(Protocol):
    """记忆检索引擎协议。负责根据查询意图从底层存储中召回记忆。"""

    async def retrieve(
        self,
        session: SessionMetadata,
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryMatch]:
        """根据用户输入，检索相关记忆。"""
        ...
