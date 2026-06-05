from abc import ABC, abstractmethod

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.types import MemorySlot, SessionMetadata


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


class BaseSlotContext(ABC):
    """中期记忆槽持久化接口"""

    @abstractmethod
    async def get_slot(
        self, session: SessionMetadata, label: str
    ) -> MemorySlot | None: ...

    @abstractmethod
    async def set_slot(self, session: SessionMetadata, slot: MemorySlot) -> None: ...

    @abstractmethod
    async def delete_slot(
        self, session: SessionMetadata, label: str, scope: str
    ) -> None: ...

    @abstractmethod
    async def list_pinned_slots(self, session: SessionMetadata) -> list[MemorySlot]: ...


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
