"""
会话记忆协议定义
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from nonebot.adapters import Bot, Event
from pydantic import BaseModel, Field

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.models import MemoryRecord


class MemoryIsolationLevel(str, Enum):
    """记忆上下文的隔离级别"""

    GLOBAL = "global"
    """全局共享：所有用户、所有群聊、所有插件看同一个记忆（极少使用）"""
    GROUP_SHARED = "group_shared"
    """群组共享：单群内所有人共享，跨群不共享 ( /p_xx/g_xx )"""
    USER_GLOBAL = "user_global"
    """用户全局：单用户跨群、跨插件共享 ( /p_xx/u_xx )"""
    GROUP_USER = "group_user"
    """群组用户：单群内的单用户共享 ( /p_xx/g_xx/u_xx )"""
    PLUGIN_USER = "plugin_user"
    """插件级用户隔离：该插件内所有 Agent 共享该用户的记忆 ( /p_xx/g_xx/u_xx/ns_xx )"""
    AGENT_USER = "agent_user"
    """Agent级用户隔离：最高级别隔离，各 Agent 间绝对物理隔离 ( /p_xx/g_xx/u_xx/ns_xx/ag_xx )"""


class SessionMetadata(BaseModel):
    """结构化会话元数据"""

    session_id: str = Field(...)
    """核心会话标识符。"""
    platform: str | None = Field(default=None)
    """平台标识。"""
    group_id: str | None = Field(default=None)
    """群组/频道 ID。"""
    user_id: str | None = Field(default=None)
    """用户 ID。"""
    namespace: str | None = Field(default=None)
    """插件/命名空间标识。"""
    agent_name: str | None = Field(default=None)
    """具体智能体标识。"""
    isolation_level: MemoryIsolationLevel | None = Field(default=None)
    """生成此会话时的隔离级别。"""

    def __str__(self) -> str:
        return self.session_id


def generate_session_meta(
    bot: Bot,
    event: Event,
    isolation_level: MemoryIsolationLevel = MemoryIsolationLevel.AGENT_USER,
    prefix: str = "",
    namespace: str | None = None,
    agent_name: str | None = None,
) -> SessionMetadata:
    """根据事件和隔离级别，自动提取生成基于路径作用域 (Scope Path) 的 SessionMetadata"""
    from nonebot_plugin_session import extract_session

    session = extract_session(bot, event)
    platform = session.platform
    user_id = session.id1
    group_id = session.id2 or session.id3

    parts = []
    if prefix:
        prefix_clean = prefix.strip("/")
        if prefix_clean:
            parts.append(prefix_clean)

    if platform:
        parts.append(f"p_{platform}")

    use_group = False
    use_user = False

    if isolation_level == MemoryIsolationLevel.GROUP_SHARED:
        use_group = True
    elif isolation_level == MemoryIsolationLevel.USER_GLOBAL:
        use_user = True
    elif isolation_level in (
        MemoryIsolationLevel.GROUP_USER,
        MemoryIsolationLevel.PLUGIN_USER,
        MemoryIsolationLevel.AGENT_USER,
    ):
        use_group = True if group_id else False
        use_user = True

    if use_group and group_id:
        parts.append(f"g_{group_id}")
    if use_user and user_id:
        parts.append(f"u_{user_id}")

    if isolation_level in (MemoryIsolationLevel.PLUGIN_USER, MemoryIsolationLevel.AGENT_USER):
        parts.append(f"ns_{namespace or 'unknown'}")

    if isolation_level == MemoryIsolationLevel.AGENT_USER:
        parts.append(f"ag_{agent_name or 'unknown'}")

    session_id = "/" + "/".join(parts)

    return SessionMetadata(
        session_id=session_id,
        platform=platform,
        group_id=group_id,
        user_id=user_id,
        namespace=namespace,
        agent_name=agent_name,
        isolation_level=isolation_level,
    )


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
