from enum import Enum
import time
from typing import Any

from pydantic import BaseModel, Field


class SlotScope(str, Enum):
    """记忆槽作用域枚举"""

    GLOBAL = "global"
    SESSION = "session"


class MemorySlot(BaseModel):
    """可编辑的持久化记忆槽 (Mid-Term Memory)"""

    label: str = Field(...)
    """槽位唯一标签标识 (如 persona, preferences)"""
    content: str = Field(default="")
    """槽位存储的具体文本内容"""
    size_limit: int = Field(default=2000)
    """槽位内容的最大字符数限制"""
    pinned: bool = Field(default=True)
    """是否固定注入到大模型的每次系统提示词中"""
    scope: SlotScope = Field(default=SlotScope.SESSION)
    """作用域：全局共享或当前会话私有"""
    created_at: float = Field(default_factory=time.time)
    """创建时间戳"""
    updated_at: float = Field(default_factory=time.time)
    """最近更新时间戳"""


class MemoryIsolationLevel(str, Enum):
    """记忆上下文的隔离级别"""

    GROUP_SHARED = "group_shared"
    """群组共享：单群内所有人共享，跨群不共享 ( /p_xx/g_xx )"""
    USER_GLOBAL = "user_global"
    """用户全局：单用户跨群、跨插件共享 ( /p_xx/u_xx )"""
    GROUP_USER = "group_user"
    """群组用户：单群内的单用户共享 ( /p_xx/g_xx/u_xx )"""
    PLUGIN_USER = "plugin_user"
    """插件级用户隔离：该插件内所有 Agent 共享该用户的记忆 ( /p_xx/g_xx/u_xx/ns_xx )"""
    AGENT_USER = "agent_user"
    """
    Agent级用户隔离：最高级别隔离，
    各 Agent 间绝对物理隔离 ( /p_xx/g_xx/u_xx/ns_xx/ag_xx )
    """


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
    scope_prefix: str = Field(default="/")
    """基于隔离级别生成的路径作用域，用于长期记忆 (RAG) 的向量检索前缀过滤。"""
    accessible_scopes: list[str] = Field(default_factory=lambda: ["/"])
    """
    当前会话有权访问的作用域列表，用于 Slice 联合检索。
    自动推导，包含从全局根路径到最深层路径的所有父节点。
    """

    def __str__(self) -> str:
        return self.session_id
