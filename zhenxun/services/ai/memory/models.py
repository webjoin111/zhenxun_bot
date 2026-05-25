"""
记忆域类型定义
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    scope_prefix: str = Field(default="/")
    """基于隔离级别生成的路径作用域，用于长期记忆 (RAG) 的向量检索前缀过滤。"""
    accessible_scopes: list[str] = Field(default_factory=lambda: ["/"])
    """当前会话有权访问的作用域列表，用于 Slice 联合检索。自动推导，包含从全局根路径到最深层路径的所有父节点。"""

    def __str__(self) -> str:
        return self.session_id


class MemoryScoringConfig(BaseModel):
    """长期记忆的复合打分与检索配置"""

    recency_weight: float = Field(default=0.3)
    """时间衰减权重"""
    semantic_weight: float = Field(default=0.5)
    """语义相似度权重"""
    importance_weight: float = Field(default=0.2)
    """重要性权重"""
    recency_half_life_days: int = Field(default=30)
    """时间衰减的半衰期(天)"""
    consolidation_threshold: float = Field(default=0.85)
    """触发记忆整合的相似度阈值 (高于此阈值的旧记忆将参与合并判断)"""


class ShortTermConfig(BaseModel):
    """短期对话记忆配置"""

    enable: bool = Field(default=True)
    """是否启用短期对话记忆上下文"""
    backend: str | Any | None = Field(default=None)
    """指定底层短期记忆数据库注册名称，或直接传入 BaseChatContext 实例。为空则使用全局默认"""
    isolation_level: MemoryIsolationLevel = Field(
        default=MemoryIsolationLevel.AGENT_USER
    )
    """记忆隔离级别"""


class LongTermConfig(BaseModel):
    """长期向量记忆配置"""

    enable: bool = Field(default=False)
    """是否启用长期记忆（开启后自动赋予 Agent 存取记忆的工具，并附加 RAG 召回能力）"""
    backend: str | Any | None = Field(default=None)
    """指定底层长期向量数据库 (Storage) 注册名称，或直接传入 StorageBackend 实例。为空则使用全局默认"""
    scope: str | None = Field(default=None)
    """长期记忆的独立作用域前缀，为 None 则不启用长期向量记忆"""
    embedder: str | Any | None = Field(default=None)
    """指定底层向量化引擎 (Embedder) 实例，若为字符串则视为 API 模型名称。为空则使用全局默认"""
    async_write: bool = Field(default=True)
    """是否开启长期记忆后台异步写入队列防阻塞"""
    auto_consolidate: bool = Field(default=True)
    """是否开启大模型记忆反思与融合"""
    consolidator: str | Any | None = Field(default=None)
    """指定底层记忆融合器 (Consolidator) 注册名称，或直接传入 Consolidator 实例。为空则使用全局默认（不融合，仅追加）"""


class ContextCompressionConfig(BaseModel):
    """上下文压缩与管理配置"""

    threshold: float | None = Field(default=None)
    """(局部重写) 触发记忆压缩的 Token 阈值"""
    max_history_turns: int | None = Field(default=None)
    """(局部重写) 触发记忆压缩的对话轮数上限"""
    vision_window: int | None = Field(default=None)
    """多模态滑动窗口大小。0表示关闭该功能，>0表示仅保留最近N轮包含多模态数据的消息，None表示跟随全局配置。"""
    policy: list[Any] | None = Field(default=None)
    """核心记忆压缩策略管线 (List[BaseMemoryReducer])。为 None 时将应用全局默认策略。"""


class MemoryConfig(BaseModel):
    """统一的记忆配置项声明 (Declarative Memory Config)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    """短期对话记忆配置"""
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    """长期向量记忆配置"""
    compression: ContextCompressionConfig = Field(
        default_factory=ContextCompressionConfig
    )
    """上下文压缩与管理配置"""


__all__ = [
    "ContextCompressionConfig",
    "LongTermConfig",
    "MemoryConfig",
    "MemoryIsolationLevel",
    "MemoryScoringConfig",
    "SessionMetadata",
    "ShortTermConfig",
]
