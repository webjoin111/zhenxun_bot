"""
记忆域类型定义
"""

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.context.memory.interfaces import (
    BaseChatContext,
    BaseMemoryIngestionMiddleware,
    BaseMemoryReducer,
    BaseSlotContext,
)
from zhenxun.services.ai.context.memory.types import (
    MemoryIsolationLevel,
    MemorySlot,
    SessionMetadata,
)
from zhenxun.services.ai.context.rag.backends import Embedder, StorageBackend
from zhenxun.services.ai.context.rag.consolidation import Consolidator


class SlotMemoryConfig(BaseModel):
    """槽位记忆 (Memory Slots) 配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable: bool = Field(default=False)
    """是否启用中期记忆槽"""
    default_slots: list[MemorySlot] = Field(default_factory=list)
    """首次初始化时自动写入的默认槽位列表"""
    backend: str | BaseSlotContext | None = Field(default=None)
    """
    指定底层槽位记忆数据库注册名称，或直接传入 BaseSlotContext 实例。
    为空则使用全局默认
    """


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
    reinforcement_weight: float = Field(default=0.2)
    """访问强化的加权权重 (被检索越多得分越高)"""
    capacity_limit: int = Field(default=500)
    """单用户/群组长期记忆容量软上限，超载后触发惰性清理"""
    evict_ratio: float = Field(default=0.2)
    """触发容量上限后，淘汰冷数据的比例"""


class ShortTermConfig(BaseModel):
    """短期对话记忆配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable: bool = Field(default=True)
    """是否启用短期对话记忆上下文"""
    backend: str | BaseChatContext | None = Field(default=None)
    """
    指定底层短期记忆数据库注册名称，或直接传入 BaseChatContext 实例。
    为空则使用全局默认
    """
    isolation_level: MemoryIsolationLevel = Field(
        default=MemoryIsolationLevel.AGENT_USER
    )
    """记忆隔离级别"""


class LongTermConfig(BaseModel):
    """长期向量记忆配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable: bool = Field(default=False)
    """是否启用长期记忆（开启后自动赋予 Agent 存取记忆的工具，并附加 RAG 召回能力）"""
    backend: str | StorageBackend | None = Field(default=None)
    """
    指定底层长期向量数据库 (Storage) 注册名称，或直接传入 StorageBackend 实例。
    为空则使用全局默认
    """
    scope: str | None = Field(default=None)
    """长期记忆的独立作用域前缀，为 None 则不启用长期向量记忆"""
    embedder: str | Embedder | None = Field(default=None)
    """
    指定底层向量化引擎 (Embedder) 实例，若为字符串则视为 API 模型名称。
    为空则使用全局默认
    """
    async_write: bool = Field(default=True)
    """是否开启长期记忆后台异步写入队列防阻塞"""
    auto_consolidate: bool = Field(default=True)
    """是否开启大模型记忆反思与融合"""
    consolidator: str | Consolidator | None = Field(default=None)
    """
    指定底层记忆融合器 (Consolidator) 注册名称，或直接传入 Consolidator 实例。
    为空则使用全局默认（不融合，仅追加）
    """


class ContextCompressionConfig(BaseModel):
    """上下文压缩与管理配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float | None = Field(default=None)
    """(局部重写) 触发记忆压缩的 Token 阈值"""
    max_history_turns: int | None = Field(default=None)
    """(局部重写) 触发记忆压缩的对话轮数上限。设为 0 表示不限制轮数。"""
    vision_window: int | None = Field(default=None)
    """多模态滑动窗口大小。0表示关闭该功能，>0表示仅保留最近N轮包含多模态数据的消息，None表示跟随全局配置。"""
    policy: list[BaseMemoryReducer] | None = Field(default=None)
    """核心记忆压缩策略管线 (List[BaseMemoryReducer])。为 None 时将应用全局默认策略。"""


class IngestionConfig(BaseModel):
    """记忆入库管线配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    middlewares: list[BaseMemoryIngestionMiddleware] = Field(default_factory=list)
    """入库中间件列表（按顺序依次执行清洗过滤）"""


class MemoryConfig(BaseModel):
    """统一的记忆配置项声明 (Declarative Memory Config)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    """短期对话记忆配置"""
    slots: SlotMemoryConfig = Field(default_factory=SlotMemoryConfig)
    """槽位记忆配置"""
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    """长期向量记忆配置"""
    compression: ContextCompressionConfig = Field(
        default_factory=ContextCompressionConfig
    )
    """上下文压缩与管理配置"""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    """记忆入库前的清洗与过滤管线配置"""


__all__ = [
    "BaseMemoryIngestionMiddleware",
    "ContextCompressionConfig",
    "IngestionConfig",
    "LongTermConfig",
    "MemoryConfig",
    "MemoryIsolationLevel",
    "MemoryScoringConfig",
    "SessionMetadata",
    "ShortTermConfig",
]
