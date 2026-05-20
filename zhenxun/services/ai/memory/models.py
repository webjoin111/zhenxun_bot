"""
记忆域类型定义
"""

from enum import Enum
import time
from typing import Any, Literal

from nonebot.adapters import Bot, Event
from pydantic import BaseModel, ConfigDict, Field


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
    scope_prefix: str = Field(default="/")
    """基于隔离级别生成的路径作用域，用于长期记忆 (RAG) 的向量检索前缀过滤。"""

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

    if isolation_level in (
        MemoryIsolationLevel.PLUGIN_USER,
        MemoryIsolationLevel.AGENT_USER,
    ):
        parts.append(f"ns_{namespace or 'unknown'}")

    if isolation_level == MemoryIsolationLevel.AGENT_USER:
        parts.append(f"ag_{agent_name or 'unknown'}")

    session_id = "/" + "/".join(parts)
    scope_prefix = session_id

    return SessionMetadata(
        session_id=session_id,
        scope_prefix=scope_prefix,
        platform=platform,
        group_id=group_id,
        user_id=user_id,
        namespace=namespace,
        agent_name=agent_name,
        isolation_level=isolation_level,
    )


class MemoryQuery(BaseModel):
    """长记忆泛化查询对象"""

    text: str = Field(description="原始查询文本")
    embedding: list[float] | None = Field(
        default=None, description="用于向量检索的 Embedding 数组"
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None, description="元数据过滤条件"
    )
    limit: int = Field(default=10, description="返回的最大条数")


class MemoryRecord(BaseModel):
    """单条长期记忆记录实体"""

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().__str__())
    """记忆的唯一标识"""
    content: str = Field(...)
    """记忆的文本内容"""
    scope: str = Field(default="/")
    """记忆的作用域(如 user_id, group_id)"""
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    """重要性评分(0.0-1.0)"""
    embedding: list[float] | None = Field(default=None)
    """向量表示，用于语义相似度搜索"""
    created_at: float = Field(default_factory=time.time)
    """创建时间戳"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """附加元数据"""


class MemoryMatch(BaseModel):
    """召回的记忆匹配结果"""

    record: MemoryRecord = Field(...)
    """匹配到的记忆实体"""
    score: float = Field(...)
    """复合相关性得分"""
    match_reasons: list[str] = Field(default_factory=list)
    """匹配原因(如 semantic, recency, importance)"""


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


class ConsolidationAction(BaseModel):
    """记忆整合单步动作"""

    action: Literal["keep", "update", "delete"] = Field(
        description="对旧记忆执行的动作"
    )
    record_id: str = Field(description="目标旧记忆的 ID")
    new_content: str | None = Field(
        default=None, description="更新后的文本内容（仅在 update 时需要）"
    )


class ConsolidationPlan(BaseModel):
    """整体记忆整合计划"""

    actions: list[ConsolidationAction] = Field(
        default_factory=list, description="对历史记录的操作列表"
    )
    insert_new: bool = Field(
        default=True, description="是否将当前的新内容作为一条独立记忆插入数据库"
    )


class MemoryConfig(BaseModel):
    """统一的记忆配置项声明 (Declarative Memory Config)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enable_short_term: bool = Field(default=True)
    """是否启用短期对话记忆上下文"""
    long_term_scope: str | None = Field(default=None)
    """长期记忆的独立作用域前缀，为 None 则不启用长期向量记忆"""

    chat_backend: str | None = Field(default=None)
    """指定底层短期记忆数据库注册名称，为空则使用全局默认"""
    ltm_backend: str | None = Field(default=None)
    """指定底层长期向量数据库注册名称，为空则使用全局默认"""

    isolation_level: MemoryIsolationLevel = Field(
        default=MemoryIsolationLevel.GROUP_USER
    )
    """记忆隔离级别"""

    context_threshold: float | None = Field(default=None)
    """(局部重写) 触发记忆压缩的 Token 阈值"""
    max_history_turns: int | None = Field(default=None)
    """(局部重写) 触发记忆压缩的对话轮数上限"""

    vision_window: int | None = Field(default=None)
    """多模态滑动窗口大小。0表示关闭该功能，>0表示仅保留最近N轮包含多模态数据的消息，None表示跟随全局配置。"""

    policy: list[Any] | None = Field(default=None)
    """核心记忆压缩策略管线 (List[BaseMemoryReducer])。为 None 时将应用全局默认策略。"""


__all__ = [
    "ConsolidationAction",
    "ConsolidationPlan",
    "MemoryIsolationLevel",
    "MemoryMatch",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScoringConfig",
    "MemoryConfig",
    "SessionMetadata",
    "generate_session_meta",
]
