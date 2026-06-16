from typing import TYPE_CHECKING

from nonebot.adapters import Bot, Event
import numpy as np

if TYPE_CHECKING:
    from zhenxun.services.ai.context.memory.types import (
        MemoryIsolationLevel,
        SessionMetadata,
    )


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """使用 numpy 计算两组向量的余弦相似度"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def generate_session_meta(
    bot: Bot,
    event: Event,
    isolation_level: "MemoryIsolationLevel | None" = None,
    prefix: str = "",
    namespace: str | None = None,
    agent_name: str | None = None,
) -> "SessionMetadata":
    """根据事件和隔离级别，自动提取生成基于路径作用域 (Scope Path) 的 SessionMetadata"""
    from nonebot_plugin_session import extract_session

    from zhenxun.services.ai.context.memory.types import (
        MemoryIsolationLevel,
        MemoryQuery,
        SessionMetadata,
    )

    if isolation_level is None:
        isolation_level = MemoryIsolationLevel.AGENT_USER

    session = extract_session(bot, event)
    platform = session.platform
    user_id = session.id1
    group_id = session.id2 or session.id3

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

    query = MemoryQuery(
        base_prefix=prefix,
        platform=platform,
        group_id=group_id if use_group else None,
        user_id=user_id if use_user else None,
        namespace=namespace
        if isolation_level
        in (MemoryIsolationLevel.PLUGIN_USER, MemoryIsolationLevel.AGENT_USER)
        else None,
        agent_name=agent_name
        if isolation_level == MemoryIsolationLevel.AGENT_USER
        else None,
    )

    parts = query.get_scope_parts()
    session_id = query.scope_prefix
    scope_prefix = query.scope_prefix

    accessible_scopes = ["/"]
    current_path = ""
    for part in parts:
        current_path += f"/{part}"
        accessible_scopes.append(current_path)

    return SessionMetadata(
        session_id=session_id,
        scope_prefix=scope_prefix,
        accessible_scopes=accessible_scopes,
        platform=platform,
        group_id=group_id,
        user_id=user_id,
        namespace=namespace,
        agent_name=agent_name,
        isolation_level=isolation_level,
    )
