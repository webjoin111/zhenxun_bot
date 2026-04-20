"""
记忆域类型定义
"""

import time
from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """单条长期记忆记录实体"""

    id: str = Field(
        default_factory=lambda: __import__("uuid").uuid4().__str__(),
        description="记忆的唯一标识",
    )
    content: str = Field(description="记忆的文本内容")
    scope: str = Field(default="/", description="记忆的作用域(如 user_id, group_id)")
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="重要性评分(0.0-1.0)"
    )
    embedding: list[float] | None = Field(
        default=None, description="向量表示，用于语义相似度搜索"
    )
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class MemoryMatch(BaseModel):
    """召回的记忆匹配结果"""

    record: MemoryRecord = Field(description="匹配到的记忆实体")
    score: float = Field(description="复合相关性得分")
    match_reasons: list[str] = Field(
        default_factory=list, description="匹配原因(如 semantic, recency, importance)"
    )


class MemoryConfig(BaseModel):
    """长期记忆的复合打分与检索配置"""

    recency_weight: float = Field(default=0.3, description="时间衰减权重")
    semantic_weight: float = Field(default=0.5, description="语义相似度权重")
    importance_weight: float = Field(default=0.2, description="重要性权重")
    recency_half_life_days: int = Field(default=30, description="时间衰减的半衰期(天)")


__all__ = [
    "MemoryConfig",
    "MemoryMatch",
    "MemoryRecord",
]
