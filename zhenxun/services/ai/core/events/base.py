import time
from typing import Any
import uuid

from pydantic import BaseModel, Field


class AIEvent(BaseModel):
    """
    真寻 AI 系统事件基类。
    所有生命周期事件（工具调用、模型生成等）均继承自此。
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    source: Any = Field(
        default=None, description="触发此事件的对象来源(如 Agent, AgentExecutor)"
    )
    session_id: str | None = Field(default=None, description="用于追踪链路的会话 ID")
    namespace: str = Field(default="global", description="触发事件的插件命名空间")

    class Config:
        arbitrary_types_allowed = True
        frozen = True  # Pydantic 锁死属性：事件产生后绝对只读
