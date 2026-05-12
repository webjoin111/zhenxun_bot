from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentStreamEvent(BaseModel):
    """局部流事件基类"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolCallStart(AgentStreamEvent):
    """工具调用开始"""

    tool_name: str
    arguments: dict[str, Any]


class ToolStreamChunk(AgentStreamEvent):
    """工具流式反馈 (等同于原 ctx.emit)"""

    tool_name: str
    content: str
    metadata: dict[str, Any] | None = None


class ToolCallResultEvent(AgentStreamEvent):
    """工具调用结束"""

    tool_name: str
    result: Any
    is_error: bool


class EventStreamer:
    """基于队列的局部事件收集器"""

    def __init__(self):
        self._queue = asyncio.Queue()
        self._finished = False

    async def send(self, event: AgentStreamEvent):
        if not self._finished:
            await self._queue.put(event)

    async def end(self):
        self._finished = True
        await self._queue.put(None)

    async def __aiter__(self):
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
