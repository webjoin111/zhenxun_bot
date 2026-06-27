from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseStreamEvent(BaseModel):
    """局部流事件基类"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolCallStart(BaseStreamEvent):
    """工具调用开始"""

    tool_name: str
    """调用的工具名称"""
    arguments: dict[str, Any]
    """工具调用参数"""
    intent: str | None = None
    """从大模型调用参数中剥离出的意图 (_intent)"""


class ToolStreamChunk(BaseStreamEvent):
    """工具流式反馈"""

    tool_name: str
    """工具名称"""
    content: str
    """当前流式输出的文本片段"""
    metadata: dict[str, Any] | None = None
    """工具流式的元数据"""


class ToolCallResultEvent(BaseStreamEvent):
    """工具调用结束"""

    tool_name: str
    """工具名称"""
    result: Any
    """工具最终返回的结果"""
    is_error: bool
    """工具执行是否失败"""


class EventStreamer:
    """基于队列的局部事件收集器"""

    def __init__(self):
        self._queue = asyncio.Queue()
        self._finished = False

    async def send(self, event: BaseStreamEvent):
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
