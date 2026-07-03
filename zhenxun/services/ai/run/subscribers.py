import time
from typing import Any

from nonebot_plugin_alconna import UniMessage

from zhenxun.services.ai.core.messages import BaseContentPart, ImagePart, TextPart
from zhenxun.services.ai.core.stream_events import (
    EventBus,
    LLMEndEvent,
    LLMStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolStreamChunkEvent,
    UserCustomEvent,
)
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunStart, AgentRunSummary
from zhenxun.services.log import logger


class TelemetrySubscriber:
    """纯粹的数据观察者：默默记录时间戳与 Token 消耗"""

    def __init__(self):
        self.summary = AgentRunSummary()
        self._start_times: dict[str, float] = {}

    def attach(self, bus: EventBus):
        bus.subscribe(AgentRunStart, self.on_run_start)
        bus.subscribe(AgentRunEnd, self.on_run_end)
        bus.subscribe(LLMStartEvent, self.on_llm_start)
        bus.subscribe(LLMEndEvent, self.on_llm_end)
        bus.subscribe(ToolCallStartEvent, self.on_tool_start)
        bus.subscribe(ToolCallEndEvent, self.on_tool_end)

    async def on_run_start(self, event: AgentRunStart):
        self._start_times["run"] = time.monotonic()
        logger.debug(f"🚀 [Telemetry] 智能体 {event.agent_name} 开始运行")

    async def on_run_end(self, event: AgentRunEnd):
        dur = (time.monotonic() - self._start_times.get("run", time.monotonic())) * 1000
        self.summary.total_latency_ms = dur
        event.result.telemetry = self.summary
        logger.debug(f"🏁 [Telemetry] 智能体运行结束 (总耗时: {dur:.2f}ms)")

    async def on_llm_start(self, event: LLMStartEvent):
        self._start_times["llm"] = time.monotonic()

    async def on_llm_end(self, event: LLMEndEvent):
        dur = (time.monotonic() - self._start_times.pop("llm", time.monotonic())) * 1000
        self.summary.chats.total += 1
        self.summary.chats.total_latency_ms += dur

        response = event.response
        if response:
            stop_reason = "tool_calls" if response.tool_calls else "stop"
            self.summary.chats.by_stop_reason[stop_reason] = (
                self.summary.chats.by_stop_reason.get(stop_reason, 0) + 1
            )
        logger.debug(f"🧠 [Telemetry] 模型调用完成 (耗时: {dur:.2f}ms)")

    async def on_tool_start(self, event: ToolCallStartEvent):
        self._start_times[f"tool_{event.tool_name}"] = time.monotonic()

    async def on_tool_end(self, event: ToolCallEndEvent):
        dur = (
            time.monotonic()
            - self._start_times.pop(f"tool_{event.tool_name}", time.monotonic())
        ) * 1000
        self.summary.tools.total += 1
        self.summary.tools.total_latency_ms += dur

        tool_stat = self.summary.tools.by_name.setdefault(
            event.tool_name, {"total": 0, "ok": 0, "error": 0, "latency_ms": 0.0}
        )
        tool_stat["total"] += 1
        tool_stat["latency_ms"] += dur

        if event.is_error:
            self.summary.tools.error += 1
            tool_stat["error"] += 1
        else:
            self.summary.tools.ok += 1
            tool_stat["ok"] += 1
        logger.debug(
            f"🛠️ [Telemetry] 工具 {event.tool_name} 执行完毕 (耗时: {dur:.2f}ms)"
        )


class DefaultUISubscriber:
    """纯粹的 UI 观察者：负责将特定事件转化为群聊气泡发送"""

    def __init__(
        self, context: RunContext, reply_to: bool = False, verbose: bool = False
    ):
        self.context = context
        self.reply_to = reply_to
        self.verbose = verbose
        self.bot = context.get_bot()
        self.event = context.get_event()

    def attach(self, bus: EventBus):
        bus.subscribe(ToolStreamChunkEvent, self.on_tool_stream)
        bus.subscribe(UserCustomEvent, self.on_custom_event)

    async def _send_to_platform(self, display: Any):
        if not self.bot or not self.event or not display:
            return

        if (
            isinstance(display, list)
            and len(display) > 0
            and isinstance(display[0], BaseContentPart)
        ):
            msg = UniMessage()
            for part in display:
                if isinstance(part, TextPart) and part.text:
                    msg = msg.text(part.text)
                elif isinstance(part, ImagePart):
                    if part.raw:
                        msg = msg.image(raw=part.raw)
                    elif part.url:
                        msg = msg.image(url=part.url)
                    elif part.path:
                        msg = msg.image(path=part.path)
            display = msg

        if isinstance(display, UniMessage):
            await display.send(self.event, bot=self.bot, reply_to=self.reply_to)
        else:
            await self.bot.send(self.event, str(display))

    async def on_tool_stream(self, event: ToolStreamChunkEvent):
        if self.verbose and event.content:
            await self._send_to_platform(event.content)

    async def on_custom_event(self, event: UserCustomEvent):
        await self._send_to_platform(event.display)
