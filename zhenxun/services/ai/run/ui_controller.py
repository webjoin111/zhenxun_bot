from typing import TYPE_CHECKING, Any

from nonebot_plugin_alconna import UniMessage

from zhenxun.services.ai.core.stream_events import ToolStreamChunk

if TYPE_CHECKING:
    from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.core.messages import BaseContentPart, ImagePart, TextPart


class UIController:
    """前端 UI 富交互流式控制器 (按需生成模式)"""

    def __init__(self, context: "RunContext"):
        self.context = context

    @property
    def tool_name(self) -> str:
        """从上下文中动态获取当前调用的工具名"""
        return getattr(self.context.call, "tool_name", "UnknownTool")

    @property
    def _streamer(self) -> Any | None:
        """从运行时上下文中获取底层的事件发射器"""
        return getattr(self.context.run, "streamer", None)

    async def send_text(self, text: str, status: str = "running") -> None:
        """向前端流式反馈执行进度文本"""
        if self._streamer:
            await self._streamer.send(
                ToolStreamChunk(
                    tool_name=self.tool_name, content=text, metadata={"status": status}
                )
            )

    async def send_image(self, image: bytes | str) -> None:
        """向前端发送富文本图片气泡（字节流或 URL）"""
        if self._streamer:
            display_msg = UniMessage()
            if isinstance(image, bytes):
                display_msg = display_msg.image(raw=image)
            else:
                display_msg = display_msg.image(url=image)
            await self._streamer.send(
                ToolStreamChunk(
                    tool_name=self.tool_name,
                    content="[图片生成完毕]",
                    metadata={"display": display_msg},
                )
            )

    async def send_display(self, display: Any) -> None:
        """向前端发送任意展示载体（兼容 ToolResult.ui_display 的行为）"""
        if self._streamer and display is not None:
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

            await self._streamer.send(
                ToolStreamChunk(
                    tool_name=self.tool_name,
                    content="",
                    metadata={"display": display},
                )
            )
