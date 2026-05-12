from typing import TYPE_CHECKING, Any

from nonebot_plugin_alconna import Image as AlcImage
from nonebot_plugin_alconna import UniMessage

from zhenxun.services.ai.core.stream_events import ToolStreamChunk

if TYPE_CHECKING:
    from zhenxun.services.ai.run.context import RunContext


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
                display_msg += AlcImage(raw=image)
            else:
                display_msg += AlcImage(url=image)
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
            await self._streamer.send(
                ToolStreamChunk(
                    tool_name=self.tool_name,
                    content="",
                    metadata={"display": display},
                )
            )
