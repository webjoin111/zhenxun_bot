import base64
from io import BytesIO
import json
from typing import Any

from zhenxun.services.ai.types.tools import ToolResult


class ToolResponse:
    """声明式工具响应构建器 (Syntactic Sugar for ToolResult)"""

    @classmethod
    def text(cls, text: str) -> ToolResult:
        return ToolResult(output=text)

    @classmethod
    def data(cls, data_obj: Any) -> ToolResult:
        return ToolResult(output=data_obj)

    @classmethod
    def reply(cls, output: Any, display: Any = None, image: Any = None) -> ToolResult:
        """向大模型返回数据，同时在群聊前端展示交互气泡。支持快捷附带图片。"""
        from nonebot_plugin_alconna import Image as AlcImage
        from nonebot_plugin_alconna import Text, UniMessage

        from zhenxun.services.ai.types.messages import ImagePart, TextPart

        final_display = display or output
        final_output = output

        if image:
            uni_msg = UniMessage()
            if isinstance(final_display, UniMessage):
                uni_msg.extend(final_display)
            else:
                uni_msg += Text(str(final_display))

            img_part = None
            if isinstance(image, str) and image.startswith("base64://"):
                raw_bytes = base64.b64decode(image[9:])
                uni_msg += AlcImage(raw=BytesIO(raw_bytes))
                img_part = ImagePart(raw=raw_bytes)
            elif isinstance(image, str) and image.startswith(("http://", "https://")):
                uni_msg += AlcImage(url=image)
                img_part = ImagePart(url=image)
            elif isinstance(image, bytes):
                uni_msg += AlcImage(raw=image)
                img_part = ImagePart(raw=image)
            else:
                path_obj = __import__("pathlib").Path(image)
                uni_msg += AlcImage(path=path_obj)
                img_part = ImagePart(path=path_obj)

            final_display = uni_msg

            if isinstance(final_output, str):
                final_output = [TextPart(text=final_output), img_part]
            elif isinstance(final_output, list):
                final_output.append(img_part)

        return ToolResult(output=final_output, display=final_display)

    @classmethod
    def error(cls, reason: str, is_retryable: bool = True) -> ToolResult:
        return ToolResult(
            output=json.dumps(
                {
                    "error_type": "ExecutionError",
                    "message": reason,
                    "is_retryable": is_retryable,
                },
                ensure_ascii=False,
            ),
            display=f"❌ 执行失败: {reason}",
            is_error=True,
        )

    @classmethod
    def abort(cls, reason: str) -> ToolResult:
        return ToolResult(
            output=json.dumps(
                {"error_type": "Aborted", "message": reason}, ensure_ascii=False
            ),
            display=f"❌ 操作已中止: {reason}",
            is_error=True,
            terminate_run=True,
        )
