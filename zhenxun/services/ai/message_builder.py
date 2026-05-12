from collections.abc import Awaitable, Callable
from io import BytesIO
import mimetypes
from pathlib import Path
from typing import Any, TypeVar, cast

from nonebot.adapters import Message as PlatformMessage
from nonebot_plugin_alconna.uniseg import (
    Image,
    Segment,
    Text,
    UniMessage,
    Video,
    Voice,
)
from PIL.Image import Image as PILImageType

from zhenxun.services.ai.core.messages import (
    AudioPart,
    BaseContentPart,
    FilePart,
    ImagePart,
    LLMContentPart,
    LLMMessage,
    SystemMessage,
    TextPart,
    UserContentUnion,
    VideoPart,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump

S = TypeVar("S", bound=Segment)


class MessageBuilder:
    """
    平台消息与 LLM 内部结构转换的 Builder 门面。
    实现 Nonebot/Alconna 生态与底层 LLM 核心（types/messages）的绝对解耦。
    """

    _MESSAGE_CONVERTERS: dict[type, Callable[[Any], Awaitable[list[LLMMessage]]]] = {}

    _SEGMENT_HANDLERS: dict[
        type[Segment], Callable[[Any], Awaitable[LLMContentPart | None]]
    ] = {}

    @classmethod
    def register_segment_handler(cls, seg_type: type[S]):
        """装饰器：注册 Uniseg 消息段的处理器"""

        def decorator(func: Callable[[S], Awaitable[LLMContentPart | None]]):
            cls._SEGMENT_HANDLERS[seg_type] = func
            return func

        return decorator

    @classmethod
    def register_message_converter(cls, msg_type: type):
        """装饰器：注册全局消息体类型的转换器"""

        def decorator(func: Callable[[Any], Awaitable[list[LLMMessage]]]):
            cls._MESSAGE_CONVERTERS[msg_type] = func
            return func

        return decorator

    @classmethod
    async def content_part_from_path(
        cls, path_like: str | Path, target_api: str | None = None
    ) -> LLMContentPart | None:
        """将本地路径读取为多态消息部件"""
        try:
            import anyio

            aio_path = anyio.Path(path_like)
            if not await aio_path.exists() or not await aio_path.is_file():
                logger.warning(f"文件不存在或不是一个文件: {path_like}")
                return None

            from pathlib import Path as StdPath

            std_path = StdPath(path_like)
            resolved_aio_path = await aio_path.absolute()
            _ = resolved_aio_path

            mime_type, _ = mimetypes.guess_type(str(std_path))
            file_name = std_path.name

            if not mime_type:
                logger.warning(
                    f"无法猜测文件 {file_name} 的MIME类型，尝试作为文本处理。"
                )
                try:
                    async with await anyio.open_file(aio_path, encoding="utf-8") as f:
                        text_content = await f.read()
                    return TextPart(text=text_content)
                except Exception as e:
                    logger.error(f"读取文本文件 {file_name} 失败: {e}")
                    return None

            if mime_type.startswith("image/"):
                return ImagePart(path=std_path, mime_type=mime_type)
            elif mime_type.startswith("audio/"):
                return AudioPart(path=std_path, mime_type=mime_type)
            elif mime_type.startswith("video/"):
                return VideoPart(path=std_path, mime_type=mime_type)
            elif mime_type.startswith("text/") or mime_type in (
                "application/json",
                "application/xml",
            ):
                try:
                    async with await anyio.open_file(aio_path, encoding="utf-8") as f:
                        text_content = await f.read()
                    return TextPart(text=text_content)
                except Exception as e:
                    logger.error(f"读取文本类文件 {file_name} 失败: {e}")
                    return None
            else:
                return FilePart(
                    path=std_path,
                    mime_type=mime_type,
                    metadata={"name": file_name, "source": "local_path"},
                )
        except Exception as e:
            logger.error(f"从路径 {path_like} 创建 ContentPart 时出错: {e}")
            return None

    @classmethod
    async def _transform_to_content_part(cls, item: Any) -> UserContentUnion:
        if isinstance(item, BaseContentPart):
            from pydantic import TypeAdapter

            return TypeAdapter(UserContentUnion).validate_python(model_dump(item))
        if isinstance(item, str):
            return TextPart(text=item)
        if isinstance(item, Path):
            part = await cls.content_part_from_path(item)
            if part is None:
                raise ValueError(f"无法从路径加载内容: {item}")
            return cast(UserContentUnion, part)
        if isinstance(item, dict):
            from pydantic import TypeAdapter

            return TypeAdapter(UserContentUnion).validate_python(item)
        if PILImageType and isinstance(item, PILImageType):
            buffer = BytesIO()
            fmt = item.format or "PNG"
            item.save(buffer, format=fmt)
            mime_type = f"image/{fmt.lower()}"
            return ImagePart(raw=buffer.getvalue(), mime_type=mime_type)
        raise TypeError(f"不支持的输入类型用于构建 ContentPart: {type(item)}")

    @classmethod
    async def unimsg_to_llm_parts(cls, message: UniMessage) -> list[UserContentUnion]:
        parts: list[UserContentUnion] = []
        for seg in message:
            handler = cls._SEGMENT_HANDLERS.get(type(seg))
            if handler:
                try:
                    part = await handler(seg)
                    if part:
                        parts.append(cast(UserContentUnion, part))
                except Exception as e:
                    logger.warning(f"处理消息段 {seg} 失败: {e}", "LLMUtils")
        return parts

    @classmethod
    async def normalize_to_llm_messages(
        cls,
        message: str | Any | LLMMessage | list[Any],
        instruction: str | None = None,
    ) -> list[LLMMessage]:
        messages = []
        if instruction:
            messages.append(SystemMessage(content=[TextPart(text=instruction)]))

        for msg_type, converter in cls._MESSAGE_CONVERTERS.items():
            if isinstance(message, msg_type):
                converted_msgs = await converter(message)
                messages.extend(converted_msgs)
                return messages

        if isinstance(message, LLMMessage):
            messages.append(message)
        elif isinstance(message, list) and all(
            isinstance(m, LLMMessage) for m in message
        ):
            messages.extend(message)
        elif isinstance(message, str):
            messages.append(LLMMessage.user(message))
        elif isinstance(message, list):
            parts = []
            for item in message:
                parts.append(await cls._transform_to_content_part(item))
            messages.append(LLMMessage.user(parts))
        else:
            raise TypeError(f"不支持的消息类型: {type(message)}")
        return messages

    @classmethod
    def create_multimodal_message(
        cls,
        text: str | None = None,
        images: list[str | Path | bytes] | str | Path | bytes | None = None,
        videos: list[str | Path | bytes] | str | Path | bytes | None = None,
        audios: list[str | Path | bytes] | str | Path | bytes | None = None,
    ) -> UniMessage:
        message = UniMessage()
        if text:
            message.append(Text(text))
        if images is not None:
            cls._add_media(message, images, Image)
        if videos is not None:
            cls._add_media(message, videos, Video)
        if audios is not None:
            cls._add_media(message, audios, Voice)
        return message

    @classmethod
    def _add_media(cls, message: UniMessage, items: Any, media_class: type) -> None:
        items_list = items if isinstance(items, list) else [items]
        for item in items_list:
            if isinstance(item, str | Path):
                if str(item).startswith(("http://", "https://")):
                    message.append(media_class(url=str(item)))
                else:
                    message.append(media_class(path=Path(item)))
            elif isinstance(item, bytes):
                message.append(media_class(raw=item))

    @classmethod
    def message_to_unimessage(cls, message: PlatformMessage) -> UniMessage:
        return UniMessage.of(message)


@MessageBuilder.register_message_converter(UniMessage)
async def _convert_unimessage(msg: UniMessage) -> list[LLMMessage]:
    content_parts = await MessageBuilder.unimsg_to_llm_parts(msg)
    return [LLMMessage.user(content_parts)]


@MessageBuilder.register_segment_handler(Text)
async def _handle_text(seg: Text) -> TextPart | None:
    return TextPart(text=seg.text) if seg.text.strip() else None


@MessageBuilder.register_segment_handler(Image)
async def _handle_image(seg: Image) -> ImagePart | None:
    mime_type = getattr(seg, "mimetype", None) or "image/png"
    if hasattr(seg, "raw") and seg.raw:
        return ImagePart(
            raw=seg.raw if isinstance(seg.raw, bytes) else seg.raw.read(),
            mime_type=mime_type,
        )
    elif getattr(seg, "path", None) is not None:
        return ImagePart(path=Path(str(seg.path)), mime_type=mime_type)

    url = getattr(seg, "url", None)
    if url:
        return ImagePart(url=url, mime_type=mime_type)
    return None

