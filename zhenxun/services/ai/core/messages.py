"""
消息与响应域类型定义
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Annotated, Any, Generic, Literal, Union, cast
from typing_extensions import Self, TypeVar

from nonebot_plugin_alconna import UniMessage
from pydantic import BaseModel, ConfigDict, Field

from zhenxun.utils.pydantic_compat import (
    model_copy,
    model_dump,
    model_validator,
    parse_as,
)

T = TypeVar("T", bound=BaseModel)

from zhenxun.services.ai.core.models import ToolChoice
from zhenxun.services.ai.core.options import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)


class BaseContentPart(BaseModel):
    """多态消息内容的底层基类"""

    metadata: dict[str, Any] | None = Field(default=None)
    """该部件的内部元数据，提供如 thought_signature、解析结果等非展示用数据"""

    @classmethod
    def text_part(cls, text: str) -> "TextPart":
        """创建一个纯文本片段"""
        return TextPart(text=text)

    @classmethod
    def thought_part(cls, text: str) -> "ThoughtPart":
        """创建一个思维链 (CoT) 思考片段"""
        return ThoughtPart(thought_text=text)

    @classmethod
    def image_url_part(cls, url: str) -> "ImagePart":
        """创建一个基于外网 URL 的图片片段"""
        return ImagePart(url=url)

    @classmethod
    def image_base64_part(cls, data: str, mime_type: str = "image/png") -> "ImagePart":
        """创建一个基于 Base64 编码数据的图片片段"""
        return ImagePart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def audio_url_part(cls, url: str, mime_type: str = "audio/wav") -> "AudioPart":
        """创建一个基于外网 URL 的音频片段"""
        return AudioPart(url=url, mime_type=mime_type)

    @classmethod
    def video_url_part(cls, url: str, mime_type: str = "video/mp4") -> "VideoPart":
        """创建一个基于外网 URL 的视频片段"""
        return VideoPart(url=url, mime_type=mime_type)

    @classmethod
    def video_base64_part(cls, data: str, mime_type: str = "video/mp4") -> "VideoPart":
        """创建一个基于 Base64 编码数据的视频片段"""
        return VideoPart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def audio_base64_part(cls, data: str, mime_type: str = "audio/wav") -> "AudioPart":
        """创建一个基于 Base64 编码数据的音频片段"""
        return AudioPart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def file_uri_part(
        cls,
        file_uri: str,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "FilePart":
        """创建一个基于云端 URI (如 Google Cloud Storage gs://) 的通用文件片段"""
        return FilePart(url=file_uri, mime_type=mime_type, metadata=metadata or {})

    @classmethod
    def tool_call_part(
        cls, id: str, tool_name: str, args: dict[str, Any] | str
    ) -> "ToolCallPart":
        """创建一个工具调用请求片段"""
        return ToolCallPart(id=id, tool_name=tool_name, args=args)

    @classmethod
    def tool_return_part(cls, call_id: str, name: str, result: Any) -> "ToolReturnPart":
        """创建一个工具执行结果片段"""
        return ToolReturnPart(
            tool_call_id=call_id,
            tool_name=name,
            output=result,
        )

    async def get_raw_bytes(self) -> bytes:
        """统一获取多模态原始字节数据 (自动适配 raw、本地 path 或自动下载公网 url)"""
        raw_data = getattr(self, "raw", None)
        if isinstance(raw_data, bytes):
            return raw_data

        path_data = getattr(self, "path", None)
        if path_data is not None:
            from pathlib import Path

            if isinstance(path_data, Path):
                return path_data.read_bytes()

        url_data = getattr(self, "url", None)
        if isinstance(url_data, str):
            from zhenxun.utils.http_utils import AsyncHttpx

            content = await AsyncHttpx.get_content(url_data)
            if isinstance(content, bytes):
                return content
            return b""

        raise ValueError(
            f"{self.__class__.__name__} 未提供有效的数据源 (url, raw, path)"
        )

    async def get_base64_data(self) -> str:
        """统一获取 Base64 编码的字符串数据"""
        return base64.b64encode(await self.get_raw_bytes()).decode("utf-8")

    async def get_data_uri(self, default_mime: str = "application/octet-stream") -> str:
        """统一获取 Data URI (data:mime;base64,...) 格式的字符串"""
        mime = getattr(self, "mime_type", None) or default_mime
        return f"data:{mime};base64,{await self.get_base64_data()}"


class ImagePart(BaseContentPart):
    """图片媒体片段"""

    type: Literal["image"] = "image"
    url: str | None = None
    """外网图片 URL (需能直接访问)"""
    raw: bytes | None = None
    """图片二进制裸数据"""
    path: Path | None = None
    """本地文件系统中的图片路径"""
    mime_type: str | None = None
    """图片 MIME 类型 (如 image/jpeg)"""
    media_resolution: str | None = None
    """媒体处理强制分辨率策略"""

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("ImagePart 必须且只能提供 url, raw, path 中的一个")
        return self


class AudioPart(BaseContentPart):
    """音频媒体片段"""

    type: Literal["audio"] = "audio"
    url: str | None = None
    """外网音频 URL"""
    raw: bytes | None = None
    """音频二进制裸数据"""
    path: Path | None = None
    """本地文件系统中的音频路径"""
    mime_type: str | None = None
    """音频 MIME 类型 (如 audio/mp3)"""

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("AudioPart 必须且只能提供 url, raw, path 中的一个")
        return self


class VideoPart(BaseContentPart):
    """视频媒体片段"""

    type: Literal["video"] = "video"
    url: str | None = None
    """外网视频 URL"""
    raw: bytes | None = None
    """视频二进制裸数据"""
    path: Path | None = None
    """本地文件系统中的视频路径"""
    mime_type: str | None = None
    """视频 MIME 类型 (如 video/mp4)"""

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("VideoPart 必须且只能提供 url, raw, path 中的一个")
        return self


class FilePart(BaseContentPart):
    """通用文件片段 (如 PDF、代码文件等)"""

    type: Literal["file"] = "file"
    url: str | None = None
    """外网文件 URL (或 Google Cloud gs:// URI)"""
    raw: bytes | None = None
    """文件二进制裸数据"""
    path: Path | None = None
    """本地文件系统中的文件路径"""
    mime_type: str | None = None
    """文件 MIME 类型 (如 application/pdf)"""

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("FilePart 必须且只能提供 url, raw, path 中的一个")
        return self


class TextPart(BaseContentPart):
    """基础文本片段"""

    type: Literal["text"] = "text"
    text: str
    """具体的纯文本内容"""


class ThoughtPart(BaseContentPart):
    """思考链 (Chain of Thought) 片段，用于容纳模型的内部思维过程"""

    type: Literal["thought"] = "thought"
    thought_text: str
    """思考过程的文本"""


class ToolCallPart(BaseContentPart):
    """工具调用请求片段 (由模型发出)"""

    type: Literal["tool_call"] = "tool_call"
    id: str
    """工具调用唯一 ID"""
    tool_name: str
    """调用的函数名称"""
    args: dict[str, Any] | str
    """传递给工具的参数 (解析后的字典或原始 JSON 字符串)"""


class ToolReturnPart(BaseContentPart):
    """工具调用结果片段 (发给模型)"""

    type: Literal["tool_return"] = "tool_return"
    tool_call_id: str
    """关联的原始调用 ID"""
    tool_name: str
    """执行的工具名称"""
    output: Any
    """工具执行结果的有效载荷"""


class TextDeltaPart(BaseModel):
    """流式文本增量片段"""

    type: Literal["text_delta"] = "text_delta"
    content_delta: str
    """流式返回的文本增量字符串"""


class ThoughtDeltaPart(BaseModel):
    """流式思考链增量片段"""

    type: Literal["thought_delta"] = "thought_delta"
    content_delta: str
    """流式返回的思考过程增量字符串"""


class EmbedPayload(BaseModel):
    """单一的嵌入载体，包含一个或多个多模态片段（融合向量）"""

    parts: list[LLMContentPart] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """快速提取纯文本（用于向下兼容不支持多模态的模型）"""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart)).strip()

    @property
    def has_multimodal(self) -> bool:
        """判断是否包含图像/音频/视频/文件等非文本模态"""
        return any(not isinstance(p, TextPart) for p in self.parts)


class EmbedBatch(BaseModel):
    """一次嵌入 API 请求的批次载体"""

    payloads: list[EmbedPayload] = Field(default_factory=list)

    def to_text_only(self, context_name: str) -> list[str]:
        """降级工具：将多模态的批量向量安全剔除图片等内容，回退为纯文本数组"""
        from zhenxun.services.log import logger

        texts = []
        for payload in self.payloads:
            if payload.has_multimodal:
                logger.warning(
                    f"⚠️ 模型 {context_name} "
                    "不支持多模态嵌入，已自动剔除富媒体内容，静默降级为纯文本进行向量化..."
                )
            texts.append(payload.text if payload.text else " ")
        return texts


class ToolCallDeltaPart(BaseModel):
    """流式工具调用增量片段"""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    tool_call_id: str | None = None
    """(可选) 流式返回的工具调用唯一ID"""
    tool_name_delta: str | None = None
    """(可选) 流式返回的工具名称增量"""
    args_delta: str | None = None
    """(可选) 流式返回的 JSON 格式参数增量"""


SystemContentUnion = TextPart
"""系统消息允许的内容片段联合类型"""

UserContentUnion = TextPart | ImagePart | AudioPart | VideoPart | FilePart
"""用户消息允许的多模态内容片段联合类型"""

AssistantContentUnion = (
    TextPart
    | ThoughtPart
    | ToolCallPart
    | ToolReturnPart
    | ImagePart
    | AudioPart
    | VideoPart
    | FilePart
)
"""助手回复允许的内容片段联合类型"""

ToolContentUnion = ToolReturnPart | ImagePart | AudioPart | VideoPart | FilePart
"""工具消息允许的内容片段联合类型"""


LLMContentPart = Annotated[
    TextPart
    | ImagePart
    | AudioPart
    | VideoPart
    | FilePart
    | ThoughtPart
    | ToolCallPart
    | ToolReturnPart,
    Field(discriminator="type"),
]
"""大模型底层标准内容片段的 Annotated 联合类型"""


RoleT = TypeVar("RoleT", default=str, covariant=True)
"""泛型：消息参与者角色类型变量"""

ContentT = TypeVar("ContentT", default=LLMContentPart, covariant=True)
"""泛型：多模态片段数组的元素内容类型变量"""


AnyLLMMessage = Annotated[
    Union["SystemMessage", "UserMessage", "AssistantMessage", "ToolMessage"],
    Field(discriminator="role"),
]
"""LLM 消息类型的合集联合类型"""


PromptInput = Union[str, UniMessage, "LLMMessage", list[LLMContentPart], Any]
"""支持作为 LLM 输入的提示词对象联合类型，包括纯文本、UniMessage、LLMMessage 消息实体"""


class LLMMessage(BaseModel, Generic[RoleT, ContentT]):
    """
    LLM 消息基类与门面工厂。
    提供统一的元数据访问、魔法加法重载以及极简实例化方法。
    """

    role: RoleT
    """消息参与者角色 (如 user, assistant, system, tool)"""
    content: list[ContentT] = Field(default_factory=list)
    """容纳实际数据的多模态片段数组"""
    created_at: float = Field(default_factory=time.time)
    """消息最初被构建的 Unix 时间戳"""
    metadata: dict[str, Any] | None = Field(default=None)
    """自由存取字典，供系统内穿透传递额外状态数据"""

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """获取当前消息中包含的所有工具调用请求片段"""
        return [p for p in self.content if isinstance(p, ToolCallPart)]

    @property
    def tool_returns(self) -> list[ToolReturnPart]:
        """获取当前消息中包含的所有工具执行结果片段"""
        return [p for p in self.content if isinstance(p, ToolReturnPart)]

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, data: Any) -> Any:
        """核心拦截：外部传入 str 时自动转为 Part，保持内部类型绝对纯净"""
        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, str):
                data["content"] = (
                    [{"type": "text", "text": content}] if content.strip() else []
                )
            elif content is None:
                data["content"] = []
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, str):
                        new_content.append({"type": "text", "text": item})
                    else:
                        new_content.append(item)
                data["content"] = new_content
            elif not isinstance(content, list):
                import json

                try:
                    text_val = json.dumps(content, ensure_ascii=False)
                except Exception:
                    text_val = str(content)
                data["content"] = [{"type": "text", "text": text_val}]
        return data

    def __add__(self, other: str | LLMContentPart | "LLMMessage") -> Self:
        """极简语法糖：支持通过加号拼接文本或多模态片段。
        示例: msg = LLMMessage.user("查看图片：") + ImagePart(url="...")
        """
        new_msg = cast(Self, model_copy(self, deep=True))
        if isinstance(other, str):
            new_msg.content.append(cast(Any, TextPart(text=other)))
        elif isinstance(other, BaseContentPart):
            new_msg.content.append(cast(Any, other))
        elif isinstance(other, LLMMessage):
            new_msg.content.extend(cast(Any, other.content))
        return new_msg

    @property
    def extract_text(self) -> str:
        """便捷属性：提取当前消息中所有的纯文本"""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def source(self) -> str | None:
        """获取消息的来源标识 (存取于 metadata 中)"""
        return self.metadata.get("source") if self.metadata else None

    @source.setter
    def source(self, value: str | None):
        """设置消息的来源标识"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["source"] = value

    @property
    def source_name(self) -> str | None:
        """获取消息来源的具体名称 (例如真实用户的昵称)"""
        return self.metadata.get("source_name") if self.metadata else None

    @source_name.setter
    def source_name(self, value: str | None):
        """设置消息来源的具体名称"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["source_name"] = value

    @property
    def scope(self) -> str | None:
        """获取该消息所绑定的作用域或会话ID"""
        return self.metadata.get("scope") if self.metadata else None

    @scope.setter
    def scope(self, value: str | None):
        """设置该消息的作用域"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["scope"] = value

    @property
    def thought_signature(self) -> str | None:
        """获取连续对话中用于保持思考一致性的加密签名 (Gemini 专属)"""
        return self.metadata.get("thought_signature") if self.metadata else None

    @thought_signature.setter
    def thought_signature(self, value: str | None):
        """设置思考加密签名"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["thought_signature"] = value

    @property
    def token_cost(self) -> int | None:
        """获取该消息自身消耗的预估或真实 Token 数"""
        return self.metadata.get("token_cost") if self.metadata else None

    @token_cost.setter
    def token_cost(self, value: int | None):
        """设置 Token 消耗数"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata["token_cost"] = value

    @classmethod
    def user(
        cls,
        content: str | Sequence[UserContentUnion | dict[str, Any]],
        source: str | None = None,
        source_name: str | None = None,
        scope: str | None = None,
    ) -> "UserMessage":
        """
        工厂方法：创建一条 User (用户) 角色的消息。

        参数:
            content: 消息内容，支持纯字符串或多模态片段数组。
            source: 可选，消息来源标识。
            source_name: 可选，消息来源的可读名称。
            scope: 可选，消息关联的会话作用域。
        """
        if isinstance(content, str):
            content = [TextPart(text=content)] if content.strip() else []
        elif not isinstance(content, list):
            content = list(content)
        msg = UserMessage(content=cast(Any, content))
        msg.source = source
        msg.source_name = source_name
        msg.scope = scope
        return msg

    @classmethod
    def assistant_tool_calls(
        cls,
        tool_calls: list[ToolCallPart],
        content: str | Sequence[AssistantContentUnion | dict[str, Any]] = "",
        scope: str | None = None,
    ) -> "AssistantMessage":
        """
        工厂方法：创建一条包含工具调用请求的 Assistant (助手) 角色消息。

        参数:
            tool_calls: 大模型发出的工具调用片段列表。
            content: 伴随工具调用的其他文本或思维过程。
            scope: 可选，消息关联的会话作用域。
        """
        if isinstance(content, str):
            _content: list[Any] = [TextPart(text=content)] if content else []
        else:
            _content = list(content)
        _content.extend(tool_calls)
        msg = AssistantMessage(content=cast(Any, _content))
        msg.scope = scope
        return msg

    @classmethod
    def assistant_text_response(
        cls,
        content: str | Sequence[AssistantContentUnion | dict[str, Any]],
        scope: str | None = None,
    ) -> "AssistantMessage":
        """
        工厂方法：创建一条仅包含普通文本/思维过程的 Assistant (助手) 角色消息。

        参数:
            content: 大模型生成的文本回复内容。
            scope: 可选，消息关联的会话作用域。
        """
        if isinstance(content, str):
            content = [TextPart(text=content)] if content and content.strip() else []
        elif not isinstance(content, list):
            content = list(content)
        msg = AssistantMessage(content=cast(Any, content))
        msg.scope = scope
        return msg

    def add_text(self, text: str) -> Self:
        """链式添加文本内容"""
        self.content.append(cast(Any, TextPart(text=text)))
        return self

    def add_image_url(self, url: str) -> Self:
        """链式添加网络图片"""
        self.content.append(cast(Any, ImagePart(url=url)))
        return self

    def add_image_base64(self, b64_data: str, mime_type: str = "image/png") -> Self:
        """链式添加 Base64 图片"""
        self.content.append(
            cast(Any, ImagePart(raw=base64.b64decode(b64_data), mime_type=mime_type))
        )
        return self

    def add_file_url(self, url: str, mime_type: str | None = None) -> Self:
        """链式添加文件链接"""
        self.content.append(cast(Any, FilePart(url=url, mime_type=mime_type)))
        return self

    @classmethod
    def tool_response(
        cls,
        tool_call_id: str,
        function_name: str,
        result: Any,
        scope: str | None = None,
    ) -> "ToolMessage":
        """
        工厂方法：创建一条 Tool (工具) 角色消息，用于承载工具执行完毕后的返回结果。

        参数:
            tool_call_id: 对应的大模型发出调用请求时的 ID。
            function_name: 执行的工具名称。
            result: 工具执行的结果负载 (会被自动 JSON 序列化)。
            scope: 可选，消息关联的会话作用域。
        """
        _content = [
            ToolReturnPart(
                tool_call_id=tool_call_id, tool_name=function_name, output=result
            )
        ]

        msg = ToolMessage(content=cast(Any, _content))
        msg.scope = scope
        return msg

    @classmethod
    def system(
        cls,
        content: str | Sequence[SystemContentUnion | dict[str, Any]],
        scope: str | None = None,
    ) -> "SystemMessage":
        """
        工厂方法：创建一条 System (系统) 角色的设定消息。

        参数:
            content: 系统的 Prompt 指令。
            scope: 可选，消息关联的会话作用域。
        """
        if isinstance(content, str):
            content = [TextPart(text=content)] if content.strip() else []
        elif not isinstance(content, list):
            content = list(content)
        msg = SystemMessage(content=cast(Any, content))
        msg.scope = scope
        return msg

    def to_storage_dict(self) -> dict[str, Any]:
        """数据库瘦身存储，仅保留核心字段"""
        return model_dump(
            self,
            exclude_none=True,
            include={
                "role",
                "content",
                "metadata",
            },
        )


class SystemMessage(LLMMessage[Literal["system"], SystemContentUnion]):
    """系统消息：通常用于在对话开头向模型提供系统级指令 (System Prompt)、
    角色设定或背景上下文。"""

    role: Literal["system"] = "system"
    content: list[Annotated[SystemContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )


class UserMessage(LLMMessage[Literal["user"], UserContentUnion]):
    """用户消息：代表来自最终用户或外部触发源的输入，支持包含文本、图片、文件等多模态数据。"""

    role: Literal["user"] = "user"
    content: list[Annotated[UserContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )


class AssistantMessage(LLMMessage[Literal["assistant"], AssistantContentUnion]):
    """助手消息：代表大模型 (AI) 生成的回复。
    可能包含纯文本、思维链过程或工具调用请求。"""

    role: Literal["assistant"] = "assistant"
    content: list[Annotated[AssistantContentUnion, Field(discriminator="type")]] = (
        Field(default_factory=list)
    )


class ToolMessage(LLMMessage[Literal["tool"], ToolContentUnion]):
    """工具消息：用于承载由用户端执行工具后，将结果返回给大模型的消息容器。"""

    role: Literal["tool"] = "tool"
    content: list[Annotated[ToolContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )

    def model_post_init(self, context: Any, /) -> None:
        """验证消息的有效性"""
        if not self.tool_returns:
            raise ValueError("工具角色的消息必须包含 ToolReturnPart")


class RerankDocument(BaseModel):
    """重排候选文档 (支持纯文本或图文字典)"""

    text: str | None = None
    """被用于重排检索的文本内容"""
    image: str | None = None
    """用于多模态重排的图片内容"""


class RerankResult(BaseModel):
    """重排返回结果"""

    index: int
    """此记录对应于输入时的原始文档数组中的索引位置"""
    relevance_score: float
    """计算出的相关性得分 (越大相关度通常越高)"""
    document: RerankDocument | None = Field(default=None)
    """实际被命中的文档数据"""


class ChatResponse(BaseModel):
    """
    文本/多模态对话响应对象，确立 SSOT (单一数据源) 架构。
    """

    content_parts: list[LLMContentPart] = Field(default_factory=list)
    """经由解析、统一封装后的标准内容部件列表"""
    usage_info: dict[str, Any] | None = None
    """厂商返回的原始用量遥测字段"""
    raw_response: dict[str, Any] | None = None
    """完整的 API 层级原生响应字典 (未经框架清洗)"""
    grounding_metadata: Any | None = None
    """联网搜索、位置等基底事实归因元数据"""
    parsed_obj: Any | None = Field(default=None)
    """在结构化输出模式下，由中间件反序列化得出的强类型 Pydantic 对象实例"""

    def get_parsed_obj(self, model_class: type[T]) -> T | None:
        """
        从结构化生成结果中获取强类型的 Pydantic 解析对象，提供完善的 IDE 类型推导支持。

        参数:
            model_class: 期望提取的 Pydantic 模型类。

        返回:
            强类型的模型实例，如果不存在则返回 None
        """
        if self.parsed_obj is None:
            return None
        if isinstance(self.parsed_obj, model_class):
            return self.parsed_obj

        try:
            return parse_as(model_class, self.parsed_obj)
        except Exception:
            return self.parsed_obj

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.content_parts if isinstance(p, ToolCallPart)]

    @property
    def text(self) -> str:
        """动态视图：提取并拼接所有文本块"""
        return "".join(
            p.text for p in self.content_parts if isinstance(p, TextPart)
        ).strip()

    @property
    def thought_text(self) -> str | None:
        """动态视图：提取并拼接所有思考/推理块"""
        thoughts = [
            p.thought_text for p in self.content_parts if isinstance(p, ThoughtPart)
        ]
        return "\n".join(thoughts).strip() if thoughts else None

    @property
    def thought_signature(self) -> str | None:
        """动态视图：获取当前响应中的思考指纹"""
        for p in reversed(self.content_parts):
            if isinstance(p, ThoughtPart | ToolCallPart | ToolReturnPart):
                if p.metadata and "thought_signature" in p.metadata:
                    return p.metadata["thought_signature"]
        return None

    @property
    def images(self) -> list[bytes | Path | str]:
        """动态视图：提取响应中包含的所有图片数据"""
        imgs = []
        for p in self.content_parts:
            if isinstance(p, ImagePart):
                if p.url:
                    imgs.append(p.url)
                elif p.raw:
                    imgs.append(p.raw)
                elif p.path:
                    imgs.append(p.path)
        return imgs

    @property
    def audios(self) -> list[bytes | Path | str]:
        """动态视图：提取视频数据"""
        audios = []
        for p in self.content_parts:
            if isinstance(p, AudioPart):
                if p.url:
                    audios.append(p.url)
                elif p.raw:
                    audios.append(p.raw)
                elif p.path:
                    audios.append(p.path)
        return audios


@dataclass
class UsageInfo:
    """使用信息数据类"""

    prompt_tokens: int = 0
    """请求发送的 Token 消耗数 (含系统提示词与历史记录)"""
    completion_tokens: int = 0
    """模型回复生成的 Token 消耗数"""
    total_tokens: int = 0
    """本次交互总计产生的 Token 数"""
    cost: float = 0.0
    """(可选) 本次交互产生的实际账单估价"""
    prompt_cache_hit_tokens: int = 0
    """被上下文缓存系统命中的 Prompt Token 数 (往往价格更低)"""
    prompt_cache_miss_tokens: int = 0
    """未能命中缓存、实际执行了计算的 Prompt Token 数"""
    reasoning_tokens: int = 0
    """专门用于内部思考/推理链 (CoT) 消耗的 Token 数"""

    @property
    def efficiency_ratio(self) -> float:
        return self.completion_tokens / max(self.prompt_tokens, 1)

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        """支持 UsageInfo 相加，用于汇聚子智能体的 Token 消耗"""
        if not isinstance(other, UsageInfo):
            return self
        return UsageInfo(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost,
            prompt_cache_hit_tokens=self.prompt_cache_hit_tokens
            + other.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=self.prompt_cache_miss_tokens
            + other.prompt_cache_miss_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class LLMCodeExecution(BaseModel):
    """大模型代码执行（沙箱/本地）结果实体"""

    code: str
    """被执行的原始代码"""
    output: str | None = None
    """标准输出 (stdout)"""
    error: str | None = None
    """标准错误 (stderr) 或框架抛出的异常"""
    execution_time: float | None = None
    """代码执行耗时 (秒)"""
    files_generated: list[str] | None = None
    """代码执行过程中生成的工件(Artifacts)文件路径或名称列表"""


class LLMGroundingAttribution(BaseModel):
    """基础事实溯源引用对象 (Grounding Attribution)"""

    title: str | None = None
    """来源网页或文档的标题"""
    uri: str | None = None
    """来源内容的统一资源标识符 (URL)"""
    snippet: str | None = None
    """从来源网页中提取的、支撑当前生成内容的文本片段"""
    confidence_score: float | None = None
    """该引用来源与生成内容之间相关性的置信度分数"""


class LLMGroundingMetadata(BaseModel):
    """检索增强/搜索引擎溯源 (Grounding) 的完整元数据字典，
    用于为大模型返回的信息提供可信背书"""

    web_search_queries: list[str] | None = None
    """模型在执行检索时，实际使用的底层搜索引擎 Query 查询词列表"""
    grounding_attributions: list[LLMGroundingAttribution] | None = None
    """溯源引用的详情列表，用于在 UI 端构建点击跳转链接或角标"""
    search_suggestions: list[dict[str, Any]] | None = None
    """随搜索返回的相关搜索建议 (Search Suggestions)"""
    search_entry_point: str | None = None
    """一段 HTML/CSS 内容，可用于在客户端渲染标准的搜索引擎入口/建议组件"""
    map_widget_token: str | None = None
    """用于渲染 Google Maps 交互式地点小组件 (Places widget) 的
    上下文 Token (针对 googleMaps 工具)"""


class EmbeddingResponse(BaseModel):
    """Embedding 向量富响应对象"""

    embeddings: list[list[float]]
    """多段输入文本对应生成的高维浮点数向量数组 (二维)"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """执行向量化任务产生的 Token 用量开销统计"""
    model_name: str
    """实际用于执行本次编码任务的模型名称"""

    @property
    def vector(self) -> list[float]:
        """便捷属性：当且仅当只需提取单一文本向量时，直接返回一维向量"""
        return self.embeddings[0] if self.embeddings else []


class AudioResponse(BaseModel):
    """统一的语音合成响应对象"""

    audio_bytes: bytes
    """生成的音频二进制裸数据，可直接用于发送或保存"""
    audio_format: str
    """实际返回的音频格式 (如 mp3, wav)"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """Token 或 字符数消耗统计"""
    raw_response: Any | None = None
    """原生响应体，供高级调试使用"""
    model_name: str
    """实际执行任务的模型名称"""


class ImageResponse(BaseModel):
    """图像生成响应对象"""

    content_parts: list[LLMContentPart] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None

    @property
    def images(self) -> list[bytes | Path | str]:
        """动态视图：提取响应中包含的所有图片数据"""
        imgs = []
        for p in self.content_parts:
            if isinstance(p, ImagePart):
                if p.url:
                    imgs.append(p.url)
                elif p.raw:
                    imgs.append(p.raw)
                elif p.path:
                    imgs.append(p.path)
        return imgs

    @property
    def text(self) -> str:
        """动态视图：提取并拼接所有 TextPart 文本内容"""
        return "".join(
            p.text for p in self.content_parts if isinstance(p, TextPart)
        ).strip()


class RerankResponse(BaseModel):
    """文本重排响应对象"""

    results: list[RerankResult]


class BaseRequest(BaseModel):
    """基础请求 DTO"""

    timeout: float | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_cache_hash_payload(self) -> dict[str, Any]:
        """获取用于计算缓存 Hash 的安全载荷，排除所有运行时动态变量"""
        request_dict = model_dump(self, exclude_none=True)

        if "config" in request_dict and isinstance(request_dict["config"], dict):
            custom_kwargs = request_dict["config"].get("custom_kwargs", {})
            if "__cache_ttl__" in custom_kwargs:
                custom_kwargs.pop("__cache_ttl__")

        if "extra" in request_dict:
            request_dict["extra"] = {
                k: v
                for k, v in request_dict["extra"].items()
                if not k.startswith("_")
                and k not in ("run_context", "output_processor", "guardrails")
            }
        return request_dict


class ChatRequest(BaseRequest):
    """对话生成请求 DTO"""

    messages: list[LLMMessage]
    config: GenerationConfig | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | ToolChoice | None = None

    def get_cache_hash_payload(self) -> dict[str, Any]:
        payload = super().get_cache_hash_payload()
        for msg in payload.get("messages", []):
            msg.pop("created_at", None)
            msg.pop("token_cost", None)
            msg.pop("metadata", None)

        if "tools" in payload and isinstance(payload["tools"], list):
            safe_tools = []
            for t in payload["tools"]:
                if isinstance(t, dict | str):
                    safe_tools.append(t)
                else:
                    safe_tools.append(getattr(t, "name", type(t).__name__))
            payload["tools"] = safe_tools
        return payload


class EmbeddingRequest(BaseRequest):
    """向量嵌入请求 DTO"""

    batch: EmbedBatch
    """向量嵌入的批次载体"""
    config: LLMEmbeddingConfig | None = None
    """向量嵌入配置"""


class ImageRequest(BaseRequest):
    """图像生成请求 DTO"""

    prompt: str
    """图像生成提示词"""
    images: list[Any] | None = None
    """输入参考图像列表"""
    config: GenerationConfig | None = None
    """图像生成配置"""


class SpeechRequest(BaseRequest):
    """语音合成请求 DTO"""

    input_text: str
    """待合成的文本内容"""
    voice: str | None = None
    """发音人/音色标识 (快捷覆盖参数，为空则使用模型默认音色)"""
    config: TTSConfig | None = None
    """语音合成配置"""


class RerankRequest(BaseRequest):
    """文本重排请求 DTO"""

    query: str
    """检索查询词"""
    documents: list[str | dict[str, str]]
    """待排序的候选文档列表"""
    top_n: int = 3
    """返回的最相关文档数量"""


__all__ = [
    "AnyLLMMessage",
    "AssistantMessage",
    "AudioPart",
    "AudioResponse",
    "BaseRequest",
    "ChatRequest",
    "ChatResponse",
    "EmbedBatch",
    "EmbedPayload",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FilePart",
    "ImagePart",
    "ImageRequest",
    "ImageResponse",
    "LLMCodeExecution",
    "LLMContentPart",
    "LLMGroundingAttribution",
    "LLMGroundingMetadata",
    "LLMMessage",
    "PromptInput",
    "RerankDocument",
    "RerankRequest",
    "RerankResponse",
    "RerankResult",
    "SpeechRequest",
    "SystemMessage",
    "TextPart",
    "ThoughtPart",
    "ToolCallPart",
    "ToolMessage",
    "ToolReturnPart",
    "UsageInfo",
    "UserMessage",
    "VideoPart",
]
