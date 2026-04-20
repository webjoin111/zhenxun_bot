"""
消息与响应域类型定义
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Annotated, Any, Generic, Literal, cast
from typing_extensions import Self, TypeVar

from pydantic import BaseModel, Field, model_validator

from zhenxun.utils.pydantic_compat import model_copy, model_dump

T = TypeVar("T", bound=BaseModel)


class ResponseFormat(Enum):
    """响应格式枚举"""

    TEXT = "text"
    JSON = "json"
    MULTIMODAL = "multimodal"


class BaseContentPart(BaseModel):
    """多态消息内容的底层基类"""

    metadata: dict[str, Any] | None = Field(default=None, description="额外元数据")

    @classmethod
    def text_part(cls, text: str) -> "TextPart":
        return TextPart(text=text)

    @classmethod
    def thought_part(cls, text: str) -> "ThoughtPart":
        return ThoughtPart(thought_text=text)

    @classmethod
    def image_url_part(cls, url: str) -> "ImagePart":
        return ImagePart(url=url)

    @classmethod
    def image_base64_part(cls, data: str, mime_type: str = "image/png") -> "ImagePart":
        import base64

        return ImagePart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def audio_url_part(cls, url: str, mime_type: str = "audio/wav") -> "AudioPart":
        return AudioPart(url=url, mime_type=mime_type)

    @classmethod
    def video_url_part(cls, url: str, mime_type: str = "video/mp4") -> "VideoPart":
        return VideoPart(url=url, mime_type=mime_type)

    @classmethod
    def video_base64_part(cls, data: str, mime_type: str = "video/mp4") -> "VideoPart":
        import base64

        return VideoPart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def audio_base64_part(cls, data: str, mime_type: str = "audio/wav") -> "AudioPart":
        import base64

        return AudioPart(raw=base64.b64decode(data), mime_type=mime_type)

    @classmethod
    def file_uri_part(
        cls,
        file_uri: str,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "FilePart":
        return FilePart(url=file_uri, mime_type=mime_type, metadata=metadata or {})

    @classmethod
    def tool_call_part(
        cls, id: str, tool_name: str, args: dict[str, Any] | str
    ) -> "ToolCallPart":
        return ToolCallPart(id=id, tool_name=tool_name, args=args)

    @classmethod
    def tool_return_part(cls, call_id: str, name: str, result: Any) -> "ToolReturnPart":
        return ToolReturnPart(
            tool_call_id=call_id,
            tool_name=name,
            output=result,
        )

    @classmethod
    async def from_path(
        cls, path_like: str | Path, target_api: str | None = None
    ) -> "LLMContentPart | None":
        from zhenxun.services.ai.message_builder import MessageBuilder

        return await MessageBuilder.content_part_from_path(path_like, target_api)


class ImagePart(BaseContentPart):
    type: Literal["image"] = "image"
    url: str | None = None
    raw: bytes | None = None
    path: Path | None = None
    mime_type: str | None = None
    media_resolution: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("ImagePart 必须且只能提供 url, raw, path 中的一个")
        return self


class AudioPart(BaseContentPart):
    type: Literal["audio"] = "audio"
    url: str | None = None
    raw: bytes | None = None
    path: Path | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("AudioPart 必须且只能提供 url, raw, path 中的一个")
        return self


class VideoPart(BaseContentPart):
    type: Literal["video"] = "video"
    url: str | None = None
    raw: bytes | None = None
    path: Path | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("VideoPart 必须且只能提供 url, raw, path 中的一个")
        return self


class FilePart(BaseContentPart):
    type: Literal["file"] = "file"
    url: str | None = None
    raw: bytes | None = None
    path: Path | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        sources = [s for s in (self.url, self.raw, self.path) if s is not None]
        if len(sources) != 1:
            raise ValueError("FilePart 必须且只能提供 url, raw, path 中的一个")
        return self


class TextPart(BaseContentPart):
    type: Literal["text"] = "text"
    text: str


class ThoughtPart(BaseContentPart):
    type: Literal["thought"] = "thought"
    thought_text: str


class ToolCallPart(BaseContentPart):
    type: Literal["tool_call"] = "tool_call"
    id: str
    tool_name: str
    args: dict[str, Any] | str


class ToolReturnPart(BaseContentPart):
    type: Literal["tool_return"] = "tool_return"
    tool_call_id: str
    tool_name: str
    output: Any


class TextDeltaPart(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    content_delta: str


class ThoughtDeltaPart(BaseModel):
    type: Literal["thought_delta"] = "thought_delta"
    content_delta: str


class ToolCallDeltaPart(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    tool_call_id: str | None = None
    tool_name_delta: str | None = None
    args_delta: str | None = None


SystemContentUnion = TextPart
UserContentUnion = TextPart | ImagePart | AudioPart | VideoPart | FilePart
AssistantContentUnion = TextPart | ThoughtPart | ToolCallPart
ToolContentUnion = ToolReturnPart | ImagePart | AudioPart | VideoPart | FilePart


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


RoleT = TypeVar("RoleT", default=str)
ContentT = TypeVar("ContentT", default=LLMContentPart)


class LLMMessage(BaseModel, Generic[RoleT, ContentT]):
    """
    LLM 消息基类与门面工厂。
    提供统一的元数据访问、魔法加法重载以及极简实例化方法。
    """

    role: RoleT
    content: list[ContentT] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    metadata: dict[str, Any] | None = Field(default=None, description="额外元数据")

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.content if isinstance(p, ToolCallPart)]

    @property
    def tool_returns(self) -> list[ToolReturnPart]:
        return [p for p in self.content if isinstance(p, ToolReturnPart)]

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, data: Any) -> Any:
        """核心拦截：外部传入 str 时自动转为 Part，保持内部类型绝对纯净"""
        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, str):
                data["content"] = [{"type": "text", "text": content}]
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
        """极简语法糖：支持 msg + '字符串' 或 msg + ImagePart()"""
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
        return self.metadata.get("source") if self.metadata else None

    @source.setter
    def source(self, value: str | None):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["source"] = value

    @property
    def source_name(self) -> str | None:
        return self.metadata.get("source_name") if self.metadata else None

    @source_name.setter
    def source_name(self, value: str | None):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["source_name"] = value

    @property
    def scope(self) -> str | None:
        return self.metadata.get("scope") if self.metadata else None

    @scope.setter
    def scope(self, value: str | None):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["scope"] = value

    @property
    def thought_signature(self) -> str | None:
        return self.metadata.get("thought_signature") if self.metadata else None

    @thought_signature.setter
    def thought_signature(self, value: str | None):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["thought_signature"] = value

    @property
    def token_cost(self) -> int | None:
        return self.metadata.get("token_cost") if self.metadata else None

    @token_cost.setter
    def token_cost(self, value: int | None):
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
        if isinstance(content, str):
            content = [TextPart(text=content)]
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
        if isinstance(content, str):
            content = [TextPart(text=content)] if content else []
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
        import base64

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
        if isinstance(content, str):
            content = [TextPart(text=content)]
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
    role: Literal["system"] = "system"
    content: list[Annotated[SystemContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )


class UserMessage(LLMMessage[Literal["user"], UserContentUnion]):
    role: Literal["user"] = "user"
    content: list[Annotated[UserContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )


class AssistantMessage(LLMMessage[Literal["assistant"], AssistantContentUnion]):
    role: Literal["assistant"] = "assistant"
    content: list[Annotated[AssistantContentUnion, Field(discriminator="type")]] = (
        Field(default_factory=list)
    )


class ToolMessage(LLMMessage[Literal["tool"], ToolContentUnion]):
    role: Literal["tool"] = "tool"
    content: list[Annotated[ToolContentUnion, Field(discriminator="type")]] = Field(
        default_factory=list
    )

    def model_post_init(self, context: Any, /) -> None:
        """验证消息的有效性"""
        if not self.tool_returns:
            raise ValueError("工具角色的消息必须包含 ToolReturnPart")


AnyLLMMessage = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class RerankDocument(BaseModel):
    """重排候选文档 (支持纯文本或图文字典)"""

    text: str | None = None
    image: str | None = None


class RerankResult(BaseModel):
    """重排返回结果"""

    index: int = Field(description="原始文档在候选列表中的索引")
    relevance_score: float = Field(description="相关性得分")
    document: RerankDocument | None = Field(default=None, description="命中的文档内容")


class LLMResponse(BaseModel):
    """
    LLM 响应对象，确立 SSOT (单一数据源) 架构。
    """

    content_parts: list[LLMContentPart] = Field(
        default_factory=list, description="原始响应内容块列表"
    )
    usage_info: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    grounding_metadata: Any | None = None
    cache_info: Any | None = None
    parsed_obj: Any | None = Field(default=None, description="Pydantic 模型实例")

    def get_parsed_obj(self, model_class: type[T]) -> T | None:
        """
        获取强类型的解析对象，提供完善的 IDE 类型推导支持。

        参数:
            model_class: 期望的 Pydantic 模型类

        返回:
            强类型的模型实例，如果不存在则返回 None
        """
        if self.parsed_obj is None:
            return None
        if isinstance(self.parsed_obj, model_class):
            return self.parsed_obj
        from zhenxun.utils.pydantic_compat import parse_as

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
            if isinstance(p, (ThoughtPart, ToolCallPart, ToolReturnPart)):
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
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def efficiency_ratio(self) -> float:
        return self.completion_tokens / max(self.prompt_tokens, 1)


class LLMCodeExecution(BaseModel):
    code: str
    output: str | None = None
    error: str | None = None
    execution_time: float | None = None
    files_generated: list[str] | None = None


class LLMGroundingAttribution(BaseModel):
    title: str | None = None
    uri: str | None = None
    snippet: str | None = None
    confidence_score: float | None = None


class LLMGroundingMetadata(BaseModel):
    web_search_queries: list[str] | None = None
    grounding_attributions: list[LLMGroundingAttribution] | None = None
    search_suggestions: list[dict[str, Any]] | None = None
    search_entry_point: str | None = None
    map_widget_token: str | None = None


class LLMCacheInfo(BaseModel):
    cache_hit: bool = False
    cache_key: str | None = None
    cache_ttl: int | None = None
    created_at: str | None = None


__all__ = [
    "AnyLLMMessage",
    "AssistantMessage",
    "AudioPart",
    "FilePart",
    "ImagePart",
    "LLMCacheInfo",
    "LLMCodeExecution",
    "LLMContentPart",
    "LLMGroundingAttribution",
    "LLMGroundingMetadata",
    "LLMMessage",
    "LLMResponse",
    "RerankDocument",
    "RerankResult",
    "ResponseFormat",
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
