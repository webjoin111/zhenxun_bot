from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.core.models import ModelCapabilities, ModelDetail
from zhenxun.services.ai.llm.adapters.base import RequestData, ResponseData

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import LLMEmbeddingConfig, TTSConfig
    from zhenxun.services.ai.core.messages import AudioResponse, RerankResult
    from zhenxun.services.ai.core.models import ToolChoice, ToolDefinition
    from zhenxun.services.ai.llm.adapters.base import BaseAdapter
    from zhenxun.services.ai.llm.service import LLMModel


class ConfigMapper(ABC):
    @abstractmethod
    def map_config(
        self,
        config: GenerationConfig,
        model_detail: ModelDetail | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> dict[str, Any]:
        """将通用生成配置转换为特定 API 的参数字典"""
        ...


class MessageConverter(ABC):
    @abstractmethod
    def convert_messages(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """将通用消息列表转换为特定 API 的消息格式"""
        ...


class ToolSerializer(ABC):
    @abstractmethod
    def serialize_tools(self, tools: list[ToolDefinition]) -> Any:
        """将通用工具定义转换为特定 API 的工具格式"""
        ...

    @abstractmethod
    def sanitize_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """对 JSON Schema 进行特定 API 的清洗和格式化"""
        ...


class ResponseParser(ABC):
    @abstractmethod
    def parse(self, response_json: dict[str, Any]) -> ResponseData:
        """将特定 API 的响应解析为通用响应数据"""
        ...


class BaseTextHandler(ABC):
    """
    文本对话生成处理器接口。
    负责将真寻的通用消息与工具列表转换为底层 API 的请求格式，并解析响应。
    """

    @abstractmethod
    async def prepare_text_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: "ToolChoice | str | dict[str, Any] | None" = None,
    ) -> RequestData: ...

    @abstractmethod
    def parse_text_response(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        response_json: dict[str, Any],
        is_advanced: bool = False,
    ) -> ResponseData: ...


class BaseEmbeddingHandler(ABC):
    """
    文本嵌入向量处理器接口。
    """

    @abstractmethod
    def prepare_embedding_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        texts: list[str],
        config: "LLMEmbeddingConfig",
    ) -> RequestData: ...

    @abstractmethod
    def parse_embedding_response(
        self, adapter: "BaseAdapter", response_json: dict[str, Any]
    ) -> list[list[float]]: ...


class BaseImageHandler(ABC):
    """
    图像生成/编辑处理器接口。
    """

    @abstractmethod
    def prepare_image_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        prompt: str,
        images: list[Any] | None = None,
        config: GenerationConfig | None = None,
    ) -> RequestData: ...

    @abstractmethod
    def parse_image_response(
        self, adapter: "BaseAdapter", response_json: dict[str, Any]
    ) -> ResponseData: ...


class BaseRerankHandler(ABC):
    """
    文本重排处理器接口。
    """

    @abstractmethod
    def prepare_rerank_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int,
    ) -> RequestData: ...

    @abstractmethod
    def parse_rerank_response(
        self, adapter: "BaseAdapter", response_json: dict[str, Any]
    ) -> list["RerankResult"]: ...


class BaseAudioHandler(ABC):
    """
    文本转语音 (TTS) 处理器接口。
    """

    @abstractmethod
    def prepare_speech_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        input_text: str,
        voice: str,
        config: "TTSConfig",
    ) -> RequestData: ...

    @abstractmethod
    async def parse_speech_response(
        self, adapter: "BaseAdapter", model: "LLMModel", raw_response: Any
    ) -> "AudioResponse": ...
