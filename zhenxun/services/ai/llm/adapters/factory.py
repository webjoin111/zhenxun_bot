"""
LLM 适配器工厂类
"""

from __future__ import annotations

import fnmatch
from typing import Any, ClassVar

from zhenxun.services.ai.core.configs import GenerationConfig, LLMEmbeddingConfig
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import EmbedBatch, LLMMessage
from zhenxun.services.ai.core.models import ToolChoice
from zhenxun.services.ai.protocols.llm import LLMModelBase

from .base import BaseAdapter, RequestData, ResponseData


class LLMAdapterFactory:
    """适配器注册与按 API 类型分发的统一入口。"""

    _adapters: ClassVar[dict[str, BaseAdapter]] = {}
    _api_type_mapping: ClassVar[dict[str, str]] = {}

    @classmethod
    def initialize(cls) -> None:
        """初始化默认适配器"""
        if cls._adapters:
            return

        from .deepseek import DeepSeekAdapter
        from .doubao import DoubaoAdapter
        from .gemini import GeminiAdapter
        from .glm import GLMAdapter
        from .jina import JinaAdapter
        from .minimax import MiniMaxAdapter
        from .openai import OpenAIAdapter

        cls.register_adapter(OpenAIAdapter())
        cls.register_adapter(DeepSeekAdapter())
        cls.register_adapter(JinaAdapter())
        cls.register_adapter(GeminiAdapter())
        cls.register_adapter(GLMAdapter())
        cls.register_adapter(SmartAdapter())
        cls.register_adapter(MiniMaxAdapter())
        cls.register_adapter(DoubaoAdapter())

    @classmethod
    def register_adapter(cls, adapter: BaseAdapter) -> None:
        """注册适配器"""
        adapter_key = adapter.api_type
        cls._adapters[adapter_key] = adapter

        for api_type in adapter.supported_api_types:
            cls._api_type_mapping[api_type] = adapter_key

    @classmethod
    def get_adapter(cls, api_type: str) -> BaseAdapter:
        """获取适配器"""
        cls.initialize()

        adapter_key = cls._api_type_mapping.get(api_type)
        if not adapter_key:
            raise LLMException(
                f"不支持的API类型: {api_type}",
                code=LLMErrorCode.UNKNOWN_API_TYPE,
                details={
                    "api_type": api_type,
                    "supported_types": list(cls._api_type_mapping.keys()),
                },
            )

        return cls._adapters[adapter_key]

    @classmethod
    def list_supported_types(cls) -> list[str]:
        """列出所有支持的API类型"""
        cls.initialize()
        return list(cls._api_type_mapping.keys())

    @classmethod
    def list_adapters(cls) -> dict[str, BaseAdapter]:
        """列出所有注册的适配器"""
        cls.initialize()
        return cls._adapters.copy()


def get_adapter_for_api_type(api_type: str) -> BaseAdapter:
    """按 API 类型获取适配器实例。"""
    return LLMAdapterFactory.get_adapter(api_type)


def register_adapter(adapter: BaseAdapter) -> None:
    """向工厂注册新的适配器实例。"""
    LLMAdapterFactory.register_adapter(adapter)


class SmartAdapter(BaseAdapter):
    """
    智能路由适配器。
    本身不处理序列化，而是根据规则委托给 OpenAIAdapter 或 GeminiAdapter。
    """

    @property
    def log_sanitization_context(self) -> str:
        """返回智能路由适配器的默认日志清洗上下文。"""
        return "openai_request"

    _ROUTING_RULES: ClassVar[list[tuple[str, str]]] = [
        ("*nano-banana*", "gemini"),
        ("*gemini*", "gemini"),
        ("*deepseek*", "deepseek"),
        ("*minimax*", "minimax"),
        ("*gpt*", "openai_responses"),
    ]
    _DEFAULT_API_TYPE: ClassVar[str] = "openai"

    def __init__(self):
        """初始化模型名到目标适配器的路由缓存。"""
        self._adapter_cache: dict[str, BaseAdapter] = {}

    @property
    def api_type(self) -> str:
        """适配器主类型标识。"""
        return "smart"

    @property
    def supported_api_types(self) -> list[str]:
        """当前适配器支持的 API 类型列表。"""
        return ["smart"]

    def _get_delegate_adapter(self, model: LLMModelBase) -> BaseAdapter:
        """
        核心路由逻辑：决定使用哪个适配器 (带缓存)
        """
        if model.model_detail.api_type:
            return get_adapter_for_api_type(model.model_detail.api_type)

        model_name = model.model_name
        if model_name in self._adapter_cache:
            return self._adapter_cache[model_name]

        target_api_type = self._DEFAULT_API_TYPE
        model_name_lower = model_name.lower()

        for pattern, api_type in self._ROUTING_RULES:
            if fnmatch.fnmatch(model_name_lower, pattern):
                target_api_type = api_type
                break

        adapter = get_adapter_for_api_type(target_api_type)
        self._adapter_cache[model_name] = adapter
        return adapter

    async def prepare_advanced_request(
        self,
        model: LLMModelBase,
        api_key: str,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    ) -> RequestData:
        """按模型路由到目标适配器并准备高级对话请求。"""
        adapter = self._get_delegate_adapter(model)
        return await adapter.prepare_advanced_request(
            model, api_key, messages, config, tools, tool_choice
        )

    def parse_response(
        self,
        model: LLMModelBase,
        response_json: dict[str, Any],
        is_advanced: bool = False,
    ) -> ResponseData:
        """按模型路由并解析文本响应。"""
        adapter = self._get_delegate_adapter(model)
        return adapter.parse_response(model, response_json, is_advanced)

    async def prepare_embedding_request(
        self,
        model: LLMModelBase,
        api_key: str,
        batch: EmbedBatch,
        config: LLMEmbeddingConfig,
    ) -> RequestData:
        """按模型路由并准备嵌入请求。"""
        adapter = self._get_delegate_adapter(model)
        return await adapter.prepare_embedding_request(model, api_key, batch, config)

    def parse_embedding_response(
        self, response_json: dict[str, Any]
    ) -> list[list[float]]:
        """使用默认 OpenAI 兼容逻辑解析嵌入响应。"""
        return get_adapter_for_api_type("openai").parse_embedding_response(
            response_json
        )

    def prepare_image_request(
        self,
        model: LLMModelBase,
        api_key: str,
        prompt: str,
        images: list[Any] | None = None,
        config: GenerationConfig | None = None,
    ) -> RequestData:
        """按模型路由并准备图像请求。"""
        adapter = self._get_delegate_adapter(model)
        return adapter.prepare_image_request(model, api_key, prompt, images, config)

    def parse_image_response(self, response_json: dict[str, Any]) -> ResponseData:
        """按响应结构选择 Gemini 或 OpenAI 的图像解析器。"""
        if "candidates" in response_json or "image_generation" in response_json:
            return get_adapter_for_api_type("gemini").parse_image_response(
                response_json
            )
        return get_adapter_for_api_type("openai").parse_image_response(response_json)

    def prepare_rerank_request(
        self,
        model: LLMModelBase,
        api_key: str,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int,
    ) -> RequestData:
        """按模型路由并准备重排请求。"""
        adapter = self._get_delegate_adapter(model)
        return adapter.prepare_rerank_request(model, api_key, query, documents, top_n)

    def parse_rerank_response(self, response_json: dict[str, Any]) -> list[Any]:
        """使用默认 OpenAI 兼容逻辑解析重排响应。"""
        return get_adapter_for_api_type("openai").parse_rerank_response(response_json)
