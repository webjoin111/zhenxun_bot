from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.core.options import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)
from zhenxun.services.ai.core.messages import (
    AudioResponse,
    EmbedBatch,
    EmbeddingResponse,
    LLMMessage,
    LLMResponse,
)
from zhenxun.services.ai.core.models import ModelCapabilities, ModelDetail, ToolChoice

if TYPE_CHECKING:
    from zhenxun.services.ai.run.models import CancellationToken


class LLMModelBase(ABC):
    """底层 LLM 模型抽象基类"""

    provider_name: str
    model_name: str
    api_type: str
    api_base: str | None
    path_prefix: str | None
    model_detail: ModelDetail
    capabilities: ModelCapabilities
    health_manager: Any
    engine: Any
    _generation_config: GenerationConfig | None

    @abstractmethod
    def _get_effective_api_type(self) -> str:
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: "CancellationToken | None" = None,
    ) -> LLMResponse:
        """生成高级响应"""
        pass

    @abstractmethod
    async def generate_speech(
        self,
        input_text: str,
        voice: str,
        config: TTSConfig | None = None,
    ) -> AudioResponse:
        """生成语音"""
        pass

    @abstractmethod
    async def generate_embeddings(
        self,
        batch: EmbedBatch,
        config: LLMEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        """生成文本或多模态嵌入向量"""
        pass
