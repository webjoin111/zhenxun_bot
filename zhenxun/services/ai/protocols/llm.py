from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.core.configs import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)
from zhenxun.services.ai.core.messages import (
    AudioResponse,
    EmbeddingResponse,
    LLMMessage,
    LLMResponse,
)

if TYPE_CHECKING:
    from zhenxun.services.ai.tools.models import ToolChoice



class LLMModelBase(ABC):
    """底层 LLM 模型抽象基类（约束 Service 实现）"""

    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any | None = None,
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
        texts: list[str],
        config: LLMEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        """生成文本嵌入向量"""
        pass

