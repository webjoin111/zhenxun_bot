"""
LLM 服务模块 - 公共 API 入口

提供统一的 AI 服务调用接口、核心数据契约和配置工具。
"""

from zhenxun.services.ai.core.configs import (
    TTSConfig,
)
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import (
    AudioResponse,
    LLMContentPart,
    LLMMessage,
    LLMResponse,
)
from zhenxun.services.ai.message_builder import MessageBuilder

from .api import (
    ModelName,
    chat,
    create_image,
    create_speech,
    embed,
    generate,
    generate_structured,
    rerank,
)
from .config import (
    IntentBuilder,
)

create_multimodal_message = MessageBuilder.create_multimodal_message

__all__ = [
    "AudioResponse",
    "IntentBuilder",
    "LLMContentPart",
    "LLMErrorCode",
    "LLMException",
    "LLMMessage",
    "LLMResponse",
    "ModelName",
    "TTSConfig",
    "chat",
    "create_image",
    "create_multimodal_message",
    "create_speech",
    "embed",
    "generate",
    "generate_structured",
    "rerank",
]
