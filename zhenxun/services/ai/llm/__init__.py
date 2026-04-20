"""
LLM 服务模块 - 公共 API 入口

提供统一的 AI 服务调用接口、核心数据契约和配置工具。
"""

from .api import (
    chat,
    code,
    create_image,
    embed,
    embed_documents,
    embed_query,
    generate,
    generate_structured,
    rerank,
    search,
)
from .config import (
    CommonOverrides,
    GenConfigBuilder,
    LLMGenerationConfig,
    register_after_llm_hook,
    register_before_llm_hook,
    register_llm_configs,
)

register_llm_configs()
from zhenxun.services.ai.message_builder import MessageBuilder
from zhenxun.services.ai.types.configs import LLMEmbeddingConfig
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import (
    LLMContentPart,
    LLMMessage,
    LLMResponse,
    UsageInfo,
)

from .api import ModelName

create_multimodal_message = MessageBuilder.create_multimodal_message

__all__ = [
    "CommonOverrides",
    "GenConfigBuilder",
    "LLMContentPart",
    "LLMEmbeddingConfig",
    "LLMErrorCode",
    "LLMException",
    "LLMGenerationConfig",
    "LLMMessage",
    "LLMResponse",
    "ModelName",
    "UsageInfo",
    "chat",
    "code",
    "create_image",
    "create_multimodal_message",
    "embed",
    "embed_documents",
    "embed_query",
    "generate",
    "generate_structured",
    "register_after_llm_hook",
    "register_before_llm_hook",
    "register_llm_configs",
    "rerank",
    "search",
]
