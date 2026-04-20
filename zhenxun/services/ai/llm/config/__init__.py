"""
LLM 配置模块

提供生成配置、预设配置和配置验证功能。
"""

from zhenxun.services.ai.config import (
    LLMConfig,
    get_gemini_safety_threshold,
    get_llm_config,
    register_ai_configs as register_llm_configs,
    set_default_model,
)
from zhenxun.services.ai.types.configs import LLMEmbeddingConfig, LLMGenerationConfig

from ..hooks import register_after_llm_hook, register_before_llm_hook
from .generation import (
    CommonOverrides,
    GenConfigBuilder,
    validate_override_params,
)
__all__ = [
    "CommonOverrides",
    "GenConfigBuilder",
    "LLMConfig",
    "LLMEmbeddingConfig",
    "LLMGenerationConfig",
    "get_gemini_safety_threshold",
    "get_llm_config",
    "register_after_llm_hook",
    "register_before_llm_hook",
    "register_llm_configs",
    "set_default_model",
    "validate_override_params",
]
