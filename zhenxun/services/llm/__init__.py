"""
向下兼容门面 (Facade)

利用 sys.modules 魔法，将对 zhenxun.services.llm 及其子模块的导入
无缝重定向到 zhenxun.services.ai.llm 下，避免破坏任何第三方插件。
同时在此提供最常用的类导出，作为 LLM 服务的统一入口。
"""

import sys
from types import ModuleType

from zhenxun.services import logger

logger.warning(
    "zhenxun.services.llm 已被重构迁移至 zhenxun.services.ai.llm，"
    "为了更好的性能和兼容性，请及时更新您的插件导入路径。"
)

import zhenxun.services.ai.core
import zhenxun.services.ai.core.configs
import zhenxun.services.ai.core.exceptions
import zhenxun.services.ai.core.messages
import zhenxun.services.ai.core.models
import zhenxun.services.ai.llm
import zhenxun.services.ai.llm.capabilities
import zhenxun.services.ai.llm.config
import zhenxun.services.ai.llm.config.generation
import zhenxun.services.ai.protocols
import zhenxun.services.ai.tools

zhenxun.services.ai.llm.config.LLMGenerationConfig = (
    zhenxun.services.ai.core.configs.GenerationConfig
)

_legacy_types_module = ModuleType("zhenxun.services.llm.types")
for _mod in (
    zhenxun.services.ai.core.configs,
    zhenxun.services.ai.core.exceptions,
    zhenxun.services.ai.core.messages,
    zhenxun.services.ai.core.models,
):
    for _name in dir(_mod):
        if not _name.startswith("_"):
            setattr(_legacy_types_module, _name, getattr(_mod, _name))

sys.modules["zhenxun.services.llm.types"] = _legacy_types_module
sys.modules["zhenxun.services.llm.types.models"] = zhenxun.services.ai.core.models
sys.modules["zhenxun.services.llm.types.protocols"] = zhenxun.services.ai.protocols
sys.modules["zhenxun.services.llm.types.exceptions"] = (
    zhenxun.services.ai.core.exceptions
)
sys.modules["zhenxun.services.llm.types.capabilities"] = (
    zhenxun.services.ai.llm.capabilities
)

prefix_old = "zhenxun.services.llm"
prefix_new = "zhenxun.services.ai.llm"

for module_name, module_obj in list(sys.modules.items()):
    if module_name.startswith(prefix_new):
        old_module_name = module_name.replace(prefix_new, prefix_old, 1)
        if old_module_name not in sys.modules:
            sys.modules[old_module_name] = module_obj

sys.modules["zhenxun.services.llm.tools"] = zhenxun.services.ai.tools


class CommonOverrides:
    """向下兼容垫片：由于该类已被废弃，此处提供空实现以防止旧插件导入报错。"""

    @staticmethod
    def _fallback(*args, **kwargs):
        from zhenxun.services.ai.llm.config.generation import IntentBuilder

        return IntentBuilder().build()

    gemini_json = _fallback
    gemini_2_5_thinking = _fallback
    gemini_3_thinking = _fallback
    gemini_structured = _fallback
    gemini_safe = _fallback
    gemini_code_execution = _fallback
    gemini_grounding = _fallback
    gemini_nano_banana = _fallback
    gemini_high_res = _fallback


from zhenxun.services.ai.chat_session import ChatSession as AI
from zhenxun.services.ai.core.configs import GenerationConfig, OutputFormatConfig


class OutputConfig(OutputFormatConfig):
    """向下兼容垫片：旧版 OutputConfig 等价于 OutputFormatConfig。"""

AIConfig = GenerationConfig
LLMGenerationConfig = GenerationConfig
zhenxun.services.ai.llm.config.generation.OutputConfig = OutputConfig
from zhenxun.services.ai.llm import *  # noqa: F403
from zhenxun.services.ai.llm.manager import (
    get_global_default_model_name,
    get_model_instance,
    list_available_models,
    list_embedding_models,
    set_global_default_model_name,
)
from zhenxun.services.ai.message_builder import MessageBuilder

create_multimodal_message = MessageBuilder.create_multimodal_message
message_to_unimessage = MessageBuilder.message_to_unimessage
unimsg_to_llm_parts = MessageBuilder.unimsg_to_llm_parts
from zhenxun.services.ai.core.exceptions import LLMException
from zhenxun.services.ai.core.messages import LLMMessage, LLMResponse
from zhenxun.services.ai.tools import tool as function_tool

__all__ = [
    "AI",
    "AIConfig",
    "CommonOverrides",
    "GenerationConfig",
    "OutputConfig",
    "LLMException",
    "LLMGenerationConfig",
    "LLMMessage",
    "LLMResponse",
    "create_multimodal_message",
    "function_tool",
    "get_global_default_model_name",
    "get_model_instance",
    "list_available_models",
    "list_embedding_models",
    "message_to_unimessage",
    "set_global_default_model_name",
    "unimsg_to_llm_parts",
]
