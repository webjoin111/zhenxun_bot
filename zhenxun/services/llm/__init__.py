"""
向下兼容门面 (Facade)

利用 sys.modules 魔法，将对 zhenxun.services.llm 及其子模块的导入
无缝重定向到 zhenxun.services.ai.llm 下，避免破坏任何第三方插件。
同时在此提供最常用的类导出，作为 LLM 服务的统一入口。
"""

import sys

from zhenxun.services import logger

logger.warning(
    "zhenxun.services.llm 已被重构迁移至 zhenxun.services.ai.llm，"
    "为了更好的性能和兼容性，请及时更新您的插件导入路径。"
)

import zhenxun.services.ai.llm
import zhenxun.services.ai.llm.adapters
import zhenxun.services.ai.llm.config.generation
import zhenxun.services.ai.llm.engine
import zhenxun.services.ai.protocols
import zhenxun.services.ai.tools
import zhenxun.services.ai.types

sys.modules["zhenxun.services.llm.types"] = zhenxun.services.ai.types
sys.modules["zhenxun.services.llm.types.models"] = zhenxun.services.ai.types
sys.modules["zhenxun.services.llm.types.protocols"] = zhenxun.services.ai.protocols
sys.modules["zhenxun.services.llm.types.exceptions"] = zhenxun.services.ai.types
sys.modules["zhenxun.services.llm.types.capabilities"] = zhenxun.services.ai.types

sys.modules["zhenxun.services.agent.core.types"] = zhenxun.services.ai.types
sys.modules["zhenxun.services.sandbox.models"] = zhenxun.services.ai.types

prefix_old = "zhenxun.services.llm"
prefix_new = "zhenxun.services.ai.llm"

for module_name, module_obj in list(sys.modules.items()):
    if module_name.startswith(prefix_new):
        old_module_name = module_name.replace(prefix_new, prefix_old, 1)
        if old_module_name not in sys.modules:
            sys.modules[old_module_name] = module_obj

sys.modules["zhenxun.services.llm.tools"] = zhenxun.services.ai.tools

from zhenxun.services.ai.chat_session import ChatSession as AI
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
from zhenxun.services.ai.tools import tool as function_tool
from zhenxun.services.ai.types.exceptions import LLMException
from zhenxun.services.ai.types.messages import LLMMessage, LLMResponse

__all__ = [
    "AI",
    "LLMException",
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
