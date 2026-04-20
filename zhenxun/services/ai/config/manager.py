from functools import lru_cache
from typing import Any

from zhenxun.configs.config import Config
from zhenxun.configs.utils import parse_as
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump

from .models import DebugLogOptions, LLMConfig, ProviderConfig

AI_CONFIG_GROUP = "AI"
PROVIDERS_CONFIG_KEY = "PROVIDERS"


def get_ai_config():
    """获取真寻配置管理器中的 AI 原始配置组"""
    return Config.get(AI_CONFIG_GROUP)


def get_default_providers() -> list[dict[str, Any]]:
    """获取默认的提供商配置"""
    return [
        {
            "name": "DeepSeek",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://api.deepseek.com",
            "api_type": "openai",
            "models": [
                {
                    "model_name": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                {
                    "model_name": "deepseek-reasoner",
                },
            ],
        },
        {
            "name": "ARK",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://ark.cn-beijing.volces.com",
            "api_type": "ark",
            "models": [
                {"model_name": "deepseek-r1-250528"},
                {"model_name": "doubao-seed-1-6-250615"},
                {"model_name": "doubao-seed-1-6-flash-250615"},
                {"model_name": "doubao-seed-1-6-thinking-250615"},
            ],
        },
        {
            "name": "siliconflow",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://api.siliconflow.cn",
            "api_type": "openai",
            "models": [
                {"model_name": "deepseek-ai/DeepSeek-V3"},
            ],
        },
        {
            "name": "GLM",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://open.bigmodel.cn",
            "api_type": "zhipu",
            "models": [
                {"model_name": "glm-4-flash"},
                {"model_name": "glm-4-plus"},
            ],
        },
        {
            "name": "Gemini",
            "api_key": [
                "AIzaSy*****************************",
                "AIzaSy*****************************",
                "AIzaSy*****************************",
            ],
            "api_base": "https://generativelanguage.googleapis.com",
            "api_type": "gemini",
            "models": [
                {"model_name": "gemini-2.5-flash"},
                {"model_name": "gemini-2.5-pro"},
                {"model_name": "gemini-2.5-flash-lite"},
            ],
        },
        {
            "name": "OpenRouter",
            "api_key": "YOUR_OPENROUTER_API_KEY",
            "api_base": "https://openrouter.ai/api",
            "api_type": "openrouter",
            "models": [
                {"model_name": "google/gemini-2.5-pro"},
                {"model_name": "google/gemini-2.5-flash"},
                {"model_name": "x-ai/grok-4"},
            ],
        },
    ]


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """获取 AI 配置单例对象"""
    ai_config = get_ai_config()
    raw_debug = ai_config.get("debug_log", False)

    if isinstance(raw_debug, bool):
        debug_log_val = DebugLogOptions(
            show_tools=raw_debug, show_schema=raw_debug, show_safety=raw_debug
        )
    else:
        debug_log_val = raw_debug

    config_data = {
        "default_model_name": ai_config.get("default_model_name"),
        "client_settings": ai_config.get("client_settings", {}),
        "debug_log": debug_log_val,
        PROVIDERS_CONFIG_KEY: ai_config.get(PROVIDERS_CONFIG_KEY, []),
        "context_settings": ai_config.get("context_settings", {}),
    }
    return parse_as(LLMConfig, config_data)


def register_ai_configs():
    """向真寻系统注册 AI 模块的所有配置项"""
    logger.info("注册 AI 模块全局配置项...")
    default_conf = LLMConfig()

    Config.add_plugin_config(AI_CONFIG_GROUP, "default_model_name", None, type=str)
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "client_settings",
        model_dump(default_conf.client_settings),
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "debug_log",
        model_dump(default_conf.debug_log),
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "context_settings",
        model_dump(default_conf.context_settings),
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        PROVIDERS_CONFIG_KEY,
        get_default_providers(),
        default_value=[],
        type=list[ProviderConfig],
    )


def set_default_model(name: str | None) -> bool:
    """设置全局默认模型"""
    if name and not get_llm_config().validate_model_name(name):
        return False
    Config.set_config(AI_CONFIG_GROUP, "default_model_name", name, auto_save=True)
    return True


def get_gemini_safety_threshold() -> str:
    """获取 Gemini 安全过滤阈值配置"""
    ai_config = get_ai_config()
    return ai_config.get("gemini_safety_threshold", "BLOCK_MEDIUM_AND_ABOVE")
