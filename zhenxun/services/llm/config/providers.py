"""
LLM 提供商配置管理

负责注册和管理 AI 服务提供商的配置项。
"""

from zhenxun.configs.config import Config
from zhenxun.services.log import logger

from ..types.models import ProviderConfig

AI_CONFIG_GROUP = "AI"
PROVIDERS_CONFIG_KEY = "PROVIDERS"


def get_ai_config():
    """获取 AI 配置组"""
    return Config.get(AI_CONFIG_GROUP)


def register_llm_configs():
    """注册 LLM 服务的配置项"""
    logger.info("注册 LLM 服务的配置项")
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "default_model_name",
        None,
        help="LLM服务全局默认使用的模型名称 (格式: ProviderName/ModelName)",
        type=str,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "proxy",
        None,
        help="LLM服务请求使用的网络代理，例如 http://127.0.0.1:7890",
        type=str,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "timeout",
        180,
        help="LLM服务API请求超时时间（秒）",
        type=int,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "max_retries_llm",
        3,
        help="LLM服务请求失败时的最大重试次数",
        type=int,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "retry_delay_llm",
        2,
        help="LLM服务请求重试的基础延迟时间（秒）",
        type=int,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        PROVIDERS_CONFIG_KEY,
        [
            {
                "name": "DeepSeek",
                "api_key": "sk-******",
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
                "name": "GLM",
                "api_key": "",
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
                    {"model_name": "gemini-2.0-flash"},
                    {"model_name": "gemini-2.5-flash-preview-05-20"},
                ],
            },
        ],
        help="配置多个 AI 服务提供商及其模型信息 (列表)",
        default_value=[],
        type=list[ProviderConfig],
    )
