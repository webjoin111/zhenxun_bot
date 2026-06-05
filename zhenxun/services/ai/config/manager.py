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
    """获取 AI 配置组"""
    return Config.get(AI_CONFIG_GROUP)


def get_default_providers() -> list[dict[str, Any]]:
    """获取默认提供商配置列表。"""
    return [
        {
            "name": "DeepSeek",
            "api_key": "YOUR_API_KEY",
            "api_base": "https://api.deepseek.com",
            "api_type": "deepseek",
            "models": [
                {
                    "model_name": "deepseek-v4-pro",
                },
                {
                    "model_name": "deepseek-v4-flash",
                },
            ],
        },
        {
            "name": "Doubao",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://ark.cn-beijing.volces.com/api",
            "api_type": "doubao",
            "models": [
                {"model_name": "doubao-seed-1-6-250615"},
                {"model_name": "doubao-seed-1-6-flash-250615"},
            ],
        },
        {
            "name": "siliconflow",
            "api_key": "YOUR_ARK_API_KEY",
            "api_base": "https://api.siliconflow.cn",
            "api_type": "openai",
            "models": [
                {"model_name": "deepseek-ai/DeepSeek-V4-Flash"},
                {"model_name": "BAAI/bge-m3"},
                {"model_name": "BAAI/bge-reranker-v2-m3"},
            ],
        },
        {
            "name": "GLM",
            "api_key": "YOUR_API_KEY",
            "api_base": "https://open.bigmodel.cn",
            "api_type": "glm",
            "models": [
                {"model_name": "glm-4.6v-flash"},
                {"model_name": "glm-5v-turbo"},
            ],
        },
        {
            "name": "Gemini",
            "api_key": [
                "AIzaSy*****************************",
                "AIzaSy*****************************",
            ],
            "api_base": "https://generativelanguage.googleapis.com",
            "api_type": "gemini",
            "models": [
                {"model_name": "gemini-3.5-flash"},
                {"model_name": "gemini-3.1-flash-lite"},
                {"model_name": "gemini-2.5-flash-image"},
                {"model_name": "gemini-embedding-2"},
                {"model_name": "gemini-3.1-flash-tts-preview"},
            ],
        },
        {
            "name": "OpenRouter",
            "api_key": "YOUR_OPENROUTER_API_KEY",
            "api_base": "https://openrouter.ai/api",
            "api_type": "openrouter",
            "models": [
                {"model_name": "google/gemini-3.1-flash-lite"},
                {"model_name": "x-ai/grok-4"},
            ],
        },
        {
            "name": "Jina",
            "api_key": "YOUR_JINA_API_KEY",
            "api_base": "https://api.jina.ai",
            "api_type": "jina",
            "models": [
                {"model_name": "jina-embeddings-v3"},
                {"model_name": "jina-embeddings-v5-omni-small"},
                {"model_name": "jina-reranker-v2-base-multilingual"},
            ],
        },
    ]


def register_llm_configs():
    """注册 LLM 服务的配置项"""
    logger.info("注册 LLM 服务的配置项")

    llm_config = LLMConfig()

    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "default_models",
        model_dump(llm_config.default_models),
        help="不同任务类型的全局默认模型配置字典",
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "client_settings",
        model_dump(llm_config.client_settings),
        help=(
            "LLM客户端高级设置。\n"
            "包含: timeout(超时秒数), max_retries(重试次数), "
            "retry_delay(重试延迟), structured_retries(结构化生成重试), proxy(代理)"
        ),
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "debug_log",
        model_dump(llm_config.debug_log),
        help=(
            "LLM日志详情开关。示例: {'show_tools': True, 'show_schema': False, "
            "'show_safety': False}"
        ),
        type=dict,
    )
    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "gemini_safety_threshold",
        "BLOCK_NONE",
        help=(
            "Gemini 安全过滤阈值 "
            "(BLOCK_LOW_AND_ABOVE: 阻止低级别及以上, "
            "BLOCK_MEDIUM_AND_ABOVE: 阻止中等级别及以上, "
            "BLOCK_ONLY_HIGH: 只阻止高级别, "
            "BLOCK_NONE: 不阻止)"
        ),
        type=str,
    )

    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "context_settings",
        model_dump(llm_config.context_settings),
        help=(
            "智能上下文管理与压缩配置。\n"
            "包含: default_strategy(默认策略: unlimited为不压缩, "
            "sliding_window为滑动窗口, llm_summary为模型总结, "
            "structured_summary为结构化总结), "
            "trigger_threshold(触发阈值), max_history_turns(最大轮数), "
            "strategy_kwargs(各模式特有参数), vision_window_size(多模态保留轮数)"
        ),
        type=dict,
    )

    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        "MODEL_GROUPS",
        llm_config.model_groups,
        help=(
            "虚拟模型路由组配置 (Virtual Router Groups)。\n"
            "键为组名，值为模型名称或其它组名的列表。\n"
            "使用 chat(model='cheap_models') 时系统将自动按列表顺序轮询和故障转移。"
        ),
        type=dict,
    )

    Config.add_plugin_config(
        AI_CONFIG_GROUP,
        PROVIDERS_CONFIG_KEY,
        get_default_providers(),
        help=(
            "配置多个 AI 服务提供商及其模型信息。\n"
            "注意：可以在特定模型配置下添加 'api_type' 以覆盖提供商的全局设置。\n"
            "支持的 api_type 包括:\n"
            "- 'openai': 标准 OpenAI 格式 (DeepSeek, SiliconFlow等)\n"
            "- 'gemini': Google Gemini API\n"
            "- 'glm': 智谱 AI (GLM)\n"
            "- 'doubao': 字节跳动火山引擎 (Doubao)\n"
            "- 'jina': Jina AI (专精于多模态嵌入与重排)\n"
            "- 'openrouter': OpenRouter 聚合平台\n"
            "- 'openai_responses': 支持新版 responses 格式的 OpenAI 兼容接口\n"
            "- 'smart': 智能路由模式 (主要用于第三方中转场景，自动根据模型名"
            "分发请求到 openai 或 gemini)"
        ),
        default_value=[],
        type=list[ProviderConfig],
    )


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """获取 LLM 配置实例"""
    ai_config = get_ai_config()

    raw_debug = ai_config.get("debug_log", False)
    if isinstance(raw_debug, bool):
        debug_log_val = DebugLogOptions(
            show_tools=raw_debug, show_schema=raw_debug, show_safety=raw_debug
        )
    else:
        debug_log_val = raw_debug

    config_data = {
        "default_models": ai_config.get("default_models", {}),
        "client_settings": ai_config.get("client_settings", {}),
        "debug_log": debug_log_val,
        PROVIDERS_CONFIG_KEY: ai_config.get(PROVIDERS_CONFIG_KEY, []),
        "context_settings": ai_config.get("context_settings", {}),
        "model_groups": ai_config.get("MODEL_GROUPS", {}),
    }

    return parse_as(LLMConfig, config_data)


def get_gemini_safety_threshold() -> str:
    """获取 Gemini 安全过滤阈值配置。"""
    ai_config = get_ai_config()
    return ai_config.get("gemini_safety_threshold", "BLOCK_MEDIUM_AND_ABOVE")


def validate_llm_config() -> tuple[bool, list[str]]:
    """验证 LLM 配置有效性并返回错误列表。"""
    errors = []

    try:
        llm_config = get_llm_config()

        if llm_config.client_settings.timeout <= 0:
            errors.append("timeout 必须大于 0")

        if llm_config.client_settings.max_retries < 0:
            errors.append("max_retries 不能小于 0")

        if llm_config.client_settings.retry_delay <= 0:
            errors.append("retry_delay 必须大于 0")

        if not llm_config.providers:
            errors.append("至少需要配置一个 AI 服务提供商")
        else:
            provider_names = set()
            for provider in llm_config.providers:
                if provider.name in provider_names:
                    errors.append(f"提供商名称重复: {provider.name}")
                provider_names.add(provider.name)

                if not provider.api_key:
                    errors.append(f"提供商 {provider.name} 缺少 API Key")

                if not provider.models:
                    errors.append(f"提供商 {provider.name} 没有配置任何模型")
                else:
                    model_names = set()
                    for model in provider.models:
                        if model.model_name in model_names:
                            errors.append(
                                f"提供商 {provider.name} 中模型名称重复: "
                                f"{model.model_name}"
                            )
                        model_names.add(model.model_name)

        for task, m_name in model_dump(llm_config.default_models).items():
            if m_name:
                if not llm_config.validate_model_name(m_name):
                    errors.append(f"任务 {task} 的默认模型 {m_name} 在配置中不存在")

    except Exception as e:
        errors.append(f"配置解析失败: {e!s}")

    return len(errors) == 0, errors


def set_default_model(task: str, provider_model_name: str | None) -> bool:
    """设置默认模型名称"""
    if provider_model_name:
        llm_config = get_llm_config()
        if not llm_config.validate_model_name(provider_model_name):
            logger.error(f"模型 {provider_model_name} 在配置中不存在")
            return False

    ai_config = get_ai_config()
    current_defaults = ai_config.get("default_models", {})
    current_defaults[task] = provider_model_name

    Config.set_config(
        AI_CONFIG_GROUP, "default_models", current_defaults, auto_save=True
    )

    if provider_model_name:
        logger.info(f"任务 {task} 的默认模型已设置为: {provider_model_name}")
    else:
        logger.info(f"任务 {task} 的默认模型已清除")

    return True
