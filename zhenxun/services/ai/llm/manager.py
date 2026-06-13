"""
LLM 模型管理器

负责模型实例的创建、缓存、配置管理和生命周期管理。
"""

import hashlib
import time
from typing import Any

from zhenxun.configs.config import Config
from zhenxun.services.ai.config import ProviderConfig, get_llm_config
from zhenxun.services.ai.core.configs import GenerationConfig, LLMEmbeddingConfig
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import EmbedBatch, EmbeddingResponse, LLMResponse
from zhenxun.services.ai.core.models import ModelDetail, ModelModality, ToolChoice
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.ai.protocols.llm import LLMModelBase
from zhenxun.services.ai.run.models import CancellationToken
from zhenxun.services.log import logger
from zhenxun.utils.manager.priority_manager import PriorityLifecycle
from zhenxun.utils.pydantic_compat import dump_json_safely, model_copy, model_dump

from .config import validate_override_params
from .core import health_manager, http_client_manager
from .engine import HttpEngine
from .service import LLMModel

AI_CONFIG_GROUP = "AI"
PROVIDERS_CONFIG_KEY = "PROVIDERS"

_model_cache: dict[str, tuple[LLMModel, float]] = {}
_cache_ttl = 3600
_max_cache_size = 10


class RoutedLLMModel(LLMModelBase):
    """
    [代理模式 Proxy] 虚拟模型路由代理类。
    实现模型的故障转移 (Fallback) 与负载轮询。向外部伪装成一个普通的单一模型。
    """

    def __init__(
        self,
        group_name: str,
        model_names: list[str],
        override_config: dict[str, Any] | GenerationConfig | None = None,
    ):
        self.group_name = group_name
        self.model_names = model_names
        self.override_config = override_config
        self.model_name = group_name

    def _get_effective_api_type(self) -> str:
        return "smart"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def generate_response(
        self,
        messages: list[Any],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict | ToolChoice | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> LLMResponse:
        from zhenxun.services.ai.llm.core import health_manager

        errors = []
        if extra is None:
            extra = {}
        extra["_is_routed_call"] = True

        start_idx = extra.get("_working_route_index", 0)
        indices_to_try = list(range(start_idx, len(self.model_names))) + list(
            range(0, start_idx)
        )

        all_nodes_bypassed = True

        for idx in indices_to_try:
            m_name = self.model_names[idx]

            if not health_manager.is_route_healthy(m_name):
                logger.debug(f"👉 [Router] 节点 '{m_name}' 熔断中，已跳过")
                errors.append(f"{m_name}(熔断中)")
                continue

            all_nodes_bypassed = False
            try:
                if idx == start_idx:
                    logger.debug(
                        f"👉 [Router] 路由组 '{self.group_name}' "
                        f"正在请求节点模型 '{m_name}'..."
                    )
                else:
                    logger.debug(
                        f"🔄 [Router] 故障转移，路由切换至备用节点: '{m_name}'..."
                    )

                async with await get_model_instance(
                    m_name, self.override_config
                ) as instance:
                    response = await instance.generate_response(
                        messages=messages,
                        config=config,
                        tools=tools,
                        tool_choice=tool_choice,
                        timeout=timeout,
                        extra=extra,
                        cancellation_token=cancellation_token,
                    )
                    extra["_working_route_index"] = idx
                    return response
            except LLMException as e:
                if e.code in [
                    LLMErrorCode.INVALID_PARAMETER,
                    LLMErrorCode.CONTENT_FILTERED,
                ] or not getattr(e, "recoverable", True):
                    logger.warning(
                        f"🚫 [Router] 节点 '{m_name}' "
                        f"返回不可恢复的业务错误 ({e.code.name})，停止故障转移。"
                    )
                    raise e
                logger.warning(
                    f"⚠️ [Router] 节点 '{m_name}' 发生错误 ({e.code.name})，"
                    "触发故障转移(Fallback)..."
                )
                errors.append(f"{m_name}({e.code.name})")
            except Exception as e:
                logger.warning(
                    f"⚠️ [Router] 节点 '{m_name}' 发生未知异常，触发故障转移: {e}"
                )
                errors.append(f"{m_name}(Error)")

        if all_nodes_bypassed:
            fallback_model = health_manager.get_best_fallback_route(self.model_names)
            logger.warning(
                f"⚠️ [Router] 组 '{self.group_name}' "
                "内所有节点均已宕机并处于冷却期！"
                f"触发全死保底机制，强制放行 '{fallback_model}' 探活..."
            )
            try:
                async with await get_model_instance(
                    fallback_model, self.override_config
                ) as instance:
                    response = await instance.generate_response(
                        messages=messages,
                        config=config,
                        tools=tools,
                        tool_choice=tool_choice,
                        timeout=timeout,
                        extra=extra,
                        cancellation_token=cancellation_token,
                    )
                    return response
            except Exception as e:
                errors.append(f"{fallback_model}(保底探活彻底失败:{e})")

        raise LLMException(
            f"路由组 '{self.group_name}' 中的所有模型节点均请求失败。"
            f"错误链路: {' -> '.join(errors)}",
            code=LLMErrorCode.GENERATION_FAILED,
        )

    async def generate_embeddings(
        self, batch: EmbedBatch, config: LLMEmbeddingConfig | None = None
    ) -> EmbeddingResponse:
        errors = []
        from zhenxun.services.ai.llm.core import health_manager

        for m_name in self.model_names:
            if not health_manager.is_route_healthy(m_name):
                continue
            try:
                async with await get_model_instance(
                    m_name, self.override_config
                ) as instance:
                    return await instance.generate_embeddings(batch, config)
            except Exception:
                errors.append(f"{m_name}(Error)")
        raise LLMException(f"路由组 '{self.group_name}' Embeddings 均失败: {errors}")

    async def rerank(
        self,
        query: str,
        documents: list[Any],
        top_n: int = 3,
        timeout: float | None = None,
    ) -> list[Any]:
        errors = []
        from zhenxun.services.ai.llm.core import health_manager

        for m_name in self.model_names:
            if not health_manager.is_route_healthy(m_name):
                continue
            try:
                async with await get_model_instance(
                    m_name, self.override_config
                ) as instance:
                    return await instance.rerank(query, documents, top_n, timeout)
            except Exception:
                errors.append(f"{m_name}(Error)")
        raise LLMException(f"路由组 '{self.group_name}' Rerank 均失败: {errors}")


def get_ai_config():
    """获取 AI 配置组"""
    return Config.get(AI_CONFIG_GROUP)


def parse_provider_model_string(name_str: str | None) -> tuple[str | None, str | None]:
    """解析 'ProviderName/ModelName' 格式的字符串"""
    if not name_str or "/" not in name_str:
        return None, None
    parts = name_str.split("/", 1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return None, None


def _get_group_name(name_str: str) -> str | None:
    """判断名称是否是组名，如果是则提取并返回组名，否则返回 None"""
    name_str = name_str.strip()
    if "/" not in name_str:
        return name_str
    return None


def _make_cache_key(
    provider_model_name: str | None,
    override_config: dict | GenerationConfig | None,
) -> str:
    """生成缓存键"""
    config_str = (
        dump_json_safely(override_config, sort_keys=True) if override_config else "None"
    )
    key_data = f"{provider_model_name}:{config_str}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_cached_model(cache_key: str) -> LLMModel | None:
    """从缓存获取模型"""
    if cache_key in _model_cache:
        model, created_time = _model_cache[cache_key]
        current_time = time.time()

        if current_time - created_time > _cache_ttl:
            del _model_cache[cache_key]
            logger.debug(f"模型缓存已过期: {cache_key}")
            return None

        if model._is_closed:
            logger.debug(
                f"缓存的模型 {cache_key} ({model.provider_name}/{model.model_name}) "
                f"处于_is_closed=True状态，重置为False以供复用。"
            )
            model._is_closed = False

        logger.debug(
            f"使用缓存的模型: {cache_key} -> {model.provider_name}/{model.model_name}"
        )
        return model
    return None


def _cache_model(cache_key: str, model: LLMModel):
    """缓存模型实例"""
    current_time = time.time()

    if len(_model_cache) >= _max_cache_size:
        oldest_key = min(_model_cache.keys(), key=lambda k: _model_cache[k][1])
        del _model_cache[oldest_key]

    _model_cache[cache_key] = (model, current_time)


def clear_model_cache():
    """清空模型缓存并释放已缓存的模型实例。"""
    global _model_cache
    _model_cache.clear()
    logger.info("已清空模型缓存")


def get_cache_stats() -> dict[str, Any]:
    """获取模型缓存统计信息。"""
    return {
        "cache_size": len(_model_cache),
        "max_cache_size": _max_cache_size,
        "cache_ttl": _cache_ttl,
        "cached_models": list(_model_cache.keys()),
    }


def get_default_api_base_for_type(api_type: str) -> str | None:
    """根据API类型获取默认的API基础地址"""
    default_api_bases = {
        "openai": "https://api.openai.com",
        "doubao": "https://ark.cn-beijing.volces.com/api",
        "deepseek": "https://api.deepseek.com",
        "jina": "https://api.jina.ai",
        "glm": "https://open.bigmodel.cn",
        "gemini": "https://generativelanguage.googleapis.com",
        "openrouter": "https://openrouter.ai/api",
        "smart": None,
        "openai_responses": None,
    }

    return default_api_bases.get(api_type)


def get_configured_providers() -> list[ProviderConfig]:
    """从配置中获取Provider列表 - 简化和修正版本"""
    ai_config = get_ai_config()
    providers = ai_config.get(PROVIDERS_CONFIG_KEY, [])

    if not isinstance(providers, list):
        logger.error(
            f"配置项 {AI_CONFIG_GROUP}.{PROVIDERS_CONFIG_KEY} 的值不是一个列表，"
            f"将使用空列表。"
        )
        return []

    valid_providers = []
    for i, item in enumerate(providers):
        if isinstance(item, ProviderConfig):
            if not item.api_base:
                default_api_base = get_default_api_base_for_type(item.api_type)
                if default_api_base:
                    item.api_base = default_api_base
            valid_providers.append(item)
        else:
            logger.warning(
                f"配置文件中第 {i + 1} 项未能正确解析为 ProviderConfig 对象，已跳过。"
                f"实际类型: {type(item)}"
            )

    return valid_providers


def find_model_config(
    provider_name: str, model_name: str
) -> tuple[ProviderConfig, ModelDetail] | None:
    """在配置中查找指定 Provider 与 ModelDetail。"""
    providers = get_configured_providers()

    for provider in providers:
        if provider.name.lower() == provider_name.lower():
            for model_detail in provider.models:
                if model_detail.model_name.lower() == model_name.lower():
                    return provider, model_detail

    return None


def _resolve_model_group(group_name: str, visited: set | None = None) -> list[str]:
    """
    递归解析模型组，展开为扁平的真实模型列表，并防止循环嵌套。
    """
    if visited is None:
        visited = set()

    if group_name in visited:
        logger.warning(f"检测到模型路由组嵌套死循环: {group_name}，已安全跳过该分支。")
        return []

    visited.add(group_name)
    llm_config = get_llm_config()
    if group_name not in llm_config.model_groups:
        logger.warning(f"模型路由组 '{group_name}' 不存在于配置中。")
        return []

    resolved_models = []
    for item in llm_config.model_groups[group_name]:
        item = item.strip()
        sub_group = _get_group_name(item)
        if sub_group:
            resolved_models.extend(_resolve_model_group(sub_group, visited.copy()))
        else:
            prov_mod = parse_provider_model_string(item)
            if prov_mod[0] and prov_mod[1]:
                if find_model_config(prov_mod[0], prov_mod[1]):
                    if item not in resolved_models:
                        resolved_models.append(item)
                else:
                    logger.warning(
                        f"⚠️ [Router] 路由组 '{group_name}' 中的模型 "
                        f"'{item}' 未在 PROVIDERS 中配置，已被自动剔除！"
                    )
            else:
                logger.warning(f"路由组 '{group_name}' 包含无效格式的项目 '{item}'。")

    return resolved_models


def list_available_models() -> list[dict[str, Any]]:
    """列出所有已配置的可用模型及其信息。"""
    providers = get_configured_providers()
    model_list = []
    for provider in providers:
        for model_detail in provider.models:
            caps = get_model_capabilities(model_detail.model_name)
            model_info = {
                "provider_name": provider.name,
                "model_name": model_detail.model_name,
                "full_name": f"{provider.name}/{model_detail.model_name}",
                "api_type": provider.api_type or "auto-detect",
                "api_base": provider.api_base,
                "is_available": model_detail.is_available,
                "is_embedding_model": caps.is_embedding_model,
                "max_input_tokens": caps.max_input_tokens,
                "available_identifiers": _get_model_identifiers(
                    provider.name, model_detail
                ),
            }
            model_list.append(model_info)
    return model_list


def _get_model_identifiers(provider_name: str, model_detail: ModelDetail) -> list[str]:
    """获取模型的所有可用标识符"""
    return [f"{provider_name}/{model_detail.model_name}"]


def list_model_identifiers() -> dict[str, list[str]]:
    """列出所有模型的可用标识符映射。"""
    providers = get_configured_providers()
    result = {}

    for provider in providers:
        for model_detail in provider.models:
            full_name = f"{provider.name}/{model_detail.model_name}"
            identifiers = _get_model_identifiers(provider.name, model_detail)
            result[full_name] = identifiers

    return result


def list_embedding_models() -> list[dict[str, Any]]:
    """列出所有支持嵌入能力的模型。"""
    all_models = list_available_models()
    return [model for model in all_models if model.get("is_embedding_model", False)]


async def get_model_instance(
    provider_model_name: str | None = None,
    override_config: dict[str, Any] | GenerationConfig | None = None,
    task: str = "chat",
) -> LLMModel:
    """[内部 API] 获取底层 LLMModel 状态机实例。"""
    cache_key = _make_cache_key(provider_model_name, override_config)
    cached_model = _get_cached_model(cache_key)
    if cached_model:
        if override_config:
            validated_override = validate_override_params(override_config)
            if cached_model._generation_config != validated_override:
                cached_model._generation_config = validated_override
                logger.debug(
                    f"对缓存模型 {provider_model_name} 应用新的覆盖配置: "
                    f"{model_dump(validated_override, exclude_none=True)}"
                )
        return cached_model

    resolved_model_name_str = provider_model_name
    if resolved_model_name_str is None:
        resolved_model_name_str = get_default_model(task)
        if resolved_model_name_str is None:
            available_models_list = list_available_models()
            if not available_models_list:
                raise LLMException(
                    "未配置任何AI模型", code=LLMErrorCode.CONFIGURATION_ERROR
                )
            resolved_model_name_str = available_models_list[0]["full_name"]
            logger.warning(f"未指定模型，使用第一个可用模型: {resolved_model_name_str}")

    group_name = _get_group_name(resolved_model_name_str)
    if group_name is not None:
        resolved_models = _resolve_model_group(group_name)
        if not resolved_models:
            raise LLMException(
                f"模型路由组 '{group_name}' 解析失败或为空，请检查配置。",
                code=LLMErrorCode.CONFIGURATION_ERROR,
            )
        return RoutedLLMModel(group_name, resolved_models, override_config)  # type: ignore

    prov_name_str, mod_name_str = parse_provider_model_string(resolved_model_name_str)
    if not prov_name_str or not mod_name_str:
        raise LLMException(
            f"无效的模型名称格式: '{resolved_model_name_str}'",
            code=LLMErrorCode.MODEL_NOT_FOUND,
        )

    config_tuple_found = find_model_config(prov_name_str, mod_name_str)
    if not config_tuple_found:
        raise LLMException(
            f"未找到模型: '{resolved_model_name_str}'. ",
            code=LLMErrorCode.MODEL_NOT_FOUND,
        )

    provider_config_found, model_detail_found = config_tuple_found

    capabilities = get_model_capabilities(model_detail_found.model_name)

    capabilities = model_copy(capabilities, deep=True)

    if model_detail_found.task_type == "image_generation":
        capabilities.output_modalities.add(ModelModality.IMAGE)
        capabilities.supports_tool_calling = False

    llm_config = get_llm_config()
    client_settings = llm_config.client_settings
    default_timeout = (
        provider_config_found.timeout
        if provider_config_found.timeout is not None
        else client_settings.timeout
    )

    config_for_http_client = ProviderConfig(
        name=provider_config_found.name,
        api_key=provider_config_found.api_key,
        models=provider_config_found.models,
        timeout=default_timeout,
        api_base=provider_config_found.api_base,
        api_type=provider_config_found.api_type,
        openai_compat=provider_config_found.openai_compat,
        temperature=provider_config_found.temperature,
        generation_max_tokens=provider_config_found.generation_max_tokens,
    )

    shared_http_client = await http_client_manager.get_client(config_for_http_client)
    engine = HttpEngine(shared_http_client)

    try:
        model_instance = LLMModel(
            provider_config=config_for_http_client,
            model_detail=model_detail_found,
            health_manager=health_manager,
            engine=engine,
            capabilities=capabilities,
        )

        if override_config:
            validated_override_params = validate_override_params(override_config)
            model_instance._generation_config = validated_override_params
            logger.debug(
                f"为新模型 {resolved_model_name_str} 应用配置覆盖: "
                f"{model_dump(validated_override_params, exclude_none=True)}"
            )

        _cache_model(cache_key, model_instance)
        logger.debug(
            f"创建并缓存了新模型: {cache_key} -> {prov_name_str}/{mod_name_str}"
        )
        return model_instance
    except LLMException:
        raise
    except Exception as e:
        logger.error(
            f"实例化 LLMModel ({resolved_model_name_str}) 时发生内部错误: {e!s}", e=e
        )
        raise LLMException(
            f"初始化模型 '{resolved_model_name_str}' 失败: {e!s}",
            code=LLMErrorCode.MODEL_INIT_FAILED,
            cause=e,
        )


def get_default_model(task: str = "chat") -> str | None:
    """根据任务类型获取默认模型名称"""
    config = get_llm_config()
    return getattr(config.default_models, task, None)


def set_global_default_model_name(task: str, provider_model_name: str | None) -> bool:
    """设置全局默认模型名称。"""
    from zhenxun.services.ai.config import set_default_model as _set_default_model

    return _set_default_model(task, provider_model_name)


async def get_key_usage_stats() -> dict[str, Any]:
    """获取所有 Provider 的 Key 使用统计。"""
    providers = get_configured_providers()
    stats = {}

    for provider in providers:
        keys = (
            [provider.api_key]
            if isinstance(provider.api_key, str)
            else provider.api_key
        )
        provider_stats = {}
        provider_state = health_manager.state.providers.get(provider.name)
        if provider_state:
            for k in keys:
                stat_data = provider_state.api_keys.get(k)
                if stat_data:
                    provider_stats[health_manager._get_key_id(k)] = model_dump(
                        stat_data
                    )
        stats[provider.name] = {
            "total_keys": len(
                [provider.api_key]
                if isinstance(provider.api_key, str)
                else provider.api_key
            ),
            "key_stats": provider_stats,
        }

    return stats


async def reset_key_status(provider_name: str, api_key: str | None = None) -> bool:
    """重置指定 Provider 的 Key 状态。"""
    providers = get_configured_providers()
    target_provider = None

    for provider in providers:
        if provider.name.lower() == provider_name.lower():
            target_provider = provider
            break

    if not target_provider:
        logger.error(f"未找到Provider: {provider_name}")
        return False

    provider_keys = (
        [target_provider.api_key]
        if isinstance(target_provider.api_key, str)
        else target_provider.api_key
    )

    if api_key:
        if api_key in provider_keys:
            await health_manager.reset_key_status(target_provider.name, api_key)
            logger.info(f"已重置Provider '{provider_name}' 的指定Key状态")
            return True
        else:
            logger.error(f"指定的Key不属于Provider '{provider_name}'")
            return False
    else:
        for key in provider_keys:
            await health_manager.reset_key_status(target_provider.name, key)
        logger.info(f"已重置Provider '{provider_name}' 的所有Key状态")
        return True


@PriorityLifecycle.on_startup(priority=10)
async def _init_llm_config_on_startup():
    """启动时初始化 LLM 配置、密钥状态并预热工具提供者管理器。"""
    logger.info("正在初始化 LLM 配置并加载遥测状态...")
    try:
        get_llm_config()
        await health_manager.initialize()
        logger.debug("LLM 配置和遥测状态初始化完成。")

        from zhenxun.services.ai.tools.engine.registry import tool_provider_manager

        logger.debug("正在预热 LLM 工具提供者管理器...")
        await tool_provider_manager.initialize()
        logger.debug("LLM 工具提供者管理器预热完成。")

    except Exception as e:
        logger.error(f"LLM 配置或遥测状态初始化时发生错误: {e}", e=e)
