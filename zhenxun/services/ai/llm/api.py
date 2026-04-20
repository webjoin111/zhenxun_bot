"""
LLM 服务的高级 API 接口 - 便捷函数入口 (无状态)
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast, overload

from pydantic import BaseModel

from zhenxun.services.ai.types.configs import (
    LLMEmbeddingConfig,
    LLMGenerationConfig,
    OutputConfig,
)
from zhenxun.services.ai.types.exceptions import (
    LLMErrorCode,
    LLMException,
    get_user_friendly_error_message,
)
from zhenxun.services.ai.types.messages import (
    LLMContentPart,
    LLMMessage,
    LLMResponse,
    RerankResult,
)
from zhenxun.services.ai.types.models import ModelName
from zhenxun.services.ai.types.tools import GeminiGoogleSearch, ToolChoice
from zhenxun.services.log import logger

from .config import CommonOverrides, GenConfigBuilder
from .hooks import _GLOBAL_AFTER_HOOKS, _GLOBAL_BEFORE_HOOKS
from .manager import get_model_instance

T = TypeVar("T", bound=BaseModel)


async def chat(
    message: str | Any | LLMMessage | list[LLMContentPart] | list[LLMMessage],
    *,
    model: ModelName = None,
    instruction: str | None = None,
    tools: list[Any] | None = None,
    tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    config: LLMGenerationConfig | GenConfigBuilder | None = None,
    timeout: float | None = None,
) -> LLMResponse:
    """
    无状态的聊天对话便捷函数，通过临时的AI会话实例与LLM模型交互。

    参数:
        message: 用户输入的消息内容，支持多种格式。
        model: 要使用的模型名称，如果为None则使用默认模型。
        instruction: 系统指令，用于指导AI的行为和回复风格。
        tools: 可用的工具列表，支持字典配置或字符串标识符。
        tool_choice: 工具选择策略，控制AI如何选择和使用工具。
        config: (可选) 生成配置对象，将与默认配置合并后传递。
        timeout: (可选) HTTP 请求超时时间（秒）。

    返回:
        LLMResponse: 包含AI回复内容、使用信息和工具调用等的完整响应对象。
    """
    try:
        from zhenxun.services.ai.message_builder import MessageBuilder

        messages = await MessageBuilder.normalize_to_llm_messages(
            message, instruction=instruction
        )
        return await generate(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
            timeout=timeout,
        )
    except LLMException:
        raise
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"执行 chat 函数失败: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(f"聊天执行失败: {friendly_msg}", cause=e)


async def code(
    prompt: str,
    *,
    model: ModelName = None,
    timeout: int | None = None,
) -> LLMResponse:
    """
    无状态的代码执行便捷函数，支持在沙箱环境中执行代码。

    参数:
        prompt: 代码执行的提示词，描述要执行的代码任务。
        model: 要使用的模型名称，默认使用Gemini/gemini-2.0-flash。
        timeout: 代码执行超时时间（秒），防止长时间运行的代码阻塞。

    返回:
        LLMResponse: 包含代码执行结果的完整响应对象。
    """
    resolved_model = model

    config = CommonOverrides.gemini_code_execution()
    if timeout:
        config.custom_params = config.custom_params or {}
        config.custom_params["code_execution_timeout"] = timeout

    return await chat(prompt, model=resolved_model, config=config)


async def embed(
    texts: list[str] | str,
    *,
    model: ModelName = None,
    config: LLMEmbeddingConfig | None = None,
) -> list[list[float]]:
    """
    无状态的文本嵌入便捷函数，将文本转换为向量表示。

    参数:
        texts: 要生成嵌入的文本内容，支持单个字符串或字符串列表。
        model: 要使用的嵌入模型名称，如果为None则使用默认模型。
        config: 嵌入配置对象。

    返回:
        list[list[float]]: 文本对应的嵌入向量列表，每个向量为浮点数列表。
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []

    final_config = config or LLMEmbeddingConfig()

    try:
        async with await get_model_instance(model) as model_instance:
            return await model_instance.generate_embeddings(texts, config=final_config)
    except LLMException:
        raise
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"文本嵌入失败: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(
            f"文本嵌入失败: {friendly_msg}",
            code=LLMErrorCode.EMBEDDING_FAILED,
            cause=e,
        )


async def rerank(
    query: str,
    documents: list[str | dict[str, str]],
    top_n: int = 3,
    *,
    model: ModelName = None,
) -> list[RerankResult]:
    """
    无状态的文本重排便捷函数。

    参数:
        query: 用户查询问题
        documents: 候选文档列表 (支持纯文本或 {"image": "url", "text": "xxx"} 图文格式)
        top_n: 返回匹配度最高的前 n 个文档
        model: 重排模型名称 (如 BAAI/bge-reranker-v2-m3)
    """
    try:
        async with await get_model_instance(model) as model_instance:
            return await model_instance.rerank(query, documents, top_n)
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"文档重排失败: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(f"文档重排失败: {friendly_msg}", cause=e)


async def embed_query(
    text: str,
    *,
    model: ModelName = None,
    dimensions: int | None = None,
) -> list[float]:
    """
    语义化便捷 API：为检索查询生成嵌入。
    """
    config = LLMEmbeddingConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=dimensions,
    )
    vectors = await embed([text], model=model, config=config)
    return vectors[0] if vectors else []


async def embed_documents(
    texts: list[str],
    *,
    model: ModelName = None,
    dimensions: int | None = None,
    title: str | None = None,
) -> list[list[float]]:
    """
    语义化便捷 API：为文档集合生成嵌入。
    """
    config = LLMEmbeddingConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=dimensions,
        title=title,
    )
    return await embed(texts, model=model, config=config)


async def generate_structured(
    message: str | Any | LLMMessage | list[LLMContentPart] | list[LLMMessage],
    response_model: type[T],
    *,
    model: ModelName = None,
    tools: list[Any] | None = None,
    tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    max_validation_retries: int | None = None,
    validation_callback: Callable[[T], Any | Awaitable[Any]] | None = None,
    error_prompt_template: str | None = None,
    auto_thinking: bool = False,
    instruction: str | None = None,
    timeout: float | None = None,
) -> T:
    """
    无状态地生成结构化响应，并自动解析为指定的Pydantic模型。

    参数:
        message: 用户输入的消息内容，支持多种格式。
        response_model: 用于解析和验证响应的Pydantic模型类。
        max_validation_retries: 校验失败时的最大重试次数，默认为 None (使用全局配置)。
        validation_callback: 自定义校验回调函数，抛出异常视为校验失败。
        error_prompt_template: 自定义错误反馈提示词模板。
        auto_thinking: 是否自动开启思维链 (CoT) 包装。适用于不支持原生思考的模型
        model: 要使用的模型名称，如果为None则使用默认模型。
        instruction: 系统指令，用于指导AI生成符合要求的结构化输出。
        timeout: HTTP 请求超时时间（秒）。

    返回:
        T: 解析后的Pydantic模型实例，类型为response_model指定的类型。
    """
    try:
        import json

        from zhenxun.services.ai.config import get_llm_config
        from zhenxun.services.ai.llm.manager import get_global_default_model_name
        from zhenxun.services.ai.llm.utils import (
            create_cot_wrapper,
            should_apply_autocot,
        )
        from zhenxun.services.ai.types.configs import (
            OutputConfig,
            StructuredOutputStrategy,
        )
        from zhenxun.services.ai.types.messages import ResponseFormat
        from zhenxun.utils.pydantic_compat import model_json_schema

        resolved_model_name = model or get_global_default_model_name()
        if max_validation_retries is None:
            max_validation_retries = get_llm_config().client_settings.structured_retries

        effective_auto_thinking = should_apply_autocot(
            auto_thinking, resolved_model_name, None
        )

        target_model: type[T] = response_model
        if effective_auto_thinking:
            target_model = cast(type[T], create_cot_wrapper(response_model))
            response_model = target_model
            cot_instruction = (
                "请务必先在 `reasoning` 字段中进行详细的一步步推理，"
                "确保逻辑正确，然后再填充 `result` 字段。"
            )
            instruction = (
                f"{instruction}\n\n{cot_instruction}"
                if instruction
                else cot_instruction
            )

        try:
            json_schema = model_json_schema(response_model)
        except AttributeError:
            json_schema = response_model.schema()

        schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2)
        prompt_prefix = f"{instruction}\n\n" if instruction else ""
        system_prompt = (
            prompt_prefix + "### 📝 [结构化输出任务]\n"
            "请严格按照指定的 **JSON Schema** 格式进行响应。具体要求：\n"
            "- **禁止**包含任何额外的解释、Markdown 装饰符或代码块包裹。\n"
            "- 必须返回一个且仅一个合法的 JSON 对象。\n\n"
            "#### 预期 Schema 定义：\n"
            f"```json\n{schema_str}\n```"
        )

        from zhenxun.services.ai.message_builder import MessageBuilder

        messages = await MessageBuilder.normalize_to_llm_messages(
            message if message is not None else [], instruction=system_prompt
        )

        structured_config = LLMGenerationConfig(
            output=OutputConfig(
                response_format=ResponseFormat.JSON,
                response_schema=json_schema,
                structured_output_strategy=StructuredOutputStrategy.NATIVE,
            )
        )

        extra_context = {
            "response_model": target_model,
            "max_validation_retries": max_validation_retries,
            "validation_callback": validation_callback,
            "error_prompt_template": error_prompt_template,
            "is_auto_thinking": effective_auto_thinking,
        }

        response = await generate(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            config=structured_config,
            timeout=timeout,
            extra=extra_context,
        )

        if not hasattr(response, "parsed_obj") or response.parsed_obj is None:
            raise LLMException("结构化输出失败：中间件未返回解析后的对象。")

        return response.parsed_obj
    except LLMException:
        raise
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"生成结构化响应失败: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(f"生成结构化响应失败: {friendly_msg}", cause=e)


async def generate(
    messages: list[LLMMessage],
    *,
    model: ModelName = None,
    tools: list[Any] | None = None,
    tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    config: LLMGenerationConfig | GenConfigBuilder | None = None,
    timeout: float | None = None,
    extra: dict[str, Any] | None = None,
) -> LLMResponse:
    """
    根据完整的消息列表生成一次性响应，这是一个无状态的底层函数。

    参数:
        messages: 完整的消息历史列表，包括系统指令、用户消息和助手回复。
        model: 要使用的模型名称，如果为None则使用默认模型。
        tools: 可用的工具列表，支持字典配置或字符串标识符。
        tool_choice: 工具选择策略，控制AI如何选择和使用工具。
        config: (可选) 生成配置对象，将与默认配置合并后传递。

    返回:
        LLMResponse: 包含AI回复内容、使用信息和工具调用等的完整响应对象。
    """
    try:
        resolved_config: LLMGenerationConfig | None = None
        if isinstance(config, GenConfigBuilder):
            resolved_config = config.build()
        else:
            resolved_config = config

        resolved_tools = None
        if tools:
            from zhenxun.services.ai.tools.core.context import RunContext
            from zhenxun.services.ai.tools.engine.registry import tool_provider_manager

            tools_to_resolve = tools if isinstance(tools, list) else [tools]
            payload = await tool_provider_manager.resolve_tools(
                tools_to_resolve, context=RunContext()
            )
            resolved_tools = list(payload.tools)
            if payload.injected_prompts:
                messages.insert(
                    0, LLMMessage.system("\n\n".join(payload.injected_prompts))
                )

        hook_kwargs = {
            "model": model,
            "config": resolved_config,
            "tools": resolved_tools,
            "session_id": "stateless",
        }
        for hook in _GLOBAL_BEFORE_HOOKS:
            messages = await hook(messages, hook_kwargs)

        async with await get_model_instance(
            model, override_config=None
        ) as model_instance:
            response = await model_instance.generate_response(
                messages,
                config=resolved_config,
                tools=resolved_tools,
                tool_choice=tool_choice,
                timeout=timeout,
                extra=extra or {},
            )

        for hook in _GLOBAL_AFTER_HOOKS:
            response = await hook(response, hook_kwargs)

        return response
    except LLMException:
        raise
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"生成响应失败: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(f"生成响应失败: {friendly_msg}", cause=e)


async def _generate_image_from_message(
    message: Any,
    model: ModelName = None,
    config: LLMGenerationConfig | GenConfigBuilder | None = None,
) -> LLMResponse:
    """
    [内部] 从 UniMessage 生成图片的核心辅助函数。
    """
    if isinstance(config, GenConfigBuilder):
        config = config.build()

    config = config or LLMGenerationConfig()

    config.validation_policy = {"require_image": True}
    if config.output is None:
        config.output = OutputConfig()
    config.output.response_modalities = ["IMAGE", "TEXT"]

    try:
        from zhenxun.services.ai.message_builder import MessageBuilder

        messages = await MessageBuilder.normalize_to_llm_messages(message)

        async with await get_model_instance(model) as model_instance:
            response = await model_instance.generate_response(messages, config=config)

            if not response.images:
                error_text = response.text or "模型未返回图片数据。"
                logger.warning(f"图片生成调用未返回图片，返回文本内容: {error_text}")

            return response
    except LLMException:
        raise
    except Exception as e:
        friendly_msg = get_user_friendly_error_message(e)
        logger.error(f"执行图片生成时发生未知错误: {e} | 建议: {friendly_msg}", e=e)
        raise LLMException(f"图片生成失败: {friendly_msg}", cause=e)


@overload
async def create_image(
    prompt: str | Any,
    *,
    images: None = None,
    model: ModelName = None,
) -> LLMResponse:
    """根据文本提示生成一张新图片。"""
    ...


@overload
async def create_image(
    prompt: str | Any,
    *,
    images: list[Path | bytes | str] | Path | bytes | str,
    model: ModelName = None,
) -> LLMResponse:
    """在给定图片的基础上，根据文本提示进行编辑或重新生成。"""
    ...


async def create_image(
    prompt: str | Any,
    *,
    images: list[Path | bytes | str] | Path | bytes | str | None = None,
    model: ModelName = None,
    config: LLMGenerationConfig | GenConfigBuilder | None = None,
) -> LLMResponse:
    """
    智能图片生成/编辑函数。
    - 如果 `images` 为 None，执行文生图。
    - 如果提供了 `images`，执行图+文生图，支持多张图片输入。
    """
    text_prompt = getattr(prompt, "extract_plain_text", lambda: str(prompt))()

    image_list = []
    if images:
        if isinstance(images, list):
            image_list.extend(images)
        else:
            image_list.append(images)

    from zhenxun.services.ai.message_builder import MessageBuilder

    message = MessageBuilder.create_multimodal_message(
        text=text_prompt, images=image_list
    )

    return await _generate_image_from_message(message, model=model, config=config)


async def search(
    query: str | Any | LLMMessage | list[LLMContentPart],
    *,
    model: ModelName = None,
    instruction: str = (
        "你是一位强大的信息检索和整合专家。请利用可用的搜索工具，"
        "根据用户的查询找到最相关的信息，并进行总结和回答。"
    ),
    config: LLMGenerationConfig | GenConfigBuilder | None = None,
) -> LLMResponse:
    """
    无状态的信息搜索便捷函数，利用搜索工具获取实时信息。

    参数:
        query: 搜索查询内容，支持多种输入格式。
        model: 要使用的模型名称，如果为None则使用默认模型。
        config: (可选) 生成配置对象，将与预设配置合并后传递。
        instruction: 搜索任务的系统指令，指导AI如何处理搜索结果。

    返回:
        LLMResponse: 包含搜索结果和AI整合回复的完整响应对象。
    """
    logger.debug("执行无状态 'search' 任务...")
    search_config = CommonOverrides.gemini_grounding()

    if isinstance(config, GenConfigBuilder):
        config = config.build()

    final_config = search_config.merge_with(config)

    return await chat(
        query,
        model=model,
        instruction=instruction,
        config=final_config,
        tools=[GeminiGoogleSearch()],
    )
