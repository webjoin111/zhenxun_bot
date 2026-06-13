import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any, cast

import httpx

from zhenxun.services.ai.core.configs import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import EmbedBatch, LLMResponse
from zhenxun.services.ai.llm.adapters.base import (
    BaseAdapter,
    RequestData,
    process_image_data,
)
from zhenxun.services.ai.llm.core import (
    HealthManager,
    RetryConfig,
    _should_retry_llm_error,
)
from zhenxun.services.ai.protocols import LLMContext
from zhenxun.services.ai.protocols.middleware import BaseLLMMiddleware, NextCall
from zhenxun.services.log import logger
from zhenxun.utils.http_utils import AsyncHttpx
from zhenxun.utils.log_sanitizer import sanitize_for_logging
from zhenxun.utils.pydantic_compat import dump_json_safely

from zhenxun.services.ai.protocols.llm import LLMModelBase


class RetryMiddleware(BaseLLMMiddleware):
    """
    重试中间件：处理异常捕获与重试循环
    """

    def __init__(self, retry_config: RetryConfig, health_manager: HealthManager):
        self.retry_config = retry_config
        self.health_manager = health_manager

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        last_exception: Exception | None = None
        is_routed = context.extra.get("_is_routed_call", False)
        max_retries = 0 if is_routed else self.retry_config.max_retries
        total_attempts = max_retries + 1

        for attempt in range(total_attempts):
            try:
                context.runtime_state["attempt"] = attempt + 1
                return await next_call(context)

            except LLMException as e:
                last_exception = e
                api_key = context.runtime_state.get("api_key")
                provider_name = context.runtime_state.get("provider_name")

                if api_key and provider_name:
                    status_code = e.details.get("status_code")
                    error_msg = f"({e.code.name}) {e.message}"
                    await self.health_manager.record_key_failure(
                        provider_name, api_key, status_code, error_msg
                    )

                if not getattr(e, "recoverable", True):
                    raise e.with_traceback(None) from None

                if not _should_retry_llm_error(e, attempt, max_retries):
                    raise e.with_traceback(None) from None

                if attempt == total_attempts - 1:
                    raise e.with_traceback(None) from None

                wait_time = self.retry_config.retry_delay
                if self.retry_config.exponential_backoff:
                    wait_time *= 2**attempt

                logger.warning(
                    f"请求失败，{wait_time:.2f}秒后重试"
                    f" (第{attempt + 1}/{self.retry_config.max_retries}次重试): {e}"
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"非预期异常，停止重试: {e}", e=e)
                raise e.with_traceback(None) from None

        if last_exception:
            raise last_exception.with_traceback(None) from None
        raise LLMException("重试循环异常结束").with_traceback(None) from None


class KeySelectionMiddleware(BaseLLMMiddleware):
    """
    密钥选择中间件：负责轮询获取可用 API Key
    """

    def __init__(
        self, health_manager: HealthManager, provider_name: str, api_keys: list[str]
    ):
        self.health_manager = health_manager
        self.provider_name = provider_name
        self.api_keys = api_keys
        self._failed_keys: set[str] = set()

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        is_routed = context.extra.get("_is_routed_call", False)
        selected_key = await self.health_manager.get_next_available_key(
            self.provider_name,
            self.api_keys,
            exclude_keys=self._failed_keys,
            strict_mode=is_routed,
        )

        if not selected_key:
            raise LLMException(
                f"提供商 {self.provider_name} 无可用 API Key",
                code=LLMErrorCode.NO_AVAILABLE_KEYS,
            )

        context.runtime_state["api_key"] = selected_key
        context.runtime_state["provider_name"] = self.provider_name

        try:
            response = await next_call(context)
            return response
        except LLMException as e:
            self._failed_keys.add(selected_key)
            masked = f"{selected_key[:8]}..."
            if isinstance(e.details, dict):
                e.details["api_key"] = masked
            raise e.with_traceback(None) from None


class LoggingMiddleware(BaseLLMMiddleware):
    """
    日志中间件：负责请求和响应的日志记录与脱敏
    """

    def __init__(
        self, provider_name: str, model_name: str, log_context: str = "Generation"
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.log_context = log_context

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        attempt = context.runtime_state.get("attempt", 1)
        api_key = context.runtime_state.get("api_key", "unknown")
        masked_key = f"{api_key[:8]}..."

        logger.info(
            f"🌐 发起LLM请求 (尝试 {attempt}) - {self.provider_name}/{self.model_name} "
            f"[{self.log_context}] Key: {masked_key}"
        )

        try:
            start_time = time.monotonic()
            response = await next_call(context)
            duration = (time.monotonic() - start_time) * 1000
            logger.info(f"🎯 LLM响应成功 [{self.log_context}] 耗时: {duration:.2f}ms")
            return response
        except Exception as e:
            raise e.with_traceback(None) from None


class EngineExecutionMiddleware(BaseLLMMiddleware):
    """
    底层引擎执行中间件：将 Adapter 数据推入 Engine 运行，并解析回调
    """

    def __init__(self, model_instance: LLMModelBase, adapter: BaseAdapter):
        self.model = model_instance
        self.adapter = adapter
        self.health_manager = model_instance.health_manager

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        api_key = context.runtime_state["api_key"]
        provider_name = self.model.provider_name

        request_data: RequestData
        gen_config: GenerationConfig | None = None
        embed_config: LLMEmbeddingConfig | None = None

        route_id = f"{self.model.provider_name}/{self.model.model_name}"

        if context.request_type == "embedding":
            embed_config = cast(LLMEmbeddingConfig, context.config)
            batch = cast(EmbedBatch, context.extra.get("embed_batch"))
            request_data = await self.adapter.prepare_embedding_request(
                model=self.model,
                api_key=api_key,
                batch=batch,
                config=embed_config,
            )
        elif context.request_type == "rerank":
            query = context.extra.get("query", "")
            documents = context.extra.get("documents", [])
            top_n = context.extra.get("top_n", 3)
            request_data = self.adapter.prepare_rerank_request(
                model=self.model,
                api_key=api_key,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        elif context.request_type == "image_generation":
            gen_config = cast(GenerationConfig, context.config)
            prompt = context.extra.get("prompt", "")
            images = context.extra.get("images", None)
            request_data = self.adapter.prepare_image_request(
                model=self.model,
                api_key=api_key,
                prompt=prompt,
                images=images,
                config=gen_config,
            )
        elif context.request_type == "speech_generation":
            tts_config = cast(TTSConfig, context.config)
            input_text = context.extra.get("input_text", "")
            voice = context.extra.get("voice", "")
            request_data = self.adapter.prepare_speech_request(
                model=self.model,
                api_key=api_key,
                input_text=input_text,
                voice=voice,
                config=tts_config,
            )
        else:
            gen_config = cast(GenerationConfig, context.config)
            request_data = await self.adapter.prepare_advanced_request(
                model=self.model,
                api_key=api_key,
                messages=context.messages,
                config=gen_config,
                tools=context.tools,
                tool_choice=context.tool_choice,
            )

        masked_key = (
            f"{api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}"
            if api_key
            else "N/A"
        )
        logger.debug(f"📡 请求URL: {request_data.url}")
        logger.debug(f"📋 请求头: {dict(request_data.headers)}")

        if self.model.api_type == "smart":
            effective_type = self.model._get_effective_api_type()
            sanitizer_req_context = f"{effective_type}_request"
        else:
            sanitizer_req_context = self.adapter.log_sanitization_context
        sanitized_body = sanitize_for_logging(
            request_data.body, context=sanitizer_req_context
        )

        if request_data.files and isinstance(sanitized_body, dict):
            file_info: list[str] = []
            file_count = 0
            if isinstance(request_data.files, list):
                file_count = len(request_data.files)
                for key, value in request_data.files:
                    filename = (
                        value[0]
                        if isinstance(value, tuple) and len(value) > 0
                        else "..."
                    )
                    file_info.append(f"{key}='{filename}'")
            elif isinstance(request_data.files, dict):
                file_count = len(request_data.files)
                file_info = list(request_data.files.keys())

            sanitized_body["[MULTIPART_FILES]"] = f"Count: {file_count} | {file_info}"

        request_body_str = dump_json_safely(
            sanitized_body, ensure_ascii=False, indent=2
        )
        logger.debug(f"📦 请求体: {request_body_str}")

        if context.cancellation_token:
            context.cancellation_token.raise_if_cancelled()

        start_time = time.monotonic()
        try:
            raw_engine_output = await self.model.engine.execute(context, request_data)

            if hasattr(raw_engine_output, "status_code"):
                logger.debug(f"📥 HTTP响应状态码: {raw_engine_output.status_code}")
                if exception := self.adapter.handle_http_error(raw_engine_output):
                    error_text = raw_engine_output.content.decode(
                        "utf-8", errors="ignore"
                    )
                    logger.debug(f"💥 完整错误响应: {error_text}")
                    await self.health_manager.record_key_failure(
                        provider_name,
                        api_key,
                        raw_engine_output.status_code,
                        error_text,
                    )
                    raise exception.with_traceback(None) from None
                if context.request_type == "speech_generation":
                    latency = (time.monotonic() - start_time) * 1000
                    await self.health_manager.record_key_success(provider_name, api_key)
                    await self.health_manager.record_route_success(route_id, latency)
                    audio_resp = await self.adapter.parse_speech_response(
                        self.model, raw_engine_output
                    )
                    return LLMResponse(parsed_obj=audio_resp)

                response_bytes = await raw_engine_output.aread()
                logger.debug(f"📦 响应体已完整读取 ({len(response_bytes)} bytes)")
                try:
                    response_json = json.loads(response_bytes)
                except json.JSONDecodeError:
                    raise LLMException(
                        "API 返回了非 JSON 格式的内容，可能是 URL "
                        "路径错误或中转站配置异常。",
                        code=LLMErrorCode.API_RESPONSE_INVALID,
                        details={
                            "raw_response": response_bytes.decode(
                                "utf-8", errors="ignore"
                            )[:500]
                        },
                    )
            else:
                response_json = cast(dict[str, Any], raw_engine_output)

            sanitizer_resp_context = sanitizer_req_context.replace(
                "_request", "_response"
            )
            if sanitizer_resp_context == sanitizer_req_context:
                sanitizer_resp_context = f"{sanitizer_req_context}_response"

            sanitized_response = sanitize_for_logging(
                response_json, context=sanitizer_resp_context
            )
            response_json_str = json.dumps(
                sanitized_response, ensure_ascii=False, indent=2
            )
            logger.debug(f"📋 响应JSON: {response_json_str}")

            if context.request_type == "embedding":
                self.adapter.validate_embedding_response(response_json)
                embeddings = self.adapter.parse_embedding_response(response_json)
                latency = (time.monotonic() - start_time) * 1000
                await self.health_manager.record_key_success(provider_name, api_key)
                await self.health_manager.record_route_success(route_id, latency)

                return LLMResponse(
                    content_parts=[],
                    raw_response=response_json,
                    cache_info={"embeddings": embeddings},
                    usage_info=response_json.get("usage")
                    or response_json.get("usageMetadata"),
                )

            if context.request_type == "rerank":
                rerank_results = self.adapter.parse_rerank_response(response_json)
                latency = (time.monotonic() - start_time) * 1000
                await self.health_manager.record_key_success(provider_name, api_key)
                await self.health_manager.record_route_success(route_id, latency)
                return LLMResponse(
                    content_parts=[],
                    raw_response=response_json,
                    cache_info={"rerank_results": rerank_results},
                )

            if context.request_type == "image_generation":
                response_data = self.adapter.parse_image_response(response_json)
                latency = (time.monotonic() - start_time) * 1000
                await self.health_manager.record_key_success(provider_name, api_key)
                await self.health_manager.record_route_success(route_id, latency)
                return LLMResponse(
                    content_parts=response_data.content_parts,
                    raw_response=response_json,
                )

            response_data = self.adapter.parse_response(
                self.model, response_json, is_advanced=True
            )

            should_rescue_image = (
                gen_config
                and gen_config.validation_policy
                and gen_config.validation_policy.get("require_image")
            )
            if (
                should_rescue_image
                and not response_data.images
                and response_data.text
                and gen_config
            ):
                markdown_matches = re.findall(
                    r"(!?\[.*?\]\((https?://[^\)]+)\))", response_data.text
                )
                if markdown_matches:
                    logger.info(
                        f"检测到 {len(markdown_matches)} "
                        "个资源链接，尝试自动下载并清洗。"
                    )
                    rescued_images = list(response_data.images)

                    downloaded_urls = set()
                    for full_tag, url in markdown_matches:
                        try:
                            if url not in downloaded_urls:
                                content = await AsyncHttpx.get_content(url)
                                rescued_images.append(process_image_data(content))
                                downloaded_urls.add(url)
                            response_data.text = response_data.text.replace(
                                full_tag, ""
                            )
                        except Exception as exc:
                            logger.warning(
                                f"自动下载生成的图片失败: {url}, 错误: {exc}"
                            )
                    response_data.images = rescued_images
                    response_data.text = response_data.text.strip()

            latency = (time.monotonic() - start_time) * 1000
            await self.health_manager.record_key_success(provider_name, api_key)
            await self.health_manager.record_route_success(route_id, latency)

            final_response = LLMResponse(
                content_parts=response_data.content_parts,
                usage_info=response_data.usage_info,
                raw_response=response_data.raw_response,
                grounding_metadata=response_data.grounding_metadata,
                cache_info=response_data.cache_info,
            )

            if context.request_type == "generation" and gen_config:
                if gen_config.response_validator:
                    try:
                        gen_config.response_validator(final_response)
                    except Exception as exc:
                        raise LLMException(
                            f"响应内容未通过自定义验证器: {exc}",
                            code=LLMErrorCode.API_RESPONSE_INVALID,
                            details={"validator_error": str(exc)},
                        ).with_traceback(None) from None

                policy = gen_config.validation_policy
                if policy:
                    effective_type = self.model._get_effective_api_type()
                    if policy.get("require_image") and not final_response.images:
                        if effective_type == "gemini" and response_data.raw_response:
                            usage_metadata = response_data.raw_response.get(
                                "usageMetadata", {}
                            )
                            prompt_token_details = usage_metadata.get(
                                "promptTokensDetails", []
                            )
                            prompt_had_image = any(
                                detail.get("modality") == "IMAGE"
                                for detail in prompt_token_details
                            )

                            if prompt_had_image:
                                raise LLMException(
                                    "响应验证失败：模型接收了图片输入但未生成图片。",
                                    code=LLMErrorCode.API_RESPONSE_INVALID,
                                    details={
                                        "policy": policy,
                                        "text_response": final_response.text,
                                        "raw_response": response_data.raw_response,
                                    },
                                )
                            else:
                                logger.debug(
                                    "Gemini提示词中未包含图片，跳过图片要求重试。"
                                )
                        else:
                            raise LLMException(
                                "响应验证失败：要求返回图片但未找到图片数据。",
                                code=LLMErrorCode.API_RESPONSE_INVALID,
                                details={
                                    "policy": policy,
                                    "text_response": final_response.text,
                                },
                            )

            return final_response

        except asyncio.CancelledError:
            logger.warning(f"网络请求已被取消: {request_data.url}")
            raise
        except Exception as e:
            is_route_failure = isinstance(
                e, (httpx.TimeoutException, httpx.NetworkError)
            )
            if isinstance(e, LLMException):
                status_code = e.details.get("status_code", 0)
                if status_code and status_code >= 500:
                    is_route_failure = True
            if is_route_failure:
                await self.health_manager.record_route_failure(route_id, str(e))

            if isinstance(e, LLMException):
                raise e.with_traceback(None) from None

            logger.error(f"解析响应失败或发生未知错误: {e}")

            if not isinstance(e, httpx.NetworkError | httpx.TimeoutException):
                await self.health_manager.record_key_failure(
                    provider_name, api_key, None, str(e)
                )

            raise LLMException(
                f"网络请求异常: {type(e).__name__} - {e}",
                code=LLMErrorCode.API_REQUEST_FAILED,
                details={"api_key": masked_key},
            ).with_traceback(None) from None
