"""
LLM 模型实现类

包含 LLM 模型的抽象基类和具体实现，负责与各种 AI 提供商的 API 交互。
"""

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
import json
from typing import Any

from zhenxun.services.log import logger

from .config import LLMGenerationConfig
from .config.providers import get_ai_config
from .core import (
    KeyStatusStore,
    LLMHttpClient,
    RetryConfig,
    http_client_manager,
    with_smart_retry,
)
from .types import (
    EmbeddingTaskType,
    LLMErrorCode,
    LLMException,
    LLMMessage,
    LLMResponse,
    LLMTool,
    ModelDetail,
    ProviderConfig,
)
from .utils import _sanitize_request_body_for_logging


class LLMModelBase(ABC):
    """LLM模型抽象基类"""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """生成文本"""
        pass

    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None = None,
        tools: list[LLMTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """生成高级响应"""
        pass

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: list[str],
        task_type: EmbeddingTaskType | str = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        **kwargs: Any,
    ) -> list[list[float]]:
        """生成文本嵌入向量"""
        pass


class LLMModel(LLMModelBase):
    """LLM 模型实现类"""

    def __init__(
        self,
        provider_config: ProviderConfig,
        model_detail: ModelDetail,
        key_store: KeyStatusStore,
        http_client: LLMHttpClient,
        config_override: LLMGenerationConfig | None = None,
    ):
        self.provider_config = provider_config
        self.model_detail = model_detail
        self.key_store = key_store
        self.http_client: LLMHttpClient = http_client
        self._generation_config = config_override

        self.provider_name = provider_config.name
        self.api_type = provider_config.api_type
        self.api_base = provider_config.api_base
        self.api_keys = (
            [provider_config.api_key]
            if isinstance(provider_config.api_key, str)
            else provider_config.api_key
        )
        self.model_name = model_detail.model_name
        self.temperature = model_detail.temperature
        self.max_tokens = model_detail.max_tokens

        self._is_closed = False

    async def _get_http_client(self) -> LLMHttpClient:
        """获取HTTP客户端"""
        if self.http_client.is_closed:
            logger.debug(
                f"LLMModel {self.provider_name}/{self.model_name} 的 HTTP 客户端已关闭,"
                "正在获取新的客户端"
            )
            self.http_client = await http_client_manager.get_client(
                self.provider_config
            )
        return self.http_client

    async def _select_api_key(self, failed_keys: set[str] | None = None) -> str:
        """选择可用的API密钥（使用轮询策略）"""
        if not self.api_keys:
            raise LLMException(
                f"提供商 {self.provider_name} 没有配置API密钥",
                code=LLMErrorCode.NO_AVAILABLE_KEYS,
            )

        selected_key = await self.key_store.get_next_available_key(
            self.provider_name, self.api_keys, failed_keys
        )

        if not selected_key:
            raise LLMException(
                f"提供商 {self.provider_name} 的所有API密钥当前都不可用",
                code=LLMErrorCode.NO_AVAILABLE_KEYS,
                details={
                    "total_keys": len(self.api_keys),
                    "failed_keys": len(failed_keys or set()),
                },
            )

        return selected_key

    async def _execute_embedding_request(
        self,
        adapter,
        texts: list[str],
        task_type: EmbeddingTaskType | str,
        http_client: LLMHttpClient,
        failed_keys: set[str] | None = None,
    ) -> list[list[float]]:
        """执行单次嵌入请求 - 供重试机制调用"""
        api_key = await self._select_api_key(failed_keys)

        try:
            request_data = adapter.prepare_embedding_request(
                model=self,
                api_key=api_key,
                texts=texts,
                task_type=task_type,
            )

            logger.info(
                f"🔤 发起嵌入请求 - 模型: {self.provider_name}/{self.model_name}"
            )
            logger.debug(f"📡 嵌入请求URL: {request_data.url}")
            masked_key = (
                f"{api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}"
            )
            logger.debug(f"🔑 API密钥: {masked_key}")
            logger.debug(f"📋 嵌入请求头: {dict(request_data.headers)}")
            logger.debug(f"📝 嵌入文本数量: {len(texts)}")

            http_response = await http_client.post(
                request_data.url,
                headers=request_data.headers,
                json=request_data.body,
            )

            if http_response.status_code != 200:
                error_text = http_response.text
                logger.error(
                    f"HTTP嵌入请求失败: {http_response.status_code} - {error_text}"
                )
                await self.key_store.record_failure(api_key, http_response.status_code)

                error_code = LLMErrorCode.API_REQUEST_FAILED
                if http_response.status_code in [401, 403]:
                    error_code = LLMErrorCode.API_KEY_INVALID
                elif http_response.status_code == 429:
                    error_code = LLMErrorCode.API_RATE_LIMITED

                raise LLMException(
                    f"HTTP嵌入请求失败: {http_response.status_code}",
                    code=error_code,
                    details={
                        "status_code": http_response.status_code,
                        "response": error_text,
                        "api_key": api_key,
                    },
                )

            try:
                response_json = http_response.json()
                adapter.validate_embedding_response(response_json)
                embeddings = adapter.parse_embedding_response(response_json)
            except Exception as e:
                logger.error(f"解析嵌入响应失败: {e}", e=e)
                await self.key_store.record_failure(api_key, None)
                if isinstance(e, LLMException):
                    raise
                else:
                    raise LLMException(
                        f"解析API嵌入响应失败: {e}",
                        code=LLMErrorCode.RESPONSE_PARSE_ERROR,
                        cause=e,
                    )

            await self.key_store.record_success(api_key)
            return embeddings

        except LLMException:
            raise
        except Exception as e:
            logger.error(f"生成嵌入时发生未预期错误: {e}", e=e)
            await self.key_store.record_failure(api_key, None)
            raise LLMException(
                f"生成嵌入失败: {e}",
                code=LLMErrorCode.EMBEDDING_FAILED,
                cause=e,
            )

    async def _execute_with_smart_retry(
        self,
        adapter,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None,
        tools: list[LLMTool] | None,
        tool_choice: str | dict[str, Any] | None,
        http_client: LLMHttpClient,
    ):
        """智能重试机制 - 使用统一的重试装饰器"""
        ai_config = get_ai_config()
        max_retries = ai_config.get("max_retries_llm", 3)
        retry_delay = ai_config.get("retry_delay_llm", 2)
        retry_config = RetryConfig(max_retries=max_retries, retry_delay=retry_delay)

        return await with_smart_retry(
            self._execute_single_request,
            adapter,
            messages,
            config,
            tools,
            tool_choice,
            http_client,
            retry_config=retry_config,
            key_store=self.key_store,
            provider_name=self.provider_name,
        )

    async def _execute_single_request(
        self,
        adapter,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None,
        tools: list[LLMTool] | None,
        tool_choice: str | dict[str, Any] | None,
        http_client: LLMHttpClient,
        failed_keys: set[str] | None = None,
    ) -> LLMResponse:
        """执行单次请求 - 供重试机制调用，直接返回 LLMResponse"""
        api_key = await self._select_api_key(failed_keys)

        try:
            request_data = await adapter.prepare_advanced_request(
                model=self,
                api_key=api_key,
                messages=messages,
                config=config,
                tools=tools,
                tool_choice=tool_choice,
            )

            logger.info(
                f"🌐 发起LLM请求 - 模型: {self.provider_name}/{self.model_name}"
            )
            logger.debug(f"📡 请求URL: {request_data.url}")
            masked_key = (
                f"{api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}"
            )
            logger.debug(f"🔑 API密钥: {masked_key}")
            logger.debug(f"📋 请求头: {dict(request_data.headers)}")

            sanitized_body = _sanitize_request_body_for_logging(request_data.body)
            request_body_str = json.dumps(sanitized_body, ensure_ascii=False, indent=2)
            logger.debug(f"📦 请求体: {request_body_str}")

            http_response = await http_client.post(
                request_data.url,
                headers=request_data.headers,
                json=request_data.body,
            )

            logger.debug(f"📥 响应状态码: {http_response.status_code}")
            logger.debug(f"📄 响应头: {dict(http_response.headers)}")

            if http_response.status_code != 200:
                error_text = http_response.text
                logger.error(
                    f"❌ HTTP请求失败: {http_response.status_code} - {error_text}"
                )
                logger.debug(f"💥 完整错误响应: {error_text}")

                await self.key_store.record_failure(api_key, http_response.status_code)

                if http_response.status_code in [401, 403]:
                    error_code = LLMErrorCode.API_KEY_INVALID
                elif http_response.status_code == 429:
                    error_code = LLMErrorCode.API_RATE_LIMITED
                elif http_response.status_code in [402, 413]:
                    error_code = LLMErrorCode.API_QUOTA_EXCEEDED
                else:
                    error_code = LLMErrorCode.API_REQUEST_FAILED

                raise LLMException(
                    f"HTTP请求失败: {http_response.status_code}",
                    code=error_code,
                    details={
                        "status_code": http_response.status_code,
                        "response": error_text,
                        "api_key": api_key,
                    },
                )

            try:
                response_json = http_response.json()
                response_json_str = json.dumps(
                    response_json, ensure_ascii=False, indent=2
                )
                logger.debug(f"📋 响应JSON: {response_json_str}")

                response_data = adapter.parse_response(
                    model=self,
                    response_json=response_json,
                    is_advanced=True,
                )

                from .types.models import LLMToolCall

                response_tool_calls = []
                if response_data.tool_calls:
                    for tc_data in response_data.tool_calls:
                        if isinstance(tc_data, LLMToolCall):
                            response_tool_calls.append(tc_data)
                        elif isinstance(tc_data, dict):
                            try:
                                response_tool_calls.append(LLMToolCall(**tc_data))
                            except Exception as e:
                                logger.warning(
                                    f"无法将工具调用数据转换为LLMToolCall: {tc_data}, "
                                    f"error: {e}"
                                )
                        else:
                            logger.warning(f"工具调用数据格式未知: {tc_data}")

                llm_response = LLMResponse(
                    text=response_data.text,
                    usage_info=response_data.usage_info,
                    raw_response=response_data.raw_response,
                    tool_calls=response_tool_calls if response_tool_calls else None,
                    code_executions=response_data.code_executions,
                    grounding_metadata=response_data.grounding_metadata,
                    cache_info=response_data.cache_info,
                )

            except Exception as e:
                logger.error(f"解析响应失败: {e}", e=e)
                await self.key_store.record_failure(api_key, None)

                if isinstance(e, LLMException):
                    raise
                else:
                    raise LLMException(
                        f"解析API响应失败: {e}",
                        code=LLMErrorCode.RESPONSE_PARSE_ERROR,
                        cause=e,
                    )

            await self.key_store.record_success(api_key)
            logger.debug(f"✅ API密钥使用成功: {masked_key}")
            logger.info(
                f"🎯 LLM响应解析完成 - 文本长度: {len(llm_response.text or '')}"
            )

            return llm_response

        except LLMException:
            raise
        except Exception as e:
            logger.error(f"生成响应时发生未预期错误: {e}", e=e)
            await self.key_store.record_failure(api_key, None)

            raise LLMException(
                f"生成响应失败: {e}",
                code=LLMErrorCode.GENERATION_FAILED,
                cause=e,
            )

    async def close(self):
        """
        标记模型实例的当前使用周期结束。
        共享的 HTTP 客户端由 LLMHttpClientManager 管理，不由 LLMModel 关闭。
        """
        if self._is_closed:
            return
        self._is_closed = True
        logger.debug(
            f"LLMModel实例的使用周期已结束: {self} (共享HTTP客户端状态不受影响)"
        )

    async def __aenter__(self):
        if self._is_closed:
            logger.debug(
                f"Re-entering context for closed LLMModel {self}. "
                f"Resetting _is_closed to False."
            )
            self._is_closed = False
        self._check_not_closed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        _ = exc_type, exc_val, exc_tb
        await self.close()

    def _check_not_closed(self):
        """检查实例是否已关闭"""
        if self._is_closed:
            raise RuntimeError(f"LLMModel实例已关闭: {self}")

    async def generate_text(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """生成文本 - 通过 generate_response 实现"""
        self._check_not_closed()

        messages: list[LLMMessage] = []

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content_text = msg.get("content", "")
                messages.append(LLMMessage(role=role, content=content_text))

        messages.append(LLMMessage.user(prompt))

        model_fields = getattr(LLMGenerationConfig, "model_fields", {})
        request_specific_config_dict = {
            k: v for k, v in kwargs.items() if k in model_fields
        }
        request_specific_config = None
        if request_specific_config_dict:
            request_specific_config = LLMGenerationConfig(
                **request_specific_config_dict
            )

        for key in request_specific_config_dict:
            kwargs.pop(key, None)

        response = await self.generate_response(
            messages,
            config=request_specific_config,
            **kwargs,
        )
        return response.text

    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None = None,
        tools: list[LLMTool] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        生成高级响应。
        """
        self._check_not_closed()

        from .adapters import get_adapter_for_api_type
        from .config.generation import create_generation_config_from_kwargs

        adapter = get_adapter_for_api_type(self.api_type)
        if not adapter:
            raise LLMException(
                f"未找到适用于 API 类型 '{self.api_type}' 的适配器",
                code=LLMErrorCode.CONFIGURATION_ERROR,
            )

        final_request_config = self._generation_config or LLMGenerationConfig()
        if kwargs:
            kwargs_config = create_generation_config_from_kwargs(**kwargs)
            merged_dict = final_request_config.to_dict()
            merged_dict.update(kwargs_config.to_dict())
            final_request_config = LLMGenerationConfig(**merged_dict)

        if config is not None:
            merged_dict = final_request_config.to_dict()
            merged_dict.update(config.to_dict())
            final_request_config = LLMGenerationConfig(**merged_dict)

        http_client = await self._get_http_client()

        async with AsyncExitStack() as stack:
            activated_tools = []
            if tools:
                for tool in tools:
                    if tool.type == "mcp" and callable(tool.mcp_session):
                        func_obj = getattr(tool.mcp_session, "func", None)
                        tool_name = (
                            getattr(func_obj, "__name__", "unknown")
                            if func_obj
                            else "unknown"
                        )
                        logger.debug(f"正在激活 MCP 工具会话: {tool_name}")

                        active_session = await stack.enter_async_context(
                            tool.mcp_session()
                        )

                        activated_tools.append(
                            LLMTool.from_mcp_session(
                                session=active_session, annotations=tool.annotations
                            )
                        )
                    else:
                        activated_tools.append(tool)

            llm_response = await self._execute_with_smart_retry(
                adapter,
                messages,
                final_request_config,
                activated_tools if activated_tools else None,
                tool_choice,
                http_client,
            )

        return llm_response

    async def generate_embeddings(
        self,
        texts: list[str],
        task_type: EmbeddingTaskType | str = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        **kwargs: Any,
    ) -> list[list[float]]:
        """生成文本嵌入向量"""
        self._check_not_closed()
        if not texts:
            return []

        from .adapters import get_adapter_for_api_type

        adapter = get_adapter_for_api_type(self.api_type)
        if not adapter:
            raise LLMException(
                f"未找到适用于 API 类型 '{self.api_type}' 的嵌入适配器",
                code=LLMErrorCode.CONFIGURATION_ERROR,
            )

        http_client = await self._get_http_client()

        ai_config = get_ai_config()
        default_max_retries = ai_config.get("max_retries_llm", 3)
        default_retry_delay = ai_config.get("retry_delay_llm", 2)
        max_retries_embed = kwargs.get(
            "max_retries_embed", max(1, default_max_retries // 2)
        )
        retry_delay_embed = kwargs.get("retry_delay_embed", default_retry_delay / 2)

        retry_config = RetryConfig(
            max_retries=max_retries_embed,
            retry_delay=retry_delay_embed,
            exponential_backoff=True,
            key_rotation=True,
        )

        return await with_smart_retry(
            self._execute_embedding_request,
            adapter,
            texts,
            task_type,
            http_client,
            retry_config=retry_config,
            key_store=self.key_store,
            provider_name=self.provider_name,
        )

    def __str__(self) -> str:
        status = "closed" if self._is_closed else "active"
        return f"LLMModel({self.provider_name}/{self.model_name}, {status})"

    def __repr__(self) -> str:
        status = "closed" if self._is_closed else "active"
        return (
            f"LLMModel(provider={self.provider_name}, model={self.model_name}, "
            f"api_type={self.api_type}, status={status})"
        )
