"""
LLM 模型实现类

包含 LLM 模型的抽象基类和具体实现，负责与各种 AI 提供商的 API 交互。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from zhenxun.services.ai.run.models import CancellationToken

from pydantic import BaseModel

from zhenxun.services.ai.config import ProviderConfig, get_llm_config
from zhenxun.services.ai.core.configs import (
    GenerationConfig,
    LLMEmbeddingConfig,
    TTSConfig,
)
from zhenxun.services.ai.core.engine.token_estimator import parse_usage_info
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import (
    AudioResponse,
    EmbedBatch,
    EmbeddingResponse,
    LLMMessage,
    LLMResponse,
    UsageInfo,
)
from zhenxun.services.ai.core.models import (
    ModelCapabilities,
    ModelDetail,
    ModelModality,
    ToolChoice,
)
from zhenxun.services.ai.llm.engine import BaseEngine
from zhenxun.services.ai.protocols import LLMContext
from zhenxun.services.ai.protocols.llm import LLMModelBase
from zhenxun.services.ai.protocols.middleware import LLMMiddleware, NextCall
from zhenxun.services.log import logger

from .core import (
    HealthManager,
    RetryConfig,
)

T = TypeVar("T", bound=BaseModel)


class LLMModel(LLMModelBase):
    """LLM 模型实现类"""

    def __init__(
        self,
        provider_config: ProviderConfig,
        model_detail: ModelDetail,
        health_manager: HealthManager,
        engine: BaseEngine,
        capabilities: ModelCapabilities,
        config_override: GenerationConfig | None = None,
    ):
        self.provider_config = provider_config
        self.model_detail = model_detail
        self.health_manager = health_manager
        self.engine: BaseEngine = engine
        self.capabilities = capabilities
        self._generation_config = config_override

        self.provider_name = provider_config.name
        self.api_type = model_detail.api_type or provider_config.api_type
        self.api_base = provider_config.api_base
        self.path_prefix = model_detail.path_prefix
        self.api_keys = (
            [provider_config.api_key]
            if isinstance(provider_config.api_key, str)
            else provider_config.api_key
        )
        self.model_name = model_detail.model_name
        self.temperature = model_detail.temperature
        self.generation_max_tokens = model_detail.generation_max_tokens

        self._is_closed = False
        self._ref_count = 0
        self._middlewares: list[LLMMiddleware] = []

    def add_middleware(self, middleware: LLMMiddleware) -> None:
        """注册一个中间件到处理管道的最外层"""
        self._middlewares.append(middleware)

    def _build_pipeline(self) -> NextCall:
        """
        构建完整的中间件调用链。顺序为：
        用户自定义中间件 -> Retry -> Logging -> KeySelection -> Network (终结者)
        """
        from .adapters import get_adapter_for_api_type
        from .middlewares import (
            EngineExecutionMiddleware,
            KeySelectionMiddleware,
            LoggingMiddleware,
            RetryMiddleware,
        )

        client_settings = get_llm_config().client_settings
        retry_config = RetryConfig(
            max_retries=client_settings.max_retries,
            retry_delay=client_settings.retry_delay,
        )
        adapter = get_adapter_for_api_type(self.api_type)

        engine_middleware = EngineExecutionMiddleware(self, adapter)

        async def terminal_handler(ctx: LLMContext) -> LLMResponse:
            async def _noop(_: LLMContext) -> LLMResponse:
                raise RuntimeError("EngineExecutionMiddleware 不应调用 next_call")

            return await engine_middleware(ctx, cast(NextCall, _noop))

        def _wrap(middleware: LLMMiddleware, next_call: NextCall) -> NextCall:
            async def _handler(inner_ctx: LLMContext) -> LLMResponse:
                return await middleware(inner_ctx, next_call)

            return cast(NextCall, _handler)

        handler: NextCall = cast(NextCall, terminal_handler)
        handler = _wrap(
            KeySelectionMiddleware(
                self.health_manager, self.provider_name, self.api_keys
            ),
            handler,
        )
        handler = _wrap(
            LoggingMiddleware(self.provider_name, self.model_name),
            handler,
        )
        handler = _wrap(
            RetryMiddleware(retry_config, self.health_manager),
            handler,
        )

        for middleware in reversed(self._middlewares):
            handler = _wrap(middleware, handler)

        return handler

    def _get_effective_api_type(self) -> str:
        """
        获取实际生效的 API 类型。
        主要用于 Smart 模式下，判断日志净化应该使用哪种格式。
        """
        if self.api_type != "smart":
            return self.api_type

        if self.model_detail.api_type:
            return self.model_detail.api_type
        if (
            "gemini" in self.model_name.lower()
            and "openai" not in self.model_name.lower()
        ):
            return "gemini"
        if "minimax" in self.model_name.lower():
            return "minimax"
        return "openai"

    async def _select_api_key(self, failed_keys: set[str] | None = None) -> str:
        """选择可用的API密钥（使用轮询策略）"""
        if not self.api_keys:
            raise LLMException(
                f"提供商 {self.provider_name} 没有配置API密钥",
                code=LLMErrorCode.NO_AVAILABLE_KEYS,
            )

        selected_key = await self.health_manager.get_next_available_key(
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

    async def close(self):
        """标记模型实例的当前使用周期结束"""
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
        self._ref_count += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        _ = exc_type, exc_val, exc_tb
        self._ref_count -= 1
        if self._ref_count <= 0:
            self._ref_count = 0
            await self.close()

    def _check_not_closed(self):
        """检查实例是否已关闭"""
        if self._is_closed:
            raise RuntimeError(f"LLMModel实例已关闭: {self}")

    async def _execute_core_generation(self, context: LLMContext) -> LLMResponse:
        """
        [内核] 执行核心生成逻辑：构建管道并执行。
        此方法作为中间件管道的终点被调用。
        """
        pipeline_handler = self._build_pipeline()
        return cast(LLMResponse, await pipeline_handler(context))

    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: "CancellationToken | None" = None,
    ) -> LLMResponse:
        """
        生成高级响应 (支持中间件管道)。
        """
        self._check_not_closed()

        filtered_messages = []
        from zhenxun.services.ai.core.messages import AudioPart, ImagePart, VideoPart
        from zhenxun.utils.pydantic_compat import model_copy

        for msg in messages:
            new_content = []
            for part in msg.content:
                if (
                    isinstance(part, ImagePart)
                    and ModelModality.IMAGE not in self.capabilities.input_modalities
                ):
                    logger.warning(
                        f"模型 {self.model_name} 不支持图像输入，已自动过滤图片内容。"
                    )
                    continue
                if (
                    isinstance(part, AudioPart)
                    and ModelModality.AUDIO not in self.capabilities.input_modalities
                ):
                    logger.warning(
                        f"模型 {self.model_name} 不支持音频输入，已自动过滤音频内容。"
                    )
                    continue
                if (
                    isinstance(part, VideoPart)
                    and ModelModality.VIDEO not in self.capabilities.input_modalities
                ):
                    logger.warning(
                        f"模型 {self.model_name} 不支持视频输入，已自动过滤视频内容。"
                    )
                    continue
                new_content.append(part)
            filtered_messages.append(model_copy(msg, update={"content": new_content}))
        messages = filtered_messages

        if self._generation_config and config:
            final_request_config = self._generation_config.merge_with(config)
        elif config:
            final_request_config = config
        else:
            final_request_config = self._generation_config or GenerationConfig()

        normalized_tools: list[Any] | None = None
        if tools:
            if isinstance(tools, dict):
                normalized_tools = list(tools.values())
            elif isinstance(tools, list):
                normalized_tools = tools
            else:
                normalized_tools = [tools]

        context = LLMContext(
            messages=messages,
            config=final_request_config,
            tools=normalized_tools,
            tool_choice=tool_choice,
            timeout=timeout,
            extra=extra or {},
            cancellation_token=cancellation_token,
        )

        capabilities = (extra or {}).get("__sys_capabilities", [])
        run_ctx = (extra or {}).get("run_context")
        if not run_ctx:
            from zhenxun.services.ai.run import RunContext

            run_ctx = RunContext()

        from zhenxun.services.ai.protocols.capabilities import CombinedCapability

        combined_cap = CombinedCapability(capabilities)

        async def inner_handler(ctx: LLMContext) -> LLMResponse:
            return await self._execute_core_generation(ctx)

        return await combined_cap.wrap_model_request(run_ctx, context, inner_handler)

    async def generate_embeddings(
        self,
        batch: EmbedBatch,
        config: LLMEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        """生成文本或多模态嵌入向量"""
        self._check_not_closed()
        if not batch.payloads:
            return EmbeddingResponse(
                embeddings=[], usage=UsageInfo(), model_name=self.model_name
            )

        final_config = config or LLMEmbeddingConfig()

        context = LLMContext(
            messages=[],
            config=final_config,
            tools=None,
            tool_choice=None,
            timeout=None,
            request_type="embedding",
            extra={"embed_batch": batch},
        )

        pipeline = self._build_pipeline()
        response = await pipeline(context)
        embeddings = (
            response.cache_info.get("embeddings") if response.cache_info else None
        )
        if embeddings is None:
            raise LLMException(
                "嵌入请求未返回 embeddings 数据",
                code=LLMErrorCode.EMBEDDING_FAILED,
            )

        usage_obj = cast(UsageInfo, parse_usage_info(response.usage_info))
        return EmbeddingResponse(
            embeddings=embeddings, usage=usage_obj, model_name=self.model_name
        )

    async def rerank(
        self,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int = 3,
        timeout: float | None = None,
    ) -> list[Any]:
        """执行文档重排"""
        self._check_not_closed()
        context = LLMContext(
            messages=[],
            config=None,
            tools=None,
            tool_choice=None,
            timeout=timeout,
            request_type="rerank",
            extra={"query": query, "documents": documents, "top_n": top_n},
        )
        pipeline = self._build_pipeline()
        response = await pipeline(context)
        return (
            response.cache_info.get("rerank_results", []) if response.cache_info else []
        )

    async def generate_speech(
        self,
        input_text: str,
        voice: str,
        config: TTSConfig | None = None,
    ) -> AudioResponse:
        """生成语音"""
        self._check_not_closed()
        if not input_text:
            raise LLMException(
                "语音合成文本不能为空", code=LLMErrorCode.INVALID_PARAMETER
            )

        final_config = config or TTSConfig()

        context = LLMContext(
            messages=[],
            config=final_config,
            tools=None,
            tool_choice=None,
            timeout=None,
            request_type="speech_generation",
            extra={"input_text": input_text, "voice": voice},
        )

        pipeline = self._build_pipeline()
        llm_response = await pipeline(context)
        return cast(AudioResponse, llm_response.parsed_obj)

    def __str__(self) -> str:
        status = "closed" if self._is_closed else "active"
        return f"LLMModel({self.provider_name}/{self.model_name}, {status})"

    def __repr__(self) -> str:
        status = "closed" if self._is_closed else "active"
        return (
            f"LLMModel(provider={self.provider_name}, model={self.model_name}, "
            f"api_type={self.api_type}, status={status})"
        )
