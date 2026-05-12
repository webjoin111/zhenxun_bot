"""
LLM 适配器基类和通用数据结构
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

import httpx
from pydantic import BaseModel, Field

from zhenxun.configs.path_config import TEMP_PATH
from zhenxun.services.ai.core.configs import LLMEmbeddingConfig
from zhenxun.services.ai.core.configs import TTSConfig
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.messages import (
    AudioResponse,
    ImagePart,
    LLMContentPart,
    RerankResult,
    TextPart,
    ThoughtPart,
)
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import GenerationConfig
    from zhenxun.services.ai.core.messages import LLMMessage
    from zhenxun.services.ai.tools.models import ToolChoice

    from ..service import LLMModel


class RequestData(BaseModel):
    """标准化的请求载体，用于向上层 HTTP 客户端传递请求参数。"""

    method: str = "POST"
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    files: dict[str, Any] | list[tuple[str, Any]] | None = None


class ResponseData(BaseModel):
    """标准化的响应载体，统一承接文本、多模态与附加元数据。"""

    content_parts: list[LLMContentPart] = Field(default_factory=list)
    usage_info: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    grounding_metadata: Any | None = None
    cache_info: Any | None = None

    @property
    def text(self) -> str:
        """提取并拼接所有 `TextPart` 文本内容。"""
        return "".join(
            p.text for p in self.content_parts if isinstance(p, TextPart)
        ).strip()

    @text.setter
    def text(self, value: str):
        """设置首个 `TextPart`，不存在则追加新的 `TextPart`。"""
        for p in self.content_parts:
            if isinstance(p, TextPart):
                p.text = value
                return
        self.content_parts.append(TextPart(text=value))

    @property
    def thought_text(self) -> str | None:
        """提取并拼接所有思维片段文本，未命中则返回 `None`。"""
        thoughts = [
            p.thought_text for p in self.content_parts if isinstance(p, ThoughtPart)
        ]
        return "\n".join(thoughts).strip() if thoughts else None

    @property
    def thought_signature(self) -> str | None:
        """从末尾向前查找思维签名，用于后续连续推理场景。"""
        for p in reversed(self.content_parts):
            if (
                hasattr(p, "metadata")
                and p.metadata
                and "thought_signature" in p.metadata
            ):
                return p.metadata["thought_signature"]
        return None

    @property
    def images(self) -> list[bytes | Path | str]:
        """收集图片内容，按 URL / 原始字节 / 本地路径顺序返回。"""
        imgs = []
        for p in self.content_parts:
            if isinstance(p, ImagePart):
                if p.url:
                    imgs.append(p.url)
                elif p.raw:
                    imgs.append(p.raw)
                elif p.path:
                    imgs.append(p.path)
        return imgs

    @images.setter
    def images(self, value: list[bytes | Path | str]):
        """覆盖图片片段并按输入类型重建 `ImagePart` 列表。"""
        self.content_parts = [
            p for p in self.content_parts if not isinstance(p, ImagePart)
        ]
        for img in value:
            if isinstance(img, str) and img.startswith(("http://", "https://")):
                self.content_parts.append(ImagePart(url=img))
            elif isinstance(img, bytes):
                self.content_parts.append(ImagePart(raw=img))
            else:
                self.content_parts.append(ImagePart(path=Path(img)))

    code_execution_results: list[dict[str, Any]] | None = None
    search_results: list[dict[str, Any]] | None = None
    function_calls: list[dict[str, Any]] | None = None
    safety_ratings: list[dict[str, Any]] | None = None
    citations: list[dict[str, Any]] | None = None


def process_image_data(image_data: bytes) -> bytes | Path:
    """处理图片二进制数据：超过 2MB 时落盘并返回文件路径。"""
    max_inline_size = 2 * 1024 * 1024
    if len(image_data) > max_inline_size:
        save_dir = TEMP_PATH / "llm"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{uuid.uuid4()}.png"
        file_path = save_dir / file_name
        file_path.write_bytes(image_data)
        logger.info(
            f"图片数据过大 ({len(image_data)} bytes)，已保存到临时文件: {file_path}",
            "LLMAdapter",
        )
        return file_path.resolve()
    return image_data


class BaseAdapter(ABC):
    """
    LLM API适配器基类 (门面模式 Facade)。
    负责维护厂商级别的通用配置(如 URL 拼接、请求头构建、通用错误拦截)，
    而将具体的模态序列化与反序列化逻辑委派给各路 Handler。
    """

    text_handler: Any = None
    image_handler: Any = None
    embedding_handler: Any = None
    rerank_handler: Any = None
    audio_handler: Any = None

    @property
    def log_sanitization_context(self) -> str:
        """用于日志清洗的上下文名称，默认 'default'"""
        return "default"

    @property
    @abstractmethod
    def api_type(self) -> str:
        """API类型标识"""
        pass

    @property
    @abstractmethod
    def supported_api_types(self) -> list[str]:
        """支持的API类型列表"""
        pass

    async def prepare_simple_request(
        self,
        model: LLMModel,
        api_key: str,
        prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> RequestData:
        """准备简单文本生成请求

        默认实现：将简单请求转换为高级请求格式
        子类可以重写此方法以提供特定的优化实现
        """
        from zhenxun.services.ai.core.messages import (
            AssistantMessage,
            SystemMessage,
            TextPart,
            UserMessage,
        )

        messages: list[Any] = []

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    messages.append(SystemMessage(content=[TextPart(text=content)]))
                elif role == "assistant":
                    messages.append(AssistantMessage(content=[TextPart(text=content)]))
                else:
                    messages.append(UserMessage(content=[TextPart(text=content)]))

        messages.append(UserMessage(content=[TextPart(text=prompt)]))

        config = model._generation_config

        return await self.prepare_advanced_request(
            model=model,
            api_key=api_key,
            messages=messages,
            config=config,
            tools=None,
            tool_choice=None,
        )

    async def prepare_advanced_request(
        self,
        model: LLMModel,
        api_key: str,
        messages: list[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    ) -> RequestData:
        """准备高级对话请求并委派给 `text_handler` 完成序列化。"""
        if self.text_handler:
            return await self.text_handler.prepare_text_request(
                adapter=self,
                model=model,
                api_key=api_key,
                messages=messages,
                config=config,
                tools=tools,
                tool_choice=tool_choice,
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 TextHandler，暂不支持文本对话能力。"
        )

    def parse_response(
        self,
        model: LLMModel,
        response_json: dict[str, Any],
        is_advanced: bool = False,
    ) -> ResponseData:
        """解析文本响应并委派给 `text_handler`。"""
        if self.text_handler:
            return self.text_handler.parse_text_response(
                adapter=self,
                model=model,
                response_json=response_json,
                is_advanced=is_advanced,
            )
        raise NotImplementedError(f"API 类型 '{self.api_type}' 未装配 TextHandler。")

    def prepare_embedding_request(
        self,
        model: LLMModel,
        api_key: str,
        texts: list[str],
        config: LLMEmbeddingConfig,
    ) -> RequestData:
        """准备文本嵌入请求并委派给 `embedding_handler`。"""
        if self.embedding_handler:
            return self.embedding_handler.prepare_embedding_request(
                adapter=self, model=model, api_key=api_key, texts=texts, config=config
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 EmbeddingHandler，暂不支持向量嵌入。"
        )

    def parse_embedding_response(
        self, response_json: dict[str, Any]
    ) -> list[list[float]]:
        """解析文本嵌入响应并委派给 `embedding_handler`。"""
        if self.embedding_handler:
            return self.embedding_handler.parse_embedding_response(
                adapter=self, response_json=response_json
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 EmbeddingHandler。"
        )

    def prepare_rerank_request(
        self,
        model: LLMModel,
        api_key: str,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int,
    ) -> RequestData:
        """准备重排请求并委派给 `rerank_handler`。"""
        if self.rerank_handler:
            return self.rerank_handler.prepare_rerank_request(
                adapter=self,
                model=model,
                api_key=api_key,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 RerankHandler，暂不支持文本重排。"
        )

    def parse_rerank_response(
        self, response_json: dict[str, Any]
    ) -> list["RerankResult"]:
        """解析重排响应并委派给 `rerank_handler`。"""
        if self.rerank_handler:
            return self.rerank_handler.parse_rerank_response(
                adapter=self, response_json=response_json
            )
        raise NotImplementedError(f"API 类型 '{self.api_type}' 未装配 RerankHandler。")

    def prepare_image_request(
        self,
        model: "LLMModel",
        api_key: str,
        prompt: str,
        images: list[Any] | None = None,
        config: "GenerationConfig | None" = None,
    ) -> RequestData:
        """准备图像请求并委派给 `image_handler`。"""
        if self.image_handler:
            return self.image_handler.prepare_image_request(
                adapter=self,
                model=model,
                api_key=api_key,
                prompt=prompt,
                images=images,
                config=config,
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 ImageHandler，暂不支持图像生成。"
        )

    def parse_image_response(self, response_json: dict[str, Any]) -> ResponseData:
        """解析图像响应并委派给 `image_handler`。"""
        if self.image_handler:
            return self.image_handler.parse_image_response(
                adapter=self, response_json=response_json
            )
        raise NotImplementedError(f"API 类型 '{self.api_type}' 未装配 ImageHandler。")

    def prepare_speech_request(
        self,
        model: "LLMModel",
        api_key: str,
        input_text: str,
        voice: str,
        config: "TTSConfig",
    ) -> RequestData:
        """准备语音生成请求并委派给 `audio_handler`。"""
        if self.audio_handler:
            return self.audio_handler.prepare_speech_request(
                adapter=self,
                model=model,
                api_key=api_key,
                input_text=input_text,
                voice=voice,
                config=config,
            )
        raise NotImplementedError(
            f"API 类型 '{self.api_type}' 未装配 AudioHandler，暂不支持语音生成。"
        )

    async def parse_speech_response(
        self, model: "LLMModel", raw_response: Any
    ) -> AudioResponse:
        """解析语音响应并委派给 `audio_handler`。注意传入的是 httpx.Response 的 raw 对象"""
        if self.audio_handler:
            return await self.audio_handler.parse_speech_response(
                adapter=self, model=model, raw_response=raw_response
            )
        raise NotImplementedError(f"API 类型 '{self.api_type}' 未装配 AudioHandler。")

    def validate_embedding_response(self, response_json: dict[str, Any]) -> None:
        """验证嵌入接口响应，检测 `error` 并转换为统一异常。"""
        if response_json.get("error"):
            error_info = response_json["error"]
            msg = (
                error_info.get("message", str(error_info))
                if isinstance(error_info, dict)
                else str(error_info)
            )
            raise LLMException(
                f"嵌入API错误: {msg}",
                code=LLMErrorCode.EMBEDDING_FAILED,
                details=response_json,
            )

    def get_api_url(self, model: LLMModel, endpoint: str) -> str:
        """拼接最终请求 URL，兼容 `path_prefix` 与端点前后斜杠。"""
        if not model.api_base:
            raise LLMException(
                f"模型 {model.model_name} 的 api_base 未设置",
                code=LLMErrorCode.CONFIGURATION_ERROR,
            )

        base_url = model.api_base.rstrip("/")
        prefix = model.path_prefix.strip("/") if model.path_prefix else ""
        ep = endpoint.lstrip("/")

        if prefix:
            return f"{base_url}/{prefix}/{ep}"
        return f"{base_url}/{ep}"

    def get_base_headers(self, api_key: str) -> dict[str, str]:
        """构建默认请求头，包含 UA、JSON 类型与 Bearer 鉴权。"""
        from zhenxun.utils.user_agent import get_user_agent

        headers = get_user_agent()
        headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def validate_response(self, response_json: dict[str, Any]) -> None:
        """统一校验文本/多模态响应并映射平台错误码。"""
        if response_json.get("error"):
            error_info = response_json["error"]

            if isinstance(error_info, dict):
                error_message = error_info.get("message", "未知错误")
                error_code = error_info.get("code", "unknown")
                error_type = error_info.get("type", "api_error")

                error_code_mapping = {
                    "invalid_api_key": LLMErrorCode.API_KEY_INVALID,
                    "authentication_failed": LLMErrorCode.API_KEY_INVALID,
                    "insufficient_quota": LLMErrorCode.API_QUOTA_EXCEEDED,
                    "rate_limit_exceeded": LLMErrorCode.API_RATE_LIMITED,
                    "quota_exceeded": LLMErrorCode.API_RATE_LIMITED,
                    "model_not_found": LLMErrorCode.MODEL_NOT_FOUND,
                    "invalid_model": LLMErrorCode.MODEL_NOT_FOUND,
                    "context_length_exceeded": LLMErrorCode.CONTEXT_LENGTH_EXCEEDED,
                    "max_tokens_exceeded": LLMErrorCode.CONTEXT_LENGTH_EXCEEDED,
                    "invalid_request_error": LLMErrorCode.INVALID_PARAMETER,
                    "invalid_parameter": LLMErrorCode.INVALID_PARAMETER,
                }

                llm_error_code = error_code_mapping.get(
                    error_code, LLMErrorCode.API_RESPONSE_INVALID
                )

                logger.error(
                    f"API返回错误: {error_message} "
                    f"(代码: {error_code}, 类型: {error_type})"
                )
            else:
                error_message = str(error_info)
                error_code = "unknown"
                llm_error_code = LLMErrorCode.API_RESPONSE_INVALID

                logger.error(f"API返回错误: {error_message}")

            raise LLMException(
                f"API请求失败: {error_message}",
                code=llm_error_code,
                details={"api_error": error_info, "error_code": error_code},
            )

        if "candidates" in response_json:
            candidates = response_json.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason")
                if finish_reason in ["SAFETY", "RECITATION"]:
                    safety_ratings = candidate.get("safetyRatings", [])
                    logger.warning(
                        f"Gemini内容被安全过滤: {finish_reason}, "
                        f"安全评级: {safety_ratings}"
                    )
                    raise LLMException(
                        f"内容被安全过滤: {finish_reason}",
                        code=LLMErrorCode.CONTENT_FILTERED,
                        details={
                            "finish_reason": finish_reason,
                            "safety_ratings": safety_ratings,
                        },
                    )

        if not response_json:
            logger.error("API返回空响应")
            raise LLMException(
                "API返回空响应",
                code=LLMErrorCode.API_RESPONSE_INVALID,
                details={"response": response_json},
            )

    def handle_http_error(self, response: httpx.Response) -> LLMException | None:
        """
        处理 HTTP 错误响应。
        如果响应状态码表示成功 (200)，返回 None；否则构造 LLMException 供外部捕获。
        """
        if response.status_code == 200:
            return None

        error_text = response.content.decode("utf-8", errors="ignore")
        error_status = ""
        error_msg = error_text
        try:
            error_json = json.loads(error_text)
            if isinstance(error_json, dict) and "error" in error_json:
                error_info = error_json["error"]
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", error_msg)
                    raw_status = error_info.get("status") or error_info.get("code")
                    error_status = str(raw_status) if raw_status is not None else ""
                elif error_info is not None:
                    error_msg = str(error_info)
                    error_status = error_msg
        except Exception:
            pass

        status_upper = error_status.upper() if error_status else ""
        text_upper = error_text.upper()

        error_code = LLMErrorCode.API_REQUEST_FAILED
        if response.status_code == 400:
            if (
                "FAILED_PRECONDITION" in status_upper
                or "LOCATION IS NOT SUPPORTED" in text_upper
            ):
                error_code = LLMErrorCode.USER_LOCATION_NOT_SUPPORTED
            elif "INVALID_ARGUMENT" in status_upper:
                error_code = LLMErrorCode.INVALID_PARAMETER
            elif "API_KEY_INVALID" in text_upper or "API KEY NOT VALID" in text_upper:
                error_code = LLMErrorCode.API_KEY_INVALID
            else:
                error_code = LLMErrorCode.INVALID_PARAMETER
        elif response.status_code in [401, 403]:
            if error_msg and (
                "country" in error_msg.lower()
                or "region" in error_msg.lower()
                or "unsupported" in error_msg.lower()
            ):
                error_code = LLMErrorCode.USER_LOCATION_NOT_SUPPORTED
            elif "PERMISSION_DENIED" in status_upper:
                error_code = LLMErrorCode.API_KEY_INVALID
            else:
                error_code = LLMErrorCode.API_KEY_INVALID
        elif response.status_code == 404:
            error_code = LLMErrorCode.MODEL_NOT_FOUND
        elif response.status_code == 429:
            if (
                "RESOURCE_EXHAUSTED" in status_upper
                or "INSUFFICIENT_QUOTA" in status_upper
                or ("quota" in error_msg.lower() if error_msg else False)
            ):
                error_code = LLMErrorCode.API_QUOTA_EXCEEDED
            else:
                error_code = LLMErrorCode.API_RATE_LIMITED
        elif response.status_code in [402, 413]:
            error_code = LLMErrorCode.API_QUOTA_EXCEEDED
        elif response.status_code == 422:
            error_code = LLMErrorCode.GENERATION_FAILED
        elif response.status_code >= 500:
            error_code = LLMErrorCode.API_TIMEOUT

        return LLMException(
            f"HTTP请求失败: {response.status_code} ({error_status or 'Unknown'})",
            code=error_code,
            details={
                "status_code": response.status_code,
                "api_status": error_status,
                "response": error_text,
            },
        )
