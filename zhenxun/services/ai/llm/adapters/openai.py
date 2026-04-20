"""
OpenAI API 适配器

支持 OpenAI、智谱AI 等 OpenAI 兼容的 API 服务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import json_repair

from zhenxun.services.ai.llm.config.generation import ImageAspectRatio
from zhenxun.services.ai.types.configs import (
    LLMEmbeddingConfig,
    StructuredOutputStrategy,
)
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import (
    ImagePart,
    RerankDocument,
    RerankResult,
    TextPart,
    ToolCallPart,
    UserMessage,
)
from zhenxun.services.ai.types.tools import ToolChoice
from zhenxun.services.log import logger
from zhenxun.utils.http_utils import AsyncHttpx

from .base import (
    BaseAdapter,
    OpenAICompatAdapter,
    RequestData,
    ResponseData,
    process_image_data,
)
from .components.interfaces import (
    ConfigMapper,
    MessageConverter,
    ResponseParser,
    ToolSerializer,
)
from .components.openai import (
    OpenAIConfigMapper,
    OpenAIMessageConverter,
    OpenAIResponseParser,
    OpenAIToolSerializer,
)

if TYPE_CHECKING:
    from zhenxun.services.ai.types.messages import LLMMessage

    from ..config.generation import LLMGenerationConfig
    from ..service import LLMModel


class APIProtocol(ABC):
    """API 协议策略基类"""

    @abstractmethod
    def get_message_converter(self) -> MessageConverter:
        """获取消息转换器"""
        pass

    @abstractmethod
    def get_tool_serializer(self) -> ToolSerializer:
        """获取工具序列化器"""
        pass

    @abstractmethod
    def get_config_mapper(self, api_type: str) -> ConfigMapper:
        """获取配置映射器"""
        pass

    @abstractmethod
    def get_response_parser(self) -> ResponseParser:
        """获取响应解析器"""
        pass

    @abstractmethod
    def build_request_body(
        self,
        model: LLMModel,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> dict[str, Any]:
        """构建不同协议下的请求体"""
        pass

    @abstractmethod
    def parse_response(self, response_json: dict[str, Any]) -> ResponseData:
        """解析不同协议下的响应"""
        pass


class StandardProtocol(APIProtocol):
    """标准 OpenAI 协议策略"""

    def __init__(self, adapter: "OpenAICompatAdapter"):
        self.adapter = adapter

    def get_message_converter(self) -> MessageConverter:
        return OpenAIMessageConverter()

    def get_tool_serializer(self) -> ToolSerializer:
        return OpenAIToolSerializer(api_type=self.adapter.api_type)

    def get_config_mapper(self, api_type: str) -> ConfigMapper:
        return OpenAIConfigMapper(api_type=api_type)

    def get_response_parser(self) -> ResponseParser:
        return OpenAIResponseParser()

    def build_request_body(
        self,
        model: LLMModel,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> dict[str, Any]:
        converter = self.get_message_converter()
        openai_messages = converter.convert_messages(messages)
        body: dict[str, Any] = {
            "model": model.model_name,
            "messages": openai_messages,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    def parse_response(self, response_json: dict[str, Any]) -> ResponseData:
        parser = self.get_response_parser()
        return parser.parse(response_json)


class ResponsesProtocol(APIProtocol):
    """/v1/responses 新版协议策略"""

    def __init__(self, adapter: "OpenAICompatAdapter"):
        self.adapter = adapter

    def get_message_converter(self) -> MessageConverter:
        from .components.openai import ResponsesMessageConverter

        return ResponsesMessageConverter()

    def get_tool_serializer(self) -> ToolSerializer:
        from .components.openai import ResponsesToolSerializer

        return ResponsesToolSerializer(api_type=self.adapter.api_type)

    def get_config_mapper(self, api_type: str) -> ConfigMapper:
        from .components.openai import ResponsesConfigMapper

        return ResponsesConfigMapper(api_type=api_type)

    def get_response_parser(self) -> ResponseParser:
        from .components.openai import ResponsesResponseParser

        return ResponsesResponseParser()

    def build_request_body(
        self,
        model: LLMModel,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
    ) -> dict[str, Any]:
        converter = self.get_message_converter()
        input_items = converter.convert_messages(messages)
        body: dict[str, Any] = {
            "model": model.model_name,
            "input": input_items,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    def parse_response(self, response_json: dict[str, Any]) -> ResponseData:
        self.adapter.validate_response(response_json)
        parser = self.get_response_parser()
        return parser.parse(response_json)


class OpenAIAdapter(OpenAICompatAdapter):
    """OpenAI兼容API适配器"""

    @property
    def api_type(self) -> str:
        return "openai"

    @property
    def supported_api_types(self) -> list[str]:
        return [
            "openai",
            "zhipu",
            "ark",
            "openrouter",
            "openai_responses",
        ]

    def get_chat_endpoint(self, model: LLMModel) -> str:
        """返回聊天完成端点"""
        if model.model_detail.endpoint:
            return model.model_detail.endpoint

        current_api_type = model.model_detail.api_type or model.api_type

        if current_api_type == "openai_responses":
            return "/v1/responses"
        if current_api_type == "ark":
            return "/api/v3/chat/completions"
        if current_api_type == "zhipu":
            return "/api/paas/v4/chat/completions"
        return "/v1/chat/completions"

    def _get_protocol_strategy(self, model: LLMModel) -> APIProtocol:
        """根据 API 类型获取对应的处理策略"""
        current_api_type = model.model_detail.api_type or model.api_type
        if current_api_type == "openai_responses":
            return ResponsesProtocol(self)
        return StandardProtocol(self)

    def get_embedding_endpoint(self, model: LLMModel) -> str:
        """根据API类型返回嵌入端点"""
        if model.api_type == "zhipu":
            return "/v4/embeddings"
        return "/v1/embeddings"

    def convert_generation_config(
        self, config: LLMGenerationConfig, model: LLMModel
    ) -> dict[str, Any]:
        protocol = self._get_protocol_strategy(model)
        mapper = protocol.get_config_mapper(api_type=self.api_type)
        return mapper.map_config(config, model.model_detail, model.capabilities)

    async def prepare_advanced_request(
        self,
        model: LLMModel,
        api_key: str,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    ) -> "RequestData":
        """根据不同协议策略构建高级请求"""
        protocol_strategy = self._get_protocol_strategy(model)
        url = self.get_api_url(model, self.get_chat_endpoint(model))
        headers = self.get_base_headers(api_key)
        if model.api_type == "openrouter":
            headers.update(
                {
                    "HTTP-Referer": "https://github.com/zhenxun-org/zhenxun_bot",
                    "X-Title": "Zhenxun Bot",
                }
            )

        default_config = getattr(model, "_generation_config", None)
        effective_config = config if config is not None else default_config
        structured_strategy = (
            effective_config.output.structured_output_strategy
            if effective_config and effective_config.output
            else None
        )
        if structured_strategy is None:
            structured_strategy = (
                StructuredOutputStrategy.TOOL_CALL
                if model.api_type == "deepseek" and model.model_name == "deepseek-chat"
                else StructuredOutputStrategy.NATIVE
            )

        openai_tools: list[dict[str, Any]] | None = None
        executables: list[Any] = []
        if tools:
            if isinstance(tools, dict):
                executables = list(tools.values())
            else:
                for tool in tools:
                    if hasattr(tool, "get_definition"):
                        executables.append(tool)

        definition_tasks = [executable.get_definition() for executable in executables]
        tool_defs: list[Any] = []
        if definition_tasks:
            import asyncio

            results = await asyncio.gather(*definition_tasks)
            tool_defs = [td for td in results if td is not None]

        if tool_defs:
            serializer = protocol_strategy.get_tool_serializer()
            openai_tools = serializer.serialize_tools(tool_defs)

        final_tool_choice = tool_choice
        if final_tool_choice is None:
            if (
                effective_config
                and effective_config.tool_config
                and effective_config.tool_config.mode == "ANY"
            ):
                allowed = effective_config.tool_config.allowed_function_names
                if allowed:
                    if len(allowed) == 1:
                        if isinstance(protocol_strategy, ResponsesProtocol):
                            final_tool_choice = {
                                "type": "function",
                                "name": allowed[0],
                            }
                        else:
                            final_tool_choice = {
                                "type": "function",
                                "function": {"name": allowed[0]},
                            }
                    else:
                        logger.warning(
                            "OpenAI API 不支持多个 allowed_function_names，降级为"
                            " required。"
                        )
                        final_tool_choice = "required"
                else:
                    final_tool_choice = "required"

        if (
            structured_strategy == StructuredOutputStrategy.TOOL_CALL
            and effective_config
            and effective_config.output
            and effective_config.output.response_schema
        ):
            serializer = protocol_strategy.get_tool_serializer()
            sanitized_schema = serializer.sanitize_schema(
                effective_config.output.response_schema
            )
            structured_tool = {
                "type": "function",
                "function": {
                    "name": "return_structured_response",
                    "description": "Output the final structured response.",
                    "parameters": sanitized_schema,
                },
            }
            if model.api_type != "deepseek":
                structured_tool["function"]["strict"] = True

            if isinstance(protocol_strategy, ResponsesProtocol):
                func_data = structured_tool.pop("function")
                structured_tool.update(func_data)

                final_tool_choice = {
                    "type": "function",
                    "name": "return_structured_response",
                }
            else:
                final_tool_choice = {
                    "type": "function",
                    "function": {"name": "return_structured_response"},
                }

            if openai_tools is None:
                openai_tools = []
            openai_tools.append(structured_tool)

        body = protocol_strategy.build_request_body(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice=final_tool_choice,
        )

        body = self.apply_config_override(model, body, config)

        if final_tool_choice is not None:
            body["tool_choice"] = final_tool_choice

        response_format = body.get("response_format", {})
        inject_prompt = (
            structured_strategy == StructuredOutputStrategy.NATIVE
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

        if inject_prompt and "messages" in body:
            messages_list = body["messages"]
            has_json_keyword = False
            for msg in messages_list:
                content = msg.get("content")
                if isinstance(content, str) and "json" in content.lower():
                    has_json_keyword = True
                    break
                if isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and "json" in part.get("text", "").lower()
                        ):
                            has_json_keyword = True
                            break
                    if has_json_keyword:
                        break

            if not has_json_keyword:
                injection_text = (
                    "请务必输出合法的 JSON 格式，避免额外的文本、Markdown 或解释。"
                )
                system_msg = next(
                    (m for m in messages_list if m.get("role") == "system"), None
                )
                if system_msg:
                    if isinstance(system_msg.get("content"), str):
                        system_msg["content"] += " " + injection_text
                    elif isinstance(system_msg.get("content"), list):
                        system_msg["content"].append(
                            {"type": "text", "text": injection_text}
                        )
                else:
                    messages_list.insert(
                        0, {"role": "system", "content": injection_text}
                    )
                body["messages"] = messages_list

        return RequestData(url=url, headers=headers, body=body)

    def parse_response(
        self,
        model: LLMModel,
        response_json: dict[str, Any],
        is_advanced: bool = False,
    ) -> ResponseData:
        """解析响应 - 使用策略模式委托处理"""
        _ = is_advanced
        protocol_strategy = self._get_protocol_strategy(model)
        response_data = protocol_strategy.parse_response(response_json)

        tool_calls = [
            p for p in response_data.content_parts if isinstance(p, ToolCallPart)
        ]
        if tool_calls:
            target_tool = next(
                (
                    tc
                    for tc in tool_calls
                    if tc.tool_name == "return_structured_response"
                ),
                None,
            )
            if target_tool:
                args_data = target_tool.args
                if isinstance(args_data, str):
                    response_data.text = json_repair.repair_json(args_data)
                else:
                    response_data.text = json.dumps(args_data, ensure_ascii=False)
                response_data.content_parts = [
                    p
                    for p in response_data.content_parts
                    if not (
                        isinstance(p, ToolCallPart)
                        and p.tool_name == "return_structured_response"
                    )
                ]

        return response_data

    def prepare_rerank_request(
        self,
        model: LLMModel,
        api_key: str,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int,
    ) -> RequestData:
        endpoint = "/api/paas/v4/rerank" if model.api_type == "zhipu" else "/v1/rerank"
        url = self.get_api_url(model, endpoint)
        headers = self.get_base_headers(api_key)

        safe_documents = []
        for doc in documents:
            if isinstance(doc, dict):
                safe_documents.append(doc.get("text", str(doc)))
            else:
                safe_documents.append(str(doc))

        body = {
            "model": model.model_name,
            "query": query,
            "documents": safe_documents,
        }
        body["top_n"] = top_n
        return RequestData(url=url, headers=headers, body=body)

    def parse_rerank_response(
        self, response_json: dict[str, Any]
    ) -> list[RerankResult]:
        self.validate_response(response_json)
        results = []
        for item in response_json.get("results", []):
            doc = item.get("document", {})
            r_doc = (
                RerankDocument(text=doc.get("text"), image=doc.get("image"))
                if isinstance(doc, dict)
                else RerankDocument(text=str(doc))
            )
            results.append(
                RerankResult(
                    index=item["index"],
                    relevance_score=item["relevance_score"],
                    document=r_doc,
                )
            )
        return results


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek 专用适配器 (基于 OpenAI 协议)"""

    @property
    def api_type(self) -> str:
        return "deepseek"

    @property
    def supported_api_types(self) -> list[str]:
        return ["deepseek"]


class OpenAIImageAdapter(BaseAdapter):
    """OpenAI 图像生成/编辑适配器"""

    @property
    def api_type(self) -> str:
        return "openai_image"

    @property
    def log_sanitization_context(self) -> str:
        return "openai_request"

    @property
    def supported_api_types(self) -> list[str]:
        return ["openai_image", "nano_banana"]

    async def prepare_advanced_request(
        self,
        model: LLMModel,
        api_key: str,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
    ) -> RequestData:
        _ = tools, tool_choice
        effective_config = config if config is not None else model._generation_config
        headers = self.get_base_headers(api_key)

        prompt = ""
        images_bytes_list: list[bytes] = []

        for msg in reversed(messages):
            if not isinstance(msg, UserMessage):
                continue
            if isinstance(msg.content, str):
                prompt = msg.content
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, TextPart) and not prompt:
                        prompt = part.text
                    elif isinstance(part, ImagePart):
                        if part.url:
                            images_bytes_list.append(
                                await AsyncHttpx.get_content(part.url)
                            )
                        elif part.raw:
                            images_bytes_list.append(part.raw)
                        elif part.path:
                            images_bytes_list.append(part.path.read_bytes())
            if prompt:
                break

        if not prompt and not images_bytes_list:
            raise LLMException(
                "图像生成需要提供 Prompt",
                code=LLMErrorCode.CONFIGURATION_ERROR,
            )

        body: dict[str, Any] = {
            "model": model.model_name,
            "prompt": prompt,
            "response_format": "b64_json",
        }

        if effective_config:
            if effective_config.visual:
                if effective_config.visual.aspect_ratio:
                    ar = effective_config.visual.aspect_ratio
                    size_map = {
                        ImageAspectRatio.SQUARE: "1024x1024",
                        ImageAspectRatio.LANDSCAPE_16_9: "1792x1024",
                        ImageAspectRatio.PORTRAIT_9_16: "1024x1792",
                    }
                    if isinstance(ar, ImageAspectRatio) and ar in size_map:
                        body["size"] = size_map[ar]
                        body["aspect_ratio"] = ar.value
                    elif isinstance(ar, str):
                        if "x" in ar:
                            body["size"] = ar
                        else:
                            body["aspect_ratio"] = ar

                if effective_config.visual.resolution:
                    res_val = effective_config.visual.resolution
                    if not isinstance(res_val, str):
                        res_val = getattr(res_val, "value", res_val)
                    body["image_size"] = res_val

            if effective_config.custom_params:
                body.update(effective_config.custom_params)

        if images_bytes_list:
            b64_images = []
            for img_bytes in images_bytes_list:
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                b64_images.append(b64_str)
            body["image"] = b64_images

        endpoint = "/v1/images/generations"
        url = self.get_api_url(model, endpoint)
        return RequestData(url=url, headers=headers, body=body)

    def parse_response(
        self,
        model: LLMModel,
        response_json: dict[str, Any],
        is_advanced: bool = False,
    ) -> ResponseData:
        _ = model, is_advanced
        self.validate_response(response_json)

        images_data: list[bytes | Path] = []
        data_list = response_json.get("data", [])

        for item in data_list:
            if "b64_json" in item:
                try:
                    b64_str = item["b64_json"]
                    if b64_str.startswith("data:"):
                        b64_str = b64_str.split(",", 1)[1]
                    img = base64.b64decode(b64_str)
                    images_data.append(process_image_data(img))
                except Exception as exc:
                    logger.error(f"Base64 解码失败: {exc}")
            elif "url" in item:
                logger.warning(
                    f"API 返回了 URL 而不是 Base64: {item.get('url', 'unknown')}"
                )

        text_summary = (
            f"已生成 {len(images_data)} 张图片。"
            if images_data
            else "图像生成接口调用成功，但未解析到图片数据。"
        )

        content_parts = []
        if text_summary:
            content_parts.append(TextPart(text=text_summary))
        for img in images_data:
            content_parts.append(
                ImagePart(raw=img) if isinstance(img, bytes) else ImagePart(path=img)
            )

        return ResponseData(
            content_parts=content_parts,
            raw_response=response_json,
        )

    def prepare_embedding_request(
        self,
        model: LLMModel,
        api_key: str,
        texts: list[str],
        config: LLMEmbeddingConfig,
    ) -> RequestData:
        raise NotImplementedError("OpenAIImageAdapter 不支持 Embedding")

    def parse_embedding_response(
        self, response_json: dict[str, Any]
    ) -> list[list[float]]:
        raise NotImplementedError("OpenAIImageAdapter 不支持 Embedding")

    def convert_generation_config(
        self, config: LLMGenerationConfig, model: LLMModel
    ) -> dict[str, Any]:
        _ = config, model
        return {}

    def prepare_rerank_request(self, *args, **kwargs) -> RequestData:
        raise NotImplementedError("OpenAIImageAdapter 不支持 Rerank")

    def parse_rerank_response(
        self, response_json: dict[str, Any]
    ) -> list[RerankResult]:
        raise NotImplementedError("OpenAIImageAdapter 不支持 Rerank")
