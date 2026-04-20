import base64
import binascii
import json
from pathlib import Path
from typing import Any

import json_repair

from zhenxun.services.ai.llm.adapters.base import ResponseData, process_image_data
from zhenxun.services.ai.llm.adapters.components.interfaces import (
    ConfigMapper,
    MessageConverter,
    ResponseParser,
    ToolSerializer,
)
from zhenxun.services.ai.llm.config.generation import (
    ImageAspectRatio,
    LLMGenerationConfig,
    ResponseFormat,
)
from zhenxun.services.ai.types.configs import StructuredOutputStrategy
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import (
    AssistantMessage,
    ImagePart,
    LLMMessage,
    SystemMessage,
    TextPart,
    ThoughtPart,
    ToolCallPart,
    ToolMessage,
    UserMessage,
)
from zhenxun.services.ai.types.models import ModelCapabilities, ModelDetail
from zhenxun.services.ai.types.tools import ToolDefinition
from zhenxun.services.log import logger


class OpenAIConfigMapper(ConfigMapper):
    def __init__(self, api_type: str = "openai"):
        self.api_type = api_type

    def map_config(
        self,
        config: LLMGenerationConfig,
        model_detail: ModelDetail | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        strategy = config.output.structured_output_strategy if config.output else None
        if strategy is None:
            strategy = (
                StructuredOutputStrategy.TOOL_CALL
                if self.api_type == "deepseek"
                else StructuredOutputStrategy.NATIVE
            )

        if config.core:
            if config.core.temperature is not None:
                params["temperature"] = config.core.temperature
            if config.core.max_tokens is not None:
                params["max_tokens"] = config.core.max_tokens
            if config.core.top_k is not None:
                params["top_k"] = config.core.top_k
            if config.core.top_p is not None:
                params["top_p"] = config.core.top_p
            if config.core.frequency_penalty is not None:
                params["frequency_penalty"] = config.core.frequency_penalty
            if config.core.presence_penalty is not None:
                params["presence_penalty"] = config.core.presence_penalty
            if config.core.stop is not None:
                params["stop"] = config.core.stop

            if config.core.repetition_penalty is not None:
                if self.api_type == "openai":
                    logger.warning("OpenAI官方API不支持repetition_penalty参数，已忽略")
                else:
                    params["repetition_penalty"] = config.core.repetition_penalty

        if config.reasoning and config.reasoning.effort:
            params["reasoning_effort"] = config.reasoning.effort.value.lower()

        if config.output:
            if isinstance(config.output.response_format, dict):
                params["response_format"] = config.output.response_format
            elif (
                config.output.response_format == ResponseFormat.JSON
                and strategy == StructuredOutputStrategy.NATIVE
            ):
                if config.output.response_schema:
                    serializer = OpenAIToolSerializer(api_type=self.api_type)
                    sanitized = serializer.sanitize_schema(
                        config.output.response_schema
                    )
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "schema": sanitized,
                            "strict": True,
                        },
                    }
                else:
                    params["response_format"] = {"type": "json_object"}

        if config.tool_config:
            mode = config.tool_config.mode
            if mode == "NONE":
                params["tool_choice"] = "none"
            elif mode == "AUTO":
                params["tool_choice"] = "auto"
            elif mode == "ANY":
                params["tool_choice"] = "required"

        if config.visual and config.visual.aspect_ratio:
            size_map = {
                ImageAspectRatio.SQUARE: "1024x1024",
                ImageAspectRatio.LANDSCAPE_16_9: "1792x1024",
                ImageAspectRatio.PORTRAIT_9_16: "1024x1792",
            }
            ar = config.visual.aspect_ratio
            if isinstance(ar, ImageAspectRatio):
                mapped_size = size_map.get(ar)
                if mapped_size:
                    params["size"] = mapped_size
            elif isinstance(ar, str):
                params["size"] = ar

        if config.custom_params:
            mapped_custom = config.custom_params.copy()
            if "repetition_penalty" in mapped_custom and self.api_type == "openai":
                mapped_custom.pop("repetition_penalty")

            if "stop" in mapped_custom:
                stop_value = mapped_custom["stop"]
                if isinstance(stop_value, str):
                    mapped_custom["stop"] = [stop_value]

            params.update(mapped_custom)

        return params


class OpenAIMessageConverter(MessageConverter):
    def convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_msg: dict[str, Any] = {"role": "system"}
            elif isinstance(msg, UserMessage):
                openai_msg: dict[str, Any] = {"role": "user"}
            elif isinstance(msg, AssistantMessage):
                openai_msg: dict[str, Any] = {"role": "assistant"}
            elif isinstance(msg, ToolMessage):
                openai_msg: dict[str, Any] = {"role": "tool"}
            else:
                openai_msg: dict[str, Any] = {"role": msg.role}

            if isinstance(msg, ToolMessage):
                returns = msg.tool_returns
                if returns:
                    openai_msg["tool_call_id"] = returns[0].tool_call_id
                    openai_msg["name"] = returns[0].tool_name
                    out_val = returns[0].output
                    openai_msg["content"] = (
                        out_val
                        if isinstance(out_val, str)
                        else json.dumps(out_val, ensure_ascii=False)
                    )
            else:
                if len(msg.content) == 1 and isinstance(msg.content[0], TextPart):
                    openai_msg["content"] = msg.content[0].text
                else:
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, TextPart):
                            content_parts.append({"type": "text", "text": part.text})
                        elif isinstance(part, ImagePart):
                            if part.url is not None:
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": part.url},
                                    }
                                )
                            elif part.raw is not None:
                                import base64

                                raw_data = part.raw
                                mime = part.mime_type or "image/png"
                                b64_str = base64.b64encode(raw_data).decode("utf-8")
                                data_uri = f"data:{mime};base64,{b64_str}"
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    }
                                )
                            elif part.path is not None:
                                import base64

                                raw_data = part.path.read_bytes()
                                mime = part.mime_type or "image/png"
                                b64_str = base64.b64encode(raw_data).decode("utf-8")
                                data_uri = f"data:{mime};base64,{b64_str}"
                                content_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    }
                                )
                    openai_msg["content"] = content_parts

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                assistant_tool_calls = []
                for call in msg.tool_calls:
                    assistant_tool_calls.append(
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.tool_name,
                                "arguments": call.args
                                if isinstance(call.args, str)
                                else json.dumps(call.args, ensure_ascii=False),
                            },
                        }
                    )
                openai_msg["tool_calls"] = assistant_tool_calls

            openai_messages.append(openai_msg)
        return openai_messages


class OpenAIToolSerializer(ToolSerializer):
    def __init__(self, api_type: str = "openai"):
        self.api_type = api_type

    def sanitize_schema(self, schema: Any) -> Any:
        """
        递归地净化 JSON Schema，移除 OpenAI API 不支持的关键字。
        """
        if isinstance(schema, list):
            return [self.sanitize_schema(item) for item in schema]
        if isinstance(schema, dict):
            schema_copy = schema.copy()

            unsupported_keys = [
                "default",
                "minLength",
                "maxLength",
                "pattern",
                "format",
                "minimum",
                "maximum",
                "multipleOf",
                "patternProperties",
                "propertyNames",
                "minItems",
                "maxItems",
                "uniqueItems",
                "$schema",
                "title",
            ]
            for key in unsupported_keys:
                schema_copy.pop(key, None)

            if "$ref" in schema_copy:
                ref_key = schema_copy["$ref"].split("/")[-1]
                defs = schema_copy.get("$defs") or schema_copy.get("definitions")
                if defs and ref_key in defs:
                    schema_copy.pop("$ref", None)
                    schema_copy.update(defs[ref_key])
                else:
                    return {"$ref": schema_copy["$ref"]}

            is_object = (
                schema_copy.get("type") == "object" or "properties" in schema_copy
            )
            if is_object:
                schema_copy["type"] = "object"
                schema_copy["additionalProperties"] = False

                properties = schema_copy.get("properties", {})
                required = schema_copy.get("required", [])
                if properties:
                    existing_req = set(required)
                    for prop in properties.keys():
                        if prop not in existing_req:
                            required.append(prop)
                    schema_copy["required"] = required

            for def_key in ["$defs", "definitions"]:
                if def_key in schema_copy and isinstance(schema_copy[def_key], dict):
                    schema_copy[def_key] = {
                        k: self.sanitize_schema(v)
                        for k, v in schema_copy[def_key].items()
                    }

            recursive_keys = ["properties", "items", "allOf", "anyOf", "oneOf"]
            for key in recursive_keys:
                if key in schema_copy:
                    if key == "properties" and isinstance(schema_copy[key], dict):
                        schema_copy[key] = {
                            k: self.sanitize_schema(v)
                            for k, v in schema_copy[key].items()
                        }
                    else:
                        schema_copy[key] = self.sanitize_schema(schema_copy[key])
            return schema_copy
        return schema

    def serialize_tools(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            raw_schema = tool.parameters.copy() if tool.parameters else {}
            sanitized_schema = self.sanitize_schema(raw_schema)

            tool_payload = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": sanitized_schema,
            }
            if self.api_type != "deepseek":
                tool_payload["strict"] = True

            openai_tools.append({"type": "function", "function": tool_payload})
        return openai_tools


class OpenAIResponseParser(ResponseParser):
    def validate_response(self, response_json: dict[str, Any]) -> None:
        if response_json.get("error"):
            error_info = response_json["error"]
            if isinstance(error_info, dict):
                error_message = error_info.get("message", "未知错误")
                error_code = error_info.get("code", "unknown")

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
            else:
                error_message = str(error_info)
                error_code = "unknown"
                llm_error_code = LLMErrorCode.API_RESPONSE_INVALID

            raise LLMException(
                f"API请求失败: {error_message}",
                code=llm_error_code,
                details={"api_error": error_info, "error_code": error_code},
            )

    def parse(self, response_json: dict[str, Any]) -> ResponseData:
        self.validate_response(response_json)

        choices = response_json.get("choices", [])
        if not choices:
            return ResponseData(raw_response=response_json)

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", None)
        refusal = message.get("refusal")

        if refusal:
            raise LLMException(
                f"模型拒绝生成请求: {refusal}",
                code=LLMErrorCode.CONTENT_FILTERED,
                details={"refusal": refusal},
                recoverable=False,
            )

        if content:
            content = content.strip()

        images_payload: list[bytes | Path] = []
        if content and content.startswith("{") and content.endswith("}"):
            try:
                content_json = json.loads(content)
                if "b64_json" in content_json:
                    b64_str = content_json["b64_json"]
                    if isinstance(b64_str, str) and b64_str.startswith("data:"):
                        b64_str = b64_str.split(",", 1)[1]
                    decoded = base64.b64decode(b64_str)
                    images_payload.append(process_image_data(decoded))
                    content = "[图片已生成]"
                elif "data" in content_json and isinstance(content_json["data"], str):
                    b64_str = content_json["data"]
                    if b64_str.startswith("data:"):
                        b64_str = b64_str.split(",", 1)[1]
                    decoded = base64.b64decode(b64_str)
                    images_payload.append(process_image_data(decoded))
                    content = "[图片已生成]"

            except (json.JSONDecodeError, KeyError, binascii.Error):
                pass
        elif (
            "images" in message
            and isinstance(message["images"], list)
            and message["images"]
        ):
            for image_info in message["images"]:
                if image_info.get("type") == "image_url":
                    image_url_obj = image_info.get("image_url", {})
                    url_str = image_url_obj.get("url", "")
                    if url_str.startswith("data:image"):
                        try:
                            b64_data = url_str.split(",", 1)[1]
                            decoded = base64.b64decode(b64_data)
                            images_payload.append(process_image_data(decoded))
                        except (IndexError, binascii.Error) as e:
                            logger.warning(f"解析OpenRouter Base64图片数据失败: {e}")

            if images_payload:
                content = content if content else "[图片已生成]"

        content_parts = []
        if content:
            content_parts.append(TextPart(text=content))
        if reasoning_content:
            content_parts.append(ThoughtPart(thought_text=reasoning_content))
        for img in images_payload:
            content_parts.append(
                ImagePart(raw=img) if isinstance(img, bytes) else ImagePart(path=img)
            )

        if message_tool_calls := message.get("tool_calls"):
            for tc_data in message_tool_calls:
                try:
                    if tc_data.get("type") == "function":
                        raw_arguments = tc_data["function"]["arguments"]
                        repaired_arguments = raw_arguments

                        if raw_arguments:
                            try:
                                json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                try:
                                    repaired_obj = json_repair.loads(raw_arguments)
                                    if isinstance(repaired_obj, dict):
                                        repaired_arguments = json.dumps(
                                            repaired_obj, ensure_ascii=False
                                        )
                                        logger.debug(
                                            f"成功修复损坏的工具参数: {raw_arguments} -> "
                                            f"{repaired_arguments}"
                                        )
                                except Exception as repair_err:
                                    logger.warning(
                                        f"尝试修复损坏的工具参数失败: {raw_arguments}, "
                                        f"错误: {repair_err}"
                                    )

                        content_parts.append(
                            ToolCallPart(
                                id=tc_data["id"],
                                tool_name=tc_data["function"]["name"],
                                args=repaired_arguments,
                            )
                        )
                except KeyError as e:
                    logger.warning(
                        f"解析OpenAI工具调用数据时缺少键: {tc_data}, 错误: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"解析OpenAI工具调用数据时出错: {tc_data}, 错误: {e}"
                    )

        usage_info = response_json.get("usage")

        return ResponseData(
            content_parts=content_parts,
            usage_info=usage_info,
            raw_response=response_json,
        )


class ResponsesConfigMapper(OpenAIConfigMapper):
    """针对 OpenAI Responses API 的配置映射器"""

    def map_config(
        self,
        config: LLMGenerationConfig,
        model_detail: ModelDetail | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> dict[str, Any]:
        params = super().map_config(config, model_detail, capabilities)

        if "response_format" in params:
            fmt = params.pop("response_format")
            if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                json_schema_dict = fmt.get("json_schema", {})
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema_dict.get("name", "structured_response"),
                        "strict": json_schema_dict.get("strict", True),
                        "schema": json_schema_dict.get("schema", {}),
                    }
                }
            elif isinstance(fmt, dict):
                params["text"] = {"format": fmt}

        return params


class ResponsesMessageConverter(MessageConverter):
    """针对 OpenAI Responses API 的消息转换器"""

    def convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role

            if isinstance(msg, ToolMessage):
                returns = msg.tool_returns
                if returns:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": returns[0].tool_call_id,
                            "output": returns[0].output
                            if isinstance(returns[0].output, str)
                            else json.dumps(returns[0].output, ensure_ascii=False),
                        }
                    )
                    continue

            content_list: list[dict[str, Any]] = []
            for part in msg.content:
                if part is None:
                    continue

                if isinstance(part, TextPart):
                    c_type = "output_text" if role == "assistant" else "input_text"
                    content_list.append({"type": c_type, "text": part.text})
                elif isinstance(part, ImagePart):
                    if part.url is not None:
                        content_list.append(
                            {"type": "input_image", "image_url": part.url}
                        )
                    elif part.raw is not None:
                        import base64

                        raw_data = part.raw
                        mime = part.mime_type or "image/png"
                        b64_str = base64.b64encode(raw_data).decode("utf-8")
                        data_uri = f"data:{mime};base64,{b64_str}"
                        content_list.append(
                            {"type": "input_image", "image_url": data_uri}
                        )
                    elif part.path is not None:
                        import base64

                        raw_data = part.path.read_bytes()
                        mime = part.mime_type or "image/png"
                        b64_str = base64.b64encode(raw_data).decode("utf-8")
                        data_uri = f"data:{mime};base64,{b64_str}"
                        content_list.append(
                            {"type": "input_image", "image_url": data_uri}
                        )
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        c_type = "output_text" if role == "assistant" else "input_text"
                        content_list.append(
                            {"type": c_type, "text": part.get("text", "")}
                        )
                    elif part_type in {"image", "image_url"}:
                        image_src = part.get("image_url") or part.get("url", "")
                        content_list.append(
                            {"type": "input_image", "image_url": image_src}
                        )

            if content_list:
                input_items.append({"role": role, "content": content_list})

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.tool_name,
                            "arguments": tc.args
                            if isinstance(tc.args, str)
                            else json.dumps(tc.args, ensure_ascii=False),
                        }
                    )

        return input_items


class ResponsesToolSerializer(OpenAIToolSerializer):
    """针对 OpenAI Responses API 的工具序列化器"""

    def serialize_tools(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        res_tools = []
        for tool in tools:
            raw_schema = tool.parameters.copy() if tool.parameters else {}
            sanitized_schema = self.sanitize_schema(raw_schema)
            tool_payload = {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": sanitized_schema,
            }
            if self.api_type != "deepseek":
                tool_payload["strict"] = True
            res_tools.append(tool_payload)
        return res_tools


class ResponsesResponseParser(OpenAIResponseParser):
    """针对 OpenAI Responses API 的响应解析器"""

    def parse(self, response_json: dict[str, Any]) -> ResponseData:
        content_parts: list[Any] = []
        text_content = ""

        for item in response_json.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text_content += content_item.get("text", "")
                    elif content_item.get("type") == "refusal":
                        raise LLMException(
                            f"模型拒绝生成: {content_item.get('refusal')}",
                            code=LLMErrorCode.CONTENT_FILTERED,
                            recoverable=False,
                        )
            elif item.get("type") == "function_call":
                content_parts.append(
                    ToolCallPart(
                        id=item.get("call_id", ""),
                        tool_name=item.get("name", ""),
                        args=item.get("arguments", "{}"),
                    )
                )

        if text_content:
            content_parts.insert(0, TextPart(text=text_content))

        return ResponseData(
            content_parts=content_parts,
            usage_info=response_json.get("usage"),
            raw_response=response_json,
        )
