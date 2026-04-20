import base64
import json
from typing import Any

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
    ReasoningEffort,
    ResponseFormat,
)
from zhenxun.services.ai.config import get_gemini_safety_threshold
from zhenxun.services.ai.llm.utils import (
    resolve_json_schema_refs,
)
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import (
    AssistantMessage,
    AudioPart,
    FilePart,
    ImagePart,
    LLMGroundingAttribution,
    LLMGroundingMetadata,
    LLMMessage,
    SystemMessage,
    TextPart,
    ThoughtPart,
    ToolCallPart,
    ToolMessage,
    ToolReturnPart,
    UserMessage,
    VideoPart,
)
from zhenxun.services.ai.types.models import ModelCapabilities, ModelDetail
from zhenxun.services.ai.types.tools import (
    ToolDefinition,
)
from zhenxun.services.log import logger
from zhenxun.utils.http_utils import AsyncHttpx


class GeminiConfigMapper(ConfigMapper):
    def map_config(
        self,
        config: LLMGenerationConfig,
        model_detail: ModelDetail | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}

        if config.core:
            if config.core.temperature is not None:
                params["temperature"] = config.core.temperature
            if config.core.max_tokens is not None:
                params["maxOutputTokens"] = config.core.max_tokens
            if config.core.top_k is not None:
                params["topK"] = config.core.top_k
            if config.core.top_p is not None:
                params["topP"] = config.core.top_p

        if config.output:
            if config.output.response_format == ResponseFormat.JSON:
                params["responseMimeType"] = "application/json"
                if config.output.response_schema:
                    params["responseJsonSchema"] = config.output.response_schema
            elif config.output.response_mime_type is not None:
                params["responseMimeType"] = config.output.response_mime_type

            if (
                config.output.response_schema is not None
                and "responseJsonSchema" not in params
            ):
                params["responseJsonSchema"] = config.output.response_schema
            if config.output.response_modalities:
                params["responseModalities"] = config.output.response_modalities

        if config.tool_config:
            fc_config: dict[str, Any] = {"mode": config.tool_config.mode}
            if (
                config.tool_config.allowed_function_names
                and config.tool_config.mode == "ANY"
            ):
                builtins = {"code_execution", "google_search", "google_map"}
                user_funcs = [
                    name
                    for name in config.tool_config.allowed_function_names
                    if name not in builtins
                ]
                if user_funcs:
                    fc_config["allowedFunctionNames"] = user_funcs
            params["toolConfig"] = {"functionCallingConfig": fc_config}

        if config.reasoning:
            thinking_config = params.setdefault("thinkingConfig", {})
            model_name = model_detail.model_name.lower() if model_detail else ""
            is_pro = "pro" in model_name
            mode = capabilities.reasoning_mode.value if capabilities else None

            if (
                mode == "level"
                or "gemini-3" in model_name
                or "gemini-exp" in model_name
            ):
                if config.reasoning.effort:
                    level_str = config.reasoning.effort.value.lower()
                    if "3.1" in model_name and is_pro and level_str == "minimal":
                        logger.warning(
                            f"模型 {model_name} 不支持 minimal 思考等级，自动提升为 low。"
                        )
                        level_str = "low"
                    thinking_config["thinkingLevel"] = level_str

            else:
                max_budget = 32768 if is_pro else 24576

                if config.reasoning.budget_tokens is not None:
                    b_tokens = int(config.reasoning.budget_tokens)
                    if b_tokens == -1:
                        thinking_config["thinkingBudget"] = -1
                    elif b_tokens == 0:
                        if is_pro:
                            logger.warning(
                                f"模型 {model_name} (Pro) 不允许关闭思考，使用动态思考(-1)。"
                            )
                            thinking_config["thinkingBudget"] = -1
                        else:
                            thinking_config["thinkingBudget"] = 0
                    else:
                        min_budget = 128 if is_pro else 0
                        clamped = max(min_budget, min(b_tokens, max_budget))
                        thinking_config["thinkingBudget"] = clamped

                elif config.reasoning.effort:
                    effort = config.reasoning.effort
                    if effort == ReasoningEffort.HIGH:
                        thinking_config["thinkingBudget"] = max_budget
                    elif effort == ReasoningEffort.MEDIUM:
                        thinking_config["thinkingBudget"] = int(max_budget * 0.5)
                    elif effort == ReasoningEffort.LOW:
                        thinking_config["thinkingBudget"] = int(max_budget * 0.1)
                    elif effort == ReasoningEffort.MINIMAL:
                        if is_pro:
                            thinking_config["thinkingBudget"] = -1
                        else:
                            thinking_config["thinkingBudget"] = 0

            if config.reasoning.show_thoughts is not None:
                thinking_config["includeThoughts"] = config.reasoning.show_thoughts
            elif capabilities and capabilities.reasoning_visibility == "visible":
                thinking_config["includeThoughts"] = True

            if not thinking_config:
                params.pop("thinkingConfig", None)

        if config.visual:
            image_config: dict[str, Any] = {}

            if config.visual.aspect_ratio is not None:
                ar_value = (
                    config.visual.aspect_ratio.value
                    if isinstance(config.visual.aspect_ratio, ImageAspectRatio)
                    else config.visual.aspect_ratio
                )
                image_config["aspectRatio"] = ar_value

            if config.visual.resolution:
                image_config["imageSize"] = config.visual.resolution

            if image_config:
                params["imageConfig"] = image_config

            if config.visual.media_resolution:
                media_value = config.visual.media_resolution.upper()
                if not media_value.startswith("MEDIA_RESOLUTION_"):
                    media_value = f"MEDIA_RESOLUTION_{media_value}"
                params["mediaResolution"] = media_value

        if config.custom_params:
            mapped_custom = config.custom_params.copy()
            if "max_tokens" in mapped_custom:
                mapped_custom["maxOutputTokens"] = mapped_custom.pop("max_tokens")
            if "top_k" in mapped_custom:
                mapped_custom["topK"] = mapped_custom.pop("top_k")
            if "top_p" in mapped_custom:
                mapped_custom["topP"] = mapped_custom.pop("top_p")

            for key in (
                "code_execution_timeout",
                "grounding_config",
                "dynamic_threshold",
                "user_location",
                "reflexion_retries",
            ):
                mapped_custom.pop(key, None)

            for unsupported in [
                "frequency_penalty",
                "presence_penalty",
                "repetition_penalty",
            ]:
                if unsupported in mapped_custom:
                    mapped_custom.pop(unsupported)

            params.update(mapped_custom)

        safety_settings: list[dict[str, Any]] = []
        if config.safety and config.safety.safety_settings:
            for category, threshold in config.safety.safety_settings.items():
                safety_settings.append({"category": category, "threshold": threshold})
        else:
            threshold = get_gemini_safety_threshold()
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]:
                safety_settings.append({"category": category, "threshold": threshold})

        if safety_settings:
            params["safetySettings"] = safety_settings

        return params


class GeminiMessageConverter(MessageConverter):
    async def convert_part(self, part: Any) -> dict[str, Any]:
        """将单个内容部分转换为 Gemini API 格式"""

        def _get_gemini_resolution_dict() -> dict[str, Any]:
            if getattr(part, "media_resolution", None):
                value = part.media_resolution.upper()
                if not value.startswith("MEDIA_RESOLUTION_"):
                    value = f"MEDIA_RESOLUTION_{value}"
                return {"media_resolution": {"level": value}}
            return {}

        if isinstance(part, TextPart):
            return {"text": part.text}

        if isinstance(part, ThoughtPart):
            return {"text": part.thought_text, "thought": True}

        if isinstance(part, ImagePart):
            import base64

            if part.url is not None:
                logger.debug(f"正在为Gemini下载并编码URL图片: {part.url}")
                try:
                    raw_data = await AsyncHttpx.get_content(part.url)
                except Exception as e:
                    logger.error(f"下载或编码URL图片失败: {e}", e=e)
                    raise ValueError(f"无法处理图片URL: {e}")
            elif part.raw is not None:
                raw_data = part.raw
            elif part.path is not None:
                raw_data = part.path.read_bytes()
            else:
                raise ValueError("ImagePart 必须且只能提供 url, raw, path 中的一个")

            mime_type = part.mime_type or "image/jpeg"
            base64_data = base64.b64encode(raw_data).decode("utf-8")
            payload = {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            payload.update(_get_gemini_resolution_dict())
            return payload

        if isinstance(part, VideoPart):
            import base64

            if part.url is not None:
                raise ValueError(
                    "Gemini API 的视频处理需要通过 File API 上传，不支持直接 URL"
                )
            elif part.raw is not None:
                raw_data = part.raw
            elif part.path is not None:
                raw_data = part.path.read_bytes()
            else:
                raise ValueError("VideoPart 必须且只能提供 url, raw, path 中的一个")
            mime_type = part.mime_type or "video/mp4"
            base64_data = base64.b64encode(raw_data).decode("utf-8")
            payload = {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            payload.update(_get_gemini_resolution_dict())
            return payload

        if isinstance(part, AudioPart):
            import base64

            if part.url is not None:
                raise ValueError(
                    "Gemini API 的音频处理需要通过 File API 上传，不支持直接 URL"
                )
            elif part.raw is not None:
                raw_data = part.raw
            elif part.path is not None:
                raw_data = part.path.read_bytes()
            else:
                raise ValueError("AudioPart 必须且只能提供 url, raw, path 中的一个")
            mime_type = part.mime_type or "audio/mp3"
            base64_data = base64.b64encode(raw_data).decode("utf-8")
            payload = {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            payload.update(_get_gemini_resolution_dict())
            return payload

        if isinstance(part, FilePart):
            if part.url is not None:
                payload = {
                    "fileData": {"mimeType": part.mime_type, "fileUri": part.url}
                }
                payload.update(_get_gemini_resolution_dict())
                return payload
            elif part.raw is not None or part.path is not None:
                file_name = (
                    part.metadata.get("name", "file") if part.metadata else "file"
                )
                return {"text": f"[文件: {file_name}]\n<已省略的二进制内容>"}

        if isinstance(part, ToolCallPart):
            payload = {
                "functionCall": {
                    "id": part.id,
                    "name": part.tool_name,
                    "args": part.args
                    if isinstance(part.args, dict)
                    else (json.loads(part.args) if part.args else {}),
                }
            }
            if part.metadata and "thought_signature" in part.metadata:
                payload["thoughtSignature"] = part.metadata["thought_signature"]
            return payload

        if isinstance(part, ToolReturnPart):
            payload = {
                "functionResponse": {
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "response": part.output
                    if isinstance(part.output, dict)
                    else {"result": part.output},
                }
            }
            if part.metadata and "thought_signature" in part.metadata:
                payload["thoughtSignature"] = part.metadata["thought_signature"]
            return payload

        raise ValueError(f"不支持的内容类型: {part.type}")

    async def convert_messages_async(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]]:
        gemini_contents: list[dict[str, Any]] = []

        for msg in messages:
            current_parts: list[dict[str, Any]] = []
            if isinstance(msg, SystemMessage):
                continue

            elif isinstance(msg, UserMessage):
                for part_obj in msg.content:
                    current_parts.append(await self.convert_part(part_obj))
                gemini_contents.append({"role": "user", "parts": current_parts})

            elif isinstance(msg, AssistantMessage):
                for part_obj in msg.content:
                    part_dict = await self.convert_part(part_obj)
                    if part_obj.metadata and "thought_signature" in part_obj.metadata:
                        part_dict["thoughtSignature"] = part_obj.metadata[
                            "thought_signature"
                        ]
                    current_parts.append(part_dict)

                if current_parts:
                    gemini_contents.append({"role": "model", "parts": current_parts})

            elif isinstance(msg, ToolMessage):
                from zhenxun.services.ai.types.messages import ToolReturnPart

                for part_obj in msg.content:
                    if isinstance(part_obj, ToolReturnPart):
                        result_obj = part_obj.output
                        if isinstance(result_obj, str):
                            try:
                                result_obj = json.loads(result_obj)
                            except json.JSONDecodeError:
                                pass
                        if not isinstance(result_obj, dict):
                            result_obj = {"result": result_obj}

                        current_parts.append(
                            {
                                "functionResponse": {
                                    "id": part_obj.tool_call_id,
                                    "name": part_obj.tool_name,
                                    "response": result_obj,
                                }
                            }
                        )
                    else:
                        part_dict = await self.convert_part(part_obj)
                        current_parts.append(part_dict)

                if current_parts:
                    if gemini_contents and gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"].extend(current_parts)
                    else:
                        gemini_contents.append({"role": "user", "parts": current_parts})

        return gemini_contents

    def convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        raise NotImplementedError("Use convert_messages_async for Gemini")


class GeminiToolSerializer(ToolSerializer):
    def sanitize_schema(self, schema: Any) -> Any:
        """
        递归地净化 JSON Schema，移除 Gemini API 不支持的关键字。
        """
        if isinstance(schema, list):
            return [self.sanitize_schema(item) for item in schema]
        if isinstance(schema, dict):
            schema_copy: dict[str, Any] = schema.copy()

            if "const" in schema_copy:
                schema_copy["enum"] = [schema_copy.pop("const")]

            if "type" in schema_copy and isinstance(schema_copy["type"], list):
                types_list = schema_copy["type"]
                if "null" in types_list:
                    schema_copy["nullable"] = True
                    types_list = [t for t in types_list if t != "null"]
                    if len(types_list) == 1:
                        schema_copy["type"] = types_list[0]
                    else:
                        schema_copy["type"] = types_list

            if "anyOf" in schema_copy:
                any_of = schema_copy["anyOf"]
                has_null = any(
                    isinstance(x, dict) and x.get("type") == "null" for x in any_of
                )
                if has_null:
                    schema_copy["nullable"] = True
                    new_any_of = [
                        x
                        for x in any_of
                        if not (isinstance(x, dict) and x.get("type") == "null")
                    ]
                    if len(new_any_of) == 1:
                        schema_copy.update(new_any_of[0])
                        schema_copy.pop("anyOf", None)
                    else:
                        schema_copy["anyOf"] = new_any_of

            unsupported_keys = [
                "exclusiveMinimum",
                "exclusiveMaximum",
                "default",
                "title",
                "additionalProperties",
                "$schema",
                "$id",
                "propertyNames",
                "patternProperties",
            ]
            for key in unsupported_keys:
                schema_copy.pop(key, None)

            if schema_copy.get("format") and schema_copy["format"] not in [
                "enum",
                "date-time",
            ]:
                schema_copy.pop("format", None)

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

    def serialize_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        function_declarations: list[dict[str, Any]] = []
        for tool_def in tools:
            raw_schema = tool_def.parameters.copy() if tool_def.parameters else {}
            resolved_schema = resolve_json_schema_refs(raw_schema)
            sanitized_schema = self.sanitize_schema(resolved_schema)
            declaration = {
                "name": tool_def.name,
                "description": tool_def.description or "",
                "parameters": sanitized_schema,
            }
            function_declarations.append(declaration)
        return function_declarations


class GeminiResponseParser(ResponseParser):
    def validate_response(self, response_json: dict[str, Any]) -> None:
        if error := response_json.get("error"):
            code = error.get("code")
            message = error.get("message", "")
            status = error.get("status")
            details = error.get("details", [])

            if code == 429 or status == "RESOURCE_EXHAUSTED":
                is_quota = any(
                    d.get("reason") in ("QUOTA_EXCEEDED", "SERVICE_DISABLED")
                    for d in details
                    if isinstance(d, dict)
                )
                if is_quota or "quota" in message.lower():
                    raise LLMException(
                        f"Gemini配额耗尽: {message}",
                        code=LLMErrorCode.API_QUOTA_EXCEEDED,
                        details=error,
                    )
                raise LLMException(
                    f"Gemini速率限制: {message}",
                    code=LLMErrorCode.API_RATE_LIMITED,
                    details=error,
                )

            if code == 400 or status in ("INVALID_ARGUMENT", "FAILED_PRECONDITION"):
                raise LLMException(
                    f"Gemini参数错误: {message}",
                    code=LLMErrorCode.INVALID_PARAMETER,
                    details=error,
                    recoverable=False,
                )

        if prompt_feedback := response_json.get("promptFeedback"):
            if block_reason := prompt_feedback.get("blockReason"):
                raise LLMException(
                    f"内容被安全过滤: {block_reason}",
                    code=LLMErrorCode.CONTENT_FILTERED,
                    details={
                        "block_reason": block_reason,
                        "safety_ratings": prompt_feedback.get("safetyRatings"),
                    },
                )

    def parse(self, response_json: dict[str, Any]) -> ResponseData:
        self.validate_response(response_json)

        if "image_generation" in response_json and isinstance(
            response_json["image_generation"], dict
        ):
            candidates_source = response_json["image_generation"]
        else:
            candidates_source = response_json

        candidates = candidates_source.get("candidates", [])
        usage_info = response_json.get("usageMetadata")

        if not candidates:
            return ResponseData(raw_response=response_json)

        candidate = candidates[0]
        thought_signature: str | None = None

        content_data = candidate.get("content", {})
        parts = content_data.get("parts", [])

        content_parts: list[Any] = []
        thought_summary_parts: list[str] = []
        answer_parts = []

        for part in parts:
            part_signature = part.get("thoughtSignature")
            if part_signature and thought_signature is None:
                thought_signature = part_signature
            part_metadata: dict[str, Any] | None = None
            if part_signature:
                part_metadata = {"thought_signature": part_signature}

            if part.get("thought") is True:
                t_text = part.get("text", "")
                thought_summary_parts.append(t_text)
                content_parts.append(ThoughtPart(thought_text=t_text))

            elif "text" in part:
                answer_parts.append(part["text"])
                c_part = TextPart(text=part["text"], metadata=part_metadata)
                content_parts.append(c_part)

            elif "thoughtSummary" in part:
                thought_summary_parts.append(part["thoughtSummary"])
                content_parts.append(ThoughtPart(thought_text=part["thoughtSummary"]))

            elif "inlineData" in part:
                inline_data = part["inlineData"]
                if "data" in inline_data:
                    decoded = base64.b64decode(inline_data["data"])
                    processed_img = process_image_data(decoded)
                    content_parts.append(
                        ImagePart(raw=processed_img)
                        if isinstance(processed_img, bytes)
                        else ImagePart(path=processed_img)
                    )

            elif "functionCall" in part or "toolCall" in part:
                fc_data = part.get("functionCall") or part.get("toolCall")
                fc_sig = part_signature
                try:
                    call_count = sum(
                        1
                        for p in content_parts
                        if getattr(p, "type", "") == "tool_call"
                    )
                    call_id = fc_data.get("id", f"call_gemini_{call_count}")
                    tc_part = ToolCallPart(
                        id=call_id,
                        tool_name=fc_data.get("name")
                        or fc_data.get("toolType", "unknown"),
                        args=fc_data.get("args", {}),
                    )
                    if fc_sig:
                        tc_part.metadata = {"thought_signature": fc_sig}
                    content_parts.append(tc_part)
                except Exception as e:
                    logger.warning(
                        f"解析Gemini functionCall时出错: {fc_data}, 错误: {e}"
                    )

        grounding_metadata_obj = None
        if grounding_data := candidate.get("groundingMetadata"):
            try:
                sep_content = None
                sep_field = grounding_data.get("searchEntryPoint")
                if isinstance(sep_field, dict):
                    sep_content = sep_field.get("renderedContent")

                attributions = []
                if chunks := grounding_data.get("groundingChunks"):
                    for chunk in chunks:
                        if web := chunk.get("web"):
                            attributions.append(
                                LLMGroundingAttribution(
                                    title=web.get("title"),
                                    uri=web.get("uri"),
                                    snippet=web.get("snippet"),
                                    confidence_score=None,
                                )
                            )

                grounding_metadata_obj = LLMGroundingMetadata(
                    web_search_queries=grounding_data.get("webSearchQueries"),
                    grounding_attributions=attributions or None,
                    search_suggestions=grounding_data.get("searchSuggestions"),
                    search_entry_point=sep_content,
                    map_widget_token=grounding_data.get("googleMapsWidgetContextToken"),
                )
            except Exception as e:
                logger.warning(f"无法解析Grounding元数据: {grounding_data}, {e}")

        return ResponseData(
            content_parts=content_parts,
            usage_info=usage_info,
            raw_response=response_json,
            grounding_metadata=grounding_metadata_obj,
        )
