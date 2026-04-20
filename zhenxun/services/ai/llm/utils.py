"""
LLM 模块的工具和转换函数
"""

from typing import Any, TypeVar

from jinja2 import Template
import json_repair
from nonebot.compat import type_validate_json
from pydantic import BaseModel, Field, ValidationError, create_model

from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import TextPart
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.ai.types.models import ReasoningMode
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_validate

T = TypeVar("T", bound=BaseModel)

DEFAULT_IVR_TEMPLATE = (
    "### ❌ [输出格式验证失败]\n"
    "你的上一次输出未能通过系统的结构化校验。请立即启动修正流程：\n\n"
    "**错误反馈报告：**\n"
    "> {error_msg}\n\n"
    "**修正要求：** 请重新输出完整的 JSON 对象，确保其语法正确并完全符合预期的 Schema 规范。"
)


def resolve_json_schema_refs(schema: dict) -> dict:
    """
    递归解析 JSON Schema 中的 $ref，将其替换为 $defs/definitions 中的定义。
    用于兼容不支持 $ref 的 Gemini API。
    """
    definitions = schema.get("$defs") or schema.get("definitions") or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                if ref_name in definitions:
                    return _resolve(definitions[ref_name])

            return {
                key: _resolve(value)
                for key, value in node.items()
                if key not in ("$defs", "definitions")
            }

        if isinstance(node, list):
            return [_resolve(item) for item in node]

        return node

    return _resolve(schema)


def extract_text_from_content(
    content: str | list[Any] | None,
) -> str:
    """
    从消息内容中提取纯文本，自动过滤非文本部分，防止污染 Prompt。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.text for part in content if isinstance(part, TextPart) and part.text
        )
    return str(content)


def parse_and_validate_json(text: str, response_model: type[T]) -> T:
    """
    通用工具：尝试将文本解析为指定的 Pydantic 模型，并统一处理异常。
    """
    try:
        return type_validate_json(response_model, text)
    except (ValidationError, ValueError) as e:
        try:
            logger.warning(f"标准JSON解析失败，尝试使用json_repair修复: {e}")
            repaired_obj = json_repair.loads(text, skip_json_loads=True)
            return model_validate(response_model, repaired_obj)
        except Exception as repair_error:
            logger.error(
                f"LLM结构化输出校验最终失败: {repair_error}",
                e=repair_error,
            )
            raise LLMException(
                "LLM返回的JSON未能通过结构验证。",
                code=LLMErrorCode.RESPONSE_PARSE_ERROR,
                details={
                    "raw_response": text,
                    "validation_error": str(repair_error),
                    "original_error": repair_error,
                },
                cause=repair_error,
            )
    except Exception as e:
        logger.error(f"解析LLM结构化输出时发生未知错误: {e}", e=e)
        raise LLMException(
            "解析LLM的JSON输出时失败。",
            code=LLMErrorCode.RESPONSE_PARSE_ERROR,
            details={"raw_response": text},
            cause=e,
        )


def create_cot_wrapper(inner_model: type[BaseModel]) -> type[BaseModel]:
    """
    [动态运行时封装]
    创建一个包含思维链 (Chain of Thought) 的包装模型。
    强制模型在生成最终 JSON 结构前，先输出一个 reasoning 字段进行思考。
    """
    wrapper_name = f"CoT_{inner_model.__name__}"

    return create_model(
        wrapper_name,
        reasoning=(
            str,
            Field(
                ...,
                min_length=10,
                description=(
                    "在生成最终结果之前，请务必在此字段中详细描述你的推理步骤、计算过程或思考逻辑。禁止留空。"
                ),
            ),
        ),
        result=(
            inner_model,
            Field(
                ...,
            ),
        ),
    )


def should_apply_autocot(
    requested: bool,
    model_name: str | None,
    config: Any,
) -> bool:
    """
    [智能决策管道]
    判断是否应该应用 AutoCoT (显式思维链包装)。
    防止在模型已有原生思维能力时进行“双重思考”。
    """
    if not requested:
        return False

    if config:
        thinking_budget = getattr(config, "thinking_budget", 0) or 0
        if thinking_budget > 0:
            return False
        if getattr(config, "thinking_level", None) is not None:
            return False

    if model_name:
        caps = get_model_capabilities(model_name)
        if caps.reasoning_mode != ReasoningMode.NONE:
            return False

    return True


def render_prompt_template(template_string: str, variables: dict[str, Any]) -> str:
    """
    统一的 Jinja2 模板渲染函数。
    如果在渲染过程中发生异常，会记录警告日志并原样返回模板字符串作为容错。
    """
    if not template_string or not variables:
        return template_string

    try:
        template = Template(template_string)
        rendered_string = template.render(**variables)
        logger.debug(f"模板渲染成功: {rendered_string}")
        return rendered_string
    except Exception as e:
        logger.warning(f"Jinja2 模板渲染失败: {e}, 模板原内容: {template_string}", e=e)
        return template_string

