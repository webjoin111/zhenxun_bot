from collections.abc import Callable
import inspect
import re
from typing import Any, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from .di import DependencyInjector


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """解析文档字符串，解耦提取主描述和参数描述"""
    if not docstring:
        return "", {}

    lines = docstring.strip().splitlines()
    main_desc_lines = []
    params: dict[str, str] = {}

    section_header = re.compile(r"^\s*(?:Args|Arguments|Parameters|参数)\s*[:：]\s*$")
    return_header = re.compile(r"^\s*(?:Returns|Return|返回)\s*[:：]\s*$")
    param_pattern = re.compile(r"^\s*(\**\w+)(?:\s*\(.*?\))?\s*[:：]\s*(.*)")

    current_section = "main"
    last_param = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_section == "main":
                main_desc_lines.append("")
            continue

        if section_header.match(line):
            current_section = "params"
            continue
        elif return_header.match(line):
            current_section = "returns"
            continue

        if current_section == "main":
            main_desc_lines.append(line.strip())
        elif current_section == "params":
            match = param_pattern.match(line)
            if match:
                param_name = match.group(1).lstrip("*")
                param_desc = match.group(2).strip()
                params[param_name] = param_desc
                last_param = param_name
            elif last_param and line.startswith(" ") and stripped:
                params[last_param] += f" {stripped}"

    main_desc = "\n".join(main_desc_lines).strip()
    return main_desc, params


def build_tool_model(
    func: Callable, strict: bool = False
) -> tuple[dict, type[BaseModel]]:
    """利用 pydantic.create_model 从函数签名动态构建绝对严谨的 JSON Schema 及验证模型"""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)
    _, doc_params = _parse_docstring(func.__doc__)

    field_definitions = {}

    for name, param in sig.parameters.items():
        if name in ("self", "cls", "args", "kwargs"):
            continue

        if DependencyInjector.can_resolve_statically(param):
            continue

        anno = type_hints.get(name, Any)
        desc = doc_params.get(name, "")
        default_val = param.default

        if isinstance(default_val, FieldInfo):
            field_info = default_val
            if not field_info.description and desc:
                field_info.description = desc
            field_definitions[name] = (anno, field_info)
        else:
            if default_val is inspect.Parameter.empty:
                field_info = Field(..., description=desc)
            else:
                field_info = Field(default=default_val, description=desc)
            field_definitions[name] = (anno, field_info)

    DynamicModel = create_model(
        f"DynamicToolModel_{func.__name__}", **field_definitions
    )
    schema_def = DynamicModel.model_json_schema(mode="serialization")

    schema_def.pop("title", None)
    if strict:
        schema_def["additionalProperties"] = False

    return schema_def, DynamicModel
