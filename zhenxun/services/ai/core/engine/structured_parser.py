from functools import reduce
import types
from typing import Any, Generic, Literal, Union, cast, get_args, get_origin

import json_repair
from nonebot.compat import type_validate_json
from pydantic import BaseModel, Field, ValidationError, create_model

from zhenxun.services.ai.core.exceptions import (
    ControlFlowException,
    GuardrailViolationError,
    SchemaParseError,
    SubmitStructuredException,
)
from zhenxun.services.ai.core.models import ToolDefinition
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.run.models import OutputDataT
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_json_schema, model_validate

DEFAULT_IVR_TEMPLATE = (
    "### ❌ [输出内容或格式验证失败]\n"
    "你的上一次输出未能通过系统的校验与规则检查。请立即启动修正流程：\n\n"
    "**错误反馈报告：**\n"
    "> {error_msg}\n\n"
    "**修正要求：** 请结合反馈报告，仔细反思你的输出内容或格式，并重新生成正确的数据以满足所有的规则与规范。"
)


class BaseOutputProcessor(Generic[OutputDataT]):
    """
    统一的结构化输出处理器。
    负责管理 Schema 生成、Prompt 约束注入以及最终的 JSON 反序列化和业务校验。
    """

    def __init__(
        self,
        response_model: type[Any],
        error_template: str | None = None,
    ):
        self.original_model = response_model
        self.error_template = error_template or DEFAULT_IVR_TEMPLATE

        self.target_model, self.is_union_wrapped = self._create_union_wrapper(
            response_model
        )

    @staticmethod
    def _create_union_wrapper(union_type: Any) -> tuple[type[BaseModel], bool]:
        """[私有方法] 如果是 Union 类型，动态构建带 kind 区分字段的模型"""
        origin = get_origin(union_type)
        union_types = [Union]
        if hasattr(types, "UnionType"):
            union_types.append(types.UnionType)

        if origin not in union_types:
            return union_type, False

        args = get_args(union_type)
        branch_models = []
        for i, arg in enumerate(args):
            name = getattr(arg, "__name__", f"Option_{i}")
            branch_model = create_model(
                f"UnionBranch_{name}",
                kind=(
                    Literal[name],
                    Field(default=name, description=f"选择输出结构为: {name}"),
                ),
                data=(arg, Field(..., description=f"{name} 的具体数据")),
            )
            branch_models.append(branch_model)

        UnionOfBranches = reduce(lambda left, right: left | right, branch_models)
        UnionWrapper = create_model(
            "UnionResponseWrapper",
            result=(
                UnionOfBranches,
                Field(..., description="根据你的决策，选择一种格式输出"),
            ),
        )
        return UnionWrapper, True

    def get_json_schema(self) -> dict[str, Any]:
        """提取目标模型的 JSON Schema"""
        try:
            return model_json_schema(self.target_model)
        except AttributeError:
            return self.target_model.schema()

    def _parse_and_validate(self, text: str) -> Any:
        """[私有方法] 执行带有容错修复的 JSON 解析与模型验证"""
        try:
            return type_validate_json(self.target_model, text)
        except (ValidationError, ValueError) as e:
            try:
                logger.warning(f"标准JSON解析失败，尝试使用json_repair修复: {e}")
                repaired_obj = json_repair.loads(text, skip_json_loads=True)
                return model_validate(self.target_model, repaired_obj)
            except Exception as repair_error:
                logger.error(
                    f"LLM结构化输出校验最终失败: {repair_error}", e=repair_error
                )
                raise SchemaParseError(
                    f"JSON格式损坏或字段不匹配，未能通过Schema验证: {repair_error}"
                )
        except Exception as e:
            logger.error(f"解析LLM结构化输出时发生未知错误: {e}", e=e)
            raise SchemaParseError(f"解析LLM的JSON输出时失败: {e}")

    async def validate_and_parse(self, text: str, context: Any = None) -> OutputDataT:
        """执行 JSON 解析与回调验证"""
        try:
            parsed_obj = self._parse_and_validate(text)

            current_obj = parsed_obj

            if getattr(self, "is_union_wrapped", False):
                branch = getattr(current_obj, "result")
                current_obj = getattr(branch, "data")

            final_obj = cast(OutputDataT, current_obj)

            return final_obj
        except Exception as e:
            raise e


class SubmitFinalResultExecutable(ToolExecutable):
    """
    动态生成的提交最终结果工具。
    用于将大模型的结构化输出拦截并终止 AgentExecutor 的循环。
    """

    _dynamic_def: Any = None
    name: str = ""

    def __init__(
        self,
        output_processor: BaseOutputProcessor,
        guardrails: list[Any] | None = None,
    ):
        self.output_processor = output_processor
        self.guardrails = guardrails or []
        self.tool_name = "submit_final_result"
        self.name = self.tool_name

    async def get_definition(self, context: Any | None = None) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        schema = self.output_processor.get_json_schema()
        return ToolDefinition(
            name=self.tool_name,
            description="当你完成所有必要的调查 and 思考后，必须且只能调用此工具来提交最终的结构化结果。提交后任务将立刻结束。",
            parameters=schema,
        )

    async def execute(self, context: Any | None = None, **kwargs) -> ToolResult:
        parse_target = kwargs
        if isinstance(kwargs, dict):
            if "kwargs" in kwargs and len(kwargs) == 1:
                parse_target = kwargs["kwargs"]
            elif "result" in kwargs and len(kwargs) == 1:
                parse_target = kwargs["result"]

        try:
            json_str = __import__("json").dumps(parse_target, ensure_ascii=False)
            final_obj = await self.output_processor.validate_and_parse(
                json_str, context=context
            )
            failed_feedbacks = []
            for v in self.guardrails:
                v_res = await v.validate(json_str, final_obj, context)
                if not v_res.success:
                    failed_feedbacks.append(v_res.feedback or "未知校验失败")

            if failed_feedbacks:
                raise GuardrailViolationError("\n".join(failed_feedbacks))

            raise SubmitStructuredException(data=final_obj)
        except ControlFlowException as e:
            raise e
        except Exception as e:
            error_msg = f"系统捕获到解析异常：\n{e}"
            raise SchemaParseError(error_msg)

    async def should_confirm(
        self, context: Any | None = None, **kwargs: Any
    ) -> str | None:
        return None
