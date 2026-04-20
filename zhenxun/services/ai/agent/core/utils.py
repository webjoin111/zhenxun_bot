from collections.abc import Callable
import json
from typing import Any

from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel

from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.types.tools import (
    ToolDefinition,
    ToolErrorResult,
    ToolErrorType,
    ToolResult,
)
from zhenxun.utils.pydantic_compat import model_dump, model_json_schema, parse_as


class HandoffExecutable(ToolExecutable):
    """
    动态生成的 Handoff (移交) 工具。
    伪装成普通工具让 LLM 调用，执行后返回特殊标记，供后续 Executor 拦截并进行上下文热切换。
    """
    _dynamic_def: Any = None

    def __init__(self, target_agent: Any, payload_model: type[BaseModel] | None = None):
        self.target_agent = target_agent
        self.target_name = target_agent.name
        self.target_desc = target_agent.instruction
        self.payload_model = payload_model
        self.tool_name = f"transfer_to_{self.target_name}"

    async def get_definition(self, context: Any | None = None) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        properties = {
            "reason": {
                "type": "string",
                "description": "向目标 Agent 解释为什么移交给它，以及它需要做什么",
            },
            "context_to_pass": {
                "type": "string",
                "description": "总结并传递必要的上下文信息，确保目标 Agent 能够无缝接手",
            },
        }
        required = ["reason", "context_to_pass"]

        if self.payload_model:
            schema = model_json_schema(self.payload_model)
            properties["payload"] = schema
            required.append("payload")

        return ToolDefinition(
            name=self.tool_name,
            description=f"将对话控制权移交给 {self.target_name}。当用户意图符合以下描述时调用此工具：{self.target_desc[:100]}...",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    async def execute(self, context=None, **kwargs) -> ToolResult:
        """
        阶段三：捕获移交时可能存在的 payload 数据。
        """
        handoff_payload = {
            "__handoff__": True,
            "target_agent": self.target_name,
            "kwargs": kwargs,
            "payload": kwargs.get("payload"),
        }
        return ToolResult(
            output=json.dumps(handoff_payload, ensure_ascii=False),
            display=f"🔄 正在将控制权移交给 {self.target_name}...",
        )

    async def should_confirm(self, **kwargs: Any) -> str | None:
        return None

class SubmitFinalResultExecutable(ToolExecutable):
    """
    动态生成的提交最终结果工具。
    用于将大模型的结构化输出拦截并终止 AgentExecutor 的循环。
    """
    _dynamic_def: Any = None

    def __init__(
        self,
        response_model: type[BaseModel],
        val_cb: Callable | None = None,
        is_auto_thinking: bool = False,
        original_model: type[BaseModel] | None = None,
    ):
        self.response_model = response_model
        self.val_cb = val_cb
        self.is_auto_thinking = is_auto_thinking
        self.original_model = original_model or response_model
        self.tool_name = "submit_final_result"

    async def get_definition(self, context: Any | None = None) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        schema = model_json_schema(self.response_model)
        return ToolDefinition(
            name=self.tool_name,
            description="当你完成所有必要的调查 and 思考后，必须且只能调用此工具来提交最终的结构化结果。提交后任务将立刻结束。",
            parameters=schema,
        )

    async def execute(self, context: Any | None = None, **kwargs) -> ToolResult:
        try:
            parse_target = kwargs
            if isinstance(kwargs, dict):
                if "kwargs" in kwargs and len(kwargs) == 1:
                    parse_target = kwargs["kwargs"]
                elif "result" in kwargs and len(kwargs) == 1:
                    parse_target = kwargs["result"]
            parsed_obj = parse_as(self.response_model, parse_target)

            if self.is_auto_thinking:
                final_obj = getattr(parsed_obj, "result")
            else:
                final_obj = parsed_obj

            if self.val_cb:
                if is_coroutine_callable(self.val_cb):
                    res = await self.val_cb(final_obj)
                else:
                    res = self.val_cb(final_obj)
                if res is not None:
                    final_obj = res

            final_dict = model_dump(final_obj)

            payload = {"__final_structured_result__": True, "data": final_dict}
            return ToolResult(
                output=json.dumps(payload, ensure_ascii=False),
                display="✅ 结构化结果校验通过，已提交。",
            )
        except Exception as e:
            error_msg = f"你的输出未能通过系统校验！请根据以下错误信息进行修正，并重新调用本工具提交：\n{e}"
            error_res = ToolErrorResult(
                error_type=ToolErrorType.INVALID_ARGUMENTS,
                message=error_msg,
                is_retryable=True,
            )
            return ToolResult(
                output=model_dump(error_res),
                display=f"❌ 结构化校验失败，已驳回要求大模型修正: {e}",
            )

    async def should_confirm(self, **kwargs: Any) -> str | None:
        return None


