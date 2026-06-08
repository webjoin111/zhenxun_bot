from abc import ABC
from collections.abc import Callable
from enum import Enum
import inspect
from typing import Any

from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel, Field

from zhenxun.services.ai.core.exceptions import (
    GuardrailFatalException,
    GuardrailViolationError,
)
from zhenxun.services.ai.core.messages import LLMMessage, LLMResponse, TextPart
from zhenxun.services.ai.run.context import RunContext


class GuardrailAction(str, Enum):
    PASS = "PASS"
    """放行"""
    REJECT = "REJECT"
    """致命拦截：直接中断大模型思考"""
    REFLECT = "REFLECT"
    """打回反思：触发自愈闭环"""
    MUTATE = "MUTATE"
    """数据变异：就地修改数据后放行"""


class GuardrailResult(BaseModel):
    """护栏验证结果的统一载体"""

    action: GuardrailAction = GuardrailAction.PASS
    """验证动作（放行、拦截、反思、变异）"""

    feedback: str | None = None
    """未通过时的校验失败反馈原因或拒绝理由"""

    mutated_text: str | None = None
    """变异后的新文本内容（MUTATE 模式下生效）"""

    mutated_obj: Any | None = None
    """变异后的新解析对象（MUTATE 模式下生效）"""

    @property
    def success(self) -> bool:
        return self.action == GuardrailAction.PASS


class BaseGuardrail(ABC):
    """大一统的业务逻辑护栏抽象基类"""

    async def validate_input(
        self, messages: list["LLMMessage"], context: "RunContext | None" = None
    ) -> GuardrailResult:
        """执行输入拦截与变异逻辑"""
        return GuardrailResult(action=GuardrailAction.PASS)

    async def validate_output(
        self,
        response: "LLMResponse | str",
        parsed_obj: Any,
        context: "RunContext | None" = None,
    ) -> GuardrailResult:
        """执行输出反思、拦截与变异逻辑"""
        return GuardrailResult(action=GuardrailAction.PASS)


class FunctionalGuardrail(BaseGuardrail):
    """包装普通 Python 函数的智能护栏"""

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        self.is_input = bool(params and params[0] == "messages")

    async def validate_input(
        self, messages: list["LLMMessage"], context: "RunContext | None" = None
    ) -> GuardrailResult:
        if not self.is_input:
            return GuardrailResult(action=GuardrailAction.PASS)

        sig = inspect.signature(self.func)
        takes_ctx = len(sig.parameters) > 1
        try:
            res = (
                (
                    await self.func(messages, context)
                    if is_coroutine_callable(self.func)
                    else self.func(messages, context)
                )
                if takes_ctx
                else (
                    await self.func(messages)
                    if is_coroutine_callable(self.func)
                    else self.func(messages)
                )
            )
            return self._parse_result(res, is_input=True)
        except (ValueError, AssertionError) as e:
            return GuardrailResult(action=GuardrailAction.REJECT, feedback=str(e))

    async def validate_output(
        self,
        response: "LLMResponse | str",
        parsed_obj: Any,
        context: "RunContext | None" = None,
    ) -> GuardrailResult:
        if self.is_input:
            return GuardrailResult(action=GuardrailAction.PASS)

        sig = inspect.signature(self.func)
        takes_ctx = len(sig.parameters) > 1
        try:
            res = (
                (
                    await self.func(context, parsed_obj)
                    if is_coroutine_callable(self.func)
                    else self.func(context, parsed_obj)
                )
                if takes_ctx
                else (
                    await self.func(parsed_obj)
                    if is_coroutine_callable(self.func)
                    else self.func(parsed_obj)
                )
            )
            return self._parse_result(res, is_input=False)
        except (ValueError, AssertionError) as e:
            return GuardrailResult(action=GuardrailAction.REFLECT, feedback=str(e))

    def _parse_result(self, res: Any, is_input: bool) -> GuardrailResult:
        """统一处理返回值类型推导"""
        if isinstance(res, GuardrailResult):
            return res

        if res is False:
            action = GuardrailAction.REJECT if is_input else GuardrailAction.REFLECT
            return GuardrailResult(
                action=action,
                feedback=f"自定义护栏函数 '{self.func.__name__}' 校验未通过",
            )
        elif isinstance(res, str):
            action = GuardrailAction.REJECT if is_input else GuardrailAction.REFLECT
            return GuardrailResult(action=action, feedback=res)

        return GuardrailResult(action=GuardrailAction.PASS)


class JudgeViolation(BaseModel):
    rule: str
    """违反的规则内容"""

    reason: str
    """违反该规则的具体原因和证据。如果没有违反，填无"""


class JudgeResponse(BaseModel):
    passed: bool
    """文本是否完全遵守了所有的规则。如果有任何一条违反，此处必须为 False"""

    violations: list[JudgeViolation] = Field(default_factory=list)
    """违反的规则列表及原因。如果没有违反，返回空列表"""


class LLMJudgeConfig(BaseModel):
    """LLM 裁判的全局设定"""

    judge_model: str | None = None
    """指定的裁判模型名称，如果为空则优先使用当前对话模型，其次为全局默认模型"""

    system_prompt_template: str | None = None
    """自定义裁判 Prompt 模板。必须包含 {rules} 和 {text} 占位符"""


class LLMGuardrail(BaseGuardrail):
    """基于 LLM-as-a-Judge 的自然语言规则裁判护栏"""

    def __init__(self, rules: list[str], config: LLMJudgeConfig | None = None):
        self.rules = rules
        self.config = config or LLMJudgeConfig()

    async def validate_output(
        self,
        response: "LLMResponse | str",
        parsed_obj: Any,
        context: "RunContext | None" = None,
    ) -> GuardrailResult:
        if not self.rules:
            return GuardrailResult(action=GuardrailAction.PASS)

        text = response if isinstance(response, str) else response.text
        from zhenxun.services.ai.llm.api import generate_structured
        from zhenxun.services.ai.llm.manager import get_default_model

        rules_str = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(self.rules)])
        if self.config.system_prompt_template:
            prompt = self.config.system_prompt_template.format(
                rules=rules_str, text=text
            )
        else:
            prompt = (
                "你是一个严格的内容风控与业务合规裁判。\n"
                "请评估以下[待检测内容]是否违反了任何一条[规则列表]。\n\n"
                f"### [规则列表]\n{rules_str}\n\n"
                f"### [待检测内容]\n{text}\n\n"
                "请严格按照规则评估。只要违反了其中任意一条，"
                "passed 必须为 false，并在 violations 中详细说明理由。"
            )

        model = self.config.judge_model
        if not model and context and context.run.current_model:
            model = context.run.current_model
        if not model:
            model = get_default_model("chat")

        try:
            res = await generate_structured(
                prompt, response_model=JudgeResponse, model=model
            )
            if res.passed:
                return GuardrailResult(action=GuardrailAction.PASS)
            feedbacks = [
                f"违反规则: 【{v.rule}】, 原因: {v.reason}" for v in res.violations
            ]
            return GuardrailResult(
                action=GuardrailAction.REFLECT, feedback="\n".join(feedbacks)
            )
        except Exception as e:
            return GuardrailResult(
                action=GuardrailAction.REFLECT, feedback=f"系统护栏裁判模型异常: {e}"
            )


GuardrailSource = Callable[..., Any] | str | BaseGuardrail | LLMJudgeConfig
"""护栏来源（函数、自然语言规则、BaseGuardrail 实例或裁判配置）"""


def parse_guardrails(guardrails: list[GuardrailSource] | None) -> list[BaseGuardrail]:
    """工具方法：将各种类型的 Guardrail 解析为标准的 BaseGuardrail 列表"""
    v_list = []
    llm_rules = []
    judge_config = None
    for v in guardrails or []:
        if isinstance(v, BaseGuardrail):
            v_list.append(v)
        elif callable(v):
            v_list.append(FunctionalGuardrail(v))
        elif isinstance(v, str):
            llm_rules.append(v)
        elif isinstance(v, LLMJudgeConfig):
            judge_config = v

    if llm_rules:
        v_list.append(LLMGuardrail(rules=llm_rules, config=judge_config))

    return v_list


class GuardrailPipeline:
    """护栏管线引擎"""

    def __init__(self, guardrails: list[BaseGuardrail]):
        self.guardrails = guardrails

    async def run_input_pipeline(
        self, messages: list["LLMMessage"], context: "RunContext | None" = None
    ) -> list["LLMMessage"]:
        """执行 Input 护栏拦截与变异"""
        for g in self.guardrails:
            res = await g.validate_input(messages, context)
            if res.action == GuardrailAction.REJECT:
                raise GuardrailFatalException(
                    guard_name=g.__class__.__name__, reason=res.feedback or "输入被拦截"
                )
            elif (
                res.action == GuardrailAction.MUTATE
                and res.mutated_text is not None
                and messages
            ):
                text_replaced = False
                for p in messages[-1].content:
                    if isinstance(p, TextPart):
                        p.text = res.mutated_text
                        text_replaced = True
                        break
                if not text_replaced:
                    messages[-1].content.append(TextPart(text=res.mutated_text))
        return messages

    async def run_output_pipeline(
        self,
        response: "LLMResponse | str",
        parsed_obj: Any,
        context: "RunContext | None" = None,
    ) -> tuple["LLMResponse | str", Any]:
        """执行 Output 护栏拦截、反思和变异"""
        feedbacks = []
        current_response = response
        current_obj = parsed_obj

        for g in self.guardrails:
            res = await g.validate_output(current_response, current_obj, context)
            if res.action == GuardrailAction.REJECT:
                raise GuardrailFatalException(
                    guard_name=g.__class__.__name__, reason=res.feedback or "输出被拒绝"
                )
            elif res.action == GuardrailAction.MUTATE:
                if res.mutated_text is not None:
                    if isinstance(current_response, str):
                        current_response = res.mutated_text
                    else:
                        text_found = False
                        for p in current_response.content_parts:
                            if isinstance(p, TextPart):
                                p.text = res.mutated_text
                                text_found = True
                                break
                        if not text_found:
                            current_response.content_parts.insert(
                                0, TextPart(text=res.mutated_text)
                            )
                if res.mutated_obj is not None:
                    current_obj = res.mutated_obj
            elif res.action == GuardrailAction.REFLECT:
                feedbacks.append(res.feedback or f"{g.__class__.__name__} 校验未通过")

        if feedbacks:
            raise GuardrailViolationError("\n".join(feedbacks))

        return current_response, current_obj
