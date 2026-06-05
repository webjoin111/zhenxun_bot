from abc import ABC, abstractmethod
from collections.abc import Callable
import inspect
from typing import Any

from nonebot.utils import is_coroutine_callable
from pydantic import BaseModel, Field

from zhenxun.services.ai.run.context import RunContext


class GuardrailResult(BaseModel):
    """护栏验证结果的统一载体"""

    success: bool
    """验证是否通过"""

    feedback: str | None = None
    """未通过时的失败反馈原因"""


class BaseGuardrail(ABC):
    """大一统的业务逻辑护栏抽象基类"""

    @abstractmethod
    async def validate(
        self, text: str, parsed_obj: Any, context: RunContext | None = None
    ) -> GuardrailResult:
        """执行验证逻辑"""
        pass


class FunctionalGuardrail(BaseGuardrail):
    """包装普通 Python 函数的护栏"""

    def __init__(self, func: Callable[..., Any]):
        self.func = func

    async def validate(
        self, text: str, parsed_obj: Any, context: RunContext | None = None
    ) -> GuardrailResult:
        sig = inspect.signature(self.func)
        takes_ctx = len(sig.parameters) > 1

        try:
            if takes_ctx:
                res = (
                    await self.func(context, parsed_obj)
                    if is_coroutine_callable(self.func)
                    else self.func(context, parsed_obj)
                )
            else:
                res = (
                    await self.func(parsed_obj)
                    if is_coroutine_callable(self.func)
                    else self.func(parsed_obj)
                )

            if res is False:
                return GuardrailResult(
                    success=False,
                    feedback=f"自定义护栏函数 '{self.func.__name__}' 校验未通过",
                )
            elif isinstance(res, str):
                return GuardrailResult(success=False, feedback=res)

            return GuardrailResult(success=True)
        except (ValueError, AssertionError) as e:
            return GuardrailResult(success=False, feedback=str(e))


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

    async def validate(
        self, text: str, parsed_obj: Any, context: RunContext | None = None
    ) -> GuardrailResult:
        if not self.rules:
            return GuardrailResult(success=True)

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
                return GuardrailResult(success=True)
            feedbacks = [
                f"违反规则: 【{v.rule}】, 原因: {v.reason}" for v in res.violations
            ]
            return GuardrailResult(success=False, feedback="\n".join(feedbacks))
        except Exception as e:
            return GuardrailResult(success=False, feedback=f"系统护栏裁判模型异常: {e}")


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
