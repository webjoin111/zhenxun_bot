from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import GenerationConfig
    from zhenxun.services.ai.core.messages import LLMResponse
    from zhenxun.services.ai.flow.agent.models import CapabilitySpec
    from zhenxun.services.ai.protocols.middleware import LLMContext
    from zhenxun.services.ai.run import AgentRunResult, RunContext

WrapRunHandler = Callable[[], Awaitable["AgentRunResult[Any]"]]
WrapModelRequestHandler = Callable[["LLMContext"], Awaitable["LLMResponse"]]
WrapToolValidateHandler = Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]
WrapToolExecuteHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class AbstractCapability:
    """
    Agent 能力组件基类协议。
    所有业务逻辑拦截（限流、权限、动态 Prompt）请在此实现。
    底层网络重试、并发控制等请勿在此处理。
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """用于 YAML/JSON 反序列化的注册标识符"""
        return cls.__name__

    @classmethod
    def from_spec(cls, **kwargs) -> "AbstractCapability":
        """从 Spec 的 kwargs 中实例化对象"""
        return cls(**kwargs)

    def __init_subclass__(cls, **kwargs):
        """自动将继承此类的所有拦截器注册到中心表"""
        super().__init_subclass__(**kwargs)
        CapabilityRegistry.register(cls)

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        """获取专用于单次运行的实例。
        默认返回自身(无状态)。若需要记录单次运行的上下文状态，请返回深/浅拷贝(如 return copy.copy(self))。
        """
        return self

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        """运行开始前触发。允许动态下发大模型配置（覆盖或合并 Agent 的默认配置）。"""
        return None

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        return []

    async def get_tools(self, context: RunContext) -> list[Any]:
        return []

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        """运行开始前/装配工具时触发。允许动态增删改当前将发往大模型的工具列表。
        默认实现：无操作，直接返回传入的工具列表。"""
        return tool_defs

    async def before_run(self, context: RunContext) -> None:
        """运行开始前触发。仅用于观察或初始化状态。"""
        pass

    async def after_run(
        self, context: RunContext, result: "AgentRunResult[Any]"
    ) -> "AgentRunResult[Any]":
        """运行成功结束后触发。可修改最终的运行结果。"""
        return result

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        """包裹整个 Agent 运行过程 (洋葱模型)。"""
        return await handler()

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> "AgentRunResult[Any]":
        """运行发生致命异常时触发。若不处理，必须重新抛出 error。可返回 AgentRunResult 实现自愈。"""
        raise error

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        """大模型发起请求前触发。可动态修改 Prompt、工具列表或生成配置。"""
        return llm_context

    async def after_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        response: LLMResponse,
    ) -> LLMResponse:
        """大模型成功返回后触发。可修改或验证大模型的原始返回对象。"""
        return response

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        """包裹单次大模型 API 请求 (洋葱模型)。"""
        return await handler(llm_context)

    async def on_model_request_error(
        self, context: RunContext, llm_context: LLMContext, error: Exception
    ) -> LLMResponse:
        """大模型请求失败（如网络超时）时触发。可调用备用模型实现故障转移，若不处理需抛出 error。"""
        raise error

    async def before_tool_validate(
        self, context: RunContext, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        """工具参数校验前触发。可清洗、修改原始参数字符串或字典。"""
        return args

    async def after_tool_validate(
        self, context: RunContext, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """工具参数校验通过后触发。接收的是反序列化后的标准字典。"""
        return args

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        """包裹工具的参数校验过程 (洋葱模型)。"""
        return await handler(args)

    async def on_tool_validate_error(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        """参数校验失败（如 Schema 不匹配）时触发。可用于交互式参数补全或自愈。"""
        raise error

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """工具实际执行前触发。可校验或篡改传入参数。"""
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        """工具成功执行后触发。可加工或过滤工具的输出结果。"""
        return result

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """包裹单一工具的执行 (洋葱模型)。"""
        return await handler(arguments)

    async def on_tool_execute_error(
        self, context: RunContext, tool_name: str, error: Exception
    ) -> Any:
        """工具执行发生异常时触发。可返回特定提示信息引导大模型自我反思 (Reflexion)，若不处理需抛出 error。"""
        raise error


class CapabilityRegistry:
    """Capability 序列化注册表"""

    _registry: dict[str, type[AbstractCapability]] = {}

    @classmethod
    def register(cls, cap_cls: type[AbstractCapability]):
        name = cap_cls.get_serialization_name()
        if name:
            cls._registry[name] = cap_cls

    @classmethod
    def get(cls, name: str) -> type[AbstractCapability] | None:
        return cls._registry.get(name)

    @classmethod
    def create_from_spec(cls, spec: "CapabilitySpec") -> AbstractCapability:
        cap_cls = cls.get(spec.name)
        if not cap_cls:
            raise ValueError(f"未知的 Capability 插件标识符: {spec.name}")
        extra_kwargs = cast(dict[str, Any], spec.model_extra or {})
        return cap_cls.from_spec(**extra_kwargs)


class ReflexionCapability(AbstractCapability):
    """自愈反思与验证引擎 (Reflexion Engine)。统一处理结构化解析失败和语义护栏拦截。"""

    async def on_tool_execute_error(self, context, tool_name, error):
        from zhenxun.services.ai.core.engine.structured_parser import (
            DEFAULT_IVR_TEMPLATE,
        )
        from zhenxun.services.ai.core.exceptions import ModelRetry, ToolRetryError
        from zhenxun.services.ai.tools.models import ToolResult

        if isinstance(error, (ToolRetryError, ModelRetry)):
            error_msg = getattr(error, "message", str(error))
            feedback_prompt = DEFAULT_IVR_TEMPLATE.format(error_msg=error_msg)
            context.run.add_system_prompt(feedback_prompt)
            return ToolResult(
                output=f"执行失败：{error_msg}",
            ).as_error()
        raise error

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        output_processor = llm_context.extra.get("output_processor")
        guardrails = llm_context.extra.get("guardrails", [])

        if not output_processor and not guardrails:
            return await handler(llm_context)

        max_retries = llm_context.extra.get("max_retries", 3)
        error_template = (
            output_processor.error_template if output_processor else "{error_msg}"
        )

        ivr_messages = list(llm_context.messages)
        last_exception: Exception | None = None

        for attempt in range(max_retries + 1):
            llm_context.messages = list(ivr_messages)
            current_response_text: str = ""

            try:
                response = await handler(llm_context)
                current_response_text = response.text

                if response.tool_calls:
                    return response

                if output_processor:
                    final_obj = await output_processor.validate_and_parse(
                        current_response_text, context=context
                    )
                else:
                    final_obj = current_response_text

                failed_feedbacks = []
                for v in guardrails:
                    v_res = await v.validate(current_response_text, final_obj, context)
                    if not v_res.success:
                        failed_feedbacks.append(v_res.feedback or "未知校验失败")

                if failed_feedbacks:
                    from zhenxun.services.ai.core.exceptions import (
                        GuardrailViolationError,
                    )

                    raise GuardrailViolationError("\n".join(failed_feedbacks))
                response.parsed_obj = final_obj
                return response

            except Exception as e:
                from typing import cast

                from zhenxun.services.ai.core.exceptions import (
                    LLMErrorCode,
                    LLMException,
                    ModelRetry,
                )
                from zhenxun.services.ai.core.messages import LLMMessage

                is_model_retry = isinstance(e, ModelRetry)
                is_llm_error = isinstance(e, LLMException)
                llm_error: LLMException | None = (
                    cast(LLMException, e) if is_llm_error else None
                )
                last_exception = e

                if (
                    not is_model_retry
                    and llm_error
                    and llm_error.code
                    not in (
                        LLMErrorCode.RESPONSE_PARSE_ERROR,
                        LLMErrorCode.API_RESPONSE_INVALID,
                    )
                ):
                    raise e

                if attempt < max_retries:
                    if is_model_retry:
                        error_msg = getattr(e, "message", str(e))
                        raw_response = current_response_text
                    else:
                        error_msg = (
                            llm_error.details.get("validation_error", str(e))
                            if llm_error
                            else str(e)
                        )
                        raw_response = current_response_text or (
                            llm_error.details.get("raw_response", "")
                            if llm_error
                            else ""
                        )

                    from zhenxun.services.log import logger

                    logger.warning(
                        f"输出校验未通过 (尝试 {attempt + 1}/{max_retries + 1})。启动反思修复闭环... 失败原因: {error_msg}"
                    )

                    if raw_response:
                        ivr_messages.append(
                            cast(
                                LLMMessage,
                                LLMMessage.assistant_text_response(raw_response),
                            )
                        )
                    from zhenxun.services.ai.core.exceptions import (
                        GuardrailViolationError,
                        SchemaParseError,
                    )

                    if isinstance(e, SchemaParseError):
                        feedback_prompt = (
                            "### ❌ [格式解析失败]\n"
                            "你输出的结构化数据（JSON）格式损坏或字段不匹配，未能通过 Schema 校验。\n\n"
                            "**解析错误报告：**\n"
                            f"> {error_msg}\n\n"
                            "**修正要求：** 请仔细检查缺失的必填字段、错误的数据类型或未闭合的括号，严格参考你可用的工具 Schema 定义，重新输出正确格式的数据。"
                        )
                    elif isinstance(e, GuardrailViolationError):
                        feedback_prompt = (
                            "### 🛡️ [业务护栏违规]\n"
                            "你输出的数据格式完全正确，但在业务逻辑层触发了合规/风控护栏。\n\n"
                            "**拦截原因报告：**\n"
                            f"> {error_msg}\n\n"
                            "**修正要求：** 请结合上述反馈报告，反思你的决策逻辑或内容生成，在保持数据格式正确的前提下，重新生成符合护栏规范的内容。"
                        )
                    else:
                        if output_processor and error_template:
                            feedback_prompt = error_template.format(error_msg=error_msg)
                        else:
                            from zhenxun.services.ai.core.engine.structured_parser import (
                                DEFAULT_IVR_TEMPLATE,
                            )

                            feedback_prompt = DEFAULT_IVR_TEMPLATE.format(
                                error_msg=error_msg
                            )
                    ivr_messages.append(
                        cast(LLMMessage, LLMMessage.user(feedback_prompt))
                    )
                    continue

                if llm_error and not getattr(llm_error, "recoverable", True):
                    raise llm_error

        from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException

        if last_exception:
            raise last_exception
        raise LLMException(
            "反思循环耗尽，未能生成符合所有校验规则的合法结果。",
            code=LLMErrorCode.GENERATION_FAILED,
        )


class HitlCapability(AbstractCapability):
    """人机协同 (Human-in-the-Loop) 能力组件"""

    async def get_tools(self, context: RunContext) -> list[Any]:
        from zhenxun.services.ai.tools.providers.builtin.hitl import HITLToolkit

        return [HITLToolkit()]


class SkillCapability(AbstractCapability):
    """技能库挂载能力组件"""

    def __init__(self):
        self.skills: list[str] = []
        self.available_skills: list[str] = []

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        prompts = []
        from zhenxun.services.ai.tools.providers.skills.manager import skill_manager

        if self.skills:
            skill_parts = []
            for skill_name in self.skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill:
                    skill_parts.append(
                        f"## Skill: {skill.name}\n\n{skill.instructions}"
                    )
                else:
                    logger.warning(
                        f"SkillCapability 请求挂载的技能 '{skill_name}' 不存在，已跳过。"
                    )

            if skill_parts:
                prompts.append(
                    "\n\n--- 挂载的专用技能手册 ---\n\n" + "\n\n".join(skill_parts)
                )

        if self.available_skills:
            catalog_parts = []
            for skill_name in self.available_skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill:
                    catalog_parts.append(
                        f"  <skill>\n    <name>{skill.id}</name>\n"
                        f"    <description>{skill.description}</description>\n  </skill>"
                    )

            if catalog_parts:
                catalog_xml = (
                    "<available_skills>\n"
                    + "\n".join(catalog_parts)
                    + "\n</available_skills>"
                )
                instruction = (
                    "### 🛠️ [外部技能调用规范]\n"
                    "以下是系统外挂技能目录。**禁止**臆造参数或直接推测调用。\n"
                    "**标准操作程序 (SOP)：**\n"
                    "1. **查阅指南**：必须首先调用 `read_skill_instructions` 获取该技能的详细指南。\n"
                    "2. **精准执行**：阅读指南后，严格按照规范使用 `run_skill_script` 执行。\n"
                    "3. **严禁盲测**：严禁在未读取指南的情况下尝试猜测参数。"
                )
                prompts.append(
                    f"\n\n--- 可选技能库 ---\n\n{instruction}\n{catalog_xml}"
                )

        return prompts

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []
        from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
        from zhenxun.services.ai.tools.providers.skills.toolkit import (
            SkillMetaToolkit,
            SkillStaticToolkit,
        )

        if self.skills:
            for skill_name in self.skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill and skill.scripts:
                    tools.append(SkillStaticToolkit(skill))

        if self.available_skills:
            tools.append(SkillMetaToolkit())

        return tools


class CombinedCapability(AbstractCapability):
    """
    组合能力容器。
    将多个 Capability 按顺序融合成一个复合的洋葱模型，处理生命周期的正序/倒序和链式调用。
    """

    def __init__(self, capabilities: list[AbstractCapability]):
        deduped = []
        seen = set()
        for c in capabilities:
            if id(c) not in seen:
                seen.add(id(c))
                deduped.append(c)
        self.capabilities = deduped

    async def for_run(self, context: RunContext) -> "AbstractCapability":
        new_caps = []
        changed = False
        for cap in self.capabilities:
            new_cap = await cap.for_run(context)
            new_caps.append(new_cap)
            if new_cap is not cap:
                changed = True

        if changed:
            return CombinedCapability(new_caps)
        return self

    async def get_generation_config(
        self, context: RunContext
    ) -> "GenerationConfig | None":
        final_config = None
        for cap in self.capabilities:
            cap_config = await cap.get_generation_config(context)
            if cap_config:
                if final_config is None:
                    final_config = cap_config
                else:
                    final_config = final_config.merge_with(cap_config)
        return final_config

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        prompts = []
        for cap in self.capabilities:
            prompts.extend(await cap.get_system_prompts(context))
        return prompts

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []
        for cap in self.capabilities:
            tools.extend(await cap.get_tools(context))
        return tools

    async def prepare_tools(
        self, context: RunContext, tool_defs: list[Any]
    ) -> list[Any]:
        current_defs = list(tool_defs)
        for cap in self.capabilities:
            res = await cap.prepare_tools(context, current_defs)
            if res is not None:
                current_defs = res
        return current_defs

    async def before_run(self, context: RunContext) -> None:
        for cap in self.capabilities:
            await cap.before_run(context)

    async def after_run(
        self, context: RunContext, result: "AgentRunResult[Any]"
    ) -> "AgentRunResult[Any]":
        for cap in reversed(self.capabilities):
            result = await cap.after_run(context, result)
        return result

    async def wrap_run(
        self, context: RunContext, handler: WrapRunHandler
    ) -> "AgentRunResult[Any]":
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped():
                    return await c.wrap_run(context, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain()

    async def on_run_error(
        self, context: RunContext, error: BaseException
    ) -> "AgentRunResult[Any]":
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_run_error(context, error)
            except BaseException as new_error:
                error = new_error
        raise error

    async def before_model_request(
        self, context: RunContext, llm_context: LLMContext
    ) -> LLMContext:
        for cap in self.capabilities:
            llm_context = await cap.before_model_request(context, llm_context)
        return llm_context

    async def after_model_request(
        self, context: RunContext, llm_context: LLMContext, response: LLMResponse
    ) -> LLMResponse:
        for cap in reversed(self.capabilities):
            response = await cap.after_model_request(context, llm_context, response)
        return response

    async def wrap_model_request(
        self,
        context: RunContext,
        llm_context: LLMContext,
        handler: WrapModelRequestHandler,
    ) -> LLMResponse:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(ctx_inner):
                    return await c.wrap_model_request(context, ctx_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(llm_context)

    async def on_model_request_error(
        self, context: RunContext, llm_context: LLMContext, error: Exception
    ) -> LLMResponse:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_model_request_error(context, llm_context, error)
            except Exception as new_error:
                error = new_error
        raise error

    async def before_tool_validate(
        self, context: RunContext, tool_name: str, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        for cap in self.capabilities:
            args = await cap.before_tool_validate(context, tool_name, args)
        return args

    async def after_tool_validate(
        self, context: RunContext, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        for cap in reversed(self.capabilities):
            args = await cap.after_tool_validate(context, tool_name, args)
        return args

    async def wrap_tool_validate(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        handler: WrapToolValidateHandler,
    ) -> dict[str, Any]:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(args_inner):
                    return await c.wrap_tool_validate(context, tool_name, args_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(args)

    async def on_tool_validate_error(
        self,
        context: RunContext,
        tool_name: str,
        args: str | dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_tool_validate_error(context, tool_name, args, error)
            except Exception as new_error:
                error = new_error
        raise error

    async def before_tool_execute(
        self, context: RunContext, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        for cap in self.capabilities:
            arguments = await cap.before_tool_execute(context, tool_name, arguments)
        return arguments

    async def after_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> Any:
        for cap in reversed(self.capabilities):
            result = await cap.after_tool_execute(context, tool_name, arguments, result)
        return result

    async def wrap_tool_execute(
        self,
        context: RunContext,
        tool_name: str,
        arguments: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        chain = handler
        for cap in reversed(self.capabilities):

            def _wrap(c, h):
                async def _wrapped(args_inner):
                    return await c.wrap_tool_execute(context, tool_name, args_inner, h)

                return _wrapped

            chain = _wrap(cap, chain)
        return await chain(arguments)

    async def on_tool_execute_error(
        self, context: RunContext, tool_name: str, error: Exception
    ) -> Any:
        for cap in reversed(self.capabilities):
            try:
                return await cap.on_tool_execute_error(context, tool_name, error)
            except Exception as new_error:
                error = new_error
        raise error
