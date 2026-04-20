from collections.abc import Callable
import json
from typing import Any, cast

from pydantic import BaseModel

from zhenxun.services.ai.engine.pipeline import DialoguePipeline
from zhenxun.services.ai.llm.config.generation import LLMGenerationConfig
from zhenxun.services.ai.llm.manager import get_model_instance
from zhenxun.services.ai.llm.utils import render_prompt_template
from zhenxun.services.ai.memory.scope import MemoryScope
from zhenxun.services.ai.protocols.memory import (
    BaseWorkingMemory,
    SessionMetadata,
)
from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.tools import RunContext
from zhenxun.services.ai.tools.providers.context_resource import (
    context_resource_manager,
)
from zhenxun.services.ai.types.agent import AgentRunResult, ExecutionConfig
from zhenxun.services.ai.types.messages import (
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    ToolMessage,
    UsageInfo,
)
from zhenxun.services.ai.types.tools import (
    GlobalToolFilter,
    ResolvedToolPayload,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy

from .executor import AgentExecutor, AgentExecutorConfig


class Agent:
    """一个高级Agent的封装，持有配置并管理其生命周期内的资源。"""

    def __init__(
        self,
        name: str,
        instruction: str = "",
        model: str | Callable[[], str] | None = None,
        tools: list | None = None,
        knowledge: list | Any | None = None,
        skills: list[str] | None = None,
        available_skills: list[str] | None = None,
        namespace: str | None = None,
        resources: list[str] | None = None,
        prompts: list[str] | None = None,
        generation_config: LLMGenerationConfig | Any | None = None,
        response_model: type[BaseModel] | None = None,
        memory_reducers: list | None = None,
        context_threshold: float | None = None,
        max_history_turns: int | None = None,
        system_prompts: list | None = None,
        result_validators: list | None = None,
        handoffs: list | None = None,
        stateless: bool = True,
    ):
        self.name = name

        self.instruction = instruction
        self.model_name = model

        _tools = tools or []
        if knowledge:
            if not isinstance(knowledge, list):
                knowledge = [knowledge]
            _tools.extend(knowledge)

        self.tool_definitions = _tools
        self.namespace = namespace

        if self.namespace is None:
            from zhenxun.utils.utils import infer_plugin_namespace

            self.namespace = infer_plugin_namespace()

        if self.namespace is None:
            self.namespace = "unknown"

        self.tool_names = [t for t in (tools or []) if isinstance(t, str)]
        self.skills = skills or []
        self.available_skills = available_skills or []
        self.resources = resources or []
        self.response_model = response_model
        self.prompts = prompts or []
        base_config = (
            model_copy(generation_config, deep=True)
            if generation_config
            else LLMGenerationConfig()
        )
        self.default_config = base_config
        self._resolved_tools: dict[str, Any] | None = None
        self.memory_reducers = memory_reducers
        self.context_threshold = context_threshold
        self.max_history_turns = max_history_turns
        self.handoffs = handoffs or []

        self.system_prompts = system_prompts or []
        self.result_validators = result_validators or []
        self.stateless = stateless

    async def __resolve_to_tools__(self) -> list[ToolExecutable]:
        """协议支持：将自身 Agent 转化为可被上级调用的工具"""
        from zhenxun.services.ai.tools.bridges.agent import AgentTool

        return [AgentTool(self)]

    async def _resolve_system_prompt(self, run_context: RunContext) -> str:
        """解析系统提示词，结合动态函数、Jinja2模板与资源管理器"""
        import inspect

        from nonebot.utils import is_coroutine_callable

        dynamic_instructions = []
        if self.instruction:
            dynamic_instructions.append(self.instruction)

        for sp_func in self.system_prompts:
            sig = inspect.signature(sp_func)
            takes_ctx = len(sig.parameters) > 0
            if takes_ctx:
                res = (
                    (await sp_func(run_context))
                    if is_coroutine_callable(sp_func)
                    else sp_func(run_context)
                )
            else:
                res = (await sp_func()) if is_coroutine_callable(sp_func) else sp_func()
            if res:
                dynamic_instructions.append(str(res))

        if self.response_model:
            dynamic_instructions.append(
                "### ⚠️ [核心任务：结构化输出要求]\n"
                "当前任务处于严格的 **结构化输出模式**。请遵循以下工作流：\n"
                "1. 执行所有必要的调查、思考和工具调用。\n"
                "2. 任务完成后，**必须且只能**调用 `submit_final_result` 工具来提交结果。\n"
                "3. **禁止**以纯文本形式直接回答用户，必须通过工具完成闭环。"
            )

        final_instruction_text = "\n\n".join(dynamic_instructions)

        if self.skills:
            from zhenxun.services.ai.tools.providers.skills.manager import skill_manager

            skill_parts = []
            for skill_name in self.skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill:
                    skill_parts.append(
                        f"## Skill: {skill.name}\n\n{skill.instructions}"
                    )
                else:
                    logger.warning(
                        f"Agent '{self.name}' 请求挂载的技能 "
                        f"'{skill_name}' 不存在，已跳过。"
                    )

            if skill_parts:
                final_instruction_text += (
                    "\n\n--- 挂载的专用技能手册 ---\n\n" + "\n\n".join(skill_parts)
                )

        if self.available_skills:
            from zhenxun.services.ai.tools.providers.skills.manager import skill_manager

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
                final_instruction_text += (
                    f"\n\n--- 可选技能库 ---\n\n{instruction}\n{catalog_xml}"
                )

        render_context = {
            "deps": run_context.deps,
            "bot": run_context.bot,
            "event": run_context.event,
            "matcher": run_context.matcher,
        }
        final_instruction = render_prompt_template(
            final_instruction_text, render_context
        )

        context_parts = []
        if self.prompts:
            context_parts.append("--- Applied Prompts ---")
            for prompt_name in self.prompts:
                content = await context_resource_manager.fetch_prompt(prompt_name)
                context_parts.append(
                    f"[Prompt {prompt_name}]\n{content}"
                    if content
                    else f"[Prompt {prompt_name}]: Not Found"
                )
        if self.resources:
            context_parts.append("--- Attached Resources ---")
            for resource_uri in self.resources:
                content = await context_resource_manager.fetch_resource(resource_uri)
                context_parts.append(
                    f"[Resource {resource_uri}]\n{content}"
                    if content
                    else f"[Resource {resource_uri}]: Not Found"
                )

        if context_parts:
            final_instruction += "\n\n" + "\n".join(context_parts)

        return final_instruction

    async def _resolve_tools(
        self,
        tool_filter: GlobalToolFilter | None,
        run_context: RunContext | None = None,
    ) -> ResolvedToolPayload:
        """解析并过滤工具集"""
        from zhenxun.services.ai.tools.engine.registry import tool_provider_manager

        defs_to_resolve = list(self.tool_definitions or [])

        from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
        from zhenxun.services.ai.tools.providers.skills.toolkit import (
            SkillMetaToolkit,
            SkillStaticToolkit,
        )

        if getattr(self, "skills", None):
            for skill_name in self.skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill and skill.scripts:
                    defs_to_resolve.append(SkillStaticToolkit(skill))

        if getattr(self, "available_skills", None):
            defs_to_resolve.append(SkillMetaToolkit())

        payload = await tool_provider_manager.resolve_tools(
            defs_to_resolve, self.namespace, context=run_context
        )

        if getattr(self, "handoffs", None):
            from .utils import HandoffExecutable

            for target_agent in self.handoffs:
                handoff_tool = HandoffExecutable(target_agent)
                payload.tools[handoff_tool.tool_name] = handoff_tool

        return payload

    def _get_validator_callback(self, run_context: RunContext):
        """统一包装结果验证器，供结构化与非结构化流程复用"""
        if not self.result_validators:
            return None

        import inspect

        from nonebot.utils import is_coroutine_callable

        async def _val_wrapper(obj: Any) -> Any:
            curr = obj
            for v in self.result_validators:
                sig = inspect.signature(v)
                takes_ctx = len(sig.parameters) > 1
                if takes_ctx:
                    res = (
                        await v(run_context, curr)
                        if is_coroutine_callable(v)
                        else v(run_context, curr)
                    )
                else:
                    res = await v(curr) if is_coroutine_callable(v) else v(curr)
                if res is not None:
                    curr = res
            return curr

        return _val_wrapper

    async def run(
        self,
        prompt: str | None = None,
        *,
        deps: Any = None,
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: ExecutionConfig | None = None,
        working_memory: BaseWorkingMemory | None = None,
        long_term_memory: MemoryScope | None = None,
        generation_config: LLMGenerationConfig | None = None,
        injected_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """
        智能体运行入口。
        内置循环以处理多智能体热切换（Handoff），将接力棒传递给下一个智能体。
        """
        from zhenxun.services.ai.agent.session import async_memory_condenser

        current_agent = self
        current_prompt = prompt
        visited_agents = {current_agent.name}
        accumulated_usage = UsageInfo()
        current_injected_state = injected_state or {}

        while True:
            result = await current_agent._run_step(
                prompt=current_prompt,
                deps=deps,
                message_history=message_history,
                tool_filter=tool_filter,
                config=config,
                working_memory=working_memory,
                long_term_memory=long_term_memory,
                generation_config=generation_config,
                injected_state=current_injected_state,
                session_id=session_id,
                cancellation_token=cancellation_token,
                **kwargs,
            )

            if result.usage:
                accumulated_usage.prompt_tokens += result.usage.prompt_tokens
                accumulated_usage.completion_tokens += result.usage.completion_tokens
                accumulated_usage.total_tokens += result.usage.total_tokens

            if not result.handoff_target:
                result.usage = accumulated_usage
                if session_id:
                    async_memory_condenser.trigger_compression(session_id)
                return result

            target_name = result.handoff_target
            if target_name in visited_agents:
                logger.warning(
                    f"⚠️ [Handoff] 检测到死循环移交 ({target_name})，强制中断。"
                )
                result.output = f"❌ 移交失败：检测到死循环移交 ({target_name})"
                result.handoff_target = None
                return result

            target_agent = next(
                (a for a in current_agent.handoffs if a.name == target_name), None
            )
            if not target_agent:
                result.output = (
                    f"❌ 移交失败：目标 Agent '{target_name}' "
                    f"未在 {current_agent.name} 的 handoffs 中声明。"
                )
                result.handoff_target = None
                return result

            visited_agents.add(target_name)
            current_agent = target_agent
            current_prompt = None

            if result.handoff_payload:
                current_injected_state = {"handoff_payload": result.handoff_payload}

    async def _run_step(
        self,
        prompt: str | None = None,
        *,
        deps: Any = None,
        message_history: list[LLMMessage] | None = None,
        tool_filter: GlobalToolFilter | None = None,
        config: ExecutionConfig | None = None,
        working_memory: BaseWorkingMemory | None = None,
        long_term_memory: MemoryScope | None = None,
        generation_config: LLMGenerationConfig | None = None,
        injected_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        """执行原子步代理逻辑"""
        import uuid

        bot = kwargs.get("bot")
        event = kwargs.get("event")
        matcher = kwargs.get("matcher")
        if isinstance(deps, dict):
            bot = bot or deps.get("bot")
            event = event or deps.get("event")
            matcher = matcher or deps.get("matcher")

        extra_data = {"user_input": prompt or ""}
        if injected_state:
            extra_data.update(injected_state)

        ai_session_id = f"ag-run-{uuid.uuid4()}"
        run_context = RunContext(  # type: ignore
            session_id=session_id or ai_session_id,
            bot=bot,
            event=event,
            matcher=matcher,
            deps=deps,
            extra=extra_data,
            cancellation_token=cancellation_token,
        )

        system_prompt = await self._resolve_system_prompt(run_context)
        tool_payload = await self._resolve_tools(tool_filter, run_context)
        effective_tools = tool_payload.tools

        if long_term_memory and prompt:
            matches = await long_term_memory.recall(prompt)
            if matches:
                fact_str = "\n".join(
                    f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
                )
                system_prompt += f"\n\n[系统补充：有关用户的长期记忆设定]\n{fact_str}"

        if tool_payload.injected_prompts:
            system_prompt += "\n\n--- 工具箱专属使用说明 ---\n\n"
            system_prompt += "\n\n".join(tool_payload.injected_prompts)

        final_gen_config = model_copy(self.default_config, deep=True)
        if generation_config:
            final_gen_config = final_gen_config.merge_with(generation_config)
        exec_config = config or ExecutionConfig()
        model_name_resolved = (
            self.model_name() if callable(self.model_name) else self.model_name
        )

        session_metadata = SessionMetadata(session_id=session_id or ai_session_id)

        if working_memory is None:
            from zhenxun.services.ai.memory.working_memory import _get_default_memory

            working_memory = _get_default_memory()

        if message_history:
            await working_memory.set_history(session_metadata, message_history)

        try:
            val_cb = self._get_validator_callback(run_context)

            target_model = self.response_model
            effective_auto_thinking = False
            if target_model is not None:
                from zhenxun.services.ai.llm.utils import (
                    create_cot_wrapper,
                    should_apply_autocot,
                )

                effective_auto_thinking = should_apply_autocot(
                    True, str(model_name_resolved), final_gen_config
                )
                if effective_auto_thinking:
                    target_model = create_cot_wrapper(target_model)
                    system_prompts_list = [
                        "\n\n[思维链要求]\n在调用 `submit_final_result` 提交结果时，",
                        "请务必先在 `reasoning` 字段中写下详细的推理过程，",
                        "再将最终答案填入 `result` 字段。",
                    ]
                    system_prompt += "".join(system_prompts_list)

                from .utils import SubmitFinalResultExecutable

                submit_tool = SubmitFinalResultExecutable(
                    response_model=target_model,
                    val_cb=val_cb,
                    is_auto_thinking=effective_auto_thinking,
                    original_model=target_model,
                )
                effective_tools[submit_tool.tool_name] = submit_tool

            pipeline = DialoguePipeline(
                model_name=str(model_name_resolved) if model_name_resolved else "",
                session_metadata=session_metadata,
                working_memory=working_memory,
                long_term_memory=long_term_memory,
                memory_reducers=self.memory_reducers,
                context_threshold=self.context_threshold,
                max_history_turns=self.max_history_turns,
            )
            messages_for_run = await pipeline.build_messages(
                user_input=prompt, system_instruction=system_prompt, base_overhead=0
            )

            if prompt and messages_for_run and messages_for_run[-1].role == "user":
                await working_memory.add_messages(
                    session_metadata, [messages_for_run[-1]]
                )

            from nonebot.utils import is_coroutine_callable

            for tk in tool_payload.toolkits:
                if hasattr(tk, "before_llm_request"):
                    if is_coroutine_callable(tk.before_llm_request):
                        await tk.before_llm_request(run_context, messages_for_run)
                    else:
                        tk.before_llm_request(run_context, messages_for_run)

            executor = AgentExecutor(
                tools=effective_tools,
                config=AgentExecutorConfig(
                    max_cycles=exec_config.max_cycles,
                    reflexion_retries=exec_config.reflexion_retries,
                ),
            )

            from contextlib import AsyncExitStack

            async with AsyncExitStack() as stack:
                for tk in tool_payload.toolkits:
                    if hasattr(tk, "enter_session"):
                        await tk.enter_session(run_context.session_id, run_context)
                        stack.push_async_callback(
                            tk.exit_session, run_context.session_id
                        )

                async with await get_model_instance(
                    str(model_name_resolved) if model_name_resolved else None,
                    override_config=None,
                ) as instance:
                    _run_result: Any = await executor.run(
                        messages=messages_for_run,
                        model_instance=instance,
                        run_context=run_context,
                        generation_config=final_gen_config,
                        cancellation_token=cancellation_token,
                    )
                    final_messages = cast(list[LLMMessage], _run_result[0])
                    handoff_target = cast(str | None, _run_result[1])
                    handoff_args = cast(dict[str, Any] | None, _run_result[2])
                    structured_data = cast(dict[str, Any] | None, _run_result[3])
                    handoff_payload = cast(dict[str, Any] | None, _run_result[4])

            new_msgs = final_messages[len(messages_for_run) :]
            if new_msgs:
                await working_memory.add_messages(session_metadata, new_msgs)

            last_msg = final_messages[-1]
            final_text = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else " ".join(
                    p.text for p in last_msg.content if p.type == "text" and p.text
                )
            )

            total_completion = sum(
                m.token_cost or 0 for m in new_msgs if isinstance(m, AssistantMessage)
            )
            usage = UsageInfo(total_tokens=total_completion)

            final_output: Any = final_text
            if self.response_model:
                if structured_data is not None:
                    from zhenxun.utils.pydantic_compat import parse_as

                    try:
                        parsed_obj = parse_as(self.response_model, structured_data)
                        final_output = parsed_obj
                    except Exception as e:
                        logger.error(f"解析结构化输出失败: {e}", e=e)
                        final_output = (
                            f"❌ 结构化解析失败: {e}\n原始数据: {structured_data}"
                        )
                else:
                    logger.error(
                        f"Agent '{self.name}' 未能调用 submit_final_result "
                        "提交结构化数据。"
                    )
                    final_output = (
                        "❌ 模型未能输出符合要求的结构化数据。\n"
                        f"模型最后回复: {final_text}"
                    )
            else:
                if val_cb:
                    final_output = await val_cb(final_text)

            if handoff_target:
                logger.info(
                    f"🔄 [Handoff State] 触发智能体移交意图: "
                    f"{self.name} -> {handoff_target}"
                )

                history_to_pass = final_messages
                if history_to_pass and isinstance(history_to_pass[0], SystemMessage):
                    history_to_pass = history_to_pass[1:]

                cleaned_history = []
                for msg in history_to_pass:
                    if isinstance(msg, AssistantMessage) and msg.tool_calls:
                        if any(
                            call.tool_name.startswith("transfer_to_")
                            for call in msg.tool_calls
                        ):
                            continue
                    if isinstance(msg, ToolMessage):
                        returns = msg.tool_returns
                        if returns and returns[0].tool_name.startswith("transfer_to_"):
                            continue
                    cleaned_history.append(msg)

                reason = (handoff_args or {}).get("reason", "无附加原因")
                context_info = (handoff_args or {}).get("context_to_pass", "无附加信息")

                payload_notice = ""
                if handoff_payload:
                    try:
                        formatted_payload = json.dumps(
                            handoff_payload, ensure_ascii=False, indent=2
                        )
                        payload_notice = (
                            "\n- 强类型状态载荷 (Payload):\n```json\n"
                            f"{formatted_payload}\n```\n"
                        )
                    except Exception:
                        payload_notice = (
                            f"\n- 强类型状态载荷 (Payload): {handoff_payload}\n"
                        )

                handoff_notice = (
                    f"\n\n[系统内部记录：当前任务已由 '{self.name}' "
                    f"移交给 '{handoff_target}']\n"
                    f"- 移交原因: {reason}\n"
                    f"- 前序上下文: {context_info}"
                    f"{payload_notice}"
                    f"(注：请结合上述上下文，严格遵循你的系统指令决定下一步行动)"
                )
                handoff_msg = LLMMessage.user(handoff_notice)
                await working_memory.add_messages(session_metadata, [handoff_msg])
                cleaned_history.append(handoff_msg)

                return AgentRunResult(
                    **{
                        "output": final_output,
                        "messages": cleaned_history,
                        "usage": usage,
                        "handoff_target": handoff_target,
                        "handoff_args": handoff_args,
                        "handoff_payload": handoff_payload,
                    }
                )

            return AgentRunResult(
                **{
                    "output": final_output,
                    "messages": new_msgs,
                    "usage": usage,
                }
            )

        except Exception as e:
            logger.error(f"Agent '{self.name}' 运行失败: {e}", e=e)
            return AgentRunResult(
                **{
                    "output": f"❌ {e}",
                    "messages": [],
                    "usage": UsageInfo(),
                }
            )
