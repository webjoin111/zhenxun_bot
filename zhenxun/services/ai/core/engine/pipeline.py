from typing import Any

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.engine.token_estimator import global_estimator
from zhenxun.services.ai.core.messages import (
    LLMMessage,
    TextPart,
)
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.ai.memory.compression import (
    CondenserPipeline,
    CondenserRegistry,
    MessageDropper,
    ToolOutputCompactor,
)
from zhenxun.services.ai.memory.interfaces import (
    BaseMemoryReducer,
)
from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy


class DialoguePipeline:
    """
    统一的对话上下文装配流水线。
    负责：消息净化、长期记忆(RAG)注入、短期记忆提取、Token估算与动态压缩。
    """

    def __init__(
        self,
        model_name: str,
        session_metadata: SessionMetadata,
        memory_facade: Any,
    ):
        self.model_name = model_name
        self.session_metadata = session_metadata
        self.memory_facade = memory_facade

        self.working_memory = memory_facade.working_memory if memory_facade else None
        self.long_term_memory = (
            memory_facade.long_term_memory if memory_facade else None
        )
        self.context_threshold = (
            memory_facade.context_threshold if memory_facade else None
        )
        self.max_history_turns = (
            memory_facade.max_history_turns if memory_facade else None
        )

        self.custom_reducers: list[BaseMemoryReducer] | None = None
        if memory_facade and memory_facade.memory_reducers is not None:
            self.custom_reducers = []
            for r in memory_facade.memory_reducers:
                if isinstance(r, str):
                    self.custom_reducers.append(CondenserRegistry.get(r))
                else:
                    self.custom_reducers.append(r)

    @staticmethod
    def sanitize_message_for_history(message: LLMMessage) -> LLMMessage:
        """
        净化存入历史记录的消息，将多模态媒体替换为文本占位符，避免重复处理和显存爆炸。
        """
        sanitized_message = model_copy(message)
        content_list = sanitized_message.content

        new_content_parts: list[Any] = []
        has_multimodal_content = False

        from zhenxun.services.ai.core.messages import (
            ThoughtPart,
            ToolCallPart,
            ToolReturnPart,
        )

        for part in content_list:
            if isinstance(part, (TextPart, ThoughtPart, ToolCallPart, ToolReturnPart)):
                new_content_parts.append(part)
            else:
                has_multimodal_content = True

        if has_multimodal_content:
            placeholder = "> *[系统注：用户发送了媒体文件，其视觉/听觉内容已在首轮分析时由上下文关联处理]*"
            text_part_found = False
            for i, part in enumerate(new_content_parts):
                if isinstance(part, TextPart):
                    new_content_parts[i] = TextPart(
                        text=f"{placeholder} {part.text or ''}".strip()
                    )
                    text_part_found = True
                    break
            if not text_part_found:
                new_content_parts.insert(0, TextPart(text=placeholder))

        sanitized_message.content = new_content_parts
        return sanitized_message

    async def _compress_history(
        self, history: list[LLMMessage], base_overhead: int = 0
    ) -> list[LLMMessage]:
        """检查 Token 并在超标时执行智能压缩管线"""
        config = get_llm_config().context_settings
        if not config.enabled and self.custom_reducers is None:
            return history

        threshold = (
            self.context_threshold
            if self.context_threshold is not None
            else config.trigger_threshold
        )
        max_turns = (
            self.max_history_turns
            if self.max_history_turns is not None
            else config.max_history_turns
        )
        caps = get_model_capabilities(self.model_name)
        max_input = caps.max_input_tokens

        limit = int(max_input * threshold) if threshold <= 1.0 else int(threshold)

        current_tokens = global_estimator.estimate_context(
            history, self.model_name, base_overhead=base_overhead
        )

        needs_compress = current_tokens > limit or (
            max_turns is not None and len(history) > max_turns
        )

        if not needs_compress:
            return history

        if needs_compress and current_tokens <= limit:
            limit = 0

        if self.custom_reducers is not None:
            reducers = self.custom_reducers
        else:
            reducers: list[BaseMemoryReducer] = [ToolOutputCompactor()]
            if config.enable_summarization:
                if getattr(config, "use_structured_summarizer", False):
                    reducers.append(CondenserRegistry.get("structured_summarizer"))
                else:
                    reducers.append(CondenserRegistry.get("llm_summarizer"))
            reducers.append(MessageDropper())

        pipeline = CondenserPipeline(reducers)

        logger.warning(
            f"⚠️ [上下文管线] 会话 {self.session_metadata.session_id} 触发自动压缩。"
            f"当前Tokens: {current_tokens}, 限制: {limit}, 条数: {len(history)}/{max_turns}"
        )
        new_history = await pipeline.run(
            history,
            target_tokens=limit,
            model_name=self.model_name,
            base_overhead=base_overhead,
        )

        return new_history

    async def build_messages(
        self,
        user_input: Any | None,
        system_instruction: str | None = None,
        message_buffer: list[LLMMessage] | None = None,
        base_overhead: int = 0,
        run_context: Any | None = None,
    ) -> list[LLMMessage]:
        """
        全量装配工作流：
        1. 提取系统提示词
        2. 获取历史记录并执行压缩
        3. 处理用户当次输入
        4. 执行 RAG 长记忆注入
        5. 拼装为最终可以发送给无状态 LLM 的数组
        """
        messages_for_run: list[LLMMessage] = []

        current_history: list[LLMMessage] = []
        if self.working_memory:
            current_history = await self.working_memory.get_history(
                self.session_metadata
            )

        current_history = await self._compress_history(current_history, base_overhead)

        normalized_user_msg = None
        if user_input:
            from zhenxun.services.ai.message_builder import MessageBuilder

            bot_inst = run_context.get_bot() if run_context else None
            event_inst = run_context.get_event() if run_context else None

            msgs = await MessageBuilder.normalize_to_llm_messages(user_input, bot=bot_inst, event=event_inst)
            if msgs:
                normalized_user_msg = msgs[-1]

        sys_instruction = system_instruction or ""
        if self.long_term_memory and normalized_user_msg:
            content_for_recall = normalized_user_msg.extract_text

            matches = await self.long_term_memory.recall(
                content_for_recall, inner_scope=self.session_metadata.scope_prefix
            )
            if matches:
                fact_str = "\n".join(
                    f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
                )
                memory_injection = f"### 🧠 [提取的长期记忆事实]\n以下是关于用户的历史事实或偏好设定，请作为回答的重要参考：\n{fact_str}"
                sys_instruction = (
                    f"{sys_instruction}\n\n{memory_injection}"
                    if sys_instruction
                    else memory_injection
                )

        if sys_instruction:
            messages_for_run.append(LLMMessage.system(sys_instruction))

        messages_for_run.extend(current_history)

        if message_buffer:
            messages_for_run.extend(message_buffer)

        if normalized_user_msg:
            messages_for_run.append(normalized_user_msg)

        return messages_for_run
