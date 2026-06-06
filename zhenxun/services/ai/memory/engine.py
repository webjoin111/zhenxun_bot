from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.memory.compression import (
    CondenserPipeline,
    MemoryPolicy,
    MultimodalPlaceholderReducer,
)
from zhenxun.services.ai.memory.manager import memory_manager
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.memory.types import SessionMetadata
from zhenxun.services.log import logger


class MemoryReader:
    """
    记忆读取器 (Memory Reader)。
    负责从数据库中提取短期上下文历史，召回长期的背景知识，并执行自动压缩。
    """

    def __init__(
        self, session_meta: SessionMetadata, memory_config: MemoryConfig | None
    ):
        self.session_meta = session_meta
        self.memory_config = memory_config

    async def get_long_term_context(self, user_input: str) -> str:
        """
        基于用户输入召回长期记忆（RAG），返回格式化后的背景提示词。
        """
        if (
            not self.memory_config
            or not self.memory_config.long_term.enable
            or not user_input
        ):
            return ""

        ltm_scope = memory_manager.get_long_term_memory(
            self.memory_config,
            namespace=self.session_meta.namespace or "global",
        )
        if not ltm_scope:
            return ""

        matches = await ltm_scope.recall(session=self.session_meta, query=user_input)
        if matches:
            fact_str = "\n".join(
                f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
            )
            logger.debug(f"🧠 [MemoryReader] 成功召回 {len(matches)} 条长期记忆。")
            return f"[系统补充：有关用户的长期记忆设定]\n{fact_str}"
        return ""

    async def get_slots_context(self) -> str:
        """
        读取并组装核心槽位记忆 (Memory Slots)，返回 XML 格式字符串供大模型使用。
        """
        if not self.memory_config or not self.memory_config.slots.enable:
            return ""
        slot_ctx = memory_manager.get_slot_context(
            self.memory_config,
            namespace=self.session_meta.namespace or "global",
        )
        if not slot_ctx:
            return ""

        if self.memory_config.slots.default_slots:
            for default_slot in self.memory_config.slots.default_slots:
                existing = await slot_ctx.get_slot(
                    self.session_meta, default_slot.label
                )
                if not existing:
                    await slot_ctx.set_slot(self.session_meta, default_slot)

        slots = await slot_ctx.list_pinned_slots(self.session_meta)
        if not slots:
            return ""

        xml_parts = ["<memory_slots>"]
        for slot in slots:
            xml_parts.append(
                f'  <slot name="{slot.label}" scope="{slot.scope.value}">\n'
                f"    {slot.content}\n"
                "  </slot>"
            )
        xml_parts.append("</memory_slots>")
        return "\n".join(xml_parts)

    async def get_short_term_context(
        self,
        model_name: str,
        override_history: list[LLMMessage] | None = None,
    ) -> list[LLMMessage]:
        """
        拉取短期对话历史，并执行 Token 压缩。
        """
        current_history: list[LLMMessage] = []
        if override_history is not None:
            current_history = list(override_history)

        chat_context = memory_manager.get_chat_context(
            self.memory_config,
            namespace=self.session_meta.namespace or "global",
        )

        if self.memory_config and self.memory_config.short_term.enable and chat_context:
            if override_history is not None:
                await chat_context.set_messages(self.session_meta, override_history)
            else:
                current_history = await chat_context.get_messages(self.session_meta)

            config = get_llm_config().context_settings
            pipeline_reducers = []

            vw = config.vision_window_size
            if (
                self.memory_config
                and self.memory_config.compression.vision_window is not None
            ):
                vw = self.memory_config.compression.vision_window
            if vw > 0:
                pipeline_reducers.append(MultimodalPlaceholderReducer(window_size=vw))

            policy = (
                self.memory_config.compression.policy if self.memory_config else None
            )
            if policy is not None:
                pipeline_reducers.extend(policy)
            else:
                threshold = config.llm_summary.trigger_threshold
                if (
                    self.memory_config
                    and self.memory_config.compression.threshold is not None
                ):
                    threshold = self.memory_config.compression.threshold

                from zhenxun.services.ai.llm.capabilities import get_model_capabilities

                caps = get_model_capabilities(model_name)
                limit = (
                    int(caps.max_input_tokens * threshold)
                    if threshold <= 1.0
                    else int(threshold)
                )

                max_turns = config.llm_summary.max_history_turns
                if (
                    self.memory_config
                    and self.memory_config.compression.max_history_turns is not None
                ):
                    max_turns = self.memory_config.compression.max_history_turns

                if config.llm_summary.enable:
                    pipeline_reducers.extend(
                        MemoryPolicy.llm_summarize(
                            trigger_tokens=limit,
                            max_turns=max_turns,
                            keep_recent_turns=config.llm_summary.keep_recent_turns,
                            summarization_model=config.llm_summary.summarization_model,
                            summarization_prompt=config.llm_summary.summarization_prompt,
                        )
                    )
                else:
                    pipeline_reducers.extend(MemoryPolicy.unlimited())

            if pipeline_reducers:
                pipeline = CondenserPipeline(pipeline_reducers)
                new_history, changed = await pipeline.run(
                    current_history, model_name=model_name, base_overhead=0
                )
                if changed:
                    await chat_context.set_messages(self.session_meta, new_history)
                    logger.info(
                        "💾 [MemoryReader] 压缩截断完毕，已同步覆写数据库。"
                        f"压缩后条数: {len(new_history)}"
                    )
                current_history = new_history

        return current_history


class MemoryWriter:
    """
    记忆写入器 (Memory Writer)。
    负责将对话增量安全地写入数据库。
    """

    def __init__(
        self, session_meta: SessionMetadata, memory_config: MemoryConfig | None
    ):
        self.session_meta = session_meta
        self.memory_config = memory_config

    async def save_new_messages(
        self,
        new_messages: list[LLMMessage],
    ):
        """将新产生的对话增量保存到数据库"""
        if not new_messages:
            return
        chat_ctx = memory_manager.get_chat_context(
            self.memory_config,
            namespace=self.session_meta.namespace or "global",
        )
        if chat_ctx and self.memory_config and self.memory_config.short_term.enable:
            await chat_ctx.add_messages(self.session_meta, new_messages)
