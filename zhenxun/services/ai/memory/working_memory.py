from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel, Field

from zhenxun.services.ai.engine.token_estimator import global_estimator
from zhenxun.services.ai.protocols.memory import (
    BaseMemoryReducer,
    BaseMessageStore,
    BaseWorkingMemory,
    SessionMetadata,
)
from zhenxun.services.ai.types.messages import (
    LLMMessage,
    SystemMessage,
    ToolMessage,
)
from zhenxun.services.log import logger


class InMemoryMessageStore(BaseMessageStore):
    def __init__(self):
        self._messages: dict[str, list[LLMMessage]] = {}

    async def get_messages(self, session: SessionMetadata) -> list[LLMMessage]:
        return self._messages.get(session.session_id, [])

    async def search(
        self, query: str, session: SessionMetadata, limit: int = 10
    ) -> list[LLMMessage]:
        results = []
        for msg in self._messages.get(session.session_id, []):
            if query in msg.extract_text:
                results.append(msg)
            if len(results) >= limit:
                break
        return results

    async def add_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        if session.session_id not in self._messages:
            self._messages[session.session_id] = []
        self._messages[session.session_id].extend(messages)

    async def set_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        self._messages[session.session_id] = list(messages)

    async def clear(self, session: SessionMetadata) -> None:
        self._messages.pop(session.session_id, None)


class ChatWorkingMemory(BaseWorkingMemory):
    def __init__(self, store: BaseMessageStore, max_messages: int = 50):
        self.store = store
        self._max_messages = max_messages

    async def _trim_history(self, session: SessionMetadata) -> None:
        history = await self.store.get_messages(session)
        if len(history) <= self._max_messages:
            return
        has_system = history and isinstance(history[0], SystemMessage)
        if has_system:
            keep_count = max(0, self._max_messages - 1)
            new_history = [history[0], *history[-keep_count:]]
        else:
            new_history = history[-self._max_messages :]
        await self.store.set_messages(session, new_history)

    async def get_history(self, session: SessionMetadata) -> list[LLMMessage]:
        return await self.store.get_messages(session)

    async def add_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        await self.store.add_messages(session, messages)
        if messages:
            await self._trim_history(session)

    async def clear_history(self, session: SessionMetadata) -> None:
        await self.store.clear(session)

    async def set_history(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        await self.store.set_messages(session, messages)


class MemoryProcessor(ABC):
    @abstractmethod
    async def process(self, session_id: str, new_messages: list[LLMMessage]) -> None:
        pass


_default_memory_factory: Callable[[], BaseWorkingMemory] | None = None
_global_default_memory_instance: BaseWorkingMemory | None = None


def set_default_memory_backend(factory: Callable[[], BaseWorkingMemory]):
    global _default_memory_factory
    _default_memory_factory = factory


def _get_default_memory() -> BaseWorkingMemory:
    global _global_default_memory_instance
    if _default_memory_factory:
        return _default_memory_factory()
    if _global_default_memory_instance is None:
        _global_default_memory_instance = ChatWorkingMemory(
            store=InMemoryMessageStore()
        )
    return _global_default_memory_instance


class ToolOutputCompactor(BaseMemoryReducer):
    async def reduce(
        self, messages, target_tokens, current_tokens, model_name, base_overhead=0
    ):
        if current_tokens <= target_tokens:
            return messages, False, current_tokens
        changed = False
        new_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                new_content = []
                msg_changed = False
                for return_part in msg.tool_returns:
                    text = (
                        str(return_part.output)
                        if not isinstance(return_part.output, str)
                        else return_part.output
                    )
                    if len(text) > 1000:
                        head = text[:300]
                        tail = text[-300:]
                        omitted = len(text) - 600
                        new_output = f"{head}\n\n...[由于上下文限制，已静默省略 {omitted} 个字符]...\n\n{tail}"
                        new_part = return_part.model_copy(update={"output": new_output})
                        new_content.append(new_part)
                        msg_changed = True
                        changed = True
                    else:
                        new_content.append(return_part)

                if msg_changed:
                    new_msg = msg.model_copy(deep=True)
                    new_msg.content = new_content
                    new_msg.token_cost = None
                    new_messages.append(new_msg)
                else:
                    new_messages.append(msg)
            else:
                new_messages.append(msg)
        if changed:
            return (
                new_messages,
                True,
                global_estimator.estimate_context(
                    new_messages, model_name, base_overhead
                ),
            )
        return messages, False, current_tokens


class MessageDropper(BaseMemoryReducer):
    async def reduce(
        self, messages, target_tokens, current_tokens, model_name, base_overhead=0
    ):
        if current_tokens <= target_tokens:
            return messages, False, current_tokens
        new_messages = list(messages)
        changed = False
        idx = 0
        while current_tokens > target_tokens and idx < len(new_messages):
            msg = new_messages[idx]
            is_pinned = msg.metadata.get("pinned", False) if msg.metadata else False
            if isinstance(msg, SystemMessage) or is_pinned:
                idx += 1
                continue
            dropped_msg = new_messages.pop(idx)
            changed = True
            current_tokens -= (
                dropped_msg.token_cost
                or global_estimator.estimate_message(dropped_msg, model_name)
            )
        if changed:
            current_tokens = global_estimator.estimate_context(
                new_messages, model_name, base_overhead
            )
        return new_messages, changed, current_tokens


class LLMSummarizerReducer(BaseMemoryReducer):
    def __init__(self, keep_recent_msgs: int = 4):
        self.keep_recent_msgs = keep_recent_msgs

    async def reduce(
        self, messages, target_tokens, current_tokens, model_name, base_overhead=0
    ):
        if current_tokens <= target_tokens:
            return messages, False, current_tokens
        from zhenxun.services.ai.config import get_llm_config

        config = get_llm_config().context_settings
        if not config.enable_summarization:
            return messages, False, current_tokens

        pinned_msgs, working_msgs, prev_summary = [], [], ""
        for msg in messages:
            is_pinned = isinstance(msg, SystemMessage) or (
                msg.metadata and msg.metadata.get("pinned", False)
            )
            if msg.metadata and msg.metadata.get("is_summary", False):
                prev_summary = msg.extract_text
            elif is_pinned:
                pinned_msgs.append(msg)
            else:
                working_msgs.append(msg)

        if len(working_msgs) <= self.keep_recent_msgs:
            return messages, False, current_tokens
        to_summarize = (
            working_msgs[: -self.keep_recent_msgs]
            if self.keep_recent_msgs > 0
            else working_msgs
        )
        to_keep = (
            working_msgs[-self.keep_recent_msgs :] if self.keep_recent_msgs > 0 else []
        )

        prompt_text = f"### 📋 [对话摘要任务]\n{config.summarization_prompt}\n\n"
        if prev_summary:
            prompt_text += "####  önceki_summary (参考先前的快照):\n"
            prompt_text += f"> {prev_summary}\n\n"
        prompt_text += "#### 待处理的历史消息流：\n"
        for m in to_summarize:
            from zhenxun.services.ai.llm.utils import extract_text_from_content

            c_str = extract_text_from_content(m.content)[:1500]
            speaker = m.source_name if m.source_name else m.role.capitalize()
            prompt_text += f"[{speaker}]: {c_str}\n"
        prompt_text += "</需要合并的旧对话记录>\n"

        from zhenxun.services.ai.llm.api import chat

        try:
            response = await chat(
                prompt_text,
                model=config.summarization_model,
                instruction="你是后台记忆整理引擎。请客观、简明输出当前对话全局摘要。",
            )
            new_summary_msg = LLMMessage.system(
                f"[全局记忆摘要(由AI生成)]\n{response.text}"
            )
            new_summary_msg.metadata = {"is_summary": True, "pinned": True}
        except Exception as e:
            logger.error(f"[LLMSummarizerReducer] 调用失败: {e}")
            return messages, False, current_tokens

        new_messages = [*pinned_msgs, new_summary_msg, *to_keep]
        return (
            new_messages,
            True,
            global_estimator.estimate_context(new_messages, model_name, base_overhead),
        )


class StructuredSummaryReducer(BaseMemoryReducer):
    """
    结构化状态摘要压缩器。
    将旧对话提炼为具有固定 JSON Schema 的结构化状态，有效消除超长对话幻觉。
    适用于跑团游戏、剧本杀或复杂长线任务。
    """

    def __init__(self, keep_recent_msgs: int = 4):
        self.keep_recent_msgs = keep_recent_msgs

    async def reduce(
        self, messages, target_tokens, current_tokens, model_name, base_overhead=0
    ):
        if current_tokens <= target_tokens:
            return messages, False, current_tokens

        from zhenxun.services.ai.config import get_llm_config

        config = get_llm_config().context_settings

        pinned_msgs, working_msgs, prev_summary = [], [], ""
        for msg in messages:
            is_pinned = isinstance(msg, SystemMessage) or (
                msg.metadata and msg.metadata.get("pinned", False)
            )
            if msg.metadata and msg.metadata.get("is_summary", False):
                prev_summary = msg.extract_text
            elif is_pinned:
                pinned_msgs.append(msg)
            else:
                working_msgs.append(msg)

        if len(working_msgs) <= self.keep_recent_msgs:
            return messages, False, current_tokens

        to_summarize = (
            working_msgs[: -self.keep_recent_msgs]
            if self.keep_recent_msgs > 0
            else working_msgs
        )
        to_keep = (
            working_msgs[-self.keep_recent_msgs :] if self.keep_recent_msgs > 0 else []
        )

        prompt_text = "你是一个专门用于长上下文状态压缩的引擎。请阅读以下先前的总结和旧对话，提取核心状态信息，并合并它们。\n\n"
        if prev_summary:
            prompt_text += f"<之前的状态摘要>\n{prev_summary}\n</之前的状态摘要>\n\n"
        prompt_text += "<需要合并的旧对话记录>\n"
        for m in to_summarize:
            from zhenxun.services.ai.llm.utils import extract_text_from_content

            c_str = extract_text_from_content(m.content)[:1500]
            speaker = m.source_name if m.source_name else m.role.capitalize()
            prompt_text += f"[{speaker}]: {c_str}\n"
        prompt_text += "</需要合并的旧对话记录>\n"

        from zhenxun.services.ai.llm.api import generate_structured

        try:

            class StateSummary(BaseModel):
                user_context: str = Field(
                    description="用户的核心意图、诉求、人设或长期记忆规则。"
                )
                completed_tasks: str = Field(
                    description="已完成的操作或已经确认的情节。"
                )
                pending_tasks: str = Field(
                    description="正在进行中的任务或尚未解答的问题。"
                )
                current_state: str = Field(
                    description="当前状态，如重要变量、玩家血量、关键物品坐标等。"
                )

            summary_obj = await generate_structured(
                prompt_text,
                response_model=StateSummary,
                model=config.summarization_model,
                instruction="请提取并合并先前的状态和最新的对话内容，保持精简，不要编造事实。",
            )

            summary_text = (
                f"👤 用户上下文: {summary_obj.user_context}\n"
                f"✅ 已完成/确认: {summary_obj.completed_tasks}\n"
                f"⏳ 待处理/疑问: {summary_obj.pending_tasks}\n"
                f"📌 当前状态: {summary_obj.current_state}"
            )

            new_summary_msg = LLMMessage.system(
                f"[结构化上下文状态摘要(由AI生成)]\n{summary_text}"
            )
            new_summary_msg.metadata = {"is_summary": True, "pinned": True}
        except Exception as e:
            logger.error(f"[StructuredSummaryReducer] 结构化总结失败: {e}")
            return messages, False, current_tokens

        new_messages = [*pinned_msgs, new_summary_msg, *to_keep]
        return (
            new_messages,
            True,
            global_estimator.estimate_context(new_messages, model_name, base_overhead),
        )


class CondenserPipeline:
    def __init__(self, reducers: list[BaseMemoryReducer]):
        self.reducers = reducers

    async def run(self, messages, target_tokens, model_name, base_overhead=0):
        current_tokens = global_estimator.estimate_context(
            messages, model_name, base_overhead
        )
        if current_tokens <= target_tokens:
            return messages
        current_messages = messages
        for reducer in self.reducers:
            current_messages, _changed, current_tokens = await reducer.reduce(
                current_messages,
                target_tokens,
                current_tokens,
                model_name,
                base_overhead,
            )
            if current_tokens <= target_tokens:
                break
        return current_messages


class CondenserRegistry:
    _reducers: dict[str, type[BaseMemoryReducer]] = {}

    @classmethod
    def register(cls, name: str, reducer_cls: type[BaseMemoryReducer]):
        cls._reducers[name] = reducer_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMemoryReducer:
        return cls._reducers[name](**kwargs)


CondenserRegistry.register("tool_compactor", ToolOutputCompactor)
CondenserRegistry.register("message_dropper", MessageDropper)
CondenserRegistry.register("llm_summarizer", LLMSummarizerReducer)
CondenserRegistry.register("structured_summarizer", StructuredSummaryReducer)
