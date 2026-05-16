import asyncio

from pydantic import BaseModel, Field

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.engine.token_estimator import global_estimator
from zhenxun.services.ai.core.messages import (
    LLMMessage,
    SystemMessage,
    ToolMessage,
)
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.memory.interfaces import (
    BaseMemoryReducer,
    BaseWorkingMemory,
)
from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.ai.memory.working_memory import _get_default_memory
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy


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
                        new_part = model_copy(
                            return_part, update={"output": new_output}
                        )
                        new_content.append(new_part)
                        msg_changed = True
                        changed = True
                    else:
                        new_content.append(return_part)

                if msg_changed:
                    new_msg = model_copy(msg, deep=True)
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
            c_str = m.extract_text[:1500]
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
    def __init__(self, keep_recent_msgs: int = 4):
        self.keep_recent_msgs = keep_recent_msgs

    async def reduce(
        self, messages, target_tokens, current_tokens, model_name, base_overhead=0
    ):
        if current_tokens <= target_tokens:
            return messages, False, current_tokens

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
            c_str = m.extract_text[:1500]
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


class AsyncMemoryCondenser:
    """负责在后台异步压缩记忆，提供并发锁保证无损合并。"""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self.memory: BaseWorkingMemory = _get_default_memory()
        self._compressing_tasks: dict[str, asyncio.Task] = {}

    def _get_lock(self, scope: str) -> asyncio.Lock:
        if scope not in self._locks:
            self._locks[scope] = asyncio.Lock()
        return self._locks[scope]

    def trigger_compression(self, scope: str):
        config = get_llm_config().context_settings
        if not config.enabled:
            return

        if scope not in self._compressing_tasks:
            task = asyncio.create_task(self._background_condense(scope))
            self._compressing_tasks[scope] = task

            def _on_done(t):
                self._compressing_tasks.pop(scope, None)

            task.add_done_callback(_on_done)

    async def _background_condense(self, scope: str):
        session = SessionMetadata(session_id=scope)
        async with self._get_lock(scope):
            current_history = await self.memory.get_history(session)
            original_len = len(current_history)

        model_name = get_global_default_model_name() or "Gemini/gemini-2.0-flash"
        caps = get_model_capabilities(model_name)
        config = get_llm_config().context_settings
        threshold_tokens = int(caps.max_input_tokens * config.trigger_threshold)

        current_tokens = global_estimator.estimate_context(current_history, model_name)
        if current_tokens <= threshold_tokens:
            return

        logger.info(
            f"[AsyncMemoryCondenser] Scope {scope} 触发后台上下文压缩 ({current_tokens}/{threshold_tokens})"
        )

        reducers = [CondenserRegistry.get("tool_compactor")]
        if config.enable_summarization:
            reducers.append(CondenserRegistry.get("llm_summarizer"))
        reducers.append(CondenserRegistry.get("message_dropper"))

        pipeline = CondenserPipeline(reducers)

        try:
            compressed_history = await pipeline.run(
                current_history, target_tokens=threshold_tokens, model_name=model_name
            )
        except Exception as e:
            logger.error(f"[AsyncMemoryCondenser] 后台压缩失败: {e}", e=e)
            return

        async with self._get_lock(scope):
            latest_history = await self.memory.get_history(session)
            new_msgs = latest_history[original_len:]
            final_history = compressed_history + new_msgs
            await self.memory.set_history(session, final_history)
            logger.info(
                f"[AsyncMemoryCondenser] Scope {scope} 后台压缩完成，已无损合并 {len(new_msgs)} 条并发新消息。"
            )


async_memory_condenser = AsyncMemoryCondenser()
