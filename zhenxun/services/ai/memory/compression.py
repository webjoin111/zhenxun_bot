from pydantic import BaseModel, Field

from zhenxun.services.ai.core.engine.token_estimator import global_estimator
from zhenxun.services.ai.core.messages import (
    AudioPart,
    FilePart,
    ImagePart,
    LLMMessage,
    SystemMessage,
    TextPart,
    ToolMessage,
    VideoPart,
)
from zhenxun.services.ai.memory.interfaces import (
    BaseMemoryReducer,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_copy


class MultimodalPlaceholderReducer(BaseMemoryReducer):
    """视觉媒体降级：将超过一定轮数的老图片/视频替换为 <图片> 占位符文本"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    @staticmethod
    def apply_multimodal_placeholder(message: LLMMessage) -> LLMMessage:
        sanitized_message = model_copy(message, deep=False)
        new_content_parts = []

        for part in sanitized_message.content:
            if isinstance(part, ImagePart):
                new_content_parts.append(TextPart(text="<图片>"))
            elif isinstance(part, AudioPart):
                new_content_parts.append(TextPart(text="<音频>"))
            elif isinstance(part, VideoPart):
                new_content_parts.append(TextPart(text="<视频>"))
            elif isinstance(part, FilePart):
                new_content_parts.append(TextPart(text="<文件>"))
            elif isinstance(part, TextPart) and "[多模态内容:" in part.text:
                new_content_parts.append(TextPart(text="<图片>"))
            else:
                new_content_parts.append(part)

        merged_parts = []
        for part in new_content_parts:
            if (
                isinstance(part, TextPart)
                and merged_parts
                and isinstance(merged_parts[-1], TextPart)
            ):
                new_text = (merged_parts[-1].text or "") + " " + (part.text or "")
                merged_parts[-1] = TextPart(text=new_text.strip())
            else:
                merged_parts.append(part)

        sanitized_message.content = merged_parts
        sanitized_message.token_cost = None
        return sanitized_message

    async def reduce(self, messages, current_tokens, model_name, base_overhead=0):
        if self.window_size <= 0:
            return messages, False, current_tokens

        processed_messages = []
        user_multimodal_count = 0
        changed = False

        for msg in reversed(messages):
            has_multimodal = False
            if isinstance(msg.content, list):
                has_multimodal = any(
                    isinstance(p, (ImagePart, AudioPart, VideoPart, FilePart))
                    or (isinstance(p, TextPart) and "[多模态内容:" in p.text)
                    for p in msg.content
                )

            if has_multimodal:
                if msg.role == "user":
                    user_multimodal_count += 1
                if user_multimodal_count > self.window_size:
                    processed_messages.append(self.apply_multimodal_placeholder(msg))
                    changed = True
                else:
                    processed_messages.append(msg)
            else:
                processed_messages.append(msg)

        if not changed:
            return messages, False, current_tokens

        processed_messages.reverse()
        new_tokens = global_estimator.estimate_context(
            processed_messages, model_name, base_overhead
        )
        return processed_messages, True, new_tokens


class ToolOutputCompactor(BaseMemoryReducer):
    async def reduce(self, messages, current_tokens, model_name, base_overhead=0):
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
    def __init__(self, trigger_tokens: int = 4000):
        self.trigger_tokens = trigger_tokens

    async def reduce(self, messages, current_tokens, model_name, base_overhead=0):
        if current_tokens <= self.trigger_tokens:
            return messages, False, current_tokens
        new_messages = list(messages)
        changed = False

        while current_tokens > self.trigger_tokens:
            user_indices = [
                i
                for i, m in enumerate(new_messages)
                if m.role == "user"
                and not (m.metadata and m.metadata.get("pinned", False))
            ]

            if len(user_indices) < 2:
                break

            start_idx = user_indices[0]
            end_idx = user_indices[1]

            del new_messages[start_idx:end_idx]
            changed = True

            current_tokens = global_estimator.estimate_context(
                new_messages, model_name, base_overhead
            )
        return new_messages, changed, current_tokens


class LLMSummarizerReducer(BaseMemoryReducer):
    def __init__(
        self,
        keep_recent_turns: int = 0,
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        summarization_model: str = "Gemini/gemini-2.5-flash",
        summarization_prompt: str = "请概括以下对话内容，保留关键的约束条件、用户偏好、已完成的任务状态和未解决的问题。",
    ):
        self.keep_recent_turns = keep_recent_turns
        self.trigger_tokens = trigger_tokens
        self.max_turns = max_turns
        self.summarization_model = summarization_model
        self.summarization_prompt = summarization_prompt

    async def reduce(self, messages, current_tokens, model_name, base_overhead=0):
        user_turns = sum(
            1
            for m in messages
            if m.role == "user"
            and not (m.metadata and m.metadata.get("is_summary", False))
        )
        is_token_exceeded = current_tokens > self.trigger_tokens
        is_turn_exceeded = self.max_turns is not None and user_turns > self.max_turns

        if not (is_token_exceeded or is_turn_exceeded):
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

        user_indices = [i for i, m in enumerate(working_msgs) if m.role == "user"]

        if len(user_indices) <= self.keep_recent_turns:
            return messages, False, current_tokens

        split_idx = (
            user_indices[-self.keep_recent_turns]
            if self.keep_recent_turns > 0
            else len(working_msgs)
        )
        to_summarize = working_msgs[:split_idx]
        to_keep = working_msgs[split_idx:]

        prompt_text = f"### 📋 [对话摘要任务]\n{self.summarization_prompt}\n\n"
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
                model=self.summarization_model,
                instruction="你是后台记忆整理引擎。请客观、简明输出当前对话全局摘要。",
            )
            new_summary_msg = LLMMessage.assistant_text_response(
                f"【历史对话摘要记忆】\n{response.text}"
            )
            new_summary_msg.metadata = {"is_summary": True, "pinned": True}
        except Exception as e:
            logger.error(
                f"[LLMSummarizerReducer] 压缩总结调用失败，已跳过本次压缩: {e}"
            )
            return messages, False, current_tokens

        new_messages = [*pinned_msgs, new_summary_msg, *to_keep]
        return (
            new_messages,
            True,
            global_estimator.estimate_context(new_messages, model_name, base_overhead),
        )


class StructuredSummaryReducer(BaseMemoryReducer):
    def __init__(
        self,
        keep_recent_turns: int = 0,
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        summarization_model: str = "Gemini/gemini-2.5-flash",
    ):
        self.keep_recent_turns = keep_recent_turns
        self.trigger_tokens = trigger_tokens
        self.max_turns = max_turns
        self.summarization_model = summarization_model

    async def reduce(self, messages, current_tokens, model_name, base_overhead=0):
        user_turns = sum(
            1
            for m in messages
            if m.role == "user"
            and not (m.metadata and m.metadata.get("is_summary", False))
        )
        is_token_exceeded = current_tokens > self.trigger_tokens
        is_turn_exceeded = self.max_turns is not None and user_turns > self.max_turns

        if not (is_token_exceeded or is_turn_exceeded):
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

        user_indices = [i for i, m in enumerate(working_msgs) if m.role == "user"]

        if len(user_indices) <= self.keep_recent_turns:
            return messages, False, current_tokens

        split_idx = (
            user_indices[-self.keep_recent_turns]
            if self.keep_recent_turns > 0
            else len(working_msgs)
        )
        to_summarize = working_msgs[:split_idx]
        to_keep = working_msgs[split_idx:]

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
                model=self.summarization_model,
                instruction="请提取并合并先前的状态和最新的对话内容，保持精简，不要编造事实。",
            )

            summary_text = (
                f"👤 用户上下文: {summary_obj.user_context}\n"
                f"✅ 已完成/确认: {summary_obj.completed_tasks}\n"
                f"⏳ 待处理/疑问: {summary_obj.pending_tasks}\n"
                f"📌 当前状态: {summary_obj.current_state}"
            )

            new_summary_msg = LLMMessage.assistant_text_response(
                f"【历史状态摘要记忆】\n{summary_text}"
            )
            new_summary_msg.metadata = {"is_summary": True, "pinned": True}
        except Exception as e:
            logger.error(
                f"[StructuredSummaryReducer] 结构化总结失败，已跳过本次压缩: {e}"
            )
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

    async def run(
        self, messages, model_name, base_overhead=0
    ) -> tuple[list[LLMMessage], bool]:
        current_tokens = global_estimator.estimate_context(
            messages, model_name, base_overhead
        )

        current_messages = messages
        any_changed = False
        for reducer in self.reducers:
            current_messages, changed, current_tokens = await reducer.reduce(
                current_messages,
                current_tokens,
                model_name,
                base_overhead,
            )
            if changed:
                any_changed = True
        return current_messages, any_changed


class MemoryPolicy:
    """
    记忆策略工厂 (Strategy Factory Facade)。
    为开发者提供开箱即用的上下文压缩管线组装方案。
    (多模态视窗 vision_window 已作为正交配置独立到 AgentMemory 中)
    """

    @staticmethod
    def unlimited() -> list[BaseMemoryReducer]:
        """无限制模式。不进行任何形式的截断和总结，适用于短对话或纯 Agent 内部流转。"""
        return []

    @staticmethod
    def llm_summarize(
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str = "Gemini/gemini-2.5-flash",
        summarization_prompt: str = "请概括以下对话内容，保留关键的约束条件、用户偏好、已完成的任务状态和未解决的问题。",
    ) -> list[BaseMemoryReducer]:
        """LLM 总结压缩模式。Token 达标后，自动将历史对话合并为一段 Summary。"""
        return [
            ToolOutputCompactor(),
            LLMSummarizerReducer(
                keep_recent_turns=keep_recent_turns,
                trigger_tokens=trigger_tokens,
                max_turns=max_turns,
                summarization_model=summarization_model,
                summarization_prompt=summarization_prompt,
            ),
            MessageDropper(trigger_tokens=trigger_tokens),
        ]

    @staticmethod
    def structured_summarize(
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str = "Gemini/gemini-2.5-flash",
    ) -> list[BaseMemoryReducer]:
        """结构化总结压缩模式。使用 JSON Schema 强制大模型提取核心状态。"""
        return [
            ToolOutputCompactor(),
            StructuredSummaryReducer(
                keep_recent_turns=keep_recent_turns,
                trigger_tokens=trigger_tokens,
                max_turns=max_turns,
                summarization_model=summarization_model,
            ),
            MessageDropper(trigger_tokens=trigger_tokens),
        ]
