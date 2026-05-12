from typing import Any, TypeVar, cast
import uuid

from nonebot.adapters import Bot, Event
from pydantic import BaseModel

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.exceptions import LLMException
from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    LLMContentPart,
    LLMMessage,
    LLMResponse,
)
from zhenxun.services.ai.core.models import ModelName
from zhenxun.services.ai.core.engine.pipeline import DialoguePipeline
from zhenxun.services.ai.core.engine.token_estimator import (
    global_estimator,
    parse_usage_info,
)
from zhenxun.services.ai.llm.api import generate, generate_structured
from zhenxun.services.ai.llm.config import IntentBuilder
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.core.templates import PromptTemplate
from zhenxun.services.ai.memory.scope import MemoryScope
from zhenxun.services.ai.memory.working_memory import _get_default_memory
from zhenxun.services.ai.message_builder import MessageBuilder
from zhenxun.services.ai.protocols.memory import (
    BaseMemoryReducer,
    BaseWorkingMemory,
    MemoryIsolationLevel,
    SessionMetadata,
    generate_session_meta,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump

T = TypeVar("T", bound=BaseModel)


class ChatSession:
    """
    极其易用的状态化对话外壳 (Stateful Chat Facade)。

    设计用于简化多轮对话管理。它会自动维护对话历史，并在达到 Token 限制时触发智能压缩。
    开发者可以直接在初始化时指定模型和工具，无需关心底层的上下文装配细节。
    """

    def __init__(
        self,
        session_id: str | None = None,
        model: str | None = None,
        default_generation_config: GenerationConfig | None = None,
        memory_reducers: list[str | BaseMemoryReducer] | None = None,
        context_threshold: float | None = None,
        max_history_turns: int | None = None,
        bot: Bot | None = None,
        event: Event | None = None,
        isolation_level: MemoryIsolationLevel = MemoryIsolationLevel.GROUP_USER,
    ):
        if bot and event:
            self.session_metadata = generate_session_meta(bot, event, isolation_level)
            self.session_id = self.session_metadata.session_id
        else:
            self.session_id = session_id or str(uuid.uuid4())
            self.session_metadata = SessionMetadata(session_id=self.session_id)

        self.model = model
        self.default_generation_config = default_generation_config or GenerationConfig()
        self.memory_reducers = memory_reducers
        self.context_threshold = context_threshold
        self.max_history_turns = max_history_turns

        self.working_memory: BaseWorkingMemory = _get_default_memory()
        self.message_buffer: list[LLMMessage] = []
        self._session_base_overhead: int | None = None

    async def clear_memory(self) -> None:
        """极简快捷方法：清空当前会话的历史记忆"""
        await self.working_memory.clear_history(self.session_metadata)

    async def get_history(self) -> list[LLMMessage]:
        """极简快捷方法：获取当前会话的历史记忆"""
        return await self.working_memory.get_history(self.session_metadata)

    def _resolve_model_name(self, model_name: str | None) -> str:
        if model_name:
            return model_name
        default_model = get_global_default_model_name()
        if default_model:
            return default_model
        raise LLMException("未指定模型名称且未设置全局默认模型")

    async def chat(
        self,
        message: str
        | Any
        | LLMMessage
        | list[LLMContentPart]
        | list[LLMMessage]
        | None,
        *,
        model: ModelName = None,
        instruction: str | None = None,
        template_vars: dict[str, Any] | None = None,
        preserve_media_in_history: bool | None = None,
        config: GenerationConfig | IntentBuilder | None = None,
        long_term_memory: MemoryScope | None = None,
        use_buffer: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """执行一次带有记忆的对话。"""
        resolved_model_name = self._resolve_model_name(model or self.model)

        final_instruction = instruction
        if final_instruction and template_vars:
            final_instruction = PromptTemplate(final_instruction).render(
                **template_vars
            )

        user_msg = None
        if message:
            msgs = await MessageBuilder.normalize_to_llm_messages(message)
            if msgs:
                user_msg = msgs[-1]

        pipeline = DialoguePipeline(
            model_name=resolved_model_name,
            session_metadata=self.session_metadata,
            working_memory=self.working_memory,
            long_term_memory=long_term_memory,
            memory_reducers=self.memory_reducers,
            context_threshold=self.context_threshold,
            max_history_turns=self.max_history_turns,
        )

        buffer_to_use = self.message_buffer if use_buffer else None

        messages_for_run = await pipeline.build_messages(
            user_input=user_msg,
            system_instruction=final_instruction,
            message_buffer=buffer_to_use,
            base_overhead=self._session_base_overhead or 0,
        )

        if use_buffer:
            self.message_buffer.clear()

        final_config = self.default_generation_config
        if isinstance(config, IntentBuilder):
            config = config.build()
        if config:
            final_config = final_config.merge_with(config)

        try:
            response = await generate(
                messages=messages_for_run,
                model=resolved_model_name,
                config=final_config,
                timeout=timeout,
            )

            usage_obj = parse_usage_info(response.usage_info)
            if usage_obj.prompt_tokens > 0:
                if self._session_base_overhead is None:
                    pure_est = global_estimator.estimate_context(
                        messages_for_run, resolved_model_name, base_overhead=0
                    )
                    overhead = usage_obj.prompt_tokens - pure_est
                    self._session_base_overhead = max(0, overhead)
                global_estimator.calibrate(
                    usage_obj.prompt_tokens, messages_for_run, resolved_model_name
                )

            should_preserve = (
                preserve_media_in_history
                if preserve_media_in_history is not None
                else False
            )

            msgs_to_save: list[LLMMessage] = []
            if user_msg:
                save_msg = (
                    user_msg
                    if should_preserve
                    else DialoguePipeline.sanitize_message_for_history(user_msg)
                )
                msgs_to_save.append(save_msg)

            if response.content_parts:
                from zhenxun.services.ai.core.messages import AssistantContentUnion

                ast_msg = AssistantMessage(
                    content=cast(list[AssistantContentUnion], response.content_parts)
                )
                msgs_to_save.append(ast_msg)

            if msgs_to_save:
                await self.working_memory.add_messages(
                    self.session_metadata, msgs_to_save
                )

            return response

        except Exception as e:
            logger.error(f"ChatSession 执行失败: {e}", e=e)
            raise

    async def generate_structured(
        self,
        message: str
        | Any
        | LLMMessage
        | list[LLMContentPart]
        | list[LLMMessage]
        | None,
        response_model: type[T],
        **kwargs,
    ) -> T:
        history = await self.get_history()
        history = await DialoguePipeline(
            model_name=self._resolve_model_name(kwargs.get("model")),
            session_metadata=self.session_metadata,
            working_memory=self.working_memory,
            context_threshold=self.context_threshold,
            max_history_turns=self.max_history_turns,
        )._compress_history(history)

        if message:
            msgs = await MessageBuilder.normalize_to_llm_messages(message)
            history.extend(msgs)

        result = await generate_structured(
            message=history, response_model=response_model, **kwargs
        )

        if message:
            user_msg = (await MessageBuilder.normalize_to_llm_messages(message))[-1]
            msgs_to_save: list[LLMMessage] = [
                DialoguePipeline.sanitize_message_for_history(user_msg)
            ]
            import json

            msgs_to_save.append(
                LLMMessage.assistant_text_response(
                    content=json.dumps(model_dump(result), ensure_ascii=False)
                )
            )
            await self.working_memory.add_messages(self.session_metadata, msgs_to_save)

        return result
