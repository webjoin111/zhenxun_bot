from collections.abc import Callable
import time
from typing import Any, cast

from nonebot.utils import is_coroutine_callable
from pydantic import TypeAdapter
from tortoise import fields

from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    LLMContentPart,
    LLMMessage,
    LLMResponse,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from zhenxun.services.ai.memory.interfaces import (
    BaseMessageStore,
    BaseWorkingMemory,
)
from zhenxun.services.ai.memory.long_term_memory import MemoryScope
from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.ai.protocols.middleware import (
    BaseLLMMiddleware,
    LLMContext,
    NextCall,
)
from zhenxun.services.db_context import Model
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump


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


class AbstractMemoryRecord(Model):
    """Tortoise ORM 短期记忆持久化基类 (Mixin)。"""

    id = fields.UUIDField(pk=True, description="主键")
    session_id = fields.CharField(max_length=255, index=True)
    role = fields.CharField(max_length=32)
    content = fields.JSONField()
    api_context = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    metadata = fields.JSONField(null=True)

    class Meta:  # type: ignore
        abstract = True


class TortoiseMessageStore(BaseMessageStore):
    def __init__(
        self,
        model_class: type[AbstractMemoryRecord],
        custom_save_hook: Callable[
            [AbstractMemoryRecord, LLMMessage, SessionMetadata], Any
        ]
        | None = None,
    ):
        self.model_class = model_class
        self.custom_save_hook = custom_save_hook

    def _row_to_message(self, row: AbstractMemoryRecord) -> LLMMessage:
        content_raw = row.content
        from zhenxun.services.ai.core.messages import TextPart

        content_parts: list[LLMContentPart] = []
        if isinstance(content_raw, list):
            adapter = TypeAdapter(LLMContentPart)
            for p in content_raw:
                if isinstance(p, dict):
                    content_parts.append(adapter.validate_python(p))
        elif isinstance(content_raw, str):
            content_parts.append(TextPart(text=content_raw))

        metadata: dict[str, Any] | None = (
            row.metadata if isinstance(row.metadata, dict) else None
        )
        kwargs = {
            "content": content_parts,
            "metadata": metadata,
            "created_at": row.created_at.timestamp() if row.created_at else time.time(),
        }
        role = row.role
        if role == "system":
            return cast(LLMMessage, SystemMessage(**kwargs))
        elif role == "user":
            return cast(LLMMessage, UserMessage(**kwargs))
        elif role == "assistant":
            return cast(LLMMessage, AssistantMessage(**kwargs))
        elif role == "tool":
            return cast(LLMMessage, ToolMessage(**kwargs))
        return cast(LLMMessage, LLMMessage(role=role, **kwargs))

    async def get_messages(self, session: SessionMetadata) -> list[LLMMessage]:
        rows = (
            await self.model_class.filter(session_id=session.session_id)
            .order_by("created_at")
            .all()
        )
        return [self._row_to_message(row) for row in rows]

    async def search(
        self, query: str, session: SessionMetadata, limit: int = 10
    ) -> list[LLMMessage]:
        rows = (
            await self.model_class.filter(
                session_id=session.session_id, content__icontains=query
            )
            .order_by("-created_at")
            .limit(limit)
            .all()
        )
        return [self._row_to_message(row) for row in reversed(rows)]

    async def add_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        if not messages:
            return
        orm_objects = []
        for msg in messages:
            content_payload = msg.content
            if isinstance(content_payload, str):
                content_payload = [{"type": "text", "text": content_payload}]
            elif isinstance(content_payload, list):
                processed_content = []
                from zhenxun.services.ai.core.messages import ThoughtPart

                for p in content_payload:
                    if isinstance(p, ThoughtPart):
                        continue
                    p_dump = (
                        model_dump(p, exclude_none=True)
                        if hasattr(p, "model_dump")
                        else (p.copy() if isinstance(p, dict) else p)
                    )
                    processed_content.append(p_dump)
                content_payload = (
                    processed_content
                    if processed_content
                    else [
                        {
                            "type": "text",
                            "text": "[仅包含思维链或工具调度，无实质文本输出]",
                        }
                    ]
                )

            orm_obj = self.model_class(
                session_id=session.session_id,
                role=msg.role,
                content=content_payload,
                api_context=None,
                metadata=msg.metadata,
            )
            if self.custom_save_hook:
                if is_coroutine_callable(self.custom_save_hook):
                    await self.custom_save_hook(orm_obj, msg, session)
                else:
                    self.custom_save_hook(orm_obj, msg, session)
            orm_objects.append(orm_obj)
        if orm_objects:
            await self.model_class.bulk_create(orm_objects)

    async def set_messages(
        self, session: SessionMetadata, messages: list[LLMMessage]
    ) -> None:
        await self.clear(session)
        await self.add_messages(session, messages)

    async def clear(self, session: SessionMetadata) -> None:
        await self.model_class.filter(session_id=session.session_id).delete()


def get_orm_working_memory(
    model_class: type[AbstractMemoryRecord],
    max_messages: int = 50,
    custom_save_hook: Callable[[AbstractMemoryRecord, LLMMessage, SessionMetadata], Any]
    | None = None,
) -> "ChatWorkingMemory":
    """
    [工厂方法] 供第三方开发者调用，
    将 Tortoise ORM 表直接包装为带有滚动窗口截断能力的工作记忆系统。
    """
    store = TortoiseMessageStore(
        model_class=model_class, custom_save_hook=custom_save_hook
    )
    return ChatWorkingMemory(store=store, max_messages=max_messages)


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


class MemoryMiddleware(BaseLLMMiddleware):
    """记忆中间件：接管大模型调用的上下文加载与保存。"""

    def __init__(
        self,
        session_meta: SessionMetadata,
        working_memory: BaseWorkingMemory | None = None,
        long_term_memory: MemoryScope | None = None,
        sanitizer: Callable[[LLMMessage], LLMMessage] | None = None,
    ):
        self.wm = working_memory
        self.ltm = long_term_memory
        self.session_meta = session_meta
        self.session_id = session_meta.session_id
        self.sanitizer = sanitizer

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        if self.ltm and context.messages:
            last_content = str(context.messages[-1].content)
            matches = await self.ltm.recall(last_content)
            if matches:
                fact_str = "\n".join(
                    f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
                )
                sys_msg = LLMMessage.system(
                    f"[系统补充：有关用户的长期记忆设定]\n{fact_str}"
                )
                context.messages.insert(0, sys_msg)
                logger.debug(f"已动态注入 {len(matches)} 条长期记忆。")

        if self.wm:
            history = await self.wm.get_history(self.session_meta)
            context.messages = history + context.messages

        response = await next_call(context)

        if self.wm and context.messages:
            user_msg = context.messages[-1]
            if self.sanitizer:
                user_msg = self.sanitizer(user_msg)
            msgs_to_save = [user_msg]
            if response.content_parts:
                ast_msg = LLMMessage(role="assistant", content=response.content_parts)
                msgs_to_save.append(ast_msg)
            await self.wm.add_messages(self.session_meta, msgs_to_save)

        return response
