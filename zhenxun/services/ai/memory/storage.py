from collections.abc import Callable
import datetime
import time
from typing import Any, cast

from nonebot.utils import is_coroutine_callable
from pydantic import TypeAdapter
from tortoise import fields
from tortoise.timezone import now

from zhenxun.services.ai.core.messages import (
    AssistantMessage,
    LLMContentPart,
    LLMMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from zhenxun.services.ai.memory.interfaces import (
    BaseChatContext,
    BaseSlotContext,
)
from zhenxun.services.ai.memory.models import SessionMetadata, MemorySlot, SlotScope
from zhenxun.services.ai.rag import BaseRecord, SearchResult
from zhenxun.services.ai.rag.engine import ScopedRAGClient
from zhenxun.services.db_context import Model
from zhenxun.utils.pydantic_compat import model_dump


class MemoryScope:
    """长期记忆的作用域视图与 RAG 管线。"""

    def __init__(
        self,
        rag_client: ScopedRAGClient,
        async_write: bool = True,
    ):
        self.rag_client = rag_client
        self.async_write = async_write

    async def remember(
        self,
        session: SessionMetadata,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """通过 RAG Ingestion Pipeline 完成记忆落盘"""
        meta = metadata.copy() if metadata else {}
        meta.update(
            {
                "scope": session.scope_prefix,
                "importance": importance,
                "created_at": time.time(),
            }
        )
        record = BaseRecord(content=content, metadata=meta)

        await self.rag_client.ingest([record], async_write=self.async_write)

    async def recall(
        self,
        session: SessionMetadata,
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """委托至 Retriever 检索与重排"""
        return await self.rag_client.search(
            query=query,
            limit=limit,
            scopes=session.accessible_scopes,
            metadata_filters=metadata_filter,
        )

    async def update(
        self,
        session: SessionMetadata,
        record_id: str,
        new_content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """原子更新：通过先删后插，确保底层向量(Embedding)能根据新文本被正确刷新"""
        deleted_count = await self.forget(session, record_ids=[record_id])
        if deleted_count > 0:
            await self.remember(
                session=session,
                content=new_content,
                importance=importance,
                metadata=metadata,
            )
            return True
        return False

    async def forget(
        self, session: SessionMetadata, record_ids: list[str] | None = None
    ) -> int:
        return await self.rag_client.delete(
            record_ids=record_ids,
        )


class InMemoryChatContext(BaseChatContext):
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


class InMemorySlotContext(BaseSlotContext):
    """内存级记忆槽存储 (主要用于测试或无状态容器)"""

    def __init__(self):
        self._slots: dict[str, dict[str, MemorySlot]] = {}

    def _get_target_session_id(self, session: SessionMetadata, scope: SlotScope) -> str:
        if scope == SlotScope.GLOBAL:
            return f"global_u_{session.user_id}" if session.user_id else f"global_s_{session.session_id}"
        return session.session_id

    async def get_slot(self, session: SessionMetadata, label: str) -> MemorySlot | None:
        session_sid = self._get_target_session_id(session, SlotScope.SESSION)
        if session_sid in self._slots and label in self._slots[session_sid]:
            return self._slots[session_sid][label]
            
        global_sid = self._get_target_session_id(session, SlotScope.GLOBAL)
        if global_sid in self._slots and label in self._slots[global_sid]:
            return self._slots[global_sid][label]
            
        return None

    async def set_slot(self, session: SessionMetadata, slot: MemorySlot) -> None:
        target_sid = self._get_target_session_id(session, slot.scope)
        if target_sid not in self._slots:
            self._slots[target_sid] = {}
        self._slots[target_sid][slot.label] = slot

    async def delete_slot(self, session: SessionMetadata, label: str, scope: str) -> None:
        target_sid = self._get_target_session_id(session, SlotScope(scope))
        if target_sid in self._slots:
            self._slots[target_sid].pop(label, None)

    async def list_pinned_slots(self, session: SessionMetadata) -> list[MemorySlot]:
        global_sid = self._get_target_session_id(session, SlotScope.GLOBAL)
        session_sid = self._get_target_session_id(session, SlotScope.SESSION)
        
        merged = {}
        if global_sid in self._slots:
            for label, slot in self._slots[global_sid].items():
                merged[label] = slot
                
        if session_sid in self._slots:
            for label, slot in self._slots[session_sid].items():
                merged[label] = slot
                
        return [s for s in merged.values() if s.pinned and s.content.strip()]


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


class TortoiseChatContext(BaseChatContext):
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
                    import base64

                    for k in list(p.keys()):
                        if k.startswith("_is_b64_"):
                            orig_k = k[8:]
                            if orig_k in p and isinstance(p[orig_k], str):
                                p[orig_k] = base64.b64decode(p[orig_k])
                            p.pop(k, None)

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

        base_time = now()

        last_msg = (
            await self.model_class.filter(session_id=session.session_id)
            .order_by("-created_at")
            .first()
        )
        if last_msg and last_msg.created_at and last_msg.created_at >= base_time:
            base_time = last_msg.created_at + datetime.timedelta(milliseconds=10)

        orm_objects = []
        for i, msg in enumerate(messages):
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

                    import base64
                    from pathlib import Path

                    if isinstance(p_dump, dict):
                        for k, v in list(p_dump.items()):
                            if isinstance(v, bytes):
                                p_dump[k] = base64.b64encode(v).decode("utf-8")
                                p_dump[f"_is_b64_{k}"] = True
                            elif isinstance(v, Path):
                                p_dump[k] = str(v)

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

            msg_time = base_time + datetime.timedelta(milliseconds=i * 10)
            orm_obj = self.model_class(
                session_id=session.session_id,
                role=msg.role,
                content=content_payload,
                api_context=None,
                metadata=msg.metadata,
                created_at=msg_time,
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


def get_orm_chat_context(
    model_class: type[AbstractMemoryRecord],
    custom_save_hook: Callable[[AbstractMemoryRecord, LLMMessage, SessionMetadata], Any]
    | None = None,
) -> TortoiseChatContext:
    """
    [工厂方法] 供第三方开发者调用，
    将 Tortoise ORM 表直接包装为对话历史记录系统。
    """
    return TortoiseChatContext(
        model_class=model_class, custom_save_hook=custom_save_hook
    )


class AbstractSlotRecord(Model):
    """Tortoise ORM 记忆槽持久化基类 (Mixin)。"""
    id = fields.CharField(pk=True, max_length=128, description="复合主键: session_id + label")
    session_id = fields.CharField(max_length=255, index=True)
    label = fields.CharField(max_length=64, index=True)
    content = fields.TextField()
    size_limit = fields.IntField(default=2000)
    pinned = fields.BooleanField(default=True)
    scope = fields.CharField(max_length=32)
    created_at = fields.FloatField()
    updated_at = fields.FloatField()

    class Meta:  # type: ignore
        abstract = True


class TortoiseSlotContext(BaseSlotContext):
    def __init__(self, model_class: type[AbstractSlotRecord]):
        self.model_class = model_class

    def _get_target_session_id(self, session: SessionMetadata, scope: SlotScope) -> str:
        if scope == SlotScope.GLOBAL:
            return f"global_u_{session.user_id}" if session.user_id else f"global_s_{session.session_id}"
        return session.session_id

    def _row_to_slot(self, row: AbstractSlotRecord) -> MemorySlot:
        return MemorySlot(
            label=row.label,
            content=row.content,
            size_limit=row.size_limit,
            pinned=row.pinned,
            scope=SlotScope(row.scope),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def get_slot(self, session: SessionMetadata, label: str) -> MemorySlot | None:
        session_sid = self._get_target_session_id(session, SlotScope.SESSION)
        row = await self.model_class.filter(session_id=session_sid, label=label).first()
        if row:
            return self._row_to_slot(row)
            
        global_sid = self._get_target_session_id(session, SlotScope.GLOBAL)
        row = await self.model_class.filter(session_id=global_sid, label=label).first()
        if row:
            return self._row_to_slot(row)
        return None

    async def set_slot(self, session: SessionMetadata, slot: MemorySlot) -> None:
        target_sid = self._get_target_session_id(session, slot.scope)
        composite_id = f"{target_sid}_{slot.label}"
        
        await self.model_class.update_or_create(
            id=composite_id,
            defaults={
                "session_id": target_sid,
                "label": slot.label,
                "content": slot.content,
                "size_limit": slot.size_limit,
                "pinned": slot.pinned,
                "scope": slot.scope.value,
                "created_at": slot.created_at,
                "updated_at": slot.updated_at,
            }
        )

    async def delete_slot(self, session: SessionMetadata, label: str, scope: str) -> None:
        target_sid = self._get_target_session_id(session, SlotScope(scope))
        composite_id = f"{target_sid}_{label}"
        await self.model_class.filter(id=composite_id).delete()

    async def list_pinned_slots(self, session: SessionMetadata) -> list[MemorySlot]:
        global_sid = self._get_target_session_id(session, SlotScope.GLOBAL)
        session_sid = self._get_target_session_id(session, SlotScope.SESSION)
        
        rows = await self.model_class.filter(
            session_id__in=[global_sid, session_sid], 
            pinned=True
        ).all()
        
        # 合并优先级：Session > Global
        merged = {}
        for row in rows:
            if row.scope == SlotScope.GLOBAL.value:
                merged[row.label] = self._row_to_slot(row)
                
        for row in rows:
            if row.scope == SlotScope.SESSION.value:
                merged[row.label] = self._row_to_slot(row)
                
        return [s for s in merged.values() if s.content.strip()]


def get_orm_slot_context(model_class: type[AbstractSlotRecord]) -> TortoiseSlotContext:
    """
    [工厂方法] 供第三方开发者调用，将 Tortoise ORM 表直接包装为记忆槽存储系统。
    """
    return TortoiseSlotContext(model_class=model_class)
