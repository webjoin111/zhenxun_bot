from collections.abc import Callable
import time
from typing import Any

from nonebot.utils import is_coroutine_callable
from pydantic import TypeAdapter
from tortoise import fields

from zhenxun.services.ai.memory.working_memory import ChatWorkingMemory
from zhenxun.services.ai.protocols.memory import BaseMessageStore, SessionMetadata
from zhenxun.services.ai.types.messages import (
    LLMContentPart, 
    LLMMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from zhenxun.services.db_context import Model
from zhenxun.utils.pydantic_compat import model_dump


class AbstractMemoryRecord(Model):
    """
    Tortoise ORM 短期记忆持久化基类 (Mixin)。
    第三方插件开发者继承此基类并指定 table_name 即可拥有持久化工作记忆能力。

    提供了 LLMMessage 与数据库互转所需的所有核心字段。
    """

    id = fields.UUIDField(pk=True, description="主键")
    session_id = fields.CharField(
        max_length=255, index=True, description="会话作用域路径 (Scope Path)"
    )
    role = fields.CharField(
        max_length=32, description="消息角色 (user/assistant/system/tool)"
    )
    content = fields.JSONField(description="序列化后的 LLMContentPart 列表或文本")

    api_context = fields.JSONField(
        null=True,
        description="API底层通信凭证集合 (包含 tool_calls, thought_signature 等)",
    )

    created_at = fields.DatetimeField(auto_now_add=True, description="消息创建时间")
    metadata = fields.JSONField(
        null=True, description="附加元数据，供开发者自由存取非索引数据"
    )

    class Meta:  # type: ignore
        abstract = True


class TortoiseMessageStore(BaseMessageStore):
    """Tortoise ORM 的工作记忆存储适配器"""

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
        """将 ORM 记录转换为标准的 LLMMessage"""
        content_raw = row.content

        api_context_raw = row.api_context
        api_context: dict[str, Any] = (
            api_context_raw if isinstance(api_context_raw, dict) else {}
        )

        global_thought_sig = api_context.get("thought_signature")

        from zhenxun.services.ai.types.messages import TextPart
        content_parts: list[LLMContentPart] = []
        if isinstance(content_raw, list):
            adapter = TypeAdapter(LLMContentPart)
            for p in content_raw:
                if not isinstance(p, dict):
                    continue
                if global_thought_sig:
                    p.setdefault("metadata", {})
                    p["metadata"]["thought_signature"] = global_thought_sig
                content_parts.append(adapter.validate_python(p))
        elif isinstance(content_raw, str):
            content_parts.append(TextPart(text=content_raw))

        metadata: dict[str, Any] | None = None
        if isinstance(row.metadata, dict):
            metadata = row.metadata

        kwargs = {
            "content": content_parts,
            "metadata": metadata,
            "created_at": row.created_at.timestamp() if row.created_at else time.time(),
        }
        
        role = row.role
        if role == "system":
            msg = SystemMessage(**kwargs)
        elif role == "user":
            msg = UserMessage(**kwargs)
        elif role == "assistant":
            msg = AssistantMessage(**kwargs)
        elif role == "tool":
            msg = ToolMessage(**kwargs)
        else:
            msg = LLMMessage(role=role, **kwargs)
        if api_context.get("thought_signature"):
            msg.thought_signature = api_context.get("thought_signature")
        return msg

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
                from zhenxun.services.ai.types.messages import ThoughtPart
                for p in content_payload:
                    if isinstance(p, ThoughtPart):
                        continue
                        
                    p_dump = (
                        model_dump(p, exclude_none=True)
                        if hasattr(p, "model_dump")
                        else (p.copy() if isinstance(p, dict) else p)
                    )

                    if isinstance(p_dump, dict):
                        if "metadata" in p_dump and isinstance(
                            p_dump["metadata"], dict
                        ):
                            allowed_types = (
                                "server_tool_call",
                                "server_tool_response",
                                "executable_code",
                                "execution_result",
                            )
                            if p_dump.get("type") not in allowed_types:
                                p_dump["metadata"].pop("thought_signature", None)
                                if not p_dump["metadata"]:
                                    p_dump.pop("metadata")

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

            api_context = {}
            if msg.thought_signature:
                api_context["thought_signature"] = msg.thought_signature

            orm_obj = self.model_class(
                session_id=session.session_id,
                role=msg.role,
                content=content_payload,
                api_context=api_context if api_context else None,
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
) -> ChatWorkingMemory:
    """
    [工厂方法] 供第三方开发者调用，将 Tortoise ORM 表直接包装为带有滚动窗口截断能力的工作记忆系统。

    参数:
        model_class: 继承了 AbstractMemoryRecord 的 Tortoise 模型。
        max_messages: 滚动窗口保留的最大消息数。
        custom_save_hook: 在持久化入库前触发的回调，允许开发者为自定义的冗余字段动态赋值。
    """
    store = TortoiseMessageStore(
        model_class=model_class, custom_save_hook=custom_save_hook
    )
    return ChatWorkingMemory(store=store, max_messages=max_messages)
