import time
from typing import Any, Literal

from zhenxun.services.ai.memory.manager import memory_manager
from zhenxun.services.ai.memory.models import (
    MemoryConfig,
)
from zhenxun.services.ai.memory.types import MemorySlot, SessionMetadata, SlotScope
from zhenxun.services.ai.tools.core.decorators import silent, tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult


class MemorySlotToolkit(BaseToolkit):
    """
    中期记忆槽工具箱。
    向大模型开放直接编辑上下文 XML 节点的能力。
    """

    default_instructions = (
        "## 记忆槽管理系统 (Memory Slots)\n"
        "系统已为你提供了类似记事本的记忆槽功能，用于保存关键偏好、人设或当前任务状态。\n"
        "1. 被记录在槽位中的内容会自动以 `<memory_slots>` "
        "的 XML 形式注入你的系统提示词。\n"
        "2. 当用户偏好改变，或者任务状态发生变迁时，"
        "请主动调用 `update_slot` 覆盖对应标签的内容。\n"
        "3. 当你要追加清单项目时，可使用 `append_slot`。"
    )

    def __init__(
        self, session_meta: SessionMetadata, memory_config: MemoryConfig, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.session_meta = session_meta
        self.memory_config = memory_config

    @property
    def _slot_ctx(self):
        return memory_manager.get_slot_context(self.memory_config)

    @tool(description="读取某个尚未展示在上下文中的记忆槽完整内容。")
    @silent()
    async def read_slot(self, label: str) -> ToolResult:
        if not self._slot_ctx:
            return ToolResult(output="错误：未配置记忆槽后端").as_error()
        slot = await self._slot_ctx.get_slot(self.session_meta, label)
        if not slot:
            return ToolResult(output=f"未找到标签为 '{label}' 的槽位。").as_error()
        return ToolResult(output=f"[{label}] 内容:\n{slot.content}")

    @tool(
        description=(
            "更新或新建记忆槽的内容（全量覆写）。"
            "若要保存长期有效的用户画像请将 scope 设为 global。"
        )
    )
    @silent()
    async def update_slot(
        self, label: str, content: str, scope: Literal["session", "global"] = "session"
    ) -> ToolResult:
        if not self._slot_ctx:
            return ToolResult(output="错误：未配置记忆槽后端").as_error()

        slot = await self._slot_ctx.get_slot(self.session_meta, label)
        if not slot:
            slot = MemorySlot(label=label, content=content, scope=SlotScope(scope))
        else:
            slot.content = content
            slot.scope = SlotScope(scope)
            slot.updated_at = time.time()

        if len(content) > slot.size_limit:
            return ToolResult(
                output=f"错误：内容长度超过限制 ({len(content)} > {slot.size_limit})。"
            ).as_error()

        await self._slot_ctx.set_slot(self.session_meta, slot)
        return ToolResult(output=f"已成功将 '{label}' 更新至记忆槽中。")

    @tool(description="在指定记忆槽的末尾追加文本（例如追加待办事项清单）。")
    @silent()
    async def append_slot(self, label: str, text: str) -> ToolResult:
        if not self._slot_ctx:
            return ToolResult(output="错误：未配置记忆槽后端").as_error()

        slot = await self._slot_ctx.get_slot(self.session_meta, label)
        if not slot:
            return ToolResult(
                output=(
                    f"错误：标签为 '{label}' 的槽位不存在，"
                    "请先使用 update_slot 创建。"
                )
            ).as_error()

        sep = "\n" if slot.content and not slot.content.endswith("\n") else ""
        new_content = f"{slot.content}{sep}{text}"

        if len(new_content) > slot.size_limit:
            return ToolResult(
                output=(
                    "错误：追加后总长度超过限制 "
                    f"({len(new_content)} > {slot.size_limit})。"
                )
            ).as_error()

        slot.content = new_content
        slot.updated_at = time.time()
        await self._slot_ctx.set_slot(self.session_meta, slot)

        return ToolResult(output=f"已成功追加至 '{label}'。")
