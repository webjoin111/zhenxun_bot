from typing import Any

from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.ai.memory.storage import MemoryScope
from zhenxun.services.ai.tools.core.decorators import silent, tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger


class MemoryManagementToolkit(BaseToolkit):
    """
    主动记忆管理工具箱 (Agentic Memory Toolkit)。
    赋予大模型自主存取、修改和删除用户长期设定的能力。
    """

    default_instructions = (
        "## 长期记忆管理系统 (LTM)\n"
        "你拥有管理用户个人设定和长期记忆的权限。请遵循以下规则：\n"
        "1. **记录新事实**：当涉及用户偏好、"
        "经历或明确要求记住新事物时，使用 `save_memory`。\n"
        "2. **更新旧事实**：如果发现用户设定的信息发生了改变"
        "（例如名字改了、喜欢的东西变了），"
        "请先通过 `search_memory` 找到旧记忆的 ID，"
        "然后直接使用 `update_memory` 覆盖。\n"
        "3. **深度记忆**：如果用户提供 URL 要求深入分析并记忆，"
        "使用 `read_url_to_memory`。\n"
        "4. **隐式服务**：除非用户询问，否则无需显式汇报『已记住』，"
        "直接在后续对话中体现即可。"
    )

    def __init__(
        self, memory_scope: MemoryScope, session_meta: SessionMetadata, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.memory_scope = memory_scope
        self.session_meta = session_meta

    @tool(
        name="save_memory",
        description="保存关于用户的重要设定、偏好或事实到长期记忆中。",
    )
    @silent()
    async def save_memory(self, content: str, importance: float = 0.5) -> ToolResult:
        await self.memory_scope.remember(
            session=self.session_meta, content=content, importance=importance
        )

        logger.info(
            f"[Agentic Memory] AI 主动存入记忆: {content} (重要性: {importance})"
        )
        return ToolResult(output="记忆已成功排入系统后台合并存入。").with_log(
            f"🧠 主动保存记忆: {content}"
        )

    @tool(
        name="search_memory",
        description=(
            "在长期记忆中主动检索关于用户的设定和事实。"
            "支持自然语言模糊搜索。"
            "可通过 filters 进行元数据的精确匹配"
            "（例如：{'source': 'web_reader'}）。"
        ),
    )
    @silent()
    async def search_memory(
        self, query: str, filters: dict[str, str] | None = None
    ) -> ToolResult:
        matches = await self.memory_scope.recall(
            session=self.session_meta, query=query, limit=5, metadata_filter=filters
        )
        if not matches:
            return ToolResult(output="未检索到相关记忆。")

        results = [
            (
                f"ID: {m.record.id} | 内容: {m.record.content} | "
                f"重要性: {m.record.metadata.get('importance', 0.5)}"
            )
            for m in matches
        ]
        return ToolResult(output="检索到的记忆如下：\n" + "\n".join(results))

    @tool(
        name="update_memory",
        description="更新指定ID的长期记忆内容。当你发现用户的某个旧设定发生改变时，请使用此工具覆盖旧记忆。",
    )
    @silent()
    async def update_memory(
        self, record_id: str, new_content: str, importance: float = 0.5
    ) -> ToolResult:
        success = await self.memory_scope.update(
            session=self.session_meta,
            record_id=record_id,
            new_content=new_content,
            importance=importance,
        )
        if success:
            logger.info(
                f"[Agentic Memory] AI 主动更新记忆: {record_id} -> {new_content}"
            )
            return ToolResult(
                output=f"记忆 {record_id} 更新成功。新内容：{new_content}"
            )
        return ToolResult(
            output=f"更新失败：未找到ID为 {record_id} 的记忆。"
        ).as_error()

    @tool(
        name="delete_memory",
        description="根据记忆的唯一ID删除已经作废或过期的用户记忆。",
    )
    @silent()
    async def delete_memory(self, record_id: str) -> ToolResult:
        deleted_count = await self.memory_scope.forget(
            session=self.session_meta, record_ids=[record_id]
        )
        if deleted_count > 0:
            logger.info(f"[Agentic Memory] AI 主动删除记忆: {record_id}")
            return ToolResult(output=f"记忆 {record_id} 删除成功。")
        return ToolResult(
            output=f"删除失败：未找到ID为 {record_id} 的记忆。"
        ).as_error()
