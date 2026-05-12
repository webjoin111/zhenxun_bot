from typing import Any

from zhenxun.services.ai.memory.scope import MemoryScope
from zhenxun.services.ai.run import Inject, RunContext
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
        "1. **核心事实记录**：当涉及用户偏好、经历或明确要求记住时，使用 `save_memory`。设定重要性（范围 0.0-1.0）。\n"
        "2. **动态更新**：如果信息失效，先通过 `search_memory` 找到旧记忆，`delete_memory` 后存入新内容。\n"
        "3. **深度记忆**：如果用户提供 URL 要求深入分析并记忆，使用 `read_url_to_memory`。\n"
        "4. **隐式服务**：除非用户询问，否则无需显式汇报『已记住』，直接在后续对话中体现即可。"
    )

    def __init__(self, memory_scope: MemoryScope, **kwargs: Any):
        super().__init__(**kwargs)
        self.memory_scope = memory_scope

    @tool(
        name="save_memory",
        description="保存关于用户的重要设定、偏好或事实到长期记忆中。",
    )
    @silent()
    async def save_memory(self, content: str, importance: float = 0.5) -> ToolResult:
        record = await self.memory_scope.remember(
            content=content, importance=importance
        )

        logger.info(
            f"[Agentic Memory] AI 主动存入记忆: {content} (重要性: {importance})"
        )
        return ToolResult(output=f"记忆已成功保存。唯一ID: {record.id}").with_log(
            f"🧠 主动保存记忆: {content}"
        )

    @tool(
        name="search_memory",
        description="在长期记忆中主动检索关于用户的设定和事实。支持自然语言模糊搜索。可通过 filters 进行元数据的精确匹配（例如：{'source': 'web_reader'}）。",
    )
    @silent()
    async def search_memory(
        self, query: str, filters: dict[str, str] | None = None
    ) -> ToolResult:
        matches = await self.memory_scope.recall(
            query=query, limit=5, metadata_filter=filters
        )
        if not matches:
            return ToolResult(output="未检索到相关记忆。")

        results = [
            f"ID: {m.record.id} | 内容: {m.record.content} | 重要性: {m.record.importance}"
            for m in matches
        ]
        return ToolResult(output="检索到的记忆如下：\n" + "\n".join(results))

    @tool(
        name="delete_memory",
        description="根据记忆的唯一ID删除已经作废或过期的用户记忆。",
    )
    @silent()
    async def delete_memory(self, record_id: str) -> ToolResult:
        deleted_count = await self.memory_scope.forget(record_ids=[record_id])
        if deleted_count > 0:
            logger.info(f"[Agentic Memory] AI 主动删除记忆: {record_id}")
            return ToolResult(output=f"记忆 {record_id} 删除成功。")
        return ToolResult(
            output=f"删除失败：未找到ID为 {record_id} 的记忆。"
        ).as_error()

    @tool(
        name="read_url_to_memory",
        description="读取指定的 URL 网页内容，并将其持久化存入长期记忆数据库中。存入后，你可以在后续任务中通过 search_memory 检索该网页的知识。",
    )
    @silent()
    async def read_url_to_memory(
        self,
        url: str,
        context: RunContext,
        ui: Inject.UI,
        tags: dict[str, str] | None = None,
    ) -> ToolResult:
        await ui.send_text(f"🌐 正在拉取网页以建立长期记忆: {url}...")
        from zhenxun.services.ai.knowledge.readers import get_reader_for_url

        reader = get_reader_for_url(url)
        if not reader:
            return ToolResult(output="该 URL 格式暂不支持读取。").as_error()

        doc = await reader.read_async(url)
        if not doc:
            return ToolResult(
                output="网页拉取或解析失败，目标网站可能设置了防爬虫机制或无法访问。"
            ).as_error()

        meta = {"source": "web_reader", "url": url}
        if tags:
            meta.update(tags)

        record = await self.memory_scope.remember(content=doc.content, metadata=meta)

        logger.info(f"[Agentic Memory] AI 将网页存入记忆: {url}")
        return ToolResult(
            output=f"网页阅读成功并已入库！提取了 {len(doc.content)} 个字符。记忆 ID: {record.id}"
        ).with_log(f"🌐 已将网页存入长期记忆: {url}")
