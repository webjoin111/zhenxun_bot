from pathlib import Path

from nonebot.adapters import Bot, Event

from zhenxun.services.ai.knowledge.vector import VectorKnowledge
from zhenxun.services.ai.memory.models import (
    MemoryIsolationLevel,
)
from zhenxun.services.ai.memory.utils import generate_session_meta
from zhenxun.services.ai.rag.backends import DictStorageBackend
from zhenxun.services.ai.rag.builder import RAGBuilder


class SimpleRAG:
    """
    极简小白 RAG 门面 (Facade)。
    提供 0 认知负担的文本/文件入库与语义检索 API。
    无需关心数据库连接、向量化模型和隔离级别策略。
    """

    _global_storage = DictStorageBackend()

    @classmethod
    def _get_kb(cls, event: Event, bot: Bot | None = None, isolation: str = "group"):
        """获取底层挂载了独立沙箱 of VectorKnowledge 实例"""
        iso_level = (
            MemoryIsolationLevel.GROUP_SHARED
            if isolation == "group"
            else MemoryIsolationLevel.USER_GLOBAL
        )

        session_meta = generate_session_meta(
            bot=bot, event=event, isolation_level=iso_level, namespace="simple_rag"
        )

        client = (
            RAGBuilder(cls._global_storage)
            .with_scope(session_meta.accessible_scopes)
            .build()
        )

        return VectorKnowledge(rag_client=client)

    @classmethod
    async def add_text(
        cls,
        text: str,
        event: Event,
        bot: Bot | None = None,
        isolation: str = "group",
        source_name: str | None = None,
    ) -> int:
        """向专属知识库注入纯文本"""
        kb = cls._get_kb(event, bot, isolation)
        from zhenxun.services.ai.rag.models import BaseRecord

        meta = {"source_name": source_name} if source_name else {}
        return await kb.add_document(BaseRecord(content=text, metadata=meta))

    @classmethod
    async def add_file(
        cls,
        file_path: str | Path,
        event: Event,
        bot: Bot | None = None,
        isolation: str = "group",
    ) -> int:
        """读取文件（支持 txt/md/csv 等）并注入专属知识库"""
        kb = cls._get_kb(event, bot, isolation)
        return await kb.add_file(file_path)

    @classmethod
    async def search(
        cls,
        query: str,
        event: Event,
        bot: Bot | None = None,
        isolation: str = "group",
        limit: int = 3,
    ) -> list[str]:
        """进行语义搜索，直接返回纯文本片段列表"""
        kb = cls._get_kb(event, bot, isolation)
        results = await kb.rag_client.search(query, limit=limit)
        return [res.record.content for res in results]

    @classmethod
    def as_tool(cls, event: Event, bot: Bot | None = None, isolation: str = "group"):
        """将当前知识库直接转化为可供 Agent 使用的 FunctionTool"""
        from zhenxun.services.ai.tools.core.tool import FunctionTool
        from zhenxun.services.ai.tools.models import ToolResult

        async def search_knowledge(query: str) -> ToolResult:
            res = await cls.search(query, event, bot, isolation)
            if not res:
                return ToolResult(output="未在知识库中检索到相关信息。")
            return ToolResult(output="\n\n---\n\n".join(res))

        return FunctionTool(
            func=search_knowledge,
            name="search_knowledge",
            description="在专属知识库中进行语义检索。当你需要查阅特定文档或以前存入的背景知识时调用。",
        )
