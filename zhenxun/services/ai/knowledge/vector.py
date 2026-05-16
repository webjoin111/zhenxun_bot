import asyncio
from pathlib import Path
from typing import Any

from zhenxun.services.ai.knowledge.base import BaseKnowledge
from zhenxun.services.ai.knowledge.chunking.document import DocumentChunking
from zhenxun.services.ai.knowledge.chunking.row import RowChunking
from zhenxun.services.ai.knowledge.chunking.strategy import ChunkingStrategy
from zhenxun.services.ai.knowledge.models import Document
from zhenxun.services.ai.knowledge.readers import get_reader_for_file
from zhenxun.services.ai.llm.api import embed
from zhenxun.services.ai.memory.models import MemoryRecord
from zhenxun.services.ai.memory.interfaces import StorageBackend
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger


class VectorKnowledge(BaseKnowledge):
    """
    原生语义向量知识库。
    将长文档切分、向量化并存入关系型/向量数据库，向大模型提供语义检索 (Semantic Search) 工具。
    """

    default_instructions = (
        "## 语义知识库\n"
        "你拥有访问外部语义向量知识库的权限。请遵循以下规则：\n"
        "1. **优先检索**：在回答专业或背景问题时，务必使用 `search_knowledge` 工具。\n"
        "2. **语义搜索**：你可以直接输入完整的问题或描述作为检索词，系统会自动进行语义匹配。\n"
        "3. **基于事实**：必须仅根据检索到的内容回答，严禁编造信息。"
    )

    def __init__(
        self,
        storage: StorageBackend,
        embedding_model: str | None = None,
        chunking_strategy: ChunkingStrategy | None = None,
        scope_prefix: str = "/knowledge",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.storage = storage
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy or DocumentChunking(chunk_size=1000)
        self.scope_prefix = scope_prefix

    async def add_document(self, document: Document) -> int:
        """
        文档注入流水线：分块 -> 并发向量化 -> 转换为 MemoryRecord -> 持久化入库
        返回成功入库的 Chunk 数量。
        """
        chunks = self.chunking_strategy.chunk(document)
        if not chunks:
            return 0

        logger.info(
            f"[VectorKnowledge] 文档 '{document.name}' 已切分为 {len(chunks)} 个 Chunk，开始向量化..."
        )

        async def process_chunk(chunk: Document):
            if self.embedding_model:
                from zhenxun.services.ai.llm.api import embed

                if chunk.content.strip():
                    try:
                        res = await embed(
                            chunk.content, task="document", model=self.embedding_model
                        )
                        if res.vector:
                            chunk.embedding = res.vector
                    except Exception as e:
                        logger.error(f"文档向量化失败 (ID: {chunk.id}): {e}")
            return chunk

        embedded_chunks = await asyncio.gather(*[process_chunk(c) for c in chunks])

        records = []
        for chunk in embedded_chunks:
            record = MemoryRecord(
                id=chunk.id or "",
                content=chunk.content,
                scope=self.scope_prefix,
                importance=0.5,
                embedding=chunk.embedding,
                metadata=chunk.meta_data,
            )
            records.append(record)

        await self.storage.save(records)
        logger.info(
            f"[VectorKnowledge] 成功将 {len(records)} 个知识 Chunk 存入底座数据库。"
        )
        return len(records)

    async def add_file(self, file_path: str | Path) -> int:
        """
        读取并注入单个文件。
        根据文件类型自动调整切块策略（CSV走RowChunking，其他走默认策略）。
        """
        path = Path(file_path)
        if not path.is_file():
            logger.error(f"[VectorKnowledge] 文件不存在: {path}")
            return 0

        reader = get_reader_for_file(path)
        if not reader:
            return 0

        doc = reader.read(path)
        if not doc:
            return 0

        original_strategy = self.chunking_strategy
        if path.suffix.lower() == ".csv":
            self.chunking_strategy = RowChunking(rows_per_chunk=30)

        added_count = await self.add_document(doc)

        self.chunking_strategy = original_strategy
        return added_count

    async def add_directory(self, dir_path: str | Path) -> int:
        """扫描目录并注入所有支持的文件"""
        total_chunks = 0
        path = Path(dir_path)
        for p in path.rglob("*"):
            if p.is_file():
                total_chunks += await self.add_file(p)
        return total_chunks

    @tool(
        name="search_knowledge",
        description="在语义知识库中搜索与输入问题最相关的内容片段。",
    )
    async def search_knowledge(self, query: str, limit: int = 5) -> ToolResult:
        if not self.embedding_model:
            query_vector = []
        else:
            try:
                query_vector = (
                    await embed(query, task="query", model=self.embedding_model)
                ).vector
            except Exception as e:
                logger.error(f"[VectorKnowledge] 检索词向量化失败: {e}")
                return ToolResult(output="检索词向量化失败，请稍后再试。").as_error()

        results = await self.storage.search(
            query_embedding=query_vector, scope_prefix=self.scope_prefix, limit=limit
        )

        if not results:
            return ToolResult(output=f"知识库中未找到与 '{query}' 紧密相关的内容。")

        formatted_results = []
        for record, score in results:
            doc_name = record.metadata.get("name", "未命名文档")
            formatted_results.append(
                f"📄 来源: {doc_name} (相关度: {score:.2f})\n片段内容:\n{record.content}"
            )

        final_text = "\n\n======\n\n".join(formatted_results)
        return ToolResult(output=final_text).with_log(
            f"语义检索 '{query}' 成功召回 {len(results)} 条记录。"
        )
