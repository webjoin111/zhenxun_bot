from pathlib import Path
from typing import Any

from zhenxun.services.ai.knowledge.base import BaseKnowledge
from zhenxun.services.ai.knowledge.readers import get_reader_for_file
from zhenxun.services.ai.rag import (
    BaseRecord,
    IngestionPipeline,
    KnowledgeScope,
    KnowledgeSlice,
    RowChunking,
)
from zhenxun.services.ai.rag.ingestion import ChunkingNode
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
        "3. **精确过滤**：如果你需要查阅特定范围，可以在 filters 参数中传入 JSON 字典进行精确匹配（如 {'source': 'local_file'}）。\n"
        "4. **基于事实**：必须仅根据检索到的内容回答，严禁编造信息。"
    )

    def __init__(
        self,
        target_scope: KnowledgeScope,
        search_slice: KnowledgeSlice,
        ingestion_pipeline: IngestionPipeline,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.target_scope = target_scope
        self.search_slice = search_slice
        self.ingestion_pipeline = ingestion_pipeline

    async def add_document(self, document: BaseRecord) -> int:
        """
        通过注入的 Ingestion Pipeline 处理并入库文档
        返回成功入库的 Chunk 数量。
        """
        records = await self.ingestion_pipeline.run([document])
        return len(records)

    async def add_file(self, file_path: str | Path) -> int:
        """
        读取并注入单个文件。
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

        import copy

        pipeline = self.ingestion_pipeline
        if path.suffix.lower() == ".csv":
            pipeline = copy.copy(self.ingestion_pipeline)
            pipeline.nodes = list(pipeline.nodes)
            for i, node in enumerate(pipeline.nodes):
                if isinstance(node, ChunkingNode):
                    pipeline.nodes[i] = ChunkingNode(RowChunking(rows_per_chunk=30))
                    break

        records = await pipeline.run([doc])
        return len(records)

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
        description="在语义知识库中搜索最相关的内容片段。可以通过 filters 字典进行额外过滤。",
    )
    async def search_knowledge(
        self, query: str, filters: dict[str, Any] | None = None, limit: int = 5
    ) -> ToolResult:
        results = await self.search_slice.search(
            query=query, limit=limit, metadata_filters=filters
        )

        if not results:
            return ToolResult(output=f"知识库中未找到与 '{query}' 紧密相关的内容。")

        formatted_results = []
        for result in results:
            doc_name = result.record.metadata.get("name", "未命名文档")
            formatted_results.append(
                f"📄 来源: {doc_name} (相关度: {result.score:.2f})\n片段内容:\n{result.record.content}"
            )

        final_text = "\n\n======\n\n".join(formatted_results)
        return ToolResult(output=final_text).with_log(
            f"语义检索 '{query}' 成功召回 {len(results)} 条记录。"
        )
