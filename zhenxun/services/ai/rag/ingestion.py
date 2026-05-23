from abc import ABC, abstractmethod
import asyncio
import re
from typing import Protocol, runtime_checkable

from zhenxun.services.ai.memory.utils import cosine_similarity
from zhenxun.services.ai.rag.models import BaseRecord
from zhenxun.services.log import logger


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, record: BaseRecord) -> list[BaseRecord]:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        cleaned_text = re.sub(r"\n+", "\n", text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        return cleaned_text.strip()

    def _create_chunk_record(
        self, original_record: BaseRecord, chunk_number: int, content: str
    ) -> BaseRecord:
        meta_data = original_record.metadata.copy()
        meta_data["chunk_index"] = chunk_number
        meta_data["chunk_size"] = len(content)
        meta_data["parent_id"] = original_record.id
        return BaseRecord(
            id=f"{original_record.id}_{chunk_number}",
            content=content,
            metadata=meta_data,
        )


class DocumentChunking(ChunkingStrategy):
    """段落语义分块策略 (按双换行切分)"""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, record: BaseRecord) -> list[BaseRecord]:
        if len(record.content) <= self.chunk_size:
            return [
                self._create_chunk_record(record, 0, self.clean_text(record.content))
            ]

        raw_paragraphs = record.content.split("\n\n")
        paragraphs = [self.clean_text(para) for para in raw_paragraphs if para.strip()]

        chunks: list[BaseRecord] = []
        current_chunk_texts = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.chunk_size and current_chunk_texts:
                chunk_content = "\n\n".join(current_chunk_texts)
                chunks.append(
                    self._create_chunk_record(record, chunk_index, chunk_content)
                )
                chunk_index += 1
                current_chunk_texts = []
                current_length = 0

            current_chunk_texts.append(para)
            current_length += para_len + 2

        if current_chunk_texts:
            chunk_content = "\n\n".join(current_chunk_texts)
            chunks.append(self._create_chunk_record(record, chunk_index, chunk_content))

        return chunks


class FixedSizeChunking(ChunkingStrategy):
    """定长分块策略 (带重叠)"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError(f"重叠长度 ({overlap}) 必须小于分块大小 ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, record: BaseRecord) -> list[BaseRecord]:
        content = self.clean_text(record.content)
        content_length = len(content)

        if content_length <= self.chunk_size:
            return [self._create_chunk_record(record, 0, content)]

        chunks: list[BaseRecord] = []
        start = 0
        chunk_index = 0

        while start < content_length:
            end = min(start + self.chunk_size, content_length)

            if end < content_length:
                while end > start and content[end] not in [
                    " ",
                    "\n",
                    "，",
                    "。",
                    "！",
                    "？",
                    ".",
                    "!",
                    "?",
                ]:
                    end -= 1
                if end == start:
                    end = start + self.chunk_size

            chunk_content = content[start:end]
            chunks.append(self._create_chunk_record(record, chunk_index, chunk_content))

            chunk_index += 1
            start = end - self.overlap

        return chunks


class RowChunking(ChunkingStrategy):
    """
    行数据分块策略 (专为 CSV/表格设计)
    核心特性：自动识别表头，并将其附加到每一个被切分的 Chunk 首部，防止上下文丢失。
    """

    def __init__(self, rows_per_chunk: int = 50):
        self.rows_per_chunk = rows_per_chunk

    def chunk(self, record: BaseRecord) -> list[BaseRecord]:
        lines = record.content.splitlines()
        lines = [line for line in lines if line.strip()]

        if not lines:
            return []

        header = lines[0]
        data_lines = lines[1:]

        if not data_lines:
            return [self._create_chunk_record(record, 0, header)]

        chunks: list[BaseRecord] = []
        chunk_index = 0

        for i in range(0, len(data_lines), self.rows_per_chunk):
            chunk_lines = [header, *data_lines[i:i + self.rows_per_chunk]]
            chunk_content = "\n".join(chunk_lines)
            chunks.append(self._create_chunk_record(record, chunk_index, chunk_content))
            chunk_index += 1

        return chunks


class DeduplicationProcessor:
    """
    入库批处理去重器 (Intra-batch Deduplication)。
    在 Chunk 存入数据库前，通过对比向量相似度，拦截高度重复的内容（如群聊复读机内容）。
    """

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not records or len(records) <= 1:
            return records

        kept_records: list[BaseRecord] = []
        dropped_count = 0

        for record in records:
            if not record.embedding:
                kept_records.append(record)
                continue

            is_duplicate = False
            for kept in kept_records:
                if not kept.embedding:
                    continue
                sim = cosine_similarity(record.embedding, kept.embedding)
                if sim >= self.threshold:
                    is_duplicate = True
                    dropped_count += 1
                    break

            if not is_duplicate:
                kept_records.append(record)

        if dropped_count > 0:
            logger.debug(
                f"🧹 [入库管线] 触发批处理去重，已拦截 {dropped_count} "
                f"个高度重复的 Chunk (阈值: {self.threshold})"
            )

        return kept_records


class ScopeInjectionNode:
    """作用域注入节点。针对独立知识库，在管线前端强制将指定前缀注入到元数据中。"""
    def __init__(self, scope_prefix: str):
        self.scope_prefix = scope_prefix

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        for r in records:
            r.metadata["scope"] = self.scope_prefix
        return records


class ConsolidationNode:
    """无状态的数据融合决策节点。在大模型研判下执行旧数据更新/删除。"""
    def __init__(self, storage, consolidator, embedder, threshold: float = 0.85):
        self.storage = storage
        self.consolidator = consolidator
        self.embedder = embedder
        self.threshold = threshold

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        from zhenxun.services.ai.rag.models import QueryRequest
        
        output_records = []
        for record in records:
            if not record.embedding:
                output_records.append(record)
                continue

            scope = record.metadata.get("scope", "/")
            rag_query = QueryRequest(text=record.content, embedding=record.embedding, limit=5)
            rag_results = await self.storage.search(rag_query, scope_prefix=scope)
            similar_records = [res.record for res in rag_results if res.score >= self.threshold]

            plan = await self.consolidator.consolidate(record.content, similar_records)
            to_delete = []
            
            for action in plan.actions:
                if action.action == "delete":
                    to_delete.append(action.record_id)
                elif action.action == "update" and action.new_content:
                    old_record = next((r for r in similar_records if r.id == action.record_id), None)
                    if old_record:
                        old_record.content = action.new_content
                        new_vecs = await self.embedder([action.new_content], task="document")
                        if new_vecs and new_vecs[0]:
                            old_record.embedding = new_vecs[0]
                        await self.storage.update(old_record)

            if to_delete:
                await self.storage.delete(record_ids=to_delete, scope_prefix=scope)

            if plan.insert_new:
                output_records.append(record)

        return output_records


@runtime_checkable
class IngestionNode(Protocol):
    """数据入库管线节点协议"""

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]: ...


class ChunkingNode:
    """分块节点"""

    def __init__(self, strategy):
        self.strategy = strategy

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        chunks = []
        for record in records:
            chunks.extend(self.strategy.chunk(record))
        return chunks


class EmbeddingNode:
    """并发向量化节点"""

    def __init__(self, embedder):
        self.embedder = embedder

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not self.embedder or not records:
            return records

        async def _embed_single(record: BaseRecord):
            if record.content.strip():
                try:
                    vecs = await self.embedder([record.content], task="document")
                    if vecs and vecs[0]:
                        record.embedding = vecs[0]
                except Exception as e:
                    logger.error(f"文档向量化失败 (ID: {record.id}): {e}")
            return record

        embedded = await asyncio.gather(*[_embed_single(r) for r in records])
        return list(embedded)


class DedupNode:
    """批次内查重节点"""

    def __init__(self, threshold: float):
        self.processor = DeduplicationProcessor(threshold=threshold)

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        return await self.processor.process(records)


class StorageWriteNode:
    """持久化落盘节点"""

    def __init__(self, storage):
        self.storage = storage

    async def process(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not records:
            return records
        await self.storage.save(records)
        logger.info(f"💾 成功将 {len(records)} 个知识块落盘。")
        return records


class IngestionPipeline:
    """统一入库流水线"""

    def __init__(self, nodes: list[IngestionNode] | None = None):
        self.nodes = nodes or []

    def add_node(self, node: IngestionNode):
        self.nodes.append(node)

    async def run(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not records:
            return []

        current_records = records
        for node in self.nodes:
            current_records = await node.process(current_records)
            if not current_records:
                break
        return current_records
