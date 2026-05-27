from abc import ABC, abstractmethod
import asyncio
import re

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
                original_end = end
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

                if end <= start + self.overlap:
                    end = original_end

            chunk_content = content[start:end]
            chunks.append(self._create_chunk_record(record, chunk_index, chunk_content))

            chunk_index += 1
            start = max(start + 1, end - self.overlap)

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
            chunk_lines = [header, *data_lines[i : i + self.rows_per_chunk]]
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


class BaseBatchNode(ABC):
    """批处理节点基类：一次性接收并处理全部记录"""

    @abstractmethod
    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]: ...


class BaseMapNode(ABC):
    """单映射节点基类：接收单条记录，引擎负责并发调度，返回None代表丢弃该数据"""

    @abstractmethod
    async def process_one(
        self, record: BaseRecord
    ) -> BaseRecord | list[BaseRecord] | None: ...


class ScopeInjectionNode(BaseBatchNode):
    """作用域注入节点。针对独立知识库，在管线前端强制将指定前缀注入到元数据中。"""

    def __init__(self, scope_prefix: str):
        self.scope_prefix = scope_prefix

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        for r in records:
            r.metadata["scope"] = self.scope_prefix
        return records


class DynamicChunkingNode(BaseBatchNode):
    """智能路由切块节点。根据记录的扩展名动态选择切块策略。"""

    def __init__(self, default_strategy: ChunkingStrategy):
        self.default_strategy = default_strategy
        self.strategies = {".csv": RowChunking(rows_per_chunk=30)}

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        chunks = []
        for record in records:
            ext = record.metadata.get("extension", "")
            strategy = self.strategies.get(ext, self.default_strategy)
            chunks.extend(strategy.chunk(record))
        return chunks


class ConsolidationNode(BaseMapNode):
    """无状态的数据融合决策节点 (Map)。请求大模型生成合并/删除/插入意图。"""

    def __init__(self, storage, consolidator, threshold: float = 0.85):
        self.storage = storage
        self.consolidator = consolidator
        self.threshold = threshold

    async def process_one(self, record: BaseRecord) -> list[BaseRecord]:
        if not record.embedding:
            return [record]

        scope = record.metadata.get("scope", "/")
        from zhenxun.services.ai.rag.models import QueryRequest

        rag_query = QueryRequest(
            text=record.content, embedding=record.embedding, limit=5
        )
        rag_results = await self.storage.search(rag_query, scope_prefix=scope)
        similar_records = [
            res.record for res in rag_results if res.score >= self.threshold
        ]

        plan = await self.consolidator.consolidate(record.content, similar_records)

        results = []
        for action in plan.actions:
            if action.action == "delete":
                results.append(
                    BaseRecord(id=action.record_id, content="", action="delete")
                )
            elif action.action == "update" and action.new_content:
                old_record = next(
                    (r for r in similar_records if r.id == action.record_id), None
                )
                if old_record:
                    old_record.content = action.new_content
                    old_record.action = "update"
                    results.append(old_record)

        if plan.insert_new:
            record.action = "insert"
            results.append(record)
        else:
            record.action = "ignore"
            results.append(record)

        return results


class EmbeddingNode(BaseBatchNode):
    """并发向量化节点"""

    def __init__(self, embedder):
        self.embedder = embedder

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not self.embedder or not records:
            return records

        texts_to_embed = [r.content for r in records if r.content.strip()]
        if not texts_to_embed:
            return records

        try:
            vecs = []
            batch_size = 80
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i : i + batch_size]
                batch_vecs = await self.embedder(batch_texts, task="document")
                vecs.extend(batch_vecs)

            vec_idx = 0
            for r in records:
                if r.content.strip():
                    if vecs and vec_idx < len(vecs) and vecs[vec_idx]:
                        r.embedding = vecs[vec_idx]
                    vec_idx += 1
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")

        return records


class UpdateEmbeddingNode(BaseBatchNode):
    """
    更新向量化节点。
    专门负责为被 Consolidator 融合更新 (action == 'update') 的记录重新生成 Embedding。
    彻底解耦模型推理与数据库提交。
    """

    def __init__(self, embedder):
        self.embedder = embedder

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not self.embedder or not records:
            return records

        records_to_re_embed = [
            r for r in records if r.action == "update" and r.content.strip()
        ]
        if not records_to_re_embed:
            return records

        texts = [r.content for r in records_to_re_embed]
        try:
            vecs = []
            batch_size = 80
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_vecs = await self.embedder(batch_texts, task="document")
                vecs.extend(batch_vecs)

            for i, r in enumerate(records_to_re_embed):
                if vecs and i < len(vecs) and vecs[i]:
                    r.embedding = vecs[i]
        except Exception as e:
            logger.error(f"融合更新记录时重新向量化失败: {e}")

        return records


class DedupNode(BaseBatchNode):
    """批次内查重节点"""

    def __init__(self, threshold: float):
        self.processor = DeduplicationProcessor(threshold=threshold)

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        return await self.processor.process(records)


class StorageCommitNode(BaseBatchNode):
    """持久化事务提交节点 (Reduce)。统一收集意图并执行并发数据库 I/O。"""

    def __init__(self, storage):
        self.storage = storage

    async def process_batch(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not records:
            return records

        to_delete = set()
        to_update = []
        to_insert = []

        for record in records:
            if record.action == "delete":
                to_delete.add(record.id)
            elif record.action == "update":
                to_update.append(record)
            elif record.action == "insert":
                to_insert.append(record)

        if to_delete:
            await self.storage.delete(record_ids=list(to_delete))

        if to_update:
            await asyncio.gather(*[self.storage.update(r) for r in to_update])

        if to_insert:
            await self.storage.save(to_insert)

        logger.info(
            "💾 RAG 事务提交完成：插入 "
            f"{len(to_insert)} 条, 更新 {len(to_update)} 条, "
            f"删除 {len(to_delete)} 条。"
        )
        return to_insert + to_update


class IndexPipeline:
    """统一入库流水线 (Map-Reduce 范式并发调度引擎)"""

    def __init__(
        self,
        nodes: list[BaseBatchNode | BaseMapNode] | None = None,
        max_workers: int = 5,
    ):
        self.nodes = nodes or []
        self.max_workers = max_workers

    def add_node(self, node: BaseBatchNode | BaseMapNode):
        self.nodes.append(node)

    async def run(self, records: list[BaseRecord]) -> list[BaseRecord]:
        if not records:
            return []

        current_records = records
        for node in self.nodes:
            if not current_records:
                break

            if isinstance(node, BaseBatchNode):
                current_records = await node.process_batch(current_records)
            elif isinstance(node, BaseMapNode):
                sem = asyncio.Semaphore(self.max_workers)
                map_node: BaseMapNode = node

                async def _process_with_sem(r: BaseRecord):
                    async with sem:
                        return await map_node.process_one(r)

                tasks = [_process_with_sem(r) for r in current_records]
                results = await asyncio.gather(*tasks)

                next_records = []
                for r in results:
                    if isinstance(r, list):
                        next_records.extend(r)
                    elif r is not None:
                        next_records.append(r)
                current_records = next_records
            else:
                raise ValueError(f"未知的管道节点类型: {type(node)}")

        return current_records
