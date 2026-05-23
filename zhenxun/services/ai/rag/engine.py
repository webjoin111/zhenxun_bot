import asyncio
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zhenxun.services.ai.memory.models import SessionMetadata

from zhenxun.services.ai.rag.backends import RagRegistry, create_storage
from zhenxun.services.ai.rag.backends.storages import StorageBackend
from zhenxun.services.ai.rag.ingestion import (
    ChunkingNode,
    ConsolidationNode,
    DedupNode,
    DocumentChunking,
    EmbeddingNode,
    IngestionPipeline,
    ScopeInjectionNode,
    StorageWriteNode,
)
from zhenxun.services.ai.rag.models import (
    BaseRecord,
    RAGConfig,
    SearchResult,
)
from zhenxun.services.ai.rag.retrieval import (
    BaseRetriever,
    LLMQueryRewritePreProcessor,
    PipelineRetriever,
    RerankRetriever,
    TimeDecayPostProcessor,
    VectorDBRetriever,
)
from zhenxun.services.log import logger


def normalize_scope_path(path: str) -> str:
    """标准化作用域路径，消除多余的斜杠并确保以 / 开头"""
    if not path or path == "/":
        return "/"
    path = re.sub(r"/+", "/", path)
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/")
    return path


class KnowledgeScope:
    """
    单一作用域视图 (Single Scope View)。
    封装了 Storage 和 Retriever，自动将所有读写操作限定在指定的 root_path 前缀下。
    """

    def __init__(
        self, storage: StorageBackend, retriever: BaseRetriever, root_path: str = "/"
    ):
        self.storage = storage
        self.retriever = retriever
        self.root_path = normalize_scope_path(root_path)

    async def save(self, records: list[BaseRecord]) -> None:
        """写入数据时，自动打上当前作用域标签"""
        for r in records:
            r.metadata["scope"] = self.root_path
        await self.storage.save(records)

    async def search(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        """搜索时，自动注入 scope_prefix 进行前缀过滤"""
        kwargs["scope_prefix"] = self.root_path
        return await self.retriever.retrieve(query, limit=limit, **kwargs)

    async def update(self, record: BaseRecord) -> None:
        record.metadata["scope"] = self.root_path
        await self.storage.update(record)

    async def delete(self, record_ids: list[str] | None = None, **kwargs: Any) -> int:
        kwargs["scope_prefix"] = self.root_path
        return await self.storage.delete(record_ids=record_ids, **kwargs)


class KnowledgeSlice:
    """
    多作用域联合切片视图 (Multi-Scope Slice View)。
    只读视图。并发向多个独立的作用域发起检索，并对结果进行合并、去重和重排。
    非常适合群聊场景：同时查询 [全局知识库, 本群知识库, 个人知识库]。
    """

    def __init__(self, retriever: BaseRetriever, scopes: list[str]):
        self.retriever = retriever
        self.scopes = [normalize_scope_path(s) for s in scopes]

    async def search(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        if not self.scopes:
            return []

        oversample_limit = limit * 2
        tasks = []

        for scope in self.scopes:
            call_kwargs = kwargs.copy()
            call_kwargs["scope_prefix"] = scope
            tasks.append(
                self.retriever.retrieve(query, limit=oversample_limit, **call_kwargs)
            )

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for res_list in results_lists:
            if isinstance(res_list, BaseException):
                logger.error(f"[KnowledgeSlice] 并发检索子任务失败: {res_list}")
                continue

            for res in res_list:
                if res.record.id not in seen_ids:
                    seen_ids.add(res.record.id)
                    all_results.append(res)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]


class RAGManager:
    """RAG 管线装配中枢"""

    @staticmethod
    def build_storage(config: RAGConfig | dict) -> StorageBackend:
        if isinstance(config, dict):
            config = RAGConfig(**config)
        return create_storage(config.storage)

    @staticmethod
    def build_retriever(
        config: RAGConfig | dict | None = None,
        storage: StorageBackend | None = None,
        embedder: Any | None = None,
        scope_prefix: str | None = None,
    ) -> BaseRetriever:
        if config is None:
            config = RAGConfig()
        elif isinstance(config, dict):
            config = RAGConfig(**config)

        if not storage:
            storage = RAGManager.build_storage(config)

        from zhenxun.services.ai.llm.manager import get_default_model

        if embedder:
            active_embedder = embedder
        else:
            model_name = config.embedder_model or get_default_model("embedding")
            if not model_name:
                raise RuntimeError(
                    "未配置默认的 Embedding 模型，请在配置文件中设置 default_models.embedding"
                )
            factory = RagRegistry.get_embedder("default")
            if not factory:
                raise RuntimeError("未找到默认的 Embedder 引擎注册")
            active_embedder = factory(model_name)

        retriever: BaseRetriever = VectorDBRetriever(
            storage=storage, embedder=active_embedder, scope_prefix=scope_prefix
        )

        rerank_model = config.rerank_model or get_default_model("rerank")
        if config.use_rerank and rerank_model:
            retriever = RerankRetriever(
                base_retriever=retriever,
                model_name=rerank_model,
                top_n=config.rerank_top_n,
            )

        pre_processors = []
        if config.use_query_rewrite:
            pre_processors.append(
                LLMQueryRewritePreProcessor(
                    model_name=config.query_rewrite_model or config.embedder_model
                )
            )

        post_processors = []
        if config.use_time_decay:
            post_processors.append(
                TimeDecayPostProcessor(half_life_days=config.half_life_days)
            )

        for pp in config.pre_processors:
            if pp.get("type") == "llm_query_rewrite":
                pre_processors.append(
                    LLMQueryRewritePreProcessor(model_name=pp.get("model_name"))
                )

        for pp in config.post_processors:
            if pp.get("type") == "time_decay":
                post_processors.append(
                    TimeDecayPostProcessor(
                        half_life_days=pp.get("half_life_days", 30),
                        decay_weight=pp.get("decay_weight", 0.3),
                        semantic_weight=pp.get("semantic_weight", 0.7),
                    )
                )

        if post_processors or pre_processors:
            return PipelineRetriever(
                base_retriever=retriever,
                post_processors=post_processors,
                pre_processors=pre_processors,
            )

        return retriever

    @staticmethod
    def build_knowledge_base(
        session_meta: "SessionMetadata",
        config: RAGConfig | dict | None = None,
        *,
        storage: StorageBackend | None = None,
        retriever: BaseRetriever | None = None,
        embedder: Any | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
        chunking_strategy: Any | None = None,
    ) -> Any:
        """
        工厂方法：基于会话元数据，组装带有隔离视图的 VectorKnowledge 对象。
        支持纯声明式配置驱动，也支持第三方高级开发者直接注入依赖实例。
        """
        from zhenxun.services.ai.knowledge.vector import VectorKnowledge

        if config is None and not (storage and retriever and ingestion_pipeline):
            raise ValueError(
                "如果不提供 config，"
                "则必须手动注入完整的 storage, retriever, ingestion_pipeline "
                "实例以完成解耦组装。"
            )

        if config is not None:
            if isinstance(config, dict):
                config = RAGConfig(**config)
        else:
            config = RAGConfig()

        active_storage = storage or RAGManager.build_storage(config)

        from zhenxun.services.ai.llm.manager import get_default_model

        if embedder:
            active_embedder = embedder
        else:
            model_name = config.embedder_model or get_default_model("embedding")
            if not model_name:
                raise RuntimeError(
                    "未配置默认的 Embedding 模型，请在配置文件中设置 default_models.embedding"
                )
            factory = RagRegistry.get_embedder("default")
            if not factory:
                raise RuntimeError("未找到默认的 Embedder 引擎注册")
            active_embedder = factory(model_name)

        active_retriever = retriever or RAGManager.build_retriever(
            config, storage=active_storage, embedder=active_embedder
        )

        target_scope = KnowledgeScope(
            active_storage, active_retriever, session_meta.scope_prefix
        )
        search_slice = KnowledgeSlice(active_retriever, session_meta.accessible_scopes)

        if not ingestion_pipeline:
            nodes = [
                ScopeInjectionNode(target_scope.root_path),
                ChunkingNode(chunking_strategy or DocumentChunking(chunk_size=1000)),
                EmbeddingNode(active_embedder),
            ]
            if config.use_dedup:
                nodes.append(DedupNode(threshold=config.dedup_threshold))

            if config.use_consolidation:
                from zhenxun.services.ai.rag.consolidation import LLMConsolidator

                nodes.append(
                    ConsolidationNode(
                        storage=active_storage,
                        consolidator=LLMConsolidator(
                            model_name=config.consolidator_model
                        ),
                        embedder=active_embedder,
                        threshold=config.consolidation_threshold,
                    )
                )

            nodes.append(StorageWriteNode(active_storage))
            active_pipeline = IngestionPipeline(nodes)
        else:
            active_pipeline = ingestion_pipeline

        return VectorKnowledge(
            target_scope=target_scope,
            search_slice=search_slice,
            ingestion_pipeline=active_pipeline,
        )
