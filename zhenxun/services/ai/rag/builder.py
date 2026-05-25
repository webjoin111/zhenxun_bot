from typing import Any

from zhenxun.services.ai.rag.backends import StorageBackend
from zhenxun.services.ai.rag.engine import ScopedRAGClient
from zhenxun.services.ai.rag.indexing import (
    ChunkingStrategy,
    ConsolidationNode,
    DedupNode,
    DocumentChunking,
    DynamicChunkingNode,
    EmbeddingNode,
    IndexPipeline,
    ScopeInjectionNode,
    StorageCommitNode,
    UpdateEmbeddingNode,
)
from zhenxun.services.ai.rag.models import RAGConfig
from zhenxun.services.ai.rag.retrieval import (
    BaseRetriever,
    LLMQueryRewritePreProcessor,
    PipelineRetriever,
    PostProcessor,
    PreProcessor,
    RerankRetriever,
    TimeDecayPostProcessor,
    VectorDBRetriever,
)
from zhenxun.services.log import logger


class RAGBuilder:
    """
    RAG 管线组装构建器 (Fluent Builder Pattern)。
    使用内部状态驱动模式 (The Memory Pattern)，内部维护私有的 RAGConfig 实例。
    """

    def __init__(
        self, storage: StorageBackend | None = None, config: RAGConfig | None = None
    ):
        self._config = config or RAGConfig()
        if storage is not None:
            self._config.storage = storage

    def with_embedder(self, embedder: Any) -> "RAGBuilder":
        """设置向量化引擎"""
        self._config.embedder = embedder
        return self

    def with_scope(self, scopes: str | list[str]) -> "RAGBuilder":
        """设置数据隔离作用域(支持单作用域或联合检索多作用域)"""
        self._config.scopes = scopes
        return self

    def with_chunking(self, strategy: ChunkingStrategy) -> "RAGBuilder":
        """设置文档切块策略"""
        self._config.chunking.strategy = strategy
        return self

    def enable_dedup(self, threshold: float = 0.98) -> "RAGBuilder":
        """开启批处理入库去重"""
        self._config.dedup.enable = True
        self._config.dedup.threshold = threshold
        return self

    def disable_dedup(self) -> "RAGBuilder":
        """关闭批处理入库去重"""
        self._config.dedup.enable = False
        return self

    def enable_consolidation(
        self, consolidator: Any, threshold: float = 0.85
    ) -> "RAGBuilder":
        """开启大模型记忆融合与反思"""
        self._config.consolidation.consolidator = consolidator
        self._config.consolidation.threshold = threshold
        return self

    def enable_rerank(self, model_name: str, top_n: int = 5) -> "RAGBuilder":
        """开启大模型交叉注意力重排"""
        self._config.rerank.enable = True
        self._config.rerank.model_name = model_name
        self._config.rerank.top_n = top_n
        return self

    def enable_query_rewrite(self, model_name: str) -> "RAGBuilder":
        """开启大模型查询意图重写"""
        self._config.query_rewrite.enable = True
        self._config.query_rewrite.model_name = model_name
        return self

    def enable_time_decay(
        self,
        half_life_days: int = 30,
        decay_weight: float = 0.3,
        semantic_weight: float = 0.7,
        importance_weight: float = 0.0,
    ) -> "RAGBuilder":
        """开启时间衰减打分后处理"""
        self._config.time_decay.enable = True
        self._config.time_decay.half_life_days = half_life_days
        self._config.time_decay.decay_weight = decay_weight
        self._config.time_decay.semantic_weight = semantic_weight
        self._config.time_decay.importance_weight = importance_weight
        return self

    def add_pre_processor(self, processor: PreProcessor) -> "RAGBuilder":
        """挂载自定义查询前处理器"""
        self._config.pre_processors.append(processor)
        return self

    def add_post_processor(self, processor: PostProcessor) -> "RAGBuilder":
        """挂载自定义检索后处理器"""
        self._config.post_processors.append(processor)
        return self

    @classmethod
    def resolve(cls, config: RAGConfig | dict | Any | None) -> RAGConfig:
        if isinstance(config, RAGConfig):
            return config
        if isinstance(config, dict):
            return RAGConfig(**config)
        return RAGConfig()

    def build(self) -> ScopedRAGClient:
        """完成所有积木组装，输出最终的客户端实体"""
        cfg = self._config
        storage = cfg.storage
        if not storage:
            raise ValueError("RAGBuilder 必须配置 storage 才能 build。")

        embedder = cfg.embedder
        if not embedder:
            from zhenxun.services.ai.llm.manager import get_default_model
            from zhenxun.services.ai.rag.backends import DefaultEmbedder

            embedder = DefaultEmbedder(model_name=get_default_model("embedding"))
            logger.debug("RAGBuilder: 未指定 Embedder，已使用系统默认 Embedder。")

        chunking_strategy = cfg.chunking.strategy or DocumentChunking()

        scopes = cfg.scopes
        nodes = [
            ScopeInjectionNode(scopes[0] if isinstance(scopes, list) else scopes),
            DynamicChunkingNode(chunking_strategy),
            EmbeddingNode(embedder),
        ]
        if cfg.dedup.enable:
            nodes.append(DedupNode(cfg.dedup.threshold))
        if cfg.consolidation.consolidator:
            nodes.append(
                ConsolidationNode(
                    storage, cfg.consolidation.consolidator, cfg.consolidation.threshold
                )
            )
            nodes.append(UpdateEmbeddingNode(embedder))

        nodes.append(StorageCommitNode(storage))
        pipeline = IndexPipeline(nodes)

        base_retriever: BaseRetriever = VectorDBRetriever(
            storage,
            embedder,
            scopes[0] if isinstance(scopes, list) else scopes,
        )

        if cfg.rerank.enable and cfg.rerank.model_name:
            base_retriever = RerankRetriever(
                base_retriever, cfg.rerank.model_name, cfg.rerank.top_n
            )

        pre_processors = list(cfg.pre_processors)
        if cfg.query_rewrite.enable and cfg.query_rewrite.model_name:
            pre_processors.append(
                LLMQueryRewritePreProcessor(model_name=cfg.query_rewrite.model_name)
            )

        post_processors = list(cfg.post_processors)
        if cfg.time_decay.enable:
            post_processors.append(
                TimeDecayPostProcessor(
                    half_life_days=cfg.time_decay.half_life_days,
                    decay_weight=cfg.time_decay.decay_weight,
                    semantic_weight=cfg.time_decay.semantic_weight,
                    importance_weight=cfg.time_decay.importance_weight,
                )
            )

        if pre_processors or post_processors:
            base_retriever = PipelineRetriever(
                base_retriever,
                pre_processors=pre_processors,
                post_processors=post_processors,
            )

        return ScopedRAGClient(
            storage=storage,
            retriever=base_retriever,
            pipeline=pipeline,
            scopes=scopes,
        )
