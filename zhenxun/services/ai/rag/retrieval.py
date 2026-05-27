from abc import abstractmethod
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from zhenxun.services.ai.llm.api import generate_structured
from zhenxun.services.ai.rag.models import QueryRequest, SearchResult
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.rag.backends.storages import StorageBackend


@runtime_checkable
class BaseRetriever(Protocol):
    """
    检索器核心协议。
    任何实现了 retrieve 方法的对象均可作为检索器
    （不仅限于向量检索，也可包含 BM25、SQL 搜索等）。
    """

    @abstractmethod
    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]: ...


@runtime_checkable
class PostProcessor(Protocol):
    """后处理器协议（如重排、时间衰减打分等）。"""

    @abstractmethod
    async def process(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]: ...


@runtime_checkable
class PreProcessor(Protocol):
    """预处理器协议（如 LLM Query 改写、意图提取等）。"""

    @abstractmethod
    async def process(self, query: str) -> list[str]:
        """接收原始查询，返回一个或多个处理/改写后的查询词"""
        ...


class FilterEvaluator:
    """纯 Python 内存求值器，用于为轻量级 Storage 提供字典精确匹配过滤"""

    @classmethod
    def evaluate(
        cls, metadata: dict[str, Any], filter_dict: dict[str, Any] | None
    ) -> bool:
        if filter_dict is None:
            return True
        return all(metadata.get(k) == v for k, v in filter_dict.items())


class VectorDBRetriever(BaseRetriever):
    """基于向量数据库的标准检索器"""

    def __init__(
        self, storage: "StorageBackend", embedder: Any, scope_prefix: str | None = None
    ):
        self.storage = storage
        self.embedder = embedder
        self.scope_prefix = scope_prefix

    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        vecs = await self.embedder([query], task="query")
        query_vec = vecs[0] if vecs else None

        req = QueryRequest(
            text=query,
            embedding=query_vec,
            limit=limit,
            search_type="dense",
            metadata_filters=kwargs.get("metadata_filters"),
        )
        effective_scope = kwargs.get("scope_prefix", self.scope_prefix)
        return await self.storage.search(req, scope_prefix=effective_scope)


class DatabaseSparseRetriever(BaseRetriever):
    """纯数据库下沉的稀疏检索器 (Keyword/FTS)"""

    def __init__(self, storage: "StorageBackend", scope_prefix: str | None = None):
        self.storage = storage
        self.scope_prefix = scope_prefix

    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        req = QueryRequest(
            text=query,
            limit=limit,
            search_type="sparse",
            metadata_filters=kwargs.get("metadata_filters"),
        )
        effective_scope = kwargs.get("scope_prefix", self.scope_prefix)
        return await self.storage.search(req, scope_prefix=effective_scope)


class RerankRetriever(BaseRetriever):
    """带大模型交叉注意力重排的高阶检索器 (Decorator Pattern)"""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str | None = None,
        top_n: int = 5,
    ):
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_n = top_n

    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        oversample_limit = max(limit * 2, 20)
        initial_results = await self.base_retriever.retrieve(
            query, limit=oversample_limit, **kwargs
        )

        if not initial_results:
            return []

        docs: list[str | dict[str, str]] = [
            res.record.content for res in initial_results
        ]

        from zhenxun.services.ai.llm.api import rerank

        try:
            reranked = await rerank(
                query=query,
                documents=docs,
                top_n=min(limit, self.top_n),
                model=self.model_name,
            )
        except Exception as e:
            logger.warning(f"Rerank 重排请求失败，将降级返回初筛结果: {e}")
            return initial_results[:limit]

        final_results = []
        for rr in reranked:
            original_res = initial_results[rr.index]
            original_res.score = rr.relevance_score
            final_results.append(original_res)

        return final_results


class QueryAnalysis(BaseModel):
    """大模型结构化提取查询意图"""

    keywords: list[str] = Field(
        description="提取出1~3个极其简短的搜索短语或名词，严格去除所有客套话、修饰词和标点。"
    )


class LLMQueryRewritePreProcessor(PreProcessor):
    """大模型查询词改写器 (Query Rewriter)"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    async def process(self, query: str) -> list[str]:
        if len(query) < 4:
            return [query]

        try:
            logger.debug(f"🤔 正在使用 LLM 重写口语化查询: '{query}'")
            res = await generate_structured(
                message=f"用户原始提问：{query}\n\n请提取核心搜索词用于向量检索数据库。",
                response_model=QueryAnalysis,
                model=self.model_name,
                instruction="你是一个资深的数据检索架构师。",
            )
            if res.keywords:
                logger.info(f"✨ 搜索词改写成功: '{query}' -> {res.keywords}")
                return res.keywords

            return [query]
        except Exception as e:
            logger.warning(f"Query 改写失败，降级使用原词: {e}")
            return [query]


class StaticSynonymPreProcessor(PreProcessor):
    """零开销静态同义词扩展器"""

    def __init__(self, synonyms: dict[str, list[str]]):
        self.synonyms = synonyms

    async def process(self, query: str) -> list[str]:
        if not self.synonyms or not query.strip():
            return [query]

        queries = [query]
        for key, value_list in self.synonyms.items():
            if key in query:
                for val in value_list:
                    expanded_q = query.replace(key, val)
                    if expanded_q not in queries:
                        queries.append(expanded_q)
                        logger.debug(
                            f"🔄 [同义词扩展] '{query}' -> 扩展查询 '{expanded_q}'"
                        )
        return queries


class PipelineRetriever(BaseRetriever):
    """支持挂载多个后处理器的流水线检索器"""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        post_processors: list[PostProcessor] | None = None,
        pre_processors: list[PreProcessor] | None = None,
    ):
        self.base_retriever = base_retriever
        self.post_processors = post_processors or []
        self.pre_processors = pre_processors or []

    async def retrieve(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[SearchResult]:
        queries_to_search = [query]
        for pp in self.pre_processors:
            queries_to_search = await pp.process(query)

        all_results = []
        seen_ids = set()

        for q in queries_to_search:
            res = await self.base_retriever.retrieve(q, limit=limit * 2, **kwargs)
            for r in res:
                if r.record.id not in seen_ids:
                    seen_ids.add(r.record.id)
                    all_results.append(r)

        results = sorted(all_results, key=lambda x: x.score, reverse=True)

        for pp in self.post_processors:
            results = await pp.process(results, query)

        return results[:limit]


class LifecyclePostProcessor(PostProcessor):
    """生命周期后处理器（融合时间衰减与惰性访问强化）"""

    def __init__(
        self,
        half_life_days: int = 30,
        decay_weight: float = 0.3,
        semantic_weight: float = 0.7,
        importance_weight: float = 0.0,
        reinforcement_weight: float = 0.2,
    ):
        self.half_life_days = half_life_days
        self.decay_weight = decay_weight
        self.semantic_weight = semantic_weight
        self.importance_weight = importance_weight
        self.reinforcement_weight = reinforcement_weight

    async def process(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        now = time.time()
        import math
        for res in results:
            created_at = res.record.metadata.get("created_at", now)
            importance = res.record.metadata.get("importance", 0.5)
            access_count = res.record.metadata.get("access_count", 0)
            last_accessed_at = res.record.metadata.get("last_accessed_at", created_at)
            age_days = max(0.0, (now - last_accessed_at) / 86400.0)
            decay = 0.5 ** (age_days / self.half_life_days)
            access_score = min(1.0, math.log1p(access_count) / 5.0)
            res.score = (
                (self.semantic_weight * res.score)
                + (self.decay_weight * decay)
                + (self.importance_weight * importance)
                + (self.reinforcement_weight * access_score)
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results
