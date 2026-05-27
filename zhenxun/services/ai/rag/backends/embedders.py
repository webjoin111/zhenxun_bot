import asyncio
from typing import Literal, Protocol

from zhenxun.services.ai.llm.api import embed as api_embed
from zhenxun.services.log import logger

EmbedTaskType = Literal[
    "general", "query", "document", "similarity", "classification", "clustering"
]


class Embedder(Protocol):
    """
    向量化引擎协议 (Callable Protocol)。
    任何实现了异步 __call__ 的对象或闭包函数均可作为 Embedder。
    """

    async def __call__(
        self, texts: list[str], task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        """
        将文本列表转换为向量列表。
        """
        ...


class DefaultEmbedder(Embedder):
    """系统默认的向量化引擎，调用大模型底座 API"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    async def __call__(
        self, texts: list[str], task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        if not texts:
            return []
        try:
            res = await api_embed(texts, model=self.model_name, task=task)
            return res.embeddings
        except Exception as e:
            logger.error(f"DefaultEmbedder 向量化失败: {e}", e=e)
            return [[] for _ in texts]


class FastEmbedder(Embedder):
    """
    基于 FastEmbed 的轻量级本地向量化引擎。
    零 PyTorch 依赖，CPU 推理极快。
    """

    def __init__(self, model_name: str | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "⚠️ 使用 FastEmbed 需要额外依赖，请在终端执行: pip install fastembed"
            )

        self.model_name = model_name or "BAAI/bge-small-zh-v1.5"
        logger.info(
            f"正在加载 FastEmbed 本地模型: {self.model_name} ... (首次加载可能需要下载)"
        )
        self.model = TextEmbedding(model_name=self.model_name)

    async def __call__(
        self, texts: list[str], task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        if not texts:
            return []

        def _sync_embed():
            return [vec.tolist() for vec in self.model.embed(texts)]

        return await asyncio.to_thread(_sync_embed)


class SentenceTransformerEmbedder(Embedder):
    """
    基于 Sentence-Transformers 的本地向量化引擎。
    支持 GPU 加速，适合重度用户。
    """

    def __init__(self, model_name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError(
                "⚠️ 使用 SentenceTransformers 需要额外依赖，"
                "请在终端执行: pip install sentence-transformers"
            )

        self.model_name = model_name or "BAAI/bge-small-zh-v1.5"
        logger.info(f"正在加载 SentenceTransformer 本地模型: {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)

    async def __call__(
        self, texts: list[str], task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        if not texts:
            return []

        def _sync_embed():
            embeddings = self.model.encode(texts)
            return embeddings.tolist()

        return await asyncio.to_thread(_sync_embed)
