from abc import ABC, abstractmethod
import asyncio
from typing import Any, Literal, Protocol, runtime_checkable

from zhenxun.services.ai.core.messages import EmbedBatch
from zhenxun.services.ai.llm.api import embed as api_embed
from zhenxun.services.ai.message_builder import MessageBuilder
from zhenxun.services.log import logger

EmbedTaskType = Literal[
    "general", "query", "document", "similarity", "classification", "clustering"
]


@runtime_checkable
class Embedder(Protocol):
    """
    向量化引擎协议 (Callable Protocol)。
    任何实现了异步 __call__ 的对象或闭包函数均可作为 Embedder。
    """

    async def __call__(
        self, input_batch: Any, task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        """
        将文本、多模态或预构建的 EmbedBatch 转换为向量列表。
        """
        ...


class DefaultEmbedder(Embedder):
    """系统默认的向量化引擎，调用大模型底座 API"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    async def __call__(
        self, input_batch: Any, task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        if not input_batch:
            return []
        try:
            res = await api_embed(input_batch, model=self.model_name, task=task)
            return res.embeddings
        except Exception as e:
            logger.error(f"DefaultEmbedder 向量化失败: {e}", e=e)
            return []


class BaseLocalEmbedder(Embedder, ABC):
    """本地向量化引擎基类，统一处理多模态降级与同步推理由协程包裹逻辑。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        """子类只需实现此同步的批量文本向量化方法即可。"""
        pass

    async def __call__(
        self, input_batch: Any, task: EmbedTaskType = "general", **kwargs
    ) -> list[list[float]]:
        if not input_batch:
            return []

        if isinstance(input_batch, EmbedBatch):
            batch = input_batch
        else:
            batch = await MessageBuilder.normalize_to_embed_batch(input_batch)

        texts = batch.to_text_only(f"本地模型 {self.model_name}")

        if not texts:
            return []

        def _sync_embed():
            return self._encode_texts(texts)

        return await asyncio.to_thread(_sync_embed)


class FastEmbedder(BaseLocalEmbedder):
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

        super().__init__(model_name or "BAAI/bge-small-zh-v1.5")
        logger.info(
            f"正在加载 FastEmbed 本地模型: {self.model_name} ... (首次加载可能需要下载)"
        )
        self.model = TextEmbedding(model_name=self.model_name)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.model.embed(texts)]


class SentenceTransformerEmbedder(BaseLocalEmbedder):
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

        super().__init__(model_name or "BAAI/bge-small-zh-v1.5")
        logger.info(f"正在加载 SentenceTransformer 本地模型: {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
