from collections.abc import Callable
from typing import Any, ClassVar

from zhenxun.services.ai.rag.backends.storages import (
    DictStorageBackend,
    LanceDBStorageBackend,
    QdrantStorageBackend,
    StorageBackend,
    TortoiseStorageBackend,
)
from zhenxun.services.ai.rag.models import (
    LanceDBStorageSpec,
    QdrantStorageSpec,
)
from zhenxun.services.log import logger
from zhenxun.services.ai.rag.backends.embedders import Embedder, DefaultEmbedder


class RagRegistry:
    """RAG 组件全局注册表 (Registry Pattern)"""

    _storage_factories: ClassVar[dict[str, Callable[..., Any]]] = {}
    _preprocessors: ClassVar[dict[str, type]] = {}
    _postprocessors: ClassVar[dict[str, type]] = {}
    _embedder_factories: ClassVar[dict[str, Callable[[str | None], Embedder]]] = {}

    @classmethod
    def register_storage(cls, name: str, factory: Callable[..., Any]) -> None:
        cls._storage_factories[name.lower()] = factory
        logger.debug(f"已注册 RAG 存储后端: '{name}'")

    @classmethod
    def get_storage(cls, name: str) -> Callable[..., Any] | None:
        return cls._storage_factories.get(name.lower())

    @classmethod
    def register_preprocessor(cls, name: str, processor_cls: type) -> None:
        cls._preprocessors[name.lower()] = processor_cls
        logger.debug(f"已注册 RAG 预处理器: '{name}'")

    @classmethod
    def get_preprocessor(cls, name: str) -> type | None:
        return cls._preprocessors.get(name.lower())

    @classmethod
    def register_postprocessor(cls, name: str, processor_cls: type) -> None:
        cls._postprocessors[name.lower()] = processor_cls
        logger.debug(f"已注册 RAG 后处理器: '{name}'")

    @classmethod
    def get_postprocessor(cls, name: str) -> type | None:
        return cls._postprocessors.get(name.lower())

    @classmethod
    def register_embedder(cls, name: str, factory: Callable[[str | None], Embedder]) -> None:
        cls._embedder_factories[name.lower()] = factory
        logger.debug(f"已注册 RAG Embedder 引擎工厂: '{name}'")

    @classmethod
    def get_embedder(cls, provider: str) -> Callable[[str | None], Embedder] | None:
        """
        获取 Embedder 工厂函数。如果未找到指定 provider，降级返回 default。
        """
        return cls._embedder_factories.get(provider.lower()) or cls._embedder_factories.get("default")


def _build_dict(spec_dict: dict[str, Any]) -> StorageBackend:
    return DictStorageBackend()


def _build_tortoise(spec_dict: dict[str, Any]) -> StorageBackend:
    model_class = spec_dict.get("model_class")
    if not model_class:
        raise ValueError("使用 tortoise 作为 RAG 存储时必须提供 model_class 参数")
    return TortoiseStorageBackend(model_class=model_class)


def _build_qdrant(spec_dict: dict[str, Any]) -> StorageBackend:
    spec = QdrantStorageSpec(**spec_dict)
    return QdrantStorageBackend(
        location=spec.location,
        collection_name=spec.collection_name,
        url=spec.url,
        port=spec.port,
        api_key=spec.api_key,
    )


def _build_lancedb(spec_dict: dict[str, Any]) -> StorageBackend:
    spec = LanceDBStorageSpec(**spec_dict)
    return LanceDBStorageBackend(uri=spec.uri, table_name=spec.table_name)


RagRegistry.register_storage("dict", _build_dict)
RagRegistry.register_storage("tortoise", _build_tortoise)
RagRegistry.register_storage("qdrant", _build_qdrant)
RagRegistry.register_storage("lancedb", _build_lancedb)


def create_storage(spec_dict: dict[str, Any]) -> StorageBackend:
    """快捷工厂函数"""
    storage_type = spec_dict.get("type")
    if not storage_type:
        raise ValueError("RAG Storage 配置字典中必须包含 'type' 字段")

    factory_func = RagRegistry.get_storage(storage_type)
    if not factory_func:
        raise ValueError(f"不支持的 RAG Storage Provider: {storage_type}")
    return factory_func(spec_dict)


def _lazy_load_fastembed(model_name: str | None) -> Embedder:
    from zhenxun.services.ai.rag.backends.embedders import FastEmbedder
    return FastEmbedder(model_name)

def _lazy_load_st(model_name: str | None) -> Embedder:
    from zhenxun.services.ai.rag.backends.embedders import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_name)


# 注册系统内置 Embedder
RagRegistry.register_embedder("api", lambda model: DefaultEmbedder(model_name=model))
RagRegistry.register_embedder("default", lambda model: DefaultEmbedder(model_name=model))
RagRegistry.register_embedder("fastembed", _lazy_load_fastembed)
RagRegistry.register_embedder("sentence-transformers", _lazy_load_st)
