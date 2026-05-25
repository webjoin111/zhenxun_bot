"""
Zhenxun AI - RAG (检索增强生成) 基础设施层
"""

from .backends.embedders import Embedder
from .backends.storages import (
    AbstractVectorRecord,
    StorageBackend,
    TortoiseStorageBackend,
)
from .builder import RAGBuilder
from .facade import SimpleRAG
from .models import BaseRecord, RAGConfig, SearchResult

__all__ = [
    "AbstractVectorRecord",
    "BaseRecord",
    "Embedder",
    "RAGBuilder",
    "RAGConfig",
    "SearchResult",
    "SimpleRAG",
    "StorageBackend",
    "TortoiseStorageBackend",
]
