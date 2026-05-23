"""
Zhenxun AI - RAG (检索增强生成) 基础设施层
"""

from .backends import RagRegistry
from .backends.embedders import DefaultEmbedder, Embedder
from .backends.storages import (
    AbstractVectorRecord,
    DictStorageBackend,
    StorageBackend,
    TortoiseStorageBackend,
)
from .engine import KnowledgeScope, KnowledgeSlice, RAGManager
from .facade import SimpleRAG
from .ingestion import (
    ChunkingStrategy,
    DocumentChunking,
    FixedSizeChunking,
    IngestionPipeline,
    RowChunking,
)
from .models import BaseRecord, QueryRequest, RAGConfig, SearchResult

__all__ = [
    "AbstractVectorRecord",
    "BaseRecord",
    "ChunkingStrategy",
    "DefaultEmbedder",
    "DictStorageBackend",
    "DocumentChunking",
    "Embedder",
    "FixedSizeChunking",
    "IngestionPipeline",
    "KnowledgeScope",
    "KnowledgeSlice",
    "QueryRequest",
    "RAGConfig",
    "RAGManager",
    "RagRegistry",
    "RowChunking",
    "SearchResult",
    "SimpleRAG",
    "StorageBackend",
    "TortoiseStorageBackend",
]
