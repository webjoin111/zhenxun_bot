from abc import ABC, abstractmethod
import re

from zhenxun.services.ai.types.knowledge import Document


class ChunkingStrategy(ABC):
    """文本分块策略基类"""

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """将单个大文档分割为多个子块文档"""
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        """基础文本清理：替换多余换行和空格"""
        cleaned_text = re.sub(r"\n+", "\n", text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        return cleaned_text.strip()

    def _create_chunk_doc(
        self, original_doc: Document, chunk_number: int, content: str
    ) -> Document:
        """为分割后的片段创建新的 Document 对象"""
        meta_data = original_doc.meta_data.copy()
        meta_data["chunk_index"] = chunk_number
        meta_data["chunk_size"] = len(content)
        meta_data["parent_id"] = original_doc.id

        return Document(name=original_doc.name, content=content, meta_data=meta_data)
