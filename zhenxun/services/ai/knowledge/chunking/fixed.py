from zhenxun.services.ai.types.knowledge import Document

from .strategy import ChunkingStrategy


class FixedSizeChunking(ChunkingStrategy):
    """定长分块策略 (带重叠)"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError(f"重叠长度 ({overlap}) 必须小于分块大小 ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Document]:
        content = self.clean_text(document.content)
        content_length = len(content)

        if content_length <= self.chunk_size:
            return [self._create_chunk_doc(document, 0, content)]

        chunks: list[Document] = []
        start = 0
        chunk_index = 0

        while start < content_length:
            end = min(start + self.chunk_size, content_length)

            if end < content_length:
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
                if end == start:
                    end = start + self.chunk_size

            chunk_content = content[start:end]
            chunks.append(self._create_chunk_doc(document, chunk_index, chunk_content))

            chunk_index += 1
            start = end - self.overlap

        return chunks
