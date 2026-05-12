from zhenxun.services.ai.knowledge.models import Document

from .strategy import ChunkingStrategy


class DocumentChunking(ChunkingStrategy):
    """段落语义分块策略 (按双换行切分)"""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Document]:
        if len(document.content) <= self.chunk_size:
            return [
                self._create_chunk_doc(document, 0, self.clean_text(document.content))
            ]

        raw_paragraphs = document.content.split("\n\n")
        paragraphs = [self.clean_text(para) for para in raw_paragraphs if para.strip()]

        chunks: list[Document] = []
        current_chunk_texts = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.chunk_size and current_chunk_texts:
                chunk_content = "\n\n".join(current_chunk_texts)
                chunks.append(
                    self._create_chunk_doc(document, chunk_index, chunk_content)
                )
                chunk_index += 1
                current_chunk_texts = []
                current_length = 0

            current_chunk_texts.append(para)
            current_length += para_len + 2

        if current_chunk_texts:
            chunk_content = "\n\n".join(current_chunk_texts)
            chunks.append(self._create_chunk_doc(document, chunk_index, chunk_content))

        return chunks

