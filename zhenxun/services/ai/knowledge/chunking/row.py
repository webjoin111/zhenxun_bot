from zhenxun.services.ai.types.knowledge import Document

from .strategy import ChunkingStrategy


class RowChunking(ChunkingStrategy):
    """
    行数据分块策略 (专为 CSV/表格设计)
    核心特性：自动识别表头，并将其附加到每一个被切分的 Chunk 首部，防止上下文丢失。
    """

    def __init__(self, rows_per_chunk: int = 50):
        self.rows_per_chunk = rows_per_chunk

    def chunk(self, document: Document) -> list[Document]:
        lines = document.content.splitlines()

        lines = [line for line in lines if line.strip()]

        if not lines:
            return []

        header = lines[0]
        data_lines = lines[1:]

        if not data_lines:
            return [self._create_chunk_doc(document, 0, header)]

        chunks: list[Document] = []
        chunk_index = 0

        for i in range(0, len(data_lines), self.rows_per_chunk):
            chunk_lines = [header] + data_lines[i : i + self.rows_per_chunk]
            chunk_content = "\n".join(chunk_lines)
            chunks.append(self._create_chunk_doc(document, chunk_index, chunk_content))
            chunk_index += 1

        return chunks
