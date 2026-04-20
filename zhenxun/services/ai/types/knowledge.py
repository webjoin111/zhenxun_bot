import hashlib
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """知识库文档基础模型"""

    content: str = Field(description="文档内容")
    id: str | None = Field(default=None, description="文档唯一标识")
    name: str | None = Field(default=None, description="文档名称/路径")
    meta_data: dict[str, Any] = Field(default_factory=dict, description="元数据")
    embedding: list[float] | None = Field(default=None, description="内容的向量表示")

    def model_post_init(self, __context: Any) -> None:
        """初始化后自动生成基于内容的稳定 ID（如果未提供）"""
        if not self.id:
            content_hash = hashlib.md5(self.content.encode("utf-8")).hexdigest()[:12]
            prefix = f"{self.name}_" if self.name else "doc_"
            self.id = f"{prefix}{content_hash}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "meta_data": self.meta_data,
        }

    async def async_embed(self, model_name: str | None = None) -> None:
        """
        异步向量化当前文档内容。
        直接调用 zhenxun 底层的 embed API。
        """
        from zhenxun.services.ai.llm.api import embed_documents
        from zhenxun.services.log import logger

        if not self.content.strip():
            return

        try:
            vectors = await embed_documents([self.content], model=model_name)
            if vectors:
                self.embedding = vectors[0]
        except Exception as e:
            logger.error(f"文档向量化失败 (ID: {self.id}): {e}")
