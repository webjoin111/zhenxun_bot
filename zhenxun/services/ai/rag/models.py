from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseRecord(BaseModel):
    """RAG 基础记录载体，没有任何业务属性"""

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    """记录的唯一标识符"""
    content: str = Field(...)
    """数据块的文本内容"""
    embedding: list[float] | None = Field(default=None)
    """数据块对应的向量嵌入"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """数据块的元数据字典"""
    action: Literal["insert", "update", "delete", "ignore"] = Field(default="insert")
    """数据块在索引管线中的操作意图"""


class SearchResult(BaseModel):
    """搜索结果"""

    record: BaseRecord
    """检索到的基础记录"""
    score: float
    """检索相似度得分"""


class QueryRequest(BaseModel):
    """通用检索请求"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(default="")
    """原始查询文本"""
    embedding: list[float] | None = Field(default=None)
    """用于向量检索的数组"""
    search_type: Literal["dense", "sparse", "hybrid"] = Field(default="dense")
    """检索类型标识：稠密向量、稀疏关键词或混合"""
    metadata_filters: dict[str, Any] | None = Field(default=None)
    """元数据精确匹配字典"""
    limit: int = Field(default=10)
    """返回的最大条数"""


StorageConfigType = dict[str, Any]


class ConsolidationAction(BaseModel):
    action: Literal["keep", "update", "delete"]
    """对旧记录执行的动作"""
    record_id: str
    """目标旧记录的 ID"""
    new_content: str | None = Field(default=None)
    """更新后的文本内容（仅在 update 时需要）"""


class ConsolidationPlan(BaseModel):
    actions: list[ConsolidationAction] = Field(default_factory=list)
    """对历史记录的操作列表"""
    insert_new: bool = Field(default=True)
    """是否将当前的新内容作为独立记录插入"""
