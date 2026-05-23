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
    metadata_filters: dict[str, Any] | None = Field(default=None)
    """元数据精确匹配字典"""
    limit: int = Field(default=10)
    """返回的最大条数"""


class DictStorageSpec(BaseModel):
    type: Literal["dict"] = "dict"
    """存储后端类型，固定为 dict"""


class TortoiseStorageSpec(BaseModel):
    type: Literal["tortoise"] = "tortoise"
    """存储后端类型，固定为 tortoise"""
    model_class: Any = Field()
    """Tortoise ORM 的模型类 (AbstractVectorRecord 的子类)"""


class QdrantStorageSpec(BaseModel):
    type: Literal["qdrant"] = "qdrant"
    """存储后端类型，固定为 qdrant"""
    location: str = Field(default=":memory:")
    """Qdrant 数据库连接地址，默认为内存模式"""
    collection_name: str = Field(default="zhenxun_rag")
    """集合（数据表）名称"""
    url: str | None = None
    """Qdrant 服务的 URL 地址"""
    port: int | None = None
    """Qdrant 服务的端口号"""
    api_key: str | None = None
    """Qdrant 服务的 API 密钥"""


class LanceDBStorageSpec(BaseModel):
    type: Literal["lancedb"] = "lancedb"
    """存储后端类型，固定为 lancedb"""
    uri: str = Field(default="./data/lancedb")
    """LanceDB 数据库文件的存储路径"""
    table_name: str = Field(default="zhenxun_rag")
    """数据表名称"""


StorageConfigType = dict[str, Any]


class LLMQueryRewriteSpec(BaseModel):
    type: Literal["llm_query_rewrite"] = "llm_query_rewrite"
    """预处理器类型，固定为 llm_query_rewrite"""
    model_name: str | None = None
    """用于重写查询的 LLM 模型名称"""


PreProcessorConfigType = dict[str, Any]


class TimeDecaySpec(BaseModel):
    type: Literal["time_decay"] = "time_decay"
    """后处理器类型，固定为 time_decay"""
    half_life_days: int = Field(default=30)
    """时间衰减的半衰期天数"""
    decay_weight: float = Field(default=0.3)
    """时间衰减权重的占比"""
    semantic_weight: float = Field(default=0.7)
    """语义相似度权重的占比"""


PostProcessorConfigType = dict[str, Any]


class RAGConfig(BaseModel):
    """局部 RAG 管线配置声明契约"""

    storage: StorageConfigType = Field(default_factory=lambda: {"type": "dict"})
    """向量存储后端配置字典"""
    embedder_model: str | None = None
    """用于向量化文本的嵌入模型名称"""
    use_rerank: bool = False
    """是否启用重排（Rerank）"""
    rerank_model: str | None = None
    """重排所使用的模型名称"""
    rerank_top_n: int = 5
    """重排后保留的前 N 条结果"""
    use_time_decay: bool = False
    """是否启用时间衰减机制"""
    half_life_days: int = 30
    """时间衰减的半衰期天数"""
    use_query_rewrite: bool = False
    """是否启用查询重写（Query Rewrite）"""
    query_rewrite_model: str | None = None
    """用于查询重写的模型名称"""
    use_dedup: bool = True
    """是否启用去重机制"""
    dedup_threshold: float = 0.98
    """相似度去重的阈值"""
    pre_processors: list[PreProcessorConfigType] = Field(default_factory=list)
    """前处理器配置列表"""
    post_processors: list[PostProcessorConfigType] = Field(default_factory=list)
    """后处理器配置列表"""
    use_consolidation: bool = False
    """是否启用大模型反思与记忆融合（Consolidation）"""
    consolidation_threshold: float = 0.85
    """触发融合的相似度阈值"""
    consolidator_model: str | None = None
    """执行融合分析的 LLM 模型名称"""


class ConsolidationAction(BaseModel):
    action: Literal["keep", "update", "delete"] = Field(description="对旧记录执行的动作")
    record_id: str = Field(description="目标旧记录的 ID")
    new_content: str | None = Field(default=None, description="更新后的文本内容（仅在 update 时需要）")

class ConsolidationPlan(BaseModel):
    actions: list[ConsolidationAction] = Field(default_factory=list, description="对历史记录的操作列表")
    insert_new: bool = Field(default=True, description="是否将当前的新内容作为独立记录插入")
