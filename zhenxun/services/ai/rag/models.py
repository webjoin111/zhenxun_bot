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


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strategy: Any | None = None
    """文档切块策略"""


class DedupConfig(BaseModel):
    enable: bool = True
    """是否开启入库去重"""
    threshold: float = 0.98
    """去重相似度阈值"""


class ConsolidationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    consolidator: Any | None = None
    """大模型记忆融合与反思器"""
    threshold: float = 0.85
    """融合相似度阈值"""


class RerankConfig(BaseModel):
    enable: bool = False
    """是否开启重排"""
    model_name: str | None = None
    """重排使用的模型名称"""
    top_n: int = 5
    """重排后保留的文档数量"""


class QueryRewriteConfig(BaseModel):
    enable: bool = False
    """是否开启查询意图重写"""
    model_name: str | None = None
    """意图重写使用的模型名称"""


class TimeDecayConfig(BaseModel):
    enable: bool = False
    """是否开启时间衰减打分后处理"""
    half_life_days: int = 30
    """半衰期天数"""
    decay_weight: float = 0.3
    """时间衰减权重"""
    semantic_weight: float = 0.7
    """语义相似度权重"""
    importance_weight: float = 0.0
    """重要性打分权重"""


class RAGConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Any | None = None
    """存储后端"""
    embedder: Any | None = None
    """向量化引擎"""
    scopes: str | list[str] = "/"
    """数据隔离作用域"""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    """文档切块配置"""
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    """去重配置"""
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    """记忆融合反思配置"""
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    """重排配置"""
    time_decay: TimeDecayConfig = Field(default_factory=TimeDecayConfig)
    """时间衰减配置"""
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)
    """查询意图重写配置"""
    pre_processors: list[Any] = Field(default_factory=list)
    """自定义查询前处理器列表"""
    post_processors: list[Any] = Field(default_factory=list)
    """自定义检索后处理器列表"""
