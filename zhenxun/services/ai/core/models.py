"""
模型自身设定域类型定义
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

ModelName = str | None


class ToolDefinition(BaseModel):
    """结构化的工具定义模型"""

    name: str = Field(...)
    """工具名称"""
    description: str = Field(...)
    """工具描述"""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """JSON Schema 参数"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """元数据"""


class ToolChoice(BaseModel):
    """工具选择配置"""

    mode: Literal["auto", "none", "any", "required"] = Field(default="auto")
    """工具选择模式"""
    allowed_function_names: list[str] | None = Field(default=None)
    """允许调用的函数名称列表"""


class ModelModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    EMBEDDING = "embedding"


class ReasoningMode(str, Enum):
    """推理/思考模式类型"""

    NONE = "none"
    BUDGET = "budget"
    LEVEL = "level"
    EFFORT = "effort"


class ModelCapabilities(BaseModel):
    """定义一个模型的核心能力。"""

    input_modalities: set[ModelModality] = Field(default={ModelModality.TEXT})
    """模型支持的输入模态集合。"""
    output_modalities: set[ModelModality] = Field(default={ModelModality.TEXT})
    """模型支持的输出模态集合。"""
    supports_tool_calling: bool = False
    """是否支持工具调用能力。"""
    is_embedding_model: bool = False
    """是否为嵌入模型。"""
    is_rerank_model: bool = False
    """是否为重排序模型。"""
    reasoning_mode: ReasoningMode = ReasoningMode.NONE
    """推理模式类型。"""
    reasoning_visibility: Literal["visible", "hidden", "none"] = "none"
    """推理过程可见性设置。"""
    max_input_tokens: int = Field(default=8192)
    """最大输入 Token 数量。"""
    max_output_tokens: int = Field(default=4096)
    """最大输出 Token 数量。"""
    max_thinking_tokens: int = Field(default=0)
    """最大思考 Token 数量。"""
    supported_native_tools: set[str] = Field(default_factory=set)
    """该模型实际支持的云端原生能力/内置工具。"""

    features: set[str] = Field(default_factory=set)
    """用于第三方插件动态注入的自定义能力标签。"""

    def accepts_input(self, modality: ModelModality) -> bool:
        """检查模型是否支持某种输入模态"""
        return modality in self.input_modalities

    def accepts_output(self, modality: ModelModality) -> bool:
        """检查模型是否支持某种输出模态"""
        return modality in self.output_modalities

    def has_feature(self, feature: str) -> bool:
        """检查模型是否具备某个扩展特性"""
        return feature in self.features


class ModelDetail(BaseModel):
    """模型详细信息"""

    model_name: str
    """模型名称。"""
    is_available: bool = True
    """模型是否可用。"""
    temperature: float | None = None
    """采样温度参数。"""
    generation_max_tokens: int | None = None
    """单次生成最大 Token 数。"""
    api_type: str | None = None
    """API 类型标识。"""
    endpoint: str | None = None
    """模型服务端点地址。"""
    task_type: str | None = Field(default=None)
    """显式声明的主任务类型 (如 'image_generation')。"""
    path_prefix: str | None = Field(default=None)
    """中转路由前缀，例如 '/cogvideox' 或 '/minimax'。"""


__all__ = [
    "ModelCapabilities",
    "ModelDetail",
    "ModelModality",
    "ModelName",
    "ReasoningMode",
    "ToolChoice",
    "ToolDefinition",
]
