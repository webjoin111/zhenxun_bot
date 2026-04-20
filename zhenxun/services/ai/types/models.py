"""
模型自身设定域类型定义
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ModelProvider(Enum):
    """模型提供商枚举"""

    OPENAI = "openai"
    GEMINI = "gemini"
    ZHIXPU = "zhipu"
    CUSTOM = "custom"


ModelName = str | None


class ModelModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
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
    output_modalities: set[ModelModality] = Field(default={ModelModality.TEXT})
    supports_tool_calling: bool = False
    is_embedding_model: bool = False
    is_rerank_model: bool = False
    reasoning_mode: ReasoningMode = ReasoningMode.NONE
    reasoning_visibility: Literal["visible", "hidden", "none"] = "none"
    max_input_tokens: int = Field(default=8192)
    max_output_tokens: int = Field(default=4096)


class ModelDetail(BaseModel):
    """模型详细信息"""

    model_name: str
    is_available: bool = True
    is_embedding_model: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    api_type: str | None = None
    endpoint: str | None = None


__all__ = [
    "ModelCapabilities",
    "ModelDetail",
    "ModelModality",
    "ModelName",
    "ModelProvider",
    "ReasoningMode",
]
