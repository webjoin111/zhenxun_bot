"""
AI 模块配置数据域类型定义
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .messages import LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from zhenxun.services.ai.llm.config.generation import GenConfigBuilder


class StructuredOutputStrategy(str, Enum):
    """结构化输出策略"""

    NATIVE = "native"
    """使用原生 API (如 OpenAI json_object/json_schema, Gemini mime_type)"""
    TOOL_CALL = "tool_call"
    """构造虚假工具调用来强制输出结构化数据 (适用于指令跟随弱但工具调用强的模型)"""
    PROMPT = "prompt"
    """仅在 Prompt 中追加 Schema 说明，依赖文本补全"""


class EmbeddingTaskType(str, Enum):
    """文本嵌入任务类型 (主要用于Gemini)"""

    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class ReasoningEffort(str, Enum):
    """推理努力程度枚举"""

    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ImageAspectRatio(str, Enum):
    """图像宽高比枚举"""

    SQUARE = "1:1"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"
    LANDSCAPE_3_2 = "3:2"
    PORTRAIT_2_3 = "2:3"


class ImageResolution(str, Enum):
    """图像分辨率/质量枚举"""

    STANDARD = "STANDARD"
    HD = "HD"


class CoreConfig(BaseModel):
    """核心生成参数"""

    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="生成温度"
    )
    max_tokens: int | None = Field(default=None, gt=0, description="最大输出token数")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="核采样参数")
    top_k: int | None = Field(default=None, gt=0, description="Top-K采样参数")
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="频率惩罚"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="存在惩罚"
    )
    repetition_penalty: float | None = Field(
        default=None, ge=0.0, le=2.0, description="重复惩罚"
    )
    stop: list[str] | str | None = Field(default=None, description="停止序列")


class ReasoningConfig(BaseModel):
    """推理能力配置"""

    effort: ReasoningEffort | None = Field(
        default=None, description="推理努力程度 (适用于 O1, Gemini 3)"
    )
    budget_tokens: int | None = Field(
        default=None, description="具体的思考 Token 预算 (适用于 Gemini 2.5)"
    )
    show_thoughts: bool | None = Field(
        default=None, description="是否在响应中显式包含思维链内容"
    )


class VisualConfig(BaseModel):
    """视觉生成配置"""

    aspect_ratio: ImageAspectRatio | str | None = Field(
        default=None, description="宽高比"
    )
    resolution: ImageResolution | str | None = Field(
        default=None, description="生成质量/分辨率"
    )
    media_resolution: str | None = Field(
        default=None,
        description="输入媒体的解析度 (Gemini 3+): 'LOW', 'MEDIUM', 'HIGH'",
    )
    style: str | None = Field(
        default=None, description="图像风格 (如 DALL-E 3 vivid/natural)"
    )


class OutputConfig(BaseModel):
    """输出格式控制"""

    response_format: ResponseFormat | dict[str, Any] | None = Field(
        default=None, description="期望的响应格式"
    )
    response_mime_type: str | None = Field(
        default=None, description="响应MIME类型（Gemini专用）"
    )
    response_schema: dict[str, Any] | None = Field(
        default=None, description="JSON响应模式"
    )
    response_modalities: list[str] | None = Field(
        default=None, description="响应模态类型 (TEXT, IMAGE, AUDIO)"
    )
    structured_output_strategy: StructuredOutputStrategy | str | None = Field(
        default=None, description="结构化输出策略 (NATIVE/TOOL_CALL/PROMPT)"
    )


class SafetyConfig(BaseModel):
    """安全设置"""

    safety_settings: dict[str, str] | None = Field(default=None, description="安全设置")


class ToolConfig(BaseModel):
    """工具调用控制配置"""

    mode: Literal["AUTO", "ANY", "NONE"] = Field(
        default="AUTO",
        description="工具调用模式: AUTO(自动), ANY(强制), NONE(禁用)",
    )
    allowed_function_names: list[str] | None = Field(
        default=None,
        description="当 mode 为 ANY 时，允许调用的函数名称白名单",
    )
    includeServerSideToolInvocations: bool | None = Field(
        default=None,
        description="是否在响应中暴露服务端工具调用轨迹 (用于支持内置与本地混合调用)",
    )


class LLMGenerationConfig(BaseModel):
    """
    LLM 生成配置
    采用组件化设计，不再扁平化参数。
    """

    core: CoreConfig | None = Field(default=None, description="基础生成参数")
    reasoning: ReasoningConfig | None = Field(default=None, description="推理能力配置")
    visual: VisualConfig | None = Field(default=None, description="视觉生成配置")
    output: OutputConfig | None = Field(default=None, description="输出格式配置")
    safety: SafetyConfig | None = Field(default=None, description="安全配置")
    tool_config: ToolConfig | None = Field(default=None, description="工具调用策略配置")

    enable_caching: bool | None = Field(default=None, description="是否启用响应缓存")
    custom_params: dict[str, Any] | None = Field(default=None, description="自定义参数")
    validation_policy: dict[str, Any] | None = Field(
        default=None, description="声明式的响应验证策略 (例如: {'require_image': True})"
    )
    response_validator: Callable[[LLMResponse], None] | None = Field(
        default=None,
        description="一个高级回调 function，用于验证响应，验证失败时应抛出异常",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def builder(cls) -> "GenConfigBuilder":
        """创建一个新的配置构建器 (需在外部导入 GenConfigBuilder)"""
        from zhenxun.services.ai.llm.config.generation import GenConfigBuilder

        return GenConfigBuilder()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，排除None值"""
        from zhenxun.utils.pydantic_compat import model_dump

        return model_dump(self, exclude_none=True)

    def merge_with(self, other: LLMGenerationConfig | None) -> LLMGenerationConfig:
        """与另一个配置对象进行深度合并"""
        from zhenxun.utils.pydantic_compat import model_copy, model_dump

        if not other:
            return model_copy(self, deep=True)

        new_config = model_copy(self, deep=True)

        def _merge_component(base_comp, override_comp, comp_cls):
            if override_comp is None:
                return base_comp
            if base_comp is None:
                return override_comp
            updates = model_dump(override_comp, exclude_none=True)
            return model_copy(base_comp, update=updates)

        new_config.core = _merge_component(new_config.core, other.core, CoreConfig)
        new_config.reasoning = _merge_component(
            new_config.reasoning, other.reasoning, ReasoningConfig
        )
        new_config.visual = _merge_component(
            new_config.visual, other.visual, VisualConfig
        )
        new_config.output = _merge_component(
            new_config.output, other.output, OutputConfig
        )
        new_config.safety = _merge_component(
            new_config.safety, other.safety, SafetyConfig
        )
        new_config.tool_config = _merge_component(
            new_config.tool_config, other.tool_config, ToolConfig
        )

        if other.enable_caching is not None:
            new_config.enable_caching = other.enable_caching
        if other.custom_params:
            if new_config.custom_params is None:
                new_config.custom_params = {}
            new_config.custom_params.update(other.custom_params)
        if other.validation_policy:
            if new_config.validation_policy is None:
                new_config.validation_policy = {}
            new_config.validation_policy.update(other.validation_policy)
        if other.response_validator:
            new_config.response_validator = other.response_validator

        return new_config


class LLMEmbeddingConfig(BaseModel):
    """Embedding 专用配置"""

    task_type: str | None = Field(default=None, description="任务类型 (Gemini/Jina)")
    output_dimensionality: int | None = Field(
        default=None, description="输出维度/压缩维度 (Gemini/Jina/OpenAI)"
    )
    title: str | None = Field(
        default=None, description="仅用于 Gemini RETRIEVAL_DOCUMENT 任务的标题"
    )
    encoding_format: str | None = Field(
        default="float", description="编码格式 (float/base64)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = [
    "CoreConfig",
    "EmbeddingTaskType",
    "ImageAspectRatio",
    "ImageResolution",
    "LLMEmbeddingConfig",
    "LLMGenerationConfig",
    "OutputConfig",
    "ReasoningConfig",
    "ReasoningEffort",
    "SafetyConfig",
    "StructuredOutputStrategy",
    "ToolConfig",
    "VisualConfig",
]
