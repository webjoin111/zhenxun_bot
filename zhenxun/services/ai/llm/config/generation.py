"""
LLM 生成配置相关类和函数
"""

from typing import Any
from typing_extensions import Self

from zhenxun.services.ai.config import get_gemini_safety_threshold
from zhenxun.services.ai.types.configs import (
    CoreConfig,
    ImageAspectRatio,
    ImageResolution,
    LLMGenerationConfig,
    OutputConfig,
    ReasoningConfig,
    ReasoningEffort,
    SafetyConfig,
    ToolConfig,
    VisualConfig,
)
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.types.messages import ResponseFormat
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_validate


class GenConfigBuilder:
    """
    LLM 生成配置的语义化构建器。
    设计原则：高频业务场景优先，低频参数命名空间化。
    """

    def __init__(self):
        self._config = LLMGenerationConfig()

    def _ensure_core(self) -> CoreConfig:
        if self._config.core is None:
            self._config.core = CoreConfig()
        return self._config.core

    def _ensure_output(self) -> OutputConfig:
        if self._config.output is None:
            self._config.output = OutputConfig()
        return self._config.output

    def _ensure_reasoning(self) -> ReasoningConfig:
        if self._config.reasoning is None:
            self._config.reasoning = ReasoningConfig()
        return self._config.reasoning

    def as_json(self, schema: dict[str, Any] | None = None) -> Self:
        """
        [高频] 强制模型输出 JSON 格式。
        """
        out = self._ensure_output()
        out.response_format = ResponseFormat.JSON
        if schema:
            out.response_schema = schema
        return self

    def enable_thinking(
        self, budget_tokens: int = -1, show_thoughts: bool = False
    ) -> Self:
        """
        [高频] 启用模型的思考/推理能力 (如 Gemini 2.0 Flash Thinking, DeepSeek R1)。
        """
        reasoning = self._ensure_reasoning()
        reasoning.budget_tokens = budget_tokens
        reasoning.show_thoughts = show_thoughts
        return self

    def config_core(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> Self:
        """
        [低频] 配置核心生成参数。
        """
        core = self._ensure_core()
        if temperature is not None:
            core.temperature = temperature
        if max_tokens is not None:
            core.max_tokens = max_tokens
        if top_p is not None:
            core.top_p = top_p
        if top_k is not None:
            core.top_k = top_k
        if stop is not None:
            core.stop = stop
        if frequency_penalty is not None:
            core.frequency_penalty = frequency_penalty
        if presence_penalty is not None:
            core.presence_penalty = presence_penalty
        return self

    def config_safety(self, settings: dict[str, str]) -> Self:
        """
        [低频] 配置安全过滤设置。
        """
        if self._config.safety is None:
            self._config.safety = SafetyConfig()
        self._config.safety.safety_settings = settings
        return self

    def enable_server_tools_context(self) -> Self:
        """
        [高频] 开启服务端工具调用轨迹上下文循环 (支持内置工具与本地函数混合调用)。
        """
        if self._config.tool_config is None:
            self._config.tool_config = ToolConfig()
        self._config.tool_config.includeServerSideToolInvocations = True
        return self

    def config_visual(
        self,
        aspect_ratio: ImageAspectRatio | str | None = None,
        resolution: ImageResolution | str | None = None,
    ) -> Self:
        """
        [低频] 配置视觉生成参数 (DALL-E 3 / Gemini Imagen)。
        """
        if self._config.visual is None:
            self._config.visual = VisualConfig()
        if aspect_ratio:
            self._config.visual.aspect_ratio = aspect_ratio
        if resolution:
            self._config.visual.resolution = resolution
        return self

    def set_custom_param(self, key: str, value: Any) -> Self:
        """设置特定于厂商的自定义参数"""
        if self._config.custom_params is None:
            self._config.custom_params = {}
        self._config.custom_params[key] = value
        return self

    def build(self) -> LLMGenerationConfig:
        """构建最终的配置对象"""
        return self._config


def validate_override_params(
    override_config: dict[str, Any] | LLMGenerationConfig | None,
) -> LLMGenerationConfig:
    """验证和标准化覆盖参数"""
    if override_config is None:
        return LLMGenerationConfig()

    if isinstance(override_config, LLMGenerationConfig):
        return override_config

    if isinstance(override_config, dict):
        try:
            return model_validate(LLMGenerationConfig, override_config)
        except Exception as e:
            logger.warning(f"覆盖配置参数验证失败: {e}")
            raise LLMException(
                f"无效的覆盖配置参数: {e}",
                code=LLMErrorCode.CONFIGURATION_ERROR,
                cause=e,
            )

    raise LLMException(
        f"不支持的配置类型: {type(override_config)}",
        code=LLMErrorCode.CONFIGURATION_ERROR,
    )


class CommonOverrides:
    """常用的配置覆盖预设"""

    @staticmethod
    def gemini_json() -> LLMGenerationConfig:
        """Gemini JSON模式：强制JSON输出"""
        return LLMGenerationConfig(
            core=CoreConfig(),
            output=OutputConfig(
                response_format=ResponseFormat.JSON,
                response_mime_type="application/json",
            ),
        )

    @staticmethod
    def gemini_2_5_thinking(tokens: int = -1) -> LLMGenerationConfig:
        """Gemini 2.5 思考模式：默认 -1 (动态思考)，0 为禁用，>=1024 为固定预算"""
        return LLMGenerationConfig(
            core=CoreConfig(temperature=1.0),
            reasoning=ReasoningConfig(budget_tokens=tokens, show_thoughts=True),
        )

    @staticmethod
    def gemini_3_thinking(level: str = "HIGH") -> LLMGenerationConfig:
        """Gemini 3 深度思考模式：使用思考等级"""
        try:
            effort = ReasoningEffort(level.upper())
        except ValueError:
            effort = ReasoningEffort.HIGH

        return LLMGenerationConfig(
            core=CoreConfig(),
            reasoning=ReasoningConfig(effort=effort, show_thoughts=True),
        )

    @staticmethod
    def gemini_structured(schema: dict[str, Any]) -> LLMGenerationConfig:
        """Gemini 结构化输出：自定义JSON模式"""
        return LLMGenerationConfig(
            core=CoreConfig(),
            output=OutputConfig(
                response_mime_type="application/json", response_schema=schema
            ),
        )

    @staticmethod
    def gemini_safe() -> LLMGenerationConfig:
        """Gemini 安全模式：使用配置的安全设置"""
        threshold = get_gemini_safety_threshold()
        return LLMGenerationConfig(
            core=CoreConfig(),
            safety=SafetyConfig(
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": threshold,
                    "HARM_CATEGORY_HATE_SPEECH": threshold,
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": threshold,
                    "HARM_CATEGORY_DANGEROUS_CONTENT": threshold,
                }
            ),
        )

    @staticmethod
    def gemini_code_execution() -> LLMGenerationConfig:
        """Gemini 代码执行模式：启用代码执行功能"""
        return LLMGenerationConfig(
            core=CoreConfig(),
            custom_params={"code_execution_timeout": 30},
        )

    @staticmethod
    def gemini_grounding() -> LLMGenerationConfig:
        """Gemini 信息来源关联模式：启用Google搜索"""
        return LLMGenerationConfig(
            core=CoreConfig(),
            custom_params={
                "grounding_config": {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC"}}
            },
        )

    @staticmethod
    def gemini_nano_banana(aspect_ratio: str = "16:9") -> LLMGenerationConfig:
        """Gemini Nano Banana Pro：自定义比例生图"""
        try:
            ar = ImageAspectRatio(aspect_ratio)
        except ValueError:
            ar = ImageAspectRatio.LANDSCAPE_16_9

        return LLMGenerationConfig(
            core=CoreConfig(),
            visual=VisualConfig(aspect_ratio=ar),
        )

    @staticmethod
    def gemini_high_res() -> LLMGenerationConfig:
        """Gemini 3: 强制使用高解析度处理输入媒体"""
        return LLMGenerationConfig(
            visual=VisualConfig(media_resolution="HIGH", resolution=ImageResolution.HD)
        )
