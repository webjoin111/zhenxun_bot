import fnmatch

from zhenxun.services.ai.types.models import (
    ModelCapabilities,
    ModelModality,
    ReasoningMode,
)

PATTERNS_GEMINI_2_5 = [
    "gemini-2.5*",
    "gemini-flash*",
    "gemini*lite*",
    "gemini-flash-latest",
]
PATTERNS_GEMINI_3 = ["gemini-3*", "gemini-exp*"]
PATTERNS_OPENAI_REASONING = ["o1-*", "o3-*", "deepseek-r1*", "deepseek-reasoner"]

STANDARD_TEXT_TOOL_CAPABILITIES = ModelCapabilities(
    input_modalities={ModelModality.TEXT},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
)
CAP_GEMINI_2_5 = ModelCapabilities(
    input_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
    },
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.BUDGET,
    reasoning_visibility="visible",
)
CAP_GEMINI_3 = ModelCapabilities(
    input_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
    },
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.LEVEL,
    reasoning_visibility="visible",
)
CAP_OPENAI_REASONING = ModelCapabilities(
    input_modalities={ModelModality.TEXT, ModelModality.IMAGE},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.EFFORT,
    reasoning_visibility="hidden",
)

CAP_RERANK_ONLY = ModelCapabilities(
    input_modalities={ModelModality.TEXT, ModelModality.IMAGE},
    is_rerank_model=True,
)

DEFAULT_PERMISSIVE_CAPABILITIES = ModelCapabilities(
    input_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
    },
    output_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
    },
    supports_tool_calling=True,
)

MODEL_ALIAS_MAPPING: dict[str, str] = {
    "deepseek-v3*": "deepseek-chat",
    "deepseek-ai/DeepSeek-V3": "deepseek-chat",
    "deepseek-r1*": "deepseek-reasoner",
}


def _build_registry() -> dict[str, ModelCapabilities]:
    """构建模型能力注册表"""
    registry: dict[str, ModelCapabilities] = {}

    def register_family(
        patterns: list[str],
        cap: ModelCapabilities,
        context_limits: dict[str, int] | None = None,
    ) -> None:
        for pattern in patterns:
            if context_limits:
                registry[pattern] = cap.model_copy(update=context_limits)
            else:
                registry[pattern] = cap

    ctx_1m_64k = {"max_input_tokens": 1048576, "max_output_tokens": 65536}
    ctx_128k_32k = {"max_input_tokens": 128000, "max_output_tokens": 32768}  # noqa: F841
    ctx_200k_100k = {"max_input_tokens": 200000, "max_output_tokens": 100000}  # noqa: F841
    ctx_128k_64k = {"max_input_tokens": 128000, "max_output_tokens": 65536}
    ctx_128k_8k = {"max_input_tokens": 128000, "max_output_tokens": 8192}
    ctx_8k_4k = {"max_input_tokens": 8192, "max_output_tokens": 4096}

    register_family(PATTERNS_GEMINI_3, CAP_GEMINI_3, ctx_1m_64k)
    register_family(PATTERNS_GEMINI_2_5, CAP_GEMINI_2_5, ctx_1m_64k)
    register_family(PATTERNS_OPENAI_REASONING, CAP_OPENAI_REASONING, ctx_128k_64k)
    register_family(
        ["gpt-4o*"],
        CAP_GEMINI_2_5,
        {"max_input_tokens": 128000, "max_output_tokens": 16384},
    )
    register_family(["gpt-4-*"], STANDARD_TEXT_TOOL_CAPABILITIES, ctx_8k_4k)
    register_family(
        ["deepseek-chat", "deepseek-v3*"], STANDARD_TEXT_TOOL_CAPABILITIES, ctx_128k_8k
    )
    register_family(["*reranker*", "*rerank*", "bge-m3*"], CAP_RERANK_ONLY)

    return registry


MODEL_CAPABILITIES_REGISTRY = _build_registry()


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """
    从注册表获取模型能力，支持别名映射和通配符匹配。
    """
    canonical_name = model_name
    for alias_pattern, c_name in MODEL_ALIAS_MAPPING.items():
        if fnmatch.fnmatch(model_name, alias_pattern):
            canonical_name = c_name
            break

    parts = canonical_name.split("/")
    names_to_check = ["/".join(parts[i:]) for i in range(len(parts))]

    for name in names_to_check:
        if name in MODEL_CAPABILITIES_REGISTRY:
            return MODEL_CAPABILITIES_REGISTRY[name]

        for pattern, capabilities in MODEL_CAPABILITIES_REGISTRY.items():
            if "*" in pattern and fnmatch.fnmatch(name, pattern):
                return capabilities

    return DEFAULT_PERMISSIVE_CAPABILITIES
