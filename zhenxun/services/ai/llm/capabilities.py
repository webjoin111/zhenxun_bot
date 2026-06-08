import fnmatch

from zhenxun.services.ai.core.models import (
    ModelCapabilities,
    ModelModality,
    ReasoningMode,
)
from zhenxun.utils.pydantic_compat import model_copy

PATTERNS_GEMINI_2_5_FLASH = [
    "gemini-2.5-flash*",
    "gemini-flash*",
    "gemini*lite*",
    "gemini-flash-latest",
]
PATTERNS_GEMINI_2_5_PRO = ["gemini-2.5-pro*"]
PATTERNS_GEMINI_3 = ["gemini-3*"]
PATTERNS_OPENAI_REASONING = ["o1-*", "o3-*"]
PATTERNS_DEEPSEEK_PRO = ["deepseek-v4-pro*"]
PATTERNS_DEEPSEEK_FLASH = ["deepseek-v4-flash*"]
PATTERNS_MINIMAX_REASONING = ["*MiniMax-M2*", "*minimax-m2*"]
PATTERNS_GEMINI_IMAGE = ["*gemini*image*"]
PATTERNS_GEMINI_EMBEDDING_2 = ["gemini-embedding-2*"]
PATTERNS_JINA_EMBEDDING = ["jina-embeddings-*"]
PATTERNS_JINA_V5_OMNI = ["jina-embeddings-v5-omni*"]
PATTERNS_JINA_RERANKER = ["jina-reranker-*", "jina-colbert-*"]

PATTERNS_1M_MODELS = [
    "mimo-v2.5*", "mimo-v2-pro*",
    "*MiniMax-M3*"
]

CAP_MULTIMODAL_EMBEDDING = ModelCapabilities(
    input_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
        ModelModality.FILE,
    },
    is_embedding_model=True,
    supports_tool_calling=False,
)

STANDARD_TEXT_TOOL_CAPABILITIES = ModelCapabilities(
    input_modalities={ModelModality.TEXT},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    supported_native_tools={
        "web_search",
        "code_execution",
        "computer_use",
        "file_search",
    },
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
    supported_native_tools={
        "web_search",
        "code_execution",
        "google_map",
        "url_context",
    },
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
    supported_native_tools={
        "web_search",
        "code_execution",
        "google_map",
        "url_context",
    },
)
CAP_OPENAI_REASONING = ModelCapabilities(
    input_modalities={ModelModality.TEXT, ModelModality.IMAGE},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.EFFORT,
    reasoning_visibility="hidden",
    supported_native_tools={
        "web_search",
        "code_execution",
        "computer_use",
        "file_search",
    },
)
CAP_DEEPSEEK_V4 = ModelCapabilities(
    input_modalities={ModelModality.TEXT},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.EFFORT,
    reasoning_visibility="visible",
)
CAP_MINIMAX_REASONING = ModelCapabilities(
    input_modalities={ModelModality.TEXT},
    output_modalities={ModelModality.TEXT},
    supports_tool_calling=True,
    reasoning_mode=ReasoningMode.EFFORT,
    reasoning_visibility="visible",
)

CAP_RERANK_ONLY = ModelCapabilities(
    input_modalities={ModelModality.TEXT, ModelModality.IMAGE},
    is_rerank_model=True,
)

CAP_GEMINI_IMAGE = ModelCapabilities(
    input_modalities={
        ModelModality.TEXT,
        ModelModality.IMAGE,
        ModelModality.AUDIO,
        ModelModality.VIDEO,
    },
    output_modalities={ModelModality.TEXT, ModelModality.IMAGE},
    supports_tool_calling=True,
    supported_native_tools={
        "web_search",
    },
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
    },
    supports_tool_calling=True,
    supported_native_tools={
        "web_search",
        "code_execution",
        "computer_use",
        "file_search",
        "google_map",
        "url_context",
    },
)

MODEL_ALIAS_MAPPING: dict[str, str] = {
    "*DeepSeek-V4-Pro*": "deepseek-v4-pro",
    "*DeepSeek-V4-Flash*": "deepseek-v4-flash",
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
                registry[pattern] = model_copy(cap, update=context_limits)
            else:
                registry[pattern] = cap

    ctx_1m = {"max_input_tokens": 1000000}
    ctx_200k = {"max_input_tokens": 204800}
    ctx_128k = {"max_input_tokens": 128000}
    ctx_8k = {"max_input_tokens": 8192}

    register_family(PATTERNS_GEMINI_IMAGE, CAP_GEMINI_IMAGE, ctx_128k)
    register_family(PATTERNS_GEMINI_3, CAP_GEMINI_3, ctx_1m)
    register_family(PATTERNS_GEMINI_2_5_FLASH, CAP_GEMINI_2_5, ctx_1m)
    register_family(PATTERNS_GEMINI_2_5_PRO, CAP_GEMINI_2_5, ctx_1m)
    register_family(PATTERNS_OPENAI_REASONING, CAP_OPENAI_REASONING, ctx_128k)
    register_family(["gpt-4o*"], CAP_GEMINI_2_5, ctx_128k)
    register_family(["gpt-4-*", "gpt-5*"], STANDARD_TEXT_TOOL_CAPABILITIES, ctx_8k)
    register_family(PATTERNS_DEEPSEEK_PRO, CAP_DEEPSEEK_V4, ctx_128k)
    register_family(PATTERNS_DEEPSEEK_FLASH, STANDARD_TEXT_TOOL_CAPABILITIES, ctx_128k)
    register_family(PATTERNS_MINIMAX_REASONING, CAP_MINIMAX_REASONING, ctx_200k)
    register_family(PATTERNS_1M_MODELS, DEFAULT_PERMISSIVE_CAPABILITIES, ctx_1m)

    register_family(["*reranker*", "*rerank*", "bge-m3*"], CAP_RERANK_ONLY)

    register_family(PATTERNS_GEMINI_EMBEDDING_2, CAP_MULTIMODAL_EMBEDDING, ctx_8k)
    register_family(PATTERNS_JINA_V5_OMNI, CAP_MULTIMODAL_EMBEDDING, ctx_8k)

    register_family(
        PATTERNS_JINA_EMBEDDING,
        ModelCapabilities(
            input_modalities={ModelModality.TEXT},
            is_embedding_model=True,
            supports_tool_calling=False,
        ),
        ctx_8k,
    )

    register_family(PATTERNS_JINA_RERANKER, CAP_RERANK_ONLY, ctx_8k)

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
