
from pydantic import BaseModel, Field

from zhenxun.services.ai.core.models import ModelDetail


class DebugLogOptions(BaseModel):
    """调试日志细粒度控制选项"""

    show_tools: bool = True
    """是否在日志中显示工具定义 (JSON Schema)"""
    show_schema: bool = True
    """是否在日志中显示结构化输出 Schema (response_format)"""
    show_safety: bool = True
    """是否在日志中显示安全设置 (safetySettings)"""

    def __bool__(self) -> bool:
        return self.show_tools or self.show_schema or self.show_safety


class ClientSettings(BaseModel):
    """LLM 客户端底层网络与重试设置"""

    timeout: int = 300
    """API 请求超时时间 (秒)"""
    max_retries: int = 3
    """请求失败时的最大重试次数"""
    retry_delay: int = 2
    """请求重试的基础延迟时间 (秒)"""
    structured_retries: int = 2
    """结构化生成校验失败时的最大重试次数 (IVR)"""


class LLMSummaryConfig(BaseModel):
    """LLM 自然语言总结压缩策略配置"""

    enable: bool = True
    """是否开启大模型对话总结以压缩上下文"""
    trigger_threshold: float = 0.8
    """触发压缩的 Token 阈值。<=1.0 为比例，>1.0 为绝对 Token 数"""
    max_history_turns: int = 0
    """触发压缩的最大历史对话轮数。设为 0 表示不限制轮数（仅受 Token 阈值控制）。"""
    summarization_model: str | None = "Gemini/gemini-3.5-flash"
    """指定用于执行总结任务的大模型名称，为空则使用全局默认"""
    summarization_prompt: str = (
        "请以客观、精炼的语言概括以下对话内容。重点保留："
        "1. 核心讨论话题及重要决定；"
        "2. 用户的个性特征、核心偏好、提及的生活背景或特殊设定；"
        "3. 双方互动的温度与情感基调。无需保留寒暄等客套话。"
    )
    """指导大模型进行总结的系统提示词"""
    keep_recent_turns: int = 3
    """在总结之外，强制原样保留的最近对话轮数"""


class ContextManagementSettings(BaseModel):
    """智能上下文管理与压缩算法设置"""

    llm_summary: LLMSummaryConfig = Field(default_factory=LLMSummaryConfig)
    """大模型自然语言总结策略"""

    vision_window_size: int = Field(default=3)
    """多模态滑动窗口大小。0表示无限制，>0表示仅保留最近N轮包含多模态真实数据的消息，超龄则自动降级为占位符"""


class ProviderConfig(BaseModel):
    """LLM 服务提供商 (接口方) 配置模型"""

    name: str
    """提供商唯一标识名称"""
    api_key: str | list[str]
    """API 密钥或密钥列表 (支持轮询)"""
    api_base: str | None = None
    """API 基础 URL 路径"""
    api_type: str = "openai"
    """API 协议类型 (openai/gemini/zhipu/etc.)"""
    openai_compat: bool = False
    """是否强制使用 OpenAI 兼容模式"""
    temperature: float | None = 0.7
    """该提供商下模型的默认温度"""
    generation_max_tokens: int | None = None
    """该提供商下模型的默认最大输出限制"""
    models: list[ModelDetail]
    """该提供商提供的具体模型列表"""
    timeout: int = 180
    """针对该提供商的特定超时时间"""


class DefaultModelsConfig(BaseModel):
    """按任务分类的默认模型配置"""
    chat: str | None = Field(default="Gemini/gemini-3.5-flash")
    embedding: str | None = Field(default="Gemini/gemini-embedding-2")
    tts: str | None = Field(default="Gemini/gemini-3.1-flash-tts-preview")
    image: str | None = Field(default="Gemini/gemini-2.5-flash-image")
    rerank: str | None = Field(default="siliconflow/BAAI/bge-reranker-v2-m3")


class AgentEngineSettings(BaseModel):
    """全局默认的 Agent 推理引擎配置"""

    max_cycles: int = 10
    """工具调用最大循环次数"""
    enable_parallel_calls: bool = True
    """允许并行工具调用"""
    reflexion_retries: int = 1
    """反思重试次数"""
    enable_fallback_summary: bool = True
    """达到最大循环次数时，是否触发大模型兜底总结（而不是直接报错）"""
    enable_hitl: bool = True
    """是否允许智能体主动挂起任务，向用户求助 (Human-in-the-Loop)"""


class LLMConfig(BaseModel):
    """AI 模块全局持久化配置总模型"""

    default_models: DefaultModelsConfig = Field(default_factory=DefaultModelsConfig)
    """全局按任务分类的默认模型路由表"""
    client_settings: ClientSettings = Field(default_factory=ClientSettings)
    """客户端通用连接配置"""
    providers: list[ProviderConfig] = Field(default_factory=list)
    """已配置的提供商列表"""
    debug_log: DebugLogOptions = Field(default_factory=DebugLogOptions)
    """日志调试开关配置"""
    context_settings: ContextManagementSettings = Field(
        default_factory=ContextManagementSettings
    )
    """上下文管理相关配置"""
    model_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "cheap_models": [
                "Gemini/gemini-3.5-flash",
                "Doubao/doubao-seed-1-6-250615",
            ],
        }
    )
    """虚拟模型路由组配置 (Virtual Router Groups)"""
    agent_settings: AgentEngineSettings = Field(default_factory=AgentEngineSettings)
    """Agent 执行引擎层核心默认参数配置"""

    def validate_model_name(self, provider_model_name: str) -> bool:
        """验证模型名称在当前配置中是否存在"""
        if "/" not in provider_model_name:
            return provider_model_name.strip() in self.model_groups
        if not provider_model_name or "/" not in provider_model_name:
            return False
        parts = provider_model_name.split("/", 1)
        p_name, m_name = parts[0], parts[1]
        for p in self.providers:
            if p.name == p_name:
                for m in p.models:
                    if m.model_name == m_name:
                        return True
        return False
