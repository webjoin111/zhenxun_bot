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
    proxy: str | None = None
    """网络代理地址，例如 http://127.0.0.1:7890"""


class ContextManagementSettings(BaseModel):
    """智能上下文管理与压缩算法设置"""

    enabled: bool = True
    """是否启用智能上下文管理"""
    enable_summarization: bool = False
    """是否启用大模型智能总结"""
    use_structured_summarizer: bool = False
    """是否使用强制结构化状态压缩（适合 RPG 游戏或长线任务）"""
    summarization_model: str = "Gemini/gemini-2.5-flash"
    """用于执行总结任务的模型名称"""
    summarization_prompt: str = "请概括以下对话内容，保留关键的约束条件、用户偏好、已完成的任务状态和未解决的问题。"
    """总结任务的系统提示词模板"""
    trigger_threshold: float = 0.8
    """触发压缩的阈值。<=1.0 为比例，>1.0 为绝对 Token 数"""
    max_history_turns: int | None = None
    """触发压缩的最大历史对话轮数"""


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
    proxy: str | None = None
    """针对该提供商的特定代理设置"""


class LLMConfig(BaseModel):
    """AI 模块全局持久化配置总模型"""

    default_model_name: str | None = None
    """全局默认使用的模型名称 (格式: ProviderName/ModelName)"""
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
    model_groups: dict[str, list[str]] = Field(default_factory=dict)
    """虚拟模型路由组配置 (Virtual Router Groups)"""

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

