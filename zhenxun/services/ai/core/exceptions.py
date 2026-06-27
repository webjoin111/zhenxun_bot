"""
自定义异常与错误码定义
"""

from typing import Any


class ModelRetry(Exception):
    """用于通知大模型修正并重试的异常"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class SchemaParseError(ModelRetry):
    """格式解析异常。当大模型返回的 JSON 损坏或不符合 Schema 时抛出。"""

    def __init__(self, message: str):
        super().__init__(message)


class GuardrailViolationError(ModelRetry):
    """护栏违规异常。当大模型返回的数据格式正确，但违反业务规则时抛出。"""

    def __init__(self, message: str):
        super().__init__(message)


class ControlFlowExit(BaseException):
    """控制流退出基类，继承自BaseException以避免被常规Exception捕获，用于静默中断。"""

    pass


class ToolFatalError(ControlFlowExit):
    """
    致命工具异常（不可恢复）。
    当工具执行遇到权限不足、严重系统故障等大模型无法通过重试解决的问题时抛出。
    这会直接熔断 Agent 推理流，并将 display_content 抛给用户。
    """

    def __init__(self, message: str, display_content: str | None = None):
        self.message = message
        self.display_content = display_content or f"❌ 工具遇到致命错误: {message}"
        super().__init__(self.message)


class GuardrailFatalException(ControlFlowExit):
    """护栏致命拦截异常 (触发 ABORT/REJECT 时抛出)"""

    def __init__(self, guard_name: str, reason: str, display: str | None = None):
        self.guard_name = guard_name
        self.reason = reason
        self.display = display or f"🛡️ 安全拦截: {reason}"
        super().__init__(f"Guardrail '{guard_name}' aborted execution: {reason}")


class ToolRetryError(Exception):
    """
    可恢复工具异常。
    当参数解析错误、业务逻辑校验失败、网络超时等问题发生时抛出。
    会被 ToolExecutor 捕获并转化为引导大模型自我反思 (Reflexion) 的 ToolResult。
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ToolFinishException(ToolFatalError):
    """
    工具执行中止异常。
    当工具开发者希望立刻停止大模型的思考循环，并直接将错误/提示信息返回给用户时抛出。
    此异常不会被大模型进行"影子自愈(Reflexion)"，而是直接熔断 Agent 执行流。
    """

    def __init__(self, message: str, display_content: str | None = None):
        super().__init__(message, display_content)


class AbortException(ControlFlowExit):
    """异常中止当前 Agent 思考流。"""

    def __init__(self, reason: str, display: Any = None):
        self.reason = reason
        self.display = display
        super().__init__(f"Aborted: {reason}")


class ConcurrencyRejectException(ControlFlowExit):
    """并发拒绝异常。当 Agent 设置为 REJECT 且正在忙碌时抛出。"""

    def __init__(self, message: str, display: Any = None):
        self.message = message
        self.display = display or "⏳ 智能体正在处理您的上一个请求，请稍后再试~"
        super().__init__(message)


class ConcurrencyInterruptException(ControlFlowExit):
    """并发打断异常。当 Agent 设置为 INTERRUPT 且被新请求打断时抛出。"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NeedsInputException(Exception):
    """
    当工具配置了 interactive=True 且缺少必要参数（或参数验证失败）时抛出此异常，
    用于交由外部中间件捕获并进行 HITL (Human-in-the-loop) 参数补充。
    """

    def __init__(
        self, missing_field: str, missing_description: str, original_kwargs: dict
    ):
        self.missing_field = missing_field
        self.missing_description = missing_description
        self.original_kwargs = original_kwargs
        super().__init__(
            f"Need input for parameter: {missing_field} - {missing_description}"
        )


class NeedsAuthException(Exception):
    """
    当工具在执行过程中发现授权失效或凭证过期时主动抛出。
    用于交由外部中间件捕获并重新发起授权 (HITL) 流程。
    """

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"Needs auth for: {provider} - {message}")


class SandboxPathEscapeError(Exception):
    """当沙箱内的路径解析结果试图逃逸出允许的工作区根目录时抛出"""

    def __init__(self, path: str, resolved_path: str | None = None, reason: str = ""):
        self.path = path
        self.resolved_path = resolved_path
        self.reason = reason
        msg = f"沙箱路径逃逸拦截: {path}"
        if resolved_path:
            msg += f" (解析至 {resolved_path})"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg)


class WorkspaceIOError(Exception):
    """沙箱文件系统读写操作失败"""

    def __init__(self, path: str, message: str, cause: Exception | None = None):
        self.path = path
        self.cause = cause
        super().__init__(f"沙箱 IO 异常 [{path}]: {message}")


class SandboxFatalError(Exception):
    """沙箱底层容器发生致命崩溃（如 OOM, 被宿主机强杀等）"""

    pass


class LLMException(Exception):
    """LLM 服务相关的基础异常类 (多态基类)"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(message)

    @property
    def is_retryable(self) -> bool:
        """是否允许在当前节点进行退避重试（如偶发网络抖动）"""
        return False

    @property
    def should_failover(self) -> bool:
        """是否允许触发节点故障转移（切换到下一个备用模型）"""
        return False

    @property
    def should_rotate_key(self) -> bool:
        """是否应该标记当前 Key 失效并轮换 API Key"""
        return False

    @property
    def user_friendly_message(self) -> str:
        """返回适合向用户展示的错误消息"""
        return "AI服务暂时不可用，请稍后再试。"

    def __str__(self) -> str:
        if self.details:
            safe_details = {k: v for k, v in self.details.items() if k != "api_key"}
            if safe_details:
                return f"{self.message} (详情: {safe_details})"
        return self.message


class InvalidRequestException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "请求参数错误或API类型不支持，请检查输入内容。"


class ContextLengthExceededException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "输入内容过长，请缩短后重试。"


class ContentFilteredException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "内容被安全过滤，请修改后重试。"


class ConfigurationException(LLMException):
    @property
    def user_friendly_message(self) -> str:
        return "AI模型配置错误或未找到，请联系管理员检查配置。"


class AuthenticationException(LLMException):
    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "API密钥无效或权限不足，请联系管理员更新配置。"


class QuotaExceededException(LLMException):
    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "API使用配额已用尽，请稍后再试或联系管理员。"


class LocationNotSupportedException(LLMException):
    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return (
            "当前网络环境不支持此 AI 模型。\n"
            "建议: 请尝试更换代理节点至支持的地区或切换备用模型。"
        )


class RateLimitException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def should_rotate_key(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "请求过于频繁，已被AI服务限流，请稍后再试。"


class UpstreamServerException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务响应异常或端点宕机，请稍后再试。"


class NetworkTimeoutException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def should_failover(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务请求超时，请稍后再试。"


class ResponseParseException(LLMException):
    @property
    def is_retryable(self) -> bool:
        return True

    @property
    def user_friendly_message(self) -> str:
        return "AI服务响应解析失败，请稍后再试。"


def get_user_friendly_error_message(error: Exception) -> str:
    """将任何异常转换为用户友好的错误消息"""
    if isinstance(error, LLMException):
        return error.user_friendly_message

    error_str = str(error).lower()

    if "timeout" in error_str or "timed out" in error_str:
        return "网络请求超时，请检查服务器网络或代理连接。"
    if "connect" in error_str and ("refused" in error_str or "error" in error_str):
        return "无法连接到 AI 服务商，请检查网络连接或代理设置。"
    if "proxy" in error_str:
        return "代理连接失败，请检查代理服务器是否正常运行。"
    if "ssl" in error_str or "certificate" in error_str:
        return "SSL 证书验证失败，请检查网络环境。"

    return f"服务暂时不可用 ({type(error).__name__})，请稍后再试。"
