"""
工具系统域类型定义
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.utils.pydantic_compat import model_dump, model_validate

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


class ToolCategory(Enum):
    """工具分类枚举"""

    FILE_SYSTEM = auto()
    NETWORK = auto()
    SYSTEM_INFO = auto()
    CALCULATION = auto()
    DATA_PROCESSING = auto()
    CUSTOM = auto()


class ToolErrorType(StrEnum):
    """结构化工具错误的类型枚举。"""

    TOOL_NOT_FOUND = "ToolNotFound"
    INVALID_ARGUMENTS = "InvalidArguments"
    EXECUTION_ERROR = "ExecutionError"
    USER_CANCELLATION = "UserCancellation"


class ToolErrorResult(BaseModel):
    """一个结构化的工具执行错误模型。"""

    error_type: ToolErrorType = Field(..., description="错误的类型。")
    message: str = Field(..., description="对错误的详细描述。")
    is_retryable: bool = Field(False, description="指示这个错误是否可能通过重试解决。")


class CodeExecutionOutcome(StrEnum):
    """代码执行结果状态枚举"""

    OUTCOME_OK = "OUTCOME_OK"
    OUTCOME_FAILED = "OUTCOME_FAILED"
    OUTCOME_DEADLINE_EXCEEDED = "OUTCOME_DEADLINE_EXCEEDED"
    OUTCOME_COMPILATION_ERROR = "OUTCOME_COMPILATION_ERROR"
    OUTCOME_RUNTIME_ERROR = "OUTCOME_RUNTIME_ERROR"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class TaskType(Enum):
    """任务类型枚举"""

    CHAT = "chat"
    CODE = "code"
    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    MULTIMODAL = "multimodal"


class ToolDefinition(BaseModel):
    """结构化的工具定义模型"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema 参数"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class ToolChoice(BaseModel):
    """工具选择配置"""

    mode: Literal["auto", "none", "any", "required"] = Field(default="auto")
    allowed_function_names: list[str] | None = Field(default=None)


class BasePlatformTool(BaseModel):
    """平台原生工具基类"""

    execution_side: Literal["client", "server"] = "server"
    tool_type: str = Field(default="unknown")

    class Config:
        extra = "forbid"

    def get_tool_declaration(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_tool_config(self) -> dict[str, Any] | None:
        return None


class GeminiCodeExecution(BasePlatformTool):
    tool_type: str = "code_execution"

    def get_tool_declaration(self) -> dict[str, Any]:
        return {"code_execution": {}}


class GeminiGoogleSearch(BasePlatformTool):
    tool_type: str = "google_search"
    mode: Literal["MODE_DYNAMIC"] = "MODE_DYNAMIC"
    dynamic_threshold: float | None = Field(default=None)

    def get_tool_declaration(self) -> dict[str, Any]:
        return {"google_search": {}}


class GeminiUrlContext(BasePlatformTool):
    tool_type: str = "url_context"

    def get_tool_declaration(self) -> dict[str, Any]:
        return {"urlContext": {}}


class GeminiGoogleMaps(BasePlatformTool):
    tool_type: str = "google_map"

    def get_tool_declaration(self) -> dict[str, Any]:
        return {"googleMap": {}}


class ToolResult(BaseModel):
    """结构化的工具执行结果模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = Field(default=None)
    """大模型实际看到的执行结果，可以是字符串、字典或任何可 JSON 序列化的对象。"""
    display: Any = Field(default=None)
    """用于在前端 UI（如群聊消息）展示给用户看的内容。如果为 None，则不展示。类型为 str 或 UniMessage。"""
    log_content: str | None = Field(default=None)
    """专门输出到后台日志的摘要内容，不会发往用户端。"""
    text: str | None = Field(default=None)
    """附加的纯文本信息。"""
    is_error: bool = Field(default=False)
    """标记本次工具执行是否发生了业务上的错误。"""
    session_state_updates: dict[str, Any] | None = Field(default=None)
    """返回一个字典，用于合并更新当前会话的上下文状态。"""
    system_prompt_append: str | None = Field(default=None)
    """追加一段系统指令到后续的对话中，用于动态引导大模型。"""
    terminate_run: bool = Field(default=False)
    """如果设为 True，将强制终止大模型的当前推理循环。"""

    def notify_user(self, display: Any) -> "ToolResult":
        """链式调用：设置在前端群聊 UI 中展示给用户的文本或数据"""
        self.display = display
        return self

    def terminate(self) -> "ToolResult":
        """链式调用：强制终止大模型的推理循环，直接返回当前结果"""
        self.terminate_run = True
        return self

    def alert_ai(self, prompt: str) -> "ToolResult":
        """链式调用：向大模型追加一段系统级的警告或引导指令"""
        if self.system_prompt_append:
            self.system_prompt_append += f"\n{prompt}"
        else:
            self.system_prompt_append = prompt
        return self

    def hide(self) -> "ToolResult":
        """链式调用：将当前工具结果设为静默，不展示给前端用户"""
        self.display = None
        return self


class ToolResultChunk(BaseModel):
    """流式工具执行结果片段模型"""

    content: str = Field(..., description="流式输出的文本片段")
    status: str = Field(
        default="running", description="当前状态 (如 running, finished)"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="携带的附加数据 (如进度比例、图片等)"
    )


@dataclass
class ToolMetadata:
    """工具元数据"""

    name: str
    description: str
    category: ToolCategory
    read_only: bool = True
    destructive: bool = False
    open_world: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)


class MCPSource(BaseModel):
    """显式定义的 MCP 工具源"""

    server_name: str
    namespace: str | None = None
    tool_whitelist: list[str] | None = None

    def __hash__(self):
        return hash(
            (self.server_name, self.namespace, tuple(self.tool_whitelist or []))
        )

class ToolOptions(BaseModel):
    """工具的高阶配置选项"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cache: bool = Field(default=False)
    """是否启用极速缓存。开启后，如果参数 Hash 相同，将直接返回上次的结果，不再执行。"""
    cache_ttl: int = Field(default=3600)
    """缓存的过期时间（秒），默认 3600 秒。"""
    require_approval: bool = Field(default=False)
    """是否需要人工审批（HITL）。开启后，执行前会在群聊发起授权确认。"""
    result_as_answer: bool = Field(default=False)
    """将工具结果直接作为大模型的最终回答，效果同 direct_reply。"""
    direct_reply: bool = Field(default=False)
    """执行后直接回复工具结果，并终止 Agent 推理循环。"""
    silent: bool = Field(default=False)
    """静默执行，工具的结果不会在前端 UI 渲染展示，仅供大模型后台参考。"""
    strict: bool = Field(default=False)
    """是否开启严格的 JSON Schema 验证模式，开启后大模型的参数将不接受额外属性。"""
    interactive: bool = Field(default=False)
    """标记为交互式工具。如果参数缺失或校验失败，会主动通过 Bot 向用户提问要求补全。"""
    max_usage_count: int | None = Field(default=None)
    """单次 Agent 会话中的最大允许调用次数，用于防止大模型陷入死循环调用。"""
    cache_function: Any | None = Field(default=None)
    """自定义的缓存判定函数，接收参数和结果，返回 True 才缓存结果。"""
    pre_hook: Any | None = Field(default=None)
    """工具执行前置钩子函数。"""
    post_hook: Any | None = Field(default=None)
    """工具执行后置钩子函数。"""
    fallback_tool: str | None = Field(default=None)
    """执行发生异常时的降级备用工具名称，系统将自动重定向至该工具。"""
    middlewares: list[Any] = Field(default_factory=list)
    """当前工具专属的中间件列表。"""
    prepare: Any | None = Field(default=None)
    """工具在发往大模型前的动态 Schema 篡改与可见性判定钩子。"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """额外扩展元数据字典，可供其他系统或自定义中间件读取。"""
    sandbox_requirements: dict[str, list[str]] | None = Field(default=None)
    """声明该工具在沙箱中执行时的环境依赖要求。"""
    args_validator: Any | None = Field(default=None)
    """参数业务级别校验钩子。若校验失败抛出异常，将自动转换为大模型重试信号。"""
    max_retries: int | None = Field(default=None)
    """工具级别的局部重试上限。优先级高于全局配置。"""

    def merge(self, other: "ToolOptions | None") -> "ToolOptions":
        """组合模式底层：合并另一个 ToolOptions，other 中的非默认值将覆盖当前值"""
        if not other:
            return self
        merged_data = model_dump(self, exclude_unset=False)
        other_data = model_dump(other, exclude_unset=True)

        if other.middlewares:
            merged_data["middlewares"] = self.middlewares + other.middlewares
        if other.prepare:
            if self.prepare:
                old_p = self.prepare
                new_p = other.prepare

                def make_chained(p1, p2):
                    async def chained(ctx, tdef):
                        from nonebot.utils import is_coroutine_callable

                        res1 = (
                            await p1(ctx, tdef)
                            if is_coroutine_callable(p1)
                            else p1(ctx, tdef)
                        )
                        if res1 is None:
                            return None
                        res2 = (
                            await p2(ctx, res1)
                            if is_coroutine_callable(p2)
                            else p2(ctx, res1)
                        )
                        return res2

                    return chained

                merged_data["prepare"] = make_chained(old_p, new_p)
            else:
                merged_data["prepare"] = other.prepare
        if other.metadata:
            merged_data["metadata"] = {**self.metadata, **other.metadata}

        for k, v in other_data.items():
            if k not in ("middlewares", "prepare", "metadata"):
                merged_data[k] = v
        return model_validate(ToolOptions, merged_data)


class ToolOverride(BaseModel):
    """工具配置动态覆盖载体"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """要覆盖的目标工具名称（在全局注册表或 Provider 中的原始名称）"""

    new_name: str | None = None
    """克隆后的新工具名称。如果为空，则保持原名称"""

    description: str | None = None
    """覆盖后的新描述。大模型将根据此新描述决定工具调用时机"""

    cache: bool | None = None
    """是否启用极速缓存"""

    cache_ttl: int | None = None
    """极速缓存过期时间(秒)"""

    require_approval: bool | None = None
    """是否拦截并要求人类审批(HITL)"""

    result_as_answer: bool | None = None
    """是否将结果直接作为大模型最终回答"""

    direct_reply: bool | None = None
    """是否直接回复结果并中断推理流"""

    silent: bool | None = None
    """是否静默执行，禁止在前端展示气泡"""

    strict: bool | None = None
    """是否强制开启 OpenAI 严格结构化输出"""

    interactive: bool | None = None
    """是否开启交互式参数补全补救"""

    max_usage_count: int | None = None
    """单次会话最大调用次数限制"""

    fallback_tool: str | None = None
    """发生异常时的透明容错降级目标工具"""

    middlewares: list[Any] | None = None
    """针对该工具实例独享的自定义中间件"""

    prepare: Any | None = None
    """覆盖执行前置结构篡改与可见性判定钩子"""

    metadata: dict[str, Any] | None = None
    """覆盖底层的额外业务字典数据"""

    args_validator: Any | None = None
    """覆盖执行前置业务参数校验钩子"""

    max_retries: int | None = None
    """覆盖工具局部的最大重试次数"""

    def to_tool_options(self) -> ToolOptions:
        settings = ToolOptions()
        for field in self.model_fields_set:
            if field in ("name", "new_name", "description"):
                continue
            setattr(settings, field, getattr(self, field))
        return settings


class GlobalToolFilter(BaseModel):
    """全局宏观工具过滤器"""

    allowed_servers: list[str] | None = None
    excluded_servers: list[str] | None = None


@dataclass
class ResolvedToolPayload:
    """解析后的工具上下文包"""

    tools: Any = field(default_factory=dict)
    injected_prompts: list[str] = field(default_factory=list)
    toolkits: list[Any] = field(default_factory=list)


__all__ = [
    "BasePlatformTool",
    "CodeExecutionOutcome",
    "GeminiCodeExecution",
    "GeminiGoogleSearch",
    "GeminiUrlContext",
    "GlobalToolFilter",
    "MCPSource",
    "ResolvedToolPayload",
    "TaskType",
    "ToolCategory",
    "ToolChoice",
    "ToolDefinition",
    "ToolErrorResult",
    "ToolErrorType",
    "ToolMetadata",
    "ToolOverride",
    "ToolResult",
    "ToolResultChunk",
]
