"""
工具系统域类型定义
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import ToolCallPart, UsageInfo
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

    error_type: ToolErrorType = Field(...)
    """错误的类型。"""
    message: str = Field(...)
    """对错误的详细描述。"""
    is_retryable: bool = Field(False)
    """指示这个错误是否可能通过重试解决。"""


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


class ToolResult(BaseModel):
    """结构化的工具执行结果模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = Field(...)
    """大模型实际看到的执行结果，可以是字符串、字典等序列化对象。"""
    usage: UsageInfo | None = Field(default=None)
    """子智能体或复杂工具消耗的 Token 统计，将向主流程冒泡累加。"""

    ui_display: Any = Field(default=None)
    """专门用于前端群聊展示的内容 (str 或 UniMessage)"""
    log_content: str | None = Field(default=None)
    """后台日志专属摘要"""
    is_error: bool = Field(default=False)
    """是否发生了业务级别的错误"""
    is_retryable: bool = Field(default=True)
    """标记该错误是否允许大模型进行自愈反思重试"""

    def show_to_user(self, display: Any) -> "ToolResult":
        """链式方法：将此结果或特定富文本推送到群聊前端渲染"""
        self.ui_display = display
        return self

    def with_log(self, log_str: str) -> "ToolResult":
        """链式方法：仅在后台控制台打印该工具的摘要说明"""
        self.log_content = log_str
        return self

    def as_error(self, is_retryable: bool = True) -> "ToolResult":
        """链式方法：标记此结果为错误，并引导大模型在下一轮进行重试自愈"""
        self.is_error = True
        self.is_retryable = is_retryable
        return self

    def as_fatal(self) -> "ToolResult":
        """链式方法：标记此结果为致命错误，立即中断大模型的思考"""
        self.is_error = True
        self.is_retryable = False
        return self


class StateSyncResult(ToolResult):
    """
    状态同步结果模型。
    除了返回工具输出外，允许开发者向大模型发送一条“系统通知”，系统会自动将其追加到上下文中，防止大模型产生幻觉。
    """

    state_notice: str | None = Field(default=None)
    """状态同步通知文本，将被自动转化为 SystemPrompt 发送给大模型。"""

    def with_state_notice(self, notice: str) -> "StateSyncResult":
        """链式方法：设置状态同步通知"""
        self.state_notice = notice
        return self


class ToolResultChunk(BaseModel):
    """流式工具执行结果片段模型"""

    content: str = Field(...)
    """流式输出的文本片段"""
    status: str = Field(default="running")
    """当前状态 (如 running, finished)"""
    metadata: dict[str, Any] | None = Field(default=None)
    """携带的附加数据 (如进度比例、图片等)"""


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


class ToolOptions(BaseModel):
    """工具的高阶配置选项"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strict: bool = Field(default=False)
    """是否开启严格的 JSON Schema 验证模式，开启后大模型的参数将不接受额外属性。"""
    max_usage_count: int | None = Field(default=None)
    """单次 Agent 会话中的最大允许调用次数，用于防止大模型陷入死循环调用。"""
    capabilities: list[Any] = Field(default_factory=list)
    """当前工具专属的生命周期拦截器 (Capabilities) 列表。"""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """额外扩展元数据字典，可供其他系统或自定义中间件读取。"""
    sandbox_requirements: dict[str, list[str]] | None = Field(default=None)
    """声明该工具在沙箱中执行时的环境依赖要求。"""
    tags: list[str] = Field(default_factory=list)
    """用于智能字符串路由和能力聚合的标签列表。"""
    max_retries: int | None = Field(default=None)
    """工具级别的局部重试上限。优先级高于全局配置。"""
    args_schema: type[BaseModel] | None = Field(default=None)
    """工具的 Pydantic 数据模型约束。大模型将以此 Schema 输出 JSON。"""

    def merge(self, other: "ToolOptions | None") -> "ToolOptions":
        """组合模式底层：合并另一个 ToolOptions，other 中的非默认值将覆盖当前值"""
        if not other:
            return self
        merged_data = model_dump(self, exclude_unset=False)
        other_data = model_dump(other, exclude_unset=True)

        if other.capabilities:
            merged_data["capabilities"] = self.capabilities + other.capabilities
        if other.metadata:
            merged_data["metadata"] = {**self.metadata, **other.metadata}
        if other.tags:
            merged_data["tags"] = list(set(self.tags + other.tags))

        for k, v in other_data.items():
            if k not in ("capabilities", "metadata", "tags"):
                merged_data[k] = v
        return model_validate(ToolOptions, merged_data)


class ToolkitConfig(BaseModel):
    """工具箱全局配置对象 (用于声明式配置解析)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prefix: str | None = Field(default=None)
    """工具名称前缀"""
    include: list[str] | None = Field(default=None)
    """允许注册的工具名白名单"""
    exclude: list[str] | None = Field(default=None)
    """排除注册的工具名黑名单"""
    global_capabilities: list[Any] = Field(default_factory=list)
    """全局拦截器"""
    global_tags: list[str] = Field(default_factory=list)
    """全局标签，该标签将被自动下发并注入到工具箱内的所有子工具中"""


class ToolOverride(BaseModel):
    """工具配置动态覆盖载体"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """要覆盖的目标工具名称（在全局注册表或 Provider 中的原始名称）"""

    new_name: str | None = None
    """克隆后的新工具名称。如果为空，则保持原名称"""

    description: str | None = None
    """覆盖后的新描述。大模型将根据此新描述决定工具调用时机"""

    options: ToolOptions | None = None
    """用于覆盖该工具底层行为的高阶配置项"""

    def to_tool_options(self) -> ToolOptions:
        return self.options or ToolOptions()

    async def resolve(self, context: Any | None = None) -> "ResolvedToolPayload":
        from zhenxun.services.ai.tools.engine.registry import tool_provider_manager
        from zhenxun.services.ai.tools.models import ResolvedToolPayload
        from zhenxun.services.log import logger

        found_tools = await tool_provider_manager.resolve_specific_tools([self.name])
        if found_tools:
            base_tool = found_tools[0]
            if hasattr(base_tool, "clone_with_options"):
                cloned_tool = base_tool.clone_with_options(self)
                if hasattr(cloned_tool, "resolve"):
                    return await cloned_tool.resolve(context)
                return ResolvedToolPayload(tools=[cloned_tool])
            else:
                logger.warning(f"工具 {self.name} 不支持动态覆盖，将原样装配。")
                if hasattr(base_tool, "resolve"):
                    return await base_tool.resolve(context)
                return ResolvedToolPayload(tools=[base_tool])

        logger.warning(f"ToolOverride 找不到目标基础工具: {self.name}")
        return ResolvedToolPayload()


class GlobalToolFilter(BaseModel):
    """全局宏观工具过滤器"""

    allowed_servers: list[str] | None = None
    """仅允许的服务端名称列表"""
    excluded_servers: list[str] | None = None
    """需要排除的服务端名称列表"""


class ValidatedToolCall(BaseModel):
    """工具调用验证结果载体（解耦验证与执行）"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: ToolCallPart = Field(...)
    """原始工具调用部件"""
    tool: Any | None = Field(default=None)
    """匹配到的目标工具实例"""
    args_valid: bool = Field(default=False)
    """参数是否成功通过校验"""
    validated_args: dict[str, Any] | None = Field(default=None)
    """通过验证并反序列化后的参数字典"""
    validation_error: Exception | None = Field(default=None)
    """验证失败时的异常信息"""


class Query(BaseModel):
    """
    工具的声明式查询对象。
    用于在 Agent 中精确或批量筛选加载特定命名空间、特定标签的工具。
    """

    name: str | None = Field(default=None, description="工具名称精确匹配")
    """如果提供，则必须与工具的名称完全一致。"""
    tags: list[str] | None = Field(default=None, description="工具标签交集匹配 (AND)")
    """如果提供，则工具必须包含这里列出的所有标签 (交集/AND匹配)。"""
    namespace: str | None = Field(
        default=None, description="搜索的命名空间，'global' 代表所有"
    )
    """必填(由系统补充或显式声明)。限制搜索的插件命名空间，'global' 将跨全插件搜索。"""
    metadata_filter: dict[str, Any] | None = Field(
        default=None, description="工具元数据精确匹配 (字典子集匹配)"
    )
    """如果提供，则工具的 metadata 必须包含这里列出的所有键值对。"""

    def match(self, tool: Any) -> bool:
        """判断某个工具或工具箱是否符合当前 Query 的筛选条件"""
        if self.name:
            tool_name = getattr(tool, "name", getattr(tool, "__class__", type).__name__)
            if tool_name != self.name:
                return False
        if self.tags:
            if hasattr(tool, "config") and hasattr(tool.config, "global_tags"):
                tool_tags = tool.config.global_tags
            else:
                tool_settings = getattr(tool, "settings", None)
                tool_tags = getattr(tool_settings, "tags", []) if tool_settings else []
            if not all(tag in tool_tags for tag in self.tags):
                return False

        if self.metadata_filter:
            tool_settings = getattr(tool, "settings", None)
            tool_metadata = (
                getattr(tool_settings, "metadata", getattr(tool, "metadata", {}))
                if tool_settings
                else getattr(tool, "metadata", {})
            )
            for k, v in self.metadata_filter.items():
                if tool_metadata.get(k) != v:
                    return False
        return True


@dataclass
class ResolvedToolPayload:
    """解析后的工具上下文包"""

    tools: list[Any] = field(default_factory=list)
    injected_prompts: list[str] = field(default_factory=list)
    toolkits: list[Any] = field(default_factory=list)


__all__ = [
    "CodeExecutionOutcome",
    "GlobalToolFilter",
    "Query",
    "ResolvedToolPayload",
    "TaskType",
    "ToolCategory",
    "ToolChoice",
    "ToolDefinition",
    "ToolErrorResult",
    "ToolErrorType",
    "ToolMetadata",
    "ToolOptions",
    "ToolOverride",
    "ToolResult",
    "ToolResultChunk",
    "ToolkitConfig",
    "ValidatedToolCall",
]
