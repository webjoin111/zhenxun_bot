from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    FUNCTION = "Function"
    STEP = "Step"
    STEPS = "Steps"
    LOOP = "Loop"
    PARALLEL = "Parallel"
    CONDITION = "Condition"
    ROUTER = "Router"


class OnReject(str, Enum):
    SKIP = "skip"
    CANCEL = "cancel"
    ELSE_BRANCH = "else"


class OnError(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    PAUSE = "pause"


class WorkflowExecutionInput(BaseModel):
    """工作流初始输入负载"""

    input: Any = Field(default=None)
    """初始输入，可以是字符串、字典或多模态列表"""

    additional_data: dict[str, Any] = Field(default_factory=dict)
    """附加元数据"""


class StepInput(BaseModel):
    """传递给每个 Step 的标准输入结构"""

    input: Any = Field(default=None)
    """继承自 Workflow 的初始输入"""

    previous_step_content: Any = Field(default=None)
    """上一个执行步骤产生的直接输出内容"""

    additional_data: dict[str, Any] = Field(default_factory=dict)
    """在生命周期中穿透传递的附加数据"""


class StepOutput(BaseModel):
    """每个 Step 的标准输出结构"""

    step_name: str | None = None
    """步骤的名称"""
    step_id: str | None = None
    """步骤的唯一标识"""
    step_type: StepType | None = None
    """步骤的节点枚举类型"""
    executor_type: str | None = None
    """底层执行器的类型标识"""
    executor_name: str | None = None
    """底层执行器的具体名称"""

    content: Any = None
    """该步骤产生的直接输出内容"""
    success: bool = True
    """标记该步骤是否执行成功"""
    error: str | None = None
    """执行失败时的异常详情"""
    stop: bool = False
    """标记是否触发了终止信号，以阻断后续流程的执行"""
    is_paused: bool = False
    """标记该步骤是否因等待外力交互 (HITL) 而处于挂起状态"""
    pause_reason: str | None = None
    """导致步骤挂起的原因描述"""

    steps: list["StepOutput"] | None = None
    """嵌套步骤的输出结果集合（如复合节点 Loop、Parallel 的内部产出）"""


class UserInputField(BaseModel):
    """HITL 交互请求所需的自定义输入字段"""

    name: str
    """交互字段的参数名称"""
    field_type: str = "str"
    """交互字段的数据类型"""
    description: str | None = None
    """展示给用户的详细引导描述"""
    value: Any | None = None
    """用户实际填入的值"""
    required: bool = True
    """标记此交互字段是否必填"""
    allowed_values: list[Any] | None = None
    """允许选择的枚举值列表"""


class StepRequirement(BaseModel):
    """HITL 挂起状态的约束条件（如等待授权、等待输入等）"""

    step_id: str
    """引发挂起步骤的唯一标识"""
    step_name: str | None = None
    """引发挂起步骤的名称"""
    step_index: int | None = None
    """引发挂起步骤的排序索引"""
    step_type: str | None = None
    """引发挂起步骤的节点类型"""

    requires_confirmation: bool = False
    """标识是否处于等待人工确认授权阶段"""
    confirmation_message: str | None = None
    """展示给审批者的确认提示信息"""
    confirmed: bool | None = None
    """是否已同意授权"""
    on_reject: OnReject | str = OnReject.CANCEL
    """用户拒绝授权时执行的挽救策略"""

    requires_user_input: bool = False
    """标识是否处于等待用户补全字段信息阶段"""
    user_input_message: str | None = None
    """展示给用户的输入提示说明"""
    user_input_schema: list[UserInputField] | None = None
    """期望用户补全的字段定义数据结构"""
    user_input: dict[str, Any] | None = None
    """用户实际提交的输入内容字典"""

    requires_route_selection: bool = False
    """标识是否处于等待用户进行路由分支选择阶段"""
    available_choices: list[str] | None = None
    """提供给用户选择的路由分支名称"""
    allow_multiple_selections: bool = False
    """是否允许用户选中多个分支并发执行"""
    selected_choices: list[str] | None = None
    """用户实际选定的路由分支列表"""

    @property
    def is_resolved(self) -> bool:
        if self.requires_confirmation and self.confirmed is None:
            return False
        if self.requires_user_input and not self.user_input:
            return False
        if self.requires_route_selection and not self.selected_choices:
            return False
        return True


class ErrorRequirement(BaseModel):
    """错误导致 HITL 挂起时的要求"""

    step_id: str
    """发生异常步骤的唯一标识"""
    step_name: str | None = None
    """发生异常步骤的名称"""
    step_index: int | None = None
    """发生异常步骤的排序索引"""
    error_message: str = ""
    """抛出的异常详情信息"""
    error_type: str | None = None
    """异常类型的分类标记"""
    retry_count: int = 0
    """此前已尝试过重试的次数"""
    decision: str | None = None
    """针对该错误进行处置的裁定指令"""


class WorkflowRunResult(BaseModel):
    """企业级工作流运行结果（包含断点快照状态）"""

    workflow_id: str
    """工作流实例运行的唯一标识"""
    workflow_name: str
    """工作流的名称"""
    status: str
    """流水线的最终运行状态 (completed, paused, error 等)"""
    original_input: Any
    """最初传入根节点的原始输入"""
    state: dict[str, Any]
    """工作流生命周期中的全局共享上下文状态字典"""
    step_outputs: dict[str, StepOutput]
    """平铺展开的所有经历过的节点步骤输出字典"""
    last_step_content: Any
    """最后一个成功执行的步骤所产出的内容"""
    final_output: StepOutput | None = None
    """工作流根节点最终包装的完整产出对象"""
    paused_step_name: str | None = None
    """若流水线处于挂起态，记录是哪个步骤引发了挂起"""


class PolicyAction(str, Enum):
    RETRY = "retry"
    CONTINUE = "continue"
    ABORT = "abort"
    FALLBACK = "fallback"


class PolicyResult(BaseModel):
    """错误策略执行结果"""

    action: PolicyAction
    """采取的具体恢复策略动作"""
    delay: float = 0.0
    """执行延迟或重试前需要等待的缓冲秒数"""
    new_input: StepInput | None = None
    """用于动态纠错自愈时替换传入的新参数结构"""
    fallback_node: Any | None = None
    """策略裁定降级时所指定的备用工作流节点"""
    healer_agent_name: str | None = None
    """执行了高级自愈的大模型或修复者名称"""


class BaseFailurePolicy(ABC):
    """错误处理策略抽象基类"""

    @abstractmethod
    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        pass


class AbortPolicy(BaseFailurePolicy):
    """直接中断策略"""

    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        return PolicyResult(action=PolicyAction.ABORT)


class SkipPolicy(BaseFailurePolicy):
    """跳过并继续策略"""

    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        return PolicyResult(action=PolicyAction.CONTINUE)


class RetryPolicy(BaseFailurePolicy):
    """退避重试策略"""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay

    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        counts = context.state.setdefault("__retry_counts__", {})
        key = f"{node.name}_{id(self)}"
        counts[key] = counts.get(key, 0) + 1

        if counts[key] <= self.max_retries:
            return PolicyResult(action=PolicyAction.RETRY, delay=self.delay)
        return PolicyResult(action=PolicyAction.ABORT)


class FallbackPolicy(BaseFailurePolicy):
    """降级路由策略"""

    def __init__(self, fallback_node: Any):
        self.fallback_node = fallback_node

    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        return PolicyResult(
            action=PolicyAction.FALLBACK, fallback_node=self.fallback_node
        )


class SelfHealingPolicy(BaseFailurePolicy):
    """大模型高级自愈策略"""

    def __init__(self, healer_model: str, max_retries: int = 2):
        self.healer_model = healer_model
        self.max_retries = max_retries

    async def handle_failure(
        self, node: Any, exception: Exception, step_input: StepInput, context: Any
    ) -> PolicyResult:
        counts = context.state.setdefault("__heal_counts__", {})
        key = f"{node.name}_{id(self)}"
        counts[key] = counts.get(key, 0) + 1

        if counts[key] > self.max_retries:
            from zhenxun.services.log import logger

            logger.warning(f"节点 '{node.name}' 自愈次数达上限，宣告失败。")
            return PolicyResult(action=PolicyAction.ABORT)

        import copy

        from zhenxun.services.ai.llm.api import generate_structured
        from zhenxun.services.log import logger

        class HealedInput(BaseModel):
            """自愈后输入结构"""

            fixed_input: str = Field(
                description="""修复后的输入参数 必须是完全合法的数据结构"""
            )

        prompt = f"""# Self-Healing Task

请修复节点 `{node.name}` 的参数错误。

## Original Input
{step_input.input}

## Exception
{exception}

## Requirements
- 分析错误原因
- 将输入修复为可被程序正确解析的格式
- 只输出修复后的结果，不要输出额外解释
"""

        try:
            logger.info(f"🩹 触发 AI 自愈分析 (节点: {node.name})...")
            res = await generate_structured(
                prompt, response_model=HealedInput, model=self.healer_model
            )

            new_input = copy.copy(step_input)
            new_input.input = res.fixed_input

            return PolicyResult(
                action=PolicyAction.RETRY,
                new_input=new_input,
                healer_agent_name=self.healer_model,
            )

        except Exception as e:
            logger.error(f"自愈过程发生大模型调用异常: {e}")
            return PolicyResult(action=PolicyAction.ABORT)
