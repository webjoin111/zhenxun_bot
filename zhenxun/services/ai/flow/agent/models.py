"""
Agent 相关静态声明类型定义
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import LLMMessage, UsageInfo
from zhenxun.services.ai.core.options import GenerationConfig
from zhenxun.services.ai.flow.base import BaseRuntimeConfig
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.engine.registry import ToolCollection
from zhenxun.services.ai.tools.models import GlobalToolFilter


class Persona(BaseModel):
    """智能体人设与上下文背景"""

    role: str = Field(...)
    """扮演的角色身份"""

    goal: str = Field(...)
    """角色的核心目标"""

    backstory: str | None = Field(default=None)
    """角色背景故事或性格设定"""

    model_config = ConfigDict(extra="ignore")


class AgentSettings(BaseRuntimeConfig):
    """
    统一的智能体宏观配置 (Consolidated Settings)。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_cycles: int = 10
    """工具调用最大循环次数"""
    enable_parallel_calls: bool = True
    """允许并行工具调用"""
    reflexion_retries: int = 1
    """反思重试次数"""
    enable_fallback_summary: bool = True
    """达到最大循环次数时，是否触发大模型兜底总结（而不是直接报错）"""

    enable_hitl: bool | None = Field(default=None)
    """是否允许智能体主动挂起任务，向用户求助 (Human-in-the-Loop)。
    若为 None 则跟随全局设置。
    """


class AgentRunProfile(BaseModel):
    """
    智能体单次运行的配置覆盖字典 (Run Profile)。
    用于在单次 run() 时覆盖 Agent 的初始静态配置。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_cycles: int | None = None
    """单次运行覆盖的最大循环次数"""
    message_history: list[LLMMessage] | None = None
    """初始化的底层对话历史记录。"""
    tool_filter: GlobalToolFilter | None = None
    """全局工具过滤器，限制本次运行可用的工具池。"""
    memory: Any | None = None
    """单次运行级别的记忆门面覆盖 (支持 bool, MemoryConfig, MemoryBuilder)。"""
    generation_config: GenerationConfig | None = None
    """单次运行覆盖的大模型生成配置。"""
    capabilities: list[Any] | None = None
    """仅针对本次运行动态注入的临时拦截器/能力组件列表。"""
    skills: Sequence[Any] | None = None
    """仅针对本次运行动态注入的临时技能集合。"""
    executor: Any | None = None
    """单次运行覆盖的核心执行引擎策略 (BaseAgentExecutor)。"""


class AgentState(BaseModel):
    """大模型思考循环的运行状态上下文 (统一载体)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    static_system_prompt: str | list[str] = ""
    """绝对不变的系统提示词（用于前缀缓存）"""
    dynamic_system_prompt: str = ""
    """包含变量与实时状态的动态提示词（用于尾部注入）"""
    tools: ToolCollection | None
    """当前轮次生效的、已完成鉴权和过滤的工具集合"""
    run_context: RunContext
    """保留依赖注入(DI)与黑板引用的全局运行时上下文"""

    messages: list[LLMMessage] = Field(default_factory=list)
    """大模型将看到的完整历史消息列表 (执行历史)"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """累计的 Token 消耗"""
    structured_result: Any | None = None
    """拦截到的结构化输出结果"""
    early_result_output: Any | None = None
    """拦截到的早期终止输出结果"""
    should_terminate: bool = False
    """标记是否应提前终止循环"""
    handoff_triggered: Any | None = None
    """标记是否触发了移交"""
    is_finished: bool = False
    """标记大模型循环是否彻底结束"""
    final_result: Any | None = None
    """最终的运行结果 (AgentRunResult)"""
    origin_msg_len: int = 0
    """初始进入循环时的消息历史长度 (用于增量保存记忆)"""
