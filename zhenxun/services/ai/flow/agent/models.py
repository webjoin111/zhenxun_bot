"""
Agent 相关静态声明类型定义
"""

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.flow.base import BaseRuntimeConfig
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.engine.registry import ToolCollection


class AgentEngineConfig(BaseModel):
    """Agent 执行引擎运行时的动态覆盖配置"""

    max_cycles: int = 10
    """工具调用最大循环次数"""
    enable_parallel_calls: bool = True
    """允许并行工具调用"""
    reflexion_retries: int = 1
    """反思重试次数"""
    enable_fallback_summary: bool = True
    """达到最大循环次数时，是否触发大模型兜底总结（而不是直接报错）"""


class Persona(BaseModel):
    """智能体人设与上下文背景"""

    role: str = Field(...)
    """扮演的角色身份"""

    goal: str = Field(...)
    """角色的核心目标"""

    backstory: str | None = Field(default=None)
    """角色背景故事或性格设定"""

    model_config = ConfigDict(extra="ignore")


class AgentRuntimeConfig(BaseRuntimeConfig):
    """智能体运行时行为与工作流配置"""

    enable_hitl: bool | None = Field(default=None)
    """是否允许智能体主动挂起任务，向用户求助 (Human-in-the-Loop)。
    若为 None 则跟随全局设置。
    """


class CapabilitySpec(BaseModel):
    """插件/拦截器声明契约，用于 YAML/JSON 配置序列化"""

    name: str = Field(...)
    """Capability 子类的注册标识符名称"""

    model_config = ConfigDict(extra="allow")


class AgentLoopContext(BaseModel):
    """传递给执行循环的静态上下文快照 (Data Contract)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    static_system_prompt: str = ""
    """绝对不变的系统提示词（用于前缀缓存）"""
    dynamic_system_prompt: str = ""
    """包含变量与实时状态的动态提示词（用于尾部注入）"""
    messages: list[LLMMessage]
    """大模型将看到的完整历史消息列表"""
    tools: ToolCollection | None
    """当前轮次生效的、已完成鉴权和过滤的工具集合"""
    run_context: RunContext
    """保留依赖注入(DI)与黑板引用的全局运行时上下文"""
