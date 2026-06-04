"""
Agent 相关静态声明类型定义
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.messages import LLMMessage
from zhenxun.services.ai.flow.base import BaseRuntimeConfig
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.engine.registry import ToolCollection


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

    enable_hitl: bool = Field(default=True)
    """是否允许智能体主动挂起任务，向用户求助 (Human-in-the-Loop)。"""

    custom_executor: Any | None = Field(default=None)
    """自定义的 AgentLoop 执行器类。用于替换底层的 AgentExecutor。"""


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


class AgentLoopConfig(BaseModel):
    """传递给执行循环的运行时配置 (Data Contract)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    """底层大模型标识 (Provider/Model)"""
    generation_config: GenerationConfig
    """大模型最终生成参数 (Temperature/Schema等)"""
    max_cycles: int = Field(default=10)
    """最大反思/工具调用循环次数"""
    reflexion_retries: int = Field(default=1)
    """错误自愈重试上限"""
    enable_fallback_summary: bool = Field(default=True)
    """达到最大循环后是否兜底总结"""
    cancellation_token: Any | None = None
    """异步控制流取消令牌"""
    event_streamer: Any | None = None
    """UI 事件流发射器"""
