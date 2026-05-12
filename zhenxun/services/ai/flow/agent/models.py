"""
Agent 相关静态声明类型定义
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.protocols.memory import MemoryIsolationLevel


class Persona(BaseModel):
    """智能体人设与上下文背景"""

    role: str = Field(...)
    """扮演的角色身份"""

    goal: str = Field(...)
    """角色的核心目标"""

    backstory: str | None = Field(default=None)
    """角色背景故事或性格设定"""

    model_config = ConfigDict(extra="ignore")


class AgentMemoryConfig(BaseModel):
    """智能体记忆与上下文压缩配置"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_reducers: list[str | Any] | None = Field(default=None)
    """记忆压缩策略列表。支持字符串标识或自定义 Reducer 实例。"""

    context_threshold: float | None = Field(default=None)
    """触发记忆压缩的 Token 阈值。<=1.0 为比例，>1.0 为绝对 Token 数。"""

    max_history_turns: int | None = Field(default=None)
    """触发记忆压缩的对话轮数上限。"""


class AgentRuntimeConfig(BaseModel):
    """智能体运行时行为与工作流配置"""

    stateless: bool = Field(default=True)
    """是否无状态执行。若为 True，将在每次请求时生成独立的临时会话。"""
    isolation_level: MemoryIsolationLevel = Field(
        default=MemoryIsolationLevel.AGENT_USER
    )
    """记忆隔离级别。默认为最高级别的智能体私有隔离。"""

    enable_hitl: bool = Field(default=True)
    """是否允许智能体主动挂起任务，向用户求助 (Human-in-the-Loop)。"""

    ui_streamer: str | None = Field(default=None)
    """自动绑定的前端UI渲染器标识符（如 'markdown'）"""


class CapabilitySpec(BaseModel):
    """插件/拦截器声明契约，用于 YAML/JSON 配置序列化"""

    name: str = Field(...)
    """Capability 子类的注册标识符名称"""

    model_config = ConfigDict(extra="allow")


class AgentSpec(BaseModel):
    """智能体声明契约，支持从字典完全实例化一个 Agent"""

    name: str | None = Field(default=None)
    """智能体名称"""

    model: str | None = Field(default=None)
    """绑定的默认大模型名称"""

    persona: Persona | None = Field(default=None)
    """智能体人设配置"""

    description: str | None = Field(default=None)
    """智能体职能描述"""

    instructions: str | list[str] | None = Field(default=None)
    """系统提示词指令"""

    tools: list[str] = Field(default_factory=list)
    """需要挂载的工具标识符列表 (字符串名称)"""

    model_settings: dict[str, Any] | None = Field(default=None)
    """模型生成配置覆盖 (温度/MaxToken等)"""

    capabilities: list[CapabilitySpec] = Field(default_factory=list)
    """需要挂载的拦截器列表"""

    model_config = ConfigDict(extra="ignore")
