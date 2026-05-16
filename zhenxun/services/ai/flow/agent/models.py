"""
Agent 相关静态声明类型定义
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.flow.base import BaseRuntimeConfig


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
