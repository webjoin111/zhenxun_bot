"""
Agent 核心类型定义
"""

from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    """
    Agent 执行引擎的配置。
    用于在单次运行中精细控制 Agent 的行为。
    """

    max_cycles: int = Field(
        default=10, description="工具调用循环的最大次数，防止无限循环。"
    )
    enable_parallel_calls: bool = Field(
        default=True, description="是否允许LLM在一次思考中请求调用多个工具。"
    )
    reflexion_retries: int = Field(
        default=1,
        description="当工具执行出错时，允许 Agent 进行自我反思和修正的最大重试次数。",
    )


class MCPSource(BaseModel):
    """
    显式定义的 MCP 工具源。
    用于在 Agent 定义中声明对外部 MCP 服务器的依赖。
    """

    server_name: str = Field(..., description="mcp_tools.json 中配置的服务器名称")
    namespace: str | None = Field(
        default=None, description="为此服务器的工具添加命名空间前缀，防止冲突"
    )
    tool_whitelist: list[str] | None = Field(
        default=None, description="仅加载该服务器下的特定工具名称(不含前缀)"
    )

    def __hash__(self):
        return hash(
            (self.server_name, self.namespace, tuple(self.tool_whitelist or []))
        )


class ToolFilter(BaseModel):
    """
    一个结构化的工具过滤器，用于在单次LLM调用中动态控制可用的工具。
    过滤顺序: 服务器黑名单 -> 服务器白名单 -> 全局工具黑名单 -> 全局工具白名单
    """

    allowed_servers: list[str] | None = Field(
        default=None, description="服务器白名单：只从这些服务器发现并加载工具。"
    )
    excluded_servers: list[str] | None = Field(
        default=None, description="服务器黑名单：完全不加载这些服务器的任何工具。"
    )
    allowed: list[str] | None = Field(
        default=None, description="全局工具白名单：只允许使用这些带前缀的完整工具名。"
    )
    excluded: list[str] | None = Field(
        default=None, description="全局工具黑名单：禁止使用这些带前缀的完整工具名。"
    )


class ReviewerConfig(BaseModel):
    """
    Agent 审查/反思配置。
    用于定义在 Agent 生成回复后，是否需要经过另一个 Agent 的审查和修正。
    """

    agent_name: str = Field(..., description="负责审查的 Agent 名称")
    prompt_template: str = Field(
        default="请审查上述回答。如果回答准确且无误，请仅回复'PASS'。否则，请提供具体的修改建议。",
        description="发送给审查者的指令模板",
    )
    max_turns: int = Field(default=3, description="最大修正轮数")
