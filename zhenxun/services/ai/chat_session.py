from typing import Any

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.messages import LLMMessage, PromptInput
from zhenxun.services.ai.flow.agent import Agent
from zhenxun.services.ai.flow.agent.models import AgentRuntimeConfig
from zhenxun.services.ai.memory.builder import MemoryBuilder
from zhenxun.services.ai.memory.models import MemoryConfig, SessionMetadata
from zhenxun.services.ai.run import AgentRunResult, RunContext


class ChatSession:
    """
    极简的状态化对话门面 (Semantic Facade)。
    """

    def __init__(
        self,
        instruction: str = "",
        model: str | None = None,
        memory: bool | MemoryConfig | MemoryBuilder = True,
        generation_config: GenerationConfig | dict | None = None,
    ):
        """
        初始化聊天会话。

        参数:
            instruction: 系统指令（System Prompt），用于定义助手的角色、语调或行为准则。
            model: 指定使用的模型名称。若不指定，将使用系统的全局默认模型。
            memory: 记忆配置。
                - bool: True 启用默认记忆（GROUP_USER 隔离），False 禁用记忆。
                - MemoryConfig: 直接传入预定义的配置对象。
                - MemoryBuilder: 直接传入预定义的配置构建器。
            generation_config: 模型生成配置（如 temperature, max_tokens 等）。
                - dict: 自动转换为 GenerationConfig 对象。
                - GenerationConfig: 直接传入配置对象。
        """
        self.agent = Agent(
            name="ChatSession",
            instruction=instruction,
            model=model,
            tools=[],
            memory=memory,
            generation_config=generation_config,
            runtime_config=AgentRuntimeConfig(stateless=False, enable_hitl=False),
        )

    def _get_implicit_session_id(self, override_sid: str | None) -> str | None:
        if override_sid:
            return override_sid
        ctx = RunContext()
        return ctx.session_id

    async def clear_memory(self, session_id: str | None = None) -> None:
        """
        清空当前会话用户的历史短期记忆。
        """
        sid = self._get_implicit_session_id(session_id)
        if sid and self.agent.memory_config:
            from zhenxun.services.ai.memory.manager import memory_manager

            await (
                memory_manager.cleaner()
                .session(sid)
                .config(self.agent.memory_config)
                .clear_short_term()
            )

    async def get_history(self, session_id: str | None = None) -> list[LLMMessage]:
        """获取当前用户的历史记忆"""
        sid = self._get_implicit_session_id(session_id)
        if sid and self.agent.memory_config:
            from zhenxun.services.ai.memory.manager import memory_manager

            chat_ctx = memory_manager.get_chat_context(self.agent.memory_config)
            if chat_ctx:
                return await chat_ctx.get_messages(SessionMetadata(session_id=sid))
        return []

    async def chat(self, prompt: PromptInput, **kwargs: Any) -> AgentRunResult[str]:
        """
        进行一次基础的上下文对话。

        该方法是 Agent 运行的轻量级门面，内部会将请求代理给底层的 Agent 引擎。

        参数:
            prompt: 用户的输入内容，可以是纯文本字符串，也可以是多模态结构体。
            **kwargs: 传递给 Agent.run() 的附加参数 (如 session_id, context 等)。

        返回:
            AgentRunResult: 包含文本输出、Token 消耗以及更新后的上下文消息列表。
        """
        return await self.agent.run(prompt=prompt, **kwargs)
