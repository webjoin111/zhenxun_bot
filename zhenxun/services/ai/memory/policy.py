from zhenxun.services.ai.memory.compression import (
    LLMSummarizerReducer,
    MessageDropper,
    SlidingWindowReducer,
    StructuredSummaryReducer,
    ToolOutputCompactor,
)
from zhenxun.services.ai.memory.interfaces import BaseMemoryReducer


class MemoryPolicy:
    """
    记忆策略工厂 (Strategy Factory Facade)。
    为开发者提供开箱即用的上下文压缩管线组装方案。
    (多模态视窗 vision_window 已作为正交配置独立到 AgentMemory 中)
    """

    @staticmethod
    def unlimited() -> list[BaseMemoryReducer]:
        """无限制模式。不进行任何形式的截断和总结，适用于短对话或纯 Agent 内部流转。"""
        return []

    @staticmethod
    def sliding_window(max_turns: int = 50) -> list[BaseMemoryReducer]:
        """物理滑动窗口模式。强制丢弃超过设定轮数的最早对话。"""
        return [SlidingWindowReducer(max_turns=max_turns)]

    @staticmethod
    def llm_summarize(
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str = "Gemini/gemini-2.5-flash",
        summarization_prompt: str = "请概括以下对话内容，保留关键的约束条件、用户偏好、已完成的任务状态和未解决的问题。"
    ) -> list[BaseMemoryReducer]:
        """LLM 总结压缩模式。Token 达标后，自动将历史对话合并为一段 Summary。"""
        return [
            ToolOutputCompactor(),
            LLMSummarizerReducer(
                keep_recent_turns=keep_recent_turns, 
                trigger_tokens=trigger_tokens,
                max_turns=max_turns,
                summarization_model=summarization_model,
                summarization_prompt=summarization_prompt
            ),
            MessageDropper(trigger_tokens=trigger_tokens),
        ]

    @staticmethod
    def structured_summarize(
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str = "Gemini/gemini-2.5-flash"
    ) -> list[BaseMemoryReducer]:
        """结构化总结压缩模式。使用 JSON Schema 强制大模型提取核心状态。"""
        return [
            ToolOutputCompactor(),
            StructuredSummaryReducer(
                keep_recent_turns=keep_recent_turns, 
                trigger_tokens=trigger_tokens,
                max_turns=max_turns,
                summarization_model=summarization_model
            ),
            MessageDropper(trigger_tokens=trigger_tokens),
        ]
