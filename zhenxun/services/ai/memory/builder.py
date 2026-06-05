from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Self

if TYPE_CHECKING:
    from zhenxun.services.ai.memory.interfaces import BaseChatContext, BaseSlotContext
    from zhenxun.services.ai.memory.models import MemorySlot
    from zhenxun.services.ai.rag.backends import Embedder, StorageBackend
    from zhenxun.services.ai.rag.consolidation import Consolidator

from zhenxun.services.ai.memory.compression import MemoryPolicy
from zhenxun.services.ai.memory.models import (
    ContextCompressionConfig,
    LongTermConfig,
    MemoryConfig,
    ShortTermConfig,
    SlotMemoryConfig,
)
from zhenxun.services.ai.memory.types import MemoryIsolationLevel


class MemoryBuilder:
    """
    记忆配置的链式构建器 (Fluent Builder)。
    """

    def __init__(self):
        """
        初始化 MemoryBuilder 实例。

        创建一个默认关闭短期和长期记忆，并包含默认上下文压缩配置的构建器。
        """
        self._config = MemoryConfig(
            short_term=ShortTermConfig(enable=False),
            slots=SlotMemoryConfig(enable=False),
            long_term=LongTermConfig(enable=False),
            compression=ContextCompressionConfig(),
        )

    @classmethod
    def auto(cls) -> "MemoryBuilder":
        """
        创建一个开箱即用的默认记忆配置构建器。

        默认开启隔离的短期记忆，并使用 LLM 对话摘要进行上下文压缩。

        返回:
            MemoryBuilder: 链式配置构建器实例。
        """
        return cls().with_short_term(enable=True).with_llm_summary()

    @classmethod
    def resolve(
        cls, memory: bool | MemoryConfig | "MemoryBuilder" | None
    ) -> MemoryConfig:
        if isinstance(memory, MemoryConfig):
            return memory
        if isinstance(memory, cls):
            return memory.build()
        if isinstance(memory, bool):
            return MemoryConfig(short_term=ShortTermConfig(enable=memory))

        return MemoryConfig(short_term=ShortTermConfig(enable=False))

    def with_short_term(
        self,
        enable: bool = True,
        isolation_level: MemoryIsolationLevel = MemoryIsolationLevel.AGENT_USER,
        backend: "BaseChatContext | None" = None,
    ) -> Self:
        """
        配置短期对话历史记忆。

        参数:
            enable: 是否开启短期记忆。
            isolation_level: 记忆隔离级别，决定会话历史记录的区分范围。
            backend: 短期记忆存储后端实例，如果为 None 则使用全局默认后端。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.short_term.enable = enable
        self._config.short_term.isolation_level = isolation_level
        if backend is not None:
            self._config.short_term.backend = backend
        return self

    def with_slots(
        self,
        enable: bool = True,
        default_slots: list["MemorySlot"] | None = None,
        backend: "BaseSlotContext | None" = None,
    ) -> Self:
        """
        配置核心槽位记忆 (Memory Slots)。

        参数:
            enable: 是否启用槽位记忆。
            default_slots: 首次初始化时自动写入的默认槽位列表。
            backend: 槽位记忆存储后端，如果为 None 则使用全局默认后端。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.slots.enable = enable
        if default_slots is not None:
            self._config.slots.default_slots = default_slots
        if backend is not None:
            self._config.slots.backend = backend
        return self

    def with_long_term(
        self,
        enable: bool = True,
        scope: str | None = None,
        backend: "StorageBackend | None" = None,
        embedder: "Embedder | str | None" = None,
        auto_consolidate: bool = True,
        consolidator: "Consolidator | None" = None,
        async_write: bool = True,
    ) -> Self:
        """
        配置长期向量记忆与 RAG 设定。

        参数:
            enable: 是否启用长期记忆。
            scope: 长期记忆的作用域标识，控制记忆召回与存储的边界。
            backend: 长期记忆存储后端。
            embedder: 用于向量化的文本嵌入模型实例。
            auto_consolidate: 是否开启背景自动记忆整合。
            consolidator: 记忆整合器实例。
            async_write: 是否使用异步非阻塞写入长期记忆。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.long_term.enable = enable
        self._config.long_term.scope = scope
        self._config.long_term.backend = backend
        self._config.long_term.embedder = embedder
        self._config.long_term.auto_consolidate = auto_consolidate
        self._config.long_term.consolidator = consolidator
        self._config.long_term.async_write = async_write
        return self

    def with_multimodal_window(self, window_size: int = 5) -> Self:
        """
        配置多模态历史视窗大小。

        超出此窗口的图片/视频等富媒体消息会自动转换为纯文本占位符，以节省 Token 预算。

        参数:
            window_size: 允许保留多模态信息的最新的对话轮数。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.compression.vision_window = window_size
        return self

    def with_llm_summary(
        self,
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str | None = None,
        summarization_prompt: str = "请概括以下对话内容，保留关键的约束条件、用户偏好、已完成的任务状态和未解决的问题。",  # noqa: E501
    ) -> Self:
        """
        配置使用大模型自然语言总结作为上下文压缩策略。

        参数:
            trigger_tokens: 触发压缩的 Token 门槛。
            max_turns: 压缩策略作用的最大历史对话轮数上限。
            keep_recent_turns: 在大模型总结之外，强制保留的最近原始对话轮数。
            summarization_model: 负责生成总结的大模型名称。
            summarization_prompt: 生成总结时所使用的系统提示词。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.compression.policy = MemoryPolicy.llm_summarize(
            trigger_tokens=trigger_tokens,
            max_turns=max_turns,
            keep_recent_turns=keep_recent_turns,
            summarization_model=summarization_model,
            summarization_prompt=summarization_prompt,
        )
        return self

    def with_structured_summary(
        self,
        trigger_tokens: int = 4000,
        max_turns: int | None = None,
        keep_recent_turns: int = 0,
        summarization_model: str | None = None,
    ) -> Self:
        """
        配置使用结构化 JSON 提取作为上下文压缩策略。

        通过结构化的大模型调用，强制提取会话中的待办任务、执行状态与核心偏好等。

        参数:
            trigger_tokens: 触发压缩的 Token 门槛。
            max_turns: 压缩策略作用的最大历史对话轮数上限。
            keep_recent_turns: 强制保留的最近原始对话轮数。
            summarization_model: 负责生成结构化总结的大模型名称。

        返回:
            Self: 链式构建器自身，用于链式调用。
        """
        self._config.compression.policy = MemoryPolicy.structured_summarize(
            trigger_tokens=trigger_tokens,
            max_turns=max_turns,
            keep_recent_turns=keep_recent_turns,
            summarization_model=summarization_model,
        )
        return self

    def unlimited(self) -> Self:
        """
        配置为不进行任何截断和压缩的策略。
        适用于短程会话或者具备超长上下文窗口的底层语言模型。
        """
        self._config.compression.policy = MemoryPolicy.unlimited()
        return self

    def build(self) -> MemoryConfig:
        """
        生成最终构建好的 MemoryConfig 配置对象。
        """
        return self._config
