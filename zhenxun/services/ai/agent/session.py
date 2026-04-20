import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import time
from typing import Any

from pydantic import BaseModel, Field

from zhenxun.services.ai.engine.token_estimator import global_estimator
from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.memory.working_memory import (
    CondenserPipeline,
    CondenserRegistry,
    _get_default_memory,
)
from zhenxun.services.ai.protocols.memory import BaseWorkingMemory, SessionMetadata
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.log import logger


class SessionInfo(BaseModel):
    """
    会话信息的元数据视图。
    （注：实际的 messages 列表存储在 memory.working_memory 中，以复用现有的缓存压缩管线）
    """

    session_id: str
    state: dict[str, Any] = Field(
        default_factory=dict, description="业务流转的强类型载荷"
    )
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class AsyncMemoryCondenser:
    """
    负责在后台异步压缩记忆，提供并发锁保证无损合并。
    从原 AgentSessionManager 剥离，专注于底层记忆 management 优化。
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self.memory: BaseWorkingMemory = _get_default_memory()
        self._compressing_tasks: dict[str, asyncio.Task] = {}

    def _get_lock(self, scope: str) -> asyncio.Lock:
        if scope not in self._locks:
            self._locks[scope] = asyncio.Lock()
        return self._locks[scope]

    def trigger_compression(self, scope: str):
        config = get_llm_config().context_settings
        if not config.enabled:
            return

        if scope not in self._compressing_tasks:
            task = asyncio.create_task(self._background_condense(scope))
            self._compressing_tasks[scope] = task

            def _on_done(t):
                self._compressing_tasks.pop(scope, None)

            task.add_done_callback(_on_done)

    async def _background_condense(self, scope: str):
        session = SessionMetadata(session_id=scope)
        async with self._get_lock(scope):
            current_history = await self.memory.get_history(session)
            original_len = len(current_history)

        model_name = get_global_default_model_name() or "Gemini/gemini-2.0-flash"
        caps = get_model_capabilities(model_name)
        config = get_llm_config().context_settings
        threshold_tokens = int(caps.max_input_tokens * config.trigger_threshold)

        current_tokens = global_estimator.estimate_context(current_history, model_name)
        if current_tokens <= threshold_tokens:
            return

        logger.info(
            f"[AsyncMemoryCondenser] Scope {scope} 触发后台上下文压缩 ({current_tokens}/{threshold_tokens})"
        )

        reducers = [CondenserRegistry.get("tool_compactor")]
        if config.enable_summarization:
            reducers.append(CondenserRegistry.get("llm_summarizer"))
        reducers.append(CondenserRegistry.get("message_dropper"))

        pipeline = CondenserPipeline(reducers)

        try:
            compressed_history = await pipeline.run(
                current_history, target_tokens=threshold_tokens, model_name=model_name
            )
        except Exception as e:
            logger.error(f"[AsyncMemoryCondenser] 后台压缩失败: {e}", e=e)
            return

        async with self._get_lock(scope):
            latest_history = await self.memory.get_history(session)
            new_msgs = latest_history[original_len:]
            final_history = compressed_history + new_msgs
            await self.memory.set_history(session, final_history)
            logger.info(
                f"[AsyncMemoryCondenser] Scope {scope} 后台压缩完成，已无损合并 {len(new_msgs)} 条并发新消息。"
            )


class AgentSessionManager:
    """
    Agent 会话状态管理器。
    彻底拥抱无状态：只维护业务强类型载荷 (state payload) 以及并发锁，不干涉 LLM 历史。
    """

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def get_or_create(self, session_id: str) -> SessionInfo:
        async with self._get_lock(session_id):
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionInfo(session_id=session_id)
            return self._sessions[session_id]

    async def get(self, session_id: str) -> SessionInfo | None:
        async with self._get_lock(session_id):
            return self._sessions.get(session_id)

    async def update_state(self, session_id: str, new_state: dict[str, Any]):
        async with self._get_lock(session_id):
            if session_id in self._sessions:
                self._sessions[session_id].state.update(new_state)
                self._sessions[session_id].updated_at = time.time()

    async def delete(self, session_id: str):
        async with self._get_lock(session_id):
            self._sessions.pop(session_id, None)
            await _get_default_memory().clear_history(
                SessionMetadata(session_id=session_id)
            )


session_manager = AgentSessionManager()
async_memory_condenser = AsyncMemoryCondenser()
active_session_id: ContextVar[str | None] = ContextVar(
    "active_session_id", default=None
)


@asynccontextmanager
async def agent_session_scope(session_id: str):
    """声明式上下文包装器。进入此作用域后的 Agent 都会自动吸附到指定的 SessionID 上。"""
    await session_manager.get_or_create(session_id)
    token = active_session_id.set(session_id)
    try:
        yield session_id
    finally:
        active_session_id.reset(token)

