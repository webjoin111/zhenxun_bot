import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.exceptions import ConcurrencyRejectException
from zhenxun.services.ai.flow.base import ConcurrencyPolicy

from zhenxun.services.ai.memory.models import SessionMetadata
from zhenxun.services.ai.run.models import CancellationToken


class SessionInfo(BaseModel):
    """会话信息的元数据视图。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    state: dict[str, Any] = Field(
        default_factory=dict, description="业务流转的强类型载荷"
    )
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    active_task: Any | None = Field(
        default=None, description="当前正在执行的 asyncio.Task"
    )
    cancel_token: CancellationToken | None = Field(
        default=None, description="当前任务的取消令牌"
    )


class AgentSessionManager:
    """
    Agent 会话状态管理器。
    彻底拥抱无状态：只维护业务强类型载荷 (state payload) 以及并发锁，不干涉 LLM 历史。
    """

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._exec_locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def _get_exec_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._exec_locks:
            self._exec_locks[session_id] = asyncio.Lock()
        return self._exec_locks[session_id]

    @asynccontextmanager
    async def apply_concurrency_policy(
        self,
        session_id: str,
        policy: ConcurrencyPolicy,
        cancel_token: CancellationToken,
    ):
        """应用并发策略的中央调度上下文管理器"""
        if policy == ConcurrencyPolicy.ALLOW:
            yield
            return

        exec_lock = self._get_exec_lock(session_id)

        if policy == ConcurrencyPolicy.REJECT:
            if exec_lock.locked():
                raise ConcurrencyRejectException(
                    f"会话 {session_id} 正忙，新请求被拒绝。"
                )

        elif policy == ConcurrencyPolicy.INTERRUPT:
            if exec_lock.locked():
                session = await self.get(session_id)
                if session:
                    if session.cancel_token:
                        session.cancel_token.cancel()
                    if session.active_task and not session.active_task.done():
                        session.active_task.cancel()

        elif policy == ConcurrencyPolicy.QUEUE:
            if exec_lock.locked():
                from zhenxun.services.log import logger

                logger.info(
                    f"⏳ [并发控制] 会话 {session_id} 正忙，新请求已进入后台等待队列 (QUEUE)..."
                )

        async with exec_lock:
            session = await self.get_or_create(session_id)
            session.active_task = asyncio.current_task()
            session.cancel_token = cancel_token
            try:
                yield
            finally:
                session.active_task = None
                session.cancel_token = None

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
            from zhenxun.services.ai.memory.manager import memory_manager
            from zhenxun.services.ai.memory.models import MemoryConfig, SessionMetadata
            # 获取全局默认短期记忆后端进行清理
            default_ctx = memory_manager.get_chat_context(MemoryConfig())
            if default_ctx:
                await default_ctx.clear(SessionMetadata(session_id=session_id))


session_manager = AgentSessionManager()
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
