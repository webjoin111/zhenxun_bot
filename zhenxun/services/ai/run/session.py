import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from zhenxun.services.ai.core.exceptions import ConcurrencyRejectException
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.flow.base import ConcurrencyPolicy
from zhenxun.services.ai.utils.scope import BaseScopeBuilder, ScopeSelector


class TaskStopper(BaseScopeBuilder["TaskStopper"]):
    """
    声明式任务中止器 (Fluent Task Stopper)。
    为第三方开发者提供友好的链式 API，精准中止正在运行或排队的大模型任务。
    """

    def __init__(self, manager: "AgentSessionManager"):
        super().__init__()
        self.manager = manager

    async def cancel(self) -> int:
        """执行中止动作，返回被成功中止的任务数量。"""
        return await self.manager.cancel_by_query(self._selector)


class SessionInfo(BaseModel):
    """会话信息的元数据视图。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    """会话的唯一标识符"""
    state: dict[str, Any] = Field(default_factory=dict)
    """业务流转的强类型载荷"""
    created_at: float = Field(default_factory=time.time)
    """会话创建的时间戳"""
    updated_at: float = Field(default_factory=time.time)
    """会话最后更新的时间戳"""
    active_task: Any | None = Field(default=None)
    """当前正在执行的 asyncio.Task"""
    cancel_token: CancellationToken | None = Field(default=None)
    """当前任务的取消令牌"""


class LockContext(BaseModel):
    """并发锁的执行追踪器（解决 INTERRUPT 需要跨 Session 取消任务的问题）"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    active_task: Any | None = None
    cancel_token: CancellationToken | None = None


class AgentSessionManager:
    """
    Agent 会话状态管理器。
    彻底拥抱无状态：只维护业务强类型载荷 (state payload) 以及并发锁，不干涉 LLM 历史。
    """

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._exec_locks: dict[str, asyncio.Lock] = {}
        self._lock_contexts: dict[str, LockContext] = {}
        self._live_tasks: dict[str, list[tuple[CancellationToken, Any]]] = {}

    def stopper(self) -> TaskStopper:
        """获取声明式任务中止器，供第三方开发者极速中止运行中/排队中的任务"""
        return TaskStopper(self)

    async def cancel_by_query(self, query: ScopeSelector) -> int:
        """根据查询条件取消符合条件的会话任务。返回取消的数量"""
        count = 0
        scope_prefix = query.scope_prefix

        for sid, tasks in list(self._live_tasks.items()):
            if sid.startswith(scope_prefix) or (
                query.session_id and sid == query.session_id
            ):
                for token, task in tasks:
                    if not token.is_cancelled():
                        token.cancel()
                        count += 1
                    if task and not task.done():
                        task.cancel()
                from zhenxun.services.log import logger

                if count > 0:
                    logger.info(
                        f"🛑 [TaskStopper] 已强制终止排队或执行中的会话任务: {sid}"
                    )
        return count

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
        lock_id: str,
        policy: ConcurrencyPolicy,
        cancel_token: CancellationToken,
    ):
        """应用并发策略的中央调度上下文管理器"""
        current_task = asyncio.current_task()
        task_tuple = (cancel_token, current_task)
        if session_id not in self._live_tasks:
            self._live_tasks[session_id] = []
        self._live_tasks[session_id].append(task_tuple)

        try:
            if policy == ConcurrencyPolicy.ALLOW:
                yield
                return

            exec_lock = self._get_exec_lock(lock_id)
            lock_ctx = self._lock_contexts.setdefault(lock_id, LockContext())

            if policy == ConcurrencyPolicy.REJECT:
                if exec_lock.locked():
                    raise ConcurrencyRejectException(
                        f"并发域 {lock_id} 正忙，新请求被拒绝。"
                    )

            elif policy == ConcurrencyPolicy.INTERRUPT:
                if exec_lock.locked():
                    if lock_ctx.cancel_token:
                        lock_ctx.cancel_token.cancel()
                    if lock_ctx.active_task and not lock_ctx.active_task.done():
                        lock_ctx.active_task.cancel()

            elif policy == ConcurrencyPolicy.QUEUE:
                if exec_lock.locked():
                    from zhenxun.services.log import logger

                    logger.info(
                        f"⏳ [并发控制] 锁域 {lock_id} 被占用，"
                        "新请求已进入后台等待队列 (QUEUE)..."
                    )

            async with exec_lock:
                session = await self.get_or_create(session_id)
                session.active_task = asyncio.current_task()
                session.cancel_token = cancel_token

                lock_ctx.active_task = asyncio.current_task()
                lock_ctx.cancel_token = cancel_token
                try:
                    yield
                finally:
                    session.active_task = None
                    session.cancel_token = None
                    lock_ctx.active_task = None
                    lock_ctx.cancel_token = None
        finally:
            if session_id in self._live_tasks:
                if task_tuple in self._live_tasks[session_id]:
                    self._live_tasks[session_id].remove(task_tuple)
                if not self._live_tasks[session_id]:
                    del self._live_tasks[session_id]

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
            from zhenxun.services.ai.context.memory.manager import memory_manager
            from zhenxun.services.ai.context.memory.models import MemoryConfig
            from zhenxun.services.ai.context.memory.types import SessionMetadata

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
