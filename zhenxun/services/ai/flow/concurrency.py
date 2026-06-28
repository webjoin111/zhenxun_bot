import asyncio
from contextlib import asynccontextmanager
from typing import Any

from zhenxun.services.ai.core.exceptions import ConcurrencyRejectException
from zhenxun.services.ai.core.models import CancellationToken
from zhenxun.services.ai.flow.base import ConcurrencyPolicy


@asynccontextmanager
async def apply_concurrency_policy(
    session_id: str,
    lock_id: str,
    policy: ConcurrencyPolicy,
    cancel_token: CancellationToken,
    intervention_policy: Any = None,
    message: Any = None,
):
    """应用并发策略的中央调度上下文管理器"""
    from zhenxun.services.ai.run.session import LockContext, session_manager

    current_task = asyncio.current_task()
    task_tuple = (cancel_token, current_task)
    if session_id not in session_manager.live_tasks:
        session_manager.live_tasks[session_id] = []
    session_manager.live_tasks[session_id].append(task_tuple)

    try:
        exec_lock = session_manager.get_exec_lock(lock_id)
        lock_ctx = session_manager.lock_contexts.setdefault(lock_id, LockContext())

        if exec_lock.locked():
            from zhenxun.services.ai.flow.base import InterventionPolicy

            if intervention_policy in (
                InterventionPolicy.STEER,
                InterventionPolicy.FOLLOW_UP,
            ):
                from zhenxun.services.ai.core.exceptions import (
                    InterventionHandledException,
                )

                session = await session_manager.get_or_create(session_id)

                actual_msg = message
                from zhenxun.services.ai.run.models import Task

                if isinstance(message, Task):
                    actual_msg = message.description
                elif hasattr(message, "extract_plain_text"):
                    actual_msg = message.extract_plain_text()

                if intervention_policy == InterventionPolicy.STEER:
                    session.steer_queue.enqueue(str(actual_msg))
                    raise InterventionHandledException(
                        "Steer successful",
                        display_content="💬 已将您的补充信息传递给正在思考的 AI...",
                    )
                elif intervention_policy == InterventionPolicy.FOLLOW_UP:
                    session.follow_up_queue.enqueue(str(actual_msg))
                    raise InterventionHandledException(
                        "Follow-up successful",
                        display_content="📝 已记录，AI 处理完当前任务后即刻执行...",
                    )

        if policy == ConcurrencyPolicy.ALLOW:
            yield
            return

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
            session = await session_manager.get_or_create(session_id)
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
        if session_id in session_manager.live_tasks:
            if task_tuple in session_manager.live_tasks[session_id]:
                session_manager.live_tasks[session_id].remove(task_tuple)
            if not session_manager.live_tasks[session_id]:
                del session_manager.live_tasks[session_id]
