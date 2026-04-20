import asyncio
import time

from zhenxun.services.log import logger


class ResourceLifespanMixin:
    """
    统一的资源生命周期管理 Mixin (基于 TTL 与后台看门狗)。
    支持管理单个全局资源或按 session_id 隔离的多个资源池。
    """

    ttl: int
    _last_active_times: dict[str, float]
    _watchdog_task: asyncio.Task | None
    _lifespan_lock: asyncio.Lock

    def init_lifespan(self, ttl: int = 600):
        self.ttl = ttl
        self._last_active_times = {}
        self._watchdog_task = None
        self._lifespan_lock = asyncio.Lock()

    def touch(self, resource_id: str = "default"):
        self._last_active_times[resource_id] = time.time()

    def _ensure_watchdog(self):
        if self.ttl > 0 and (self._watchdog_task is None or self._watchdog_task.done()):
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self):
        try:
            while True:
                await asyncio.sleep(max(1.0, self.ttl / 2))
                now = time.time()
                async with self._lifespan_lock:
                    expired_ids = [
                        res_id
                        for res_id, last_time in self._last_active_times.items()
                        if now - last_time > self.ttl
                    ]
                    for res_id in expired_ids:
                        logger.info(
                            f"♻️ [Watchdog] 资源 '{res_id}' 闲置超时 "
                            f"({self.ttl}s)，触发回收。"
                        )
                        await self.release_resource(res_id)
                        self._last_active_times.pop(res_id, None)

                    if not self._last_active_times:
                        break
        except asyncio.CancelledError:
            pass

    async def release_resource(self, resource_id: str):
        """需子类实现具体的销毁逻辑"""
        raise NotImplementedError
