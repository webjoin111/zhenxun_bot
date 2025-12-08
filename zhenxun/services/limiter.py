import time
from typing import Any, ClassVar

from zhenxun.models.plugin_limit_state import PluginLimitState
from zhenxun.services.log import logger
from zhenxun.utils.limiters import BaseLimiter, LimitResult

PERSISTENCE_THRESHOLD = 300


class LimiterService:
    """
    分级存储限制器服务，内存 + 数据库
    """

    _cache: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def _gen_key(cls, scope: str, subject_id: str, plugin: str, node: str) -> str:
        return f"{scope}:{subject_id}:{plugin}:{node}"

    @classmethod
    async def check(
        cls,
        limiter: BaseLimiter,
        scope: str,
        subject_id: str,
        plugin: str,
        node: str,
        **kwargs: Any,
    ) -> LimitResult:
        key = cls._gen_key(scope, subject_id, plugin, node)
        state = cls._cache.get(key)
        if state is None:
            db_record = await PluginLimitState.get_or_none(
                scope=scope, subject_id=subject_id, plugin=plugin, node=node
            )
            if db_record:
                if db_record.expire_at < time.time():
                    await db_record.delete()
                    state = {}
                else:
                    state = db_record.state or {}
            else:
                state = {}
        check_kwargs = dict(kwargs)
        result = limiter.check(state, **check_kwargs)
        if result.passed or result.new_state != state:
            cls._cache[key] = result.new_state
            time_until_expire = result.expire_at - time.time()
            if time_until_expire > PERSISTENCE_THRESHOLD:
                await PluginLimitState.update_or_create(
                    scope=scope,
                    subject_id=subject_id,
                    plugin=plugin,
                    node=node,
                    defaults={
                        "state": result.new_state,
                        "expire_at": result.expire_at,
                    },
                )
                logger.debug(f"限制器状态已持久化: {key}")
        return result

    @classmethod
    async def clear_cache(cls):
        """
        清理内存缓存 (用于辅助 GC)
        """
        cls._cache.clear()


limiter_service = LimiterService()
