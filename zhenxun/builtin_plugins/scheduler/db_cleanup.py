import time

from nonebot_plugin_apscheduler import scheduler

from zhenxun.models.plugin_limit_state import PluginLimitState
from zhenxun.services.log import logger


@scheduler.scheduled_job("cron", hour=22, minute=20, id="limit_state_cleanup")
async def cleanup_expired_limits():
    """
    定期清理过期的限制器状态
    """
    now = time.time()
    try:
        count = await PluginLimitState.filter(expire_at__lt=now).delete()
        if count > 0:
            logger.info(
                f"[Scheduler] 已清理 {count} 条过期的限制器状态记录", "DB_CLEANUP"
            )
    except Exception as e:
        logger.error(f"[Scheduler] 清理限制器状态失败: {e}", "DB_CLEANUP")
