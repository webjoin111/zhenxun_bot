"""
定时任务的生命周期管理

包含在机器人启动时加载和调度数据库中保存的任务的逻辑。
"""

from zhenxun.services.log import logger
from zhenxun.utils.manager.priority_manager import PriorityLifecycle
from zhenxun.utils.pydantic_compat import model_dump

from .engine import APSchedulerAdapter
from .manager import scheduler_manager
from .repository import ScheduleRepository
from .types import ScheduleContext


@PriorityLifecycle.on_startup(priority=90)
async def _load_schedules_from_db():
    """在服务启动时从数据库加载并调度所有任务。"""
    logger.info("正在从数据库加载并调度所有定时任务...")
    schedules = await ScheduleRepository.get_all_enabled()
    count = 0
    for schedule in schedules:
        if schedule.plugin_name in scheduler_manager._registered_tasks:
            APSchedulerAdapter.add_or_reschedule_job(schedule)
            count += 1
        else:
            logger.warning(f"跳过加载定时任务：插件 '{schedule.plugin_name}' 未注册。")
    logger.info(f"数据库定时任务加载完成，共成功加载 {count} 个任务。")

    logger.info("正在检查并注册声明式默认任务...")
    declared_count = 0
    for task_info in scheduler_manager._declared_tasks:
        plugin_name = task_info.plugin_name
        group_id = task_info.group_id
        bot_id = task_info.bot_id

        query_kwargs = {
            "plugin_name": plugin_name,
            "target_identifier": group_id or "",
            "bot_id": bot_id,
        }
        exists = await ScheduleRepository.exists(**query_kwargs)

        if not exists:
            logger.info(f"为插件 '{plugin_name}' 注册新的默认定时任务...")

            trigger_config_dict = model_dump(
                task_info.trigger, exclude={"trigger_type"}
            )

            target_type = "GROUP" if group_id else "GLOBAL"
            target_identifier = group_id or ""

            schedule = await scheduler_manager.add_schedule(
                plugin_name=plugin_name,
                target_type=target_type,
                target_identifier=target_identifier,
                trigger_type=task_info.trigger.trigger_type,
                trigger_config=trigger_config_dict,
                job_kwargs=task_info.job_kwargs,
                bot_id=bot_id,
            )
            if schedule:
                declared_count += 1
                logger.debug(f"默认任务 '{plugin_name}' 注册成功 (ID: {schedule.id})")
            else:
                logger.error(f"默认任务 '{plugin_name}' 注册失败")
        else:
            logger.debug(f"插件 '{plugin_name}' 的默认任务已存在于数据库中，跳过注册。")

    if declared_count > 0:
        logger.info(f"声明式任务检查完成，新注册了 {declared_count} 个默认任务。")

    logger.info("正在调度声明式临时任务...")
    ephemeral_count = 0
    for declaration in scheduler_manager._ephemeral_declared_tasks:
        try:
            job_id = f"runtime::{declaration.plugin_name}::{declaration.func.__name__}"

            context = ScheduleContext(
                schedule_id=0,
                plugin_name=job_id,
                bot_id=None,
                group_id=None,
                job_kwargs={},
            )

            trigger_config_dict = model_dump(
                declaration.trigger, exclude={"trigger_type"}
            )

            APSchedulerAdapter.add_ephemeral_job(
                job_id=job_id,
                func=declaration.func,
                trigger_type=declaration.trigger.trigger_type,
                trigger_config=trigger_config_dict,
                context=context,
            )
            ephemeral_count += 1
        except Exception as e:
            logger.error(f"调度临时任务 '{declaration.plugin_name}' 失败", e=e)

    if ephemeral_count > 0:
        logger.info(f"临时任务调度完成，共成功加载 {ephemeral_count} 个任务。")
