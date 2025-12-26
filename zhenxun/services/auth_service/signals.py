from tortoise.signals import post_delete, post_save

from zhenxun.models.ban_console import BanConsole
from zhenxun.models.bot_console import BotConsole
from zhenxun.models.group_console import GroupConsole, convert_module_format
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.plugin_limit import PluginLimit
from zhenxun.models.user_console import UserConsole
from zhenxun.services.log import logger
from zhenxun.utils.enum import BlockType

from .cache import AuthStateCache as Cache

LOG_CMD = "AuthSignals"


@post_save(BanConsole)
async def on_ban_save(
    sender: type[BanConsole],
    instance: BanConsole,
    created: bool,
    using_db,
    update_fields,
):
    """当封禁记录创建或更新时触发"""
    expire_time = (
        -1 if instance.duration == -1 else (instance.ban_time + instance.duration)
    )

    if instance.user_id:
        Cache.set_user_ban(instance.user_id, expire_time)
        logger.debug(f"同步用户封禁缓存: {instance.user_id}", LOG_CMD)

    if instance.group_id:
        Cache.set_group_ban(instance.group_id, expire_time)
        logger.debug(f"同步群组封禁缓存: {instance.group_id}", LOG_CMD)


@post_delete(BanConsole)
async def on_ban_delete(sender: type[BanConsole], instance: BanConsole, using_db):
    """当封禁记录删除时触发"""
    if instance.user_id:
        Cache.remove_user_ban(instance.user_id)
    if instance.group_id:
        Cache.remove_group_ban(instance.group_id)
    logger.debug(f"移除封禁缓存: {instance.user_id or instance.group_id}", LOG_CMD)


@post_save(GroupConsole)
async def on_group_save(
    sender: type[GroupConsole],
    instance: GroupConsole,
    created: bool,
    using_db,
    update_fields,
):
    disabled = (
        set(convert_module_format(instance.block_plugin))
        if instance.block_plugin
        else set()
    )
    su_disabled = (
        set(convert_module_format(instance.superuser_block_plugin))
        if instance.superuser_block_plugin
        else set()
    )
    disabled_tasks = (
        set(convert_module_format(instance.block_task))
        if instance.block_task
        else set()
    )
    su_disabled_tasks = (
        set(convert_module_format(instance.superuser_block_task))
        if instance.superuser_block_task
        else set()
    )

    Cache.update_group_rule(
        group_id=str(instance.group_id),
        level=instance.level,
        status=instance.status,
        disabled_plugins=disabled,
        superuser_disabled_plugins=su_disabled,
        disabled_tasks=disabled_tasks,
        superuser_disabled_tasks=su_disabled_tasks,
    )
    logger.debug(f"同步群组规则缓存: {instance.group_id}", LOG_CMD)


@post_delete(GroupConsole)
async def on_group_delete(sender: type[GroupConsole], instance: GroupConsole, using_db):
    Cache.remove_group_rule(str(instance.group_id))


@post_save(BotConsole)
async def on_bot_save(
    sender: type[BotConsole],
    instance: BotConsole,
    created: bool,
    using_db,
    update_fields,
):
    disabled = (
        set(BotConsole.convert_module_format(instance.block_plugins))
        if instance.block_plugins
        else set()
    )
    disabled_tasks = (
        set(BotConsole.convert_module_format(instance.block_tasks))
        if instance.block_tasks
        else set()
    )

    Cache.update_bot_rule(
        bot_id=str(instance.bot_id),
        status=instance.status,
        disabled_plugins=disabled,
        disabled_tasks=disabled_tasks
    )


@post_delete(BotConsole)
async def on_bot_delete(sender: type[BotConsole], instance: BotConsole, using_db):
    Cache.remove_bot_rule(str(instance.bot_id))


@post_save(PluginInfo)
async def on_plugin_save(
    sender: type[PluginInfo],
    instance: PluginInfo,
    created: bool,
    using_db,
    update_fields,
):
    disabled = not instance.status or instance.block_type == BlockType.ALL
    Cache.set_global_plugin_disabled(instance.module, disabled)
    logger.debug(
        f"同步全局插件状态: {instance.module} -> {'禁用' if disabled else '启用'}",
        LOG_CMD,
    )


@post_save(PluginLimit)
@post_delete(PluginLimit)
async def on_limit_change(
    sender: type[PluginLimit], instance: PluginLimit, using_db, *args, **kwargs
):
    """当限制规则变更时，重新加载该模块的所有限制"""
    limits = await PluginLimit.filter(module=instance.module, status=True).all()
    Cache.set_plugin_limits(instance.module, limits)
    logger.debug(f"重载插件限制缓存: {instance.module}", LOG_CMD)


@post_save(UserConsole)
async def on_user_save(
    sender: type[UserConsole],
    instance: UserConsole,
    created: bool,
    using_db,
    update_fields,
):
    if created:
        Cache.add_user_existence(str(instance.user_id))


def register_signals():
    """显式导入以确保装饰器生效"""
    pass
