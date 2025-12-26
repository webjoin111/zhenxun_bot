import asyncio
import time
from typing import ClassVar

from zhenxun.models.ban_console import BanConsole
from zhenxun.models.bot_console import BotConsole
from zhenxun.models.group_console import GroupConsole, convert_module_format
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.plugin_limit import PluginLimit
from zhenxun.models.user_console import UserConsole
from zhenxun.services.data_access import DataAccess
from zhenxun.services.log import logger
from zhenxun.utils.enum import BlockType

from .cache import AuthStateCache

LOG_CMD = "AuthService"


class AuthService:
    """
    鉴权服务层 (Singleton)
    负责协调 Cache 和 DB，提供高层鉴权 API。
    """

    _user_queue: ClassVar[asyncio.Queue] = asyncio.Queue()
    _background_tasks: ClassVar[set[asyncio.Task]] = set()

    @classmethod
    async def init(cls):
        """系统启动时全量加载数据到缓存"""
        logger.info("开始构建权限服务缓存...", LOG_CMD)
        start_time = time.time()

        bans = await BanConsole.all()
        for ban in bans:
            expire_time = -1 if ban.duration == -1 else (ban.ban_time + ban.duration)
            if ban.user_id:
                AuthStateCache.set_user_ban(ban.user_id, expire_time)
            if ban.group_id:
                AuthStateCache.set_group_ban(ban.group_id, expire_time)

        plugins = await PluginInfo.all()
        for p in plugins:
            disabled = not p.status or p.block_type == BlockType.ALL
            AuthStateCache.set_global_plugin_disabled(p.module, disabled)
            AuthStateCache.set_plugin_info(p.module, p)

        limits = await PluginLimit.filter(status=True).all()
        limit_map = {}
        for limit in limits:
            limit_map.setdefault(limit.module, []).append(limit)
        for module, limit_list in limit_map.items():
            AuthStateCache.set_plugin_limits(module, limit_list)

        groups = await GroupConsole.all()
        for g in groups:
            disabled = (
                set(convert_module_format(g.block_plugin)) if g.block_plugin else set()
            )
            su_disabled = (
                set(convert_module_format(g.superuser_block_plugin))
                if g.superuser_block_plugin
                else set()
            )
            disabled_tasks = (
                set(convert_module_format(g.block_task)) if g.block_task else set()
            )
            su_disabled_tasks = (
                set(convert_module_format(g.superuser_block_task))
                if g.superuser_block_task
                else set()
            )
            AuthStateCache.update_group_rule(
                str(g.group_id), g.level, g.status,
                disabled, su_disabled,
                disabled_tasks, su_disabled_tasks
            )

        bots = await BotConsole.all()
        for b in bots:
            disabled = (
                set(BotConsole.convert_module_format(b.block_plugins))
                if b.block_plugins
                else set()
            )
            disabled_tasks = (
                set(BotConsole.convert_module_format(b.block_tasks))
                if b.block_tasks
                else set()
            )
            AuthStateCache.update_bot_rule(str(b.bot_id), b.status,
                                           disabled, disabled_tasks)

        users = await UserConsole.all().values_list("user_id", flat=True)
        for uid in users:
            AuthStateCache.add_user_existence(str(uid))

        logger.info(f"缓存构建完成，耗时 {time.time() - start_time:.2f}s", LOG_CMD)

        task = asyncio.create_task(cls._user_creation_loop())
        cls._background_tasks.add(task)
        task.add_done_callback(cls._background_tasks.discard)

    @classmethod
    def is_user_banned(cls, user_id: str) -> bool:
        """
        检查用户是否被 Ban (包含惰性过期检查)
        返回: True (被 Ban), False (正常)
        """
        expire = AuthStateCache.get_user_ban_expire(user_id)
        if expire is None:
            return False

        if expire == -1:
            return True

        if time.time() > expire:
            task = asyncio.create_task(cls._cleanup_expired_ban(user_id=user_id))
            cls._background_tasks.add(task)
            task.add_done_callback(cls._background_tasks.discard)
            return False

        return True

    @classmethod
    def is_group_banned(cls, group_id: str) -> bool:
        """检查群组是否被 Ban"""
        expire = AuthStateCache.get_group_ban_expire(group_id)
        if expire is None:
            return False

        if expire == -1:
            return True

        if time.time() > expire:
            task = asyncio.create_task(cls._cleanup_expired_ban(group_id=group_id))
            cls._background_tasks.add(task)
            task.add_done_callback(cls._background_tasks.discard)
            return False

        return True

    @classmethod
    async def _cleanup_expired_ban(
        cls, user_id: str | None = None, group_id: str | None = None
    ):
        """[内部] 异步清理过期的 Ban 记录 (数据库 + 缓存)"""
        try:
            if user_id:
                await BanConsole.filter(user_id=user_id, group_id__isnull=True).delete()
            if group_id:
                await BanConsole.filter(
                    group_id=group_id, user_id__isnull=True
                ).delete()
        except Exception as e:
            logger.error(f"清理过期 Ban 失败: {e}", LOG_CMD)

    @classmethod
    def check_plugin_permission(
        cls, module: str, bot_id: str, group_id: str | None = None
    ) -> bool:
        """
        检查插件是否允许运行 (层级：Global -> Bot -> Group)
        返回: True (允许), False (被禁用)
        """
        if AuthStateCache.is_plugin_globally_disabled(module):
            return False

        bot_rule = AuthStateCache.get_bot_rule(bot_id)
        if bot_rule:
            if not bot_rule.status:
                return False
            if module in bot_rule.disabled_plugins:
                return False

        if group_id:
            group_rule = AuthStateCache.get_group_rule(group_id)
            if group_rule:
                if not group_rule.status and module not in [
                    "plugin_switch",
                    "admin_help",
                ]:
                    return False
                if module in group_rule.disabled_plugins:
                    return False
                if module in group_rule.superuser_disabled_plugins:
                    return False

        return True

    @classmethod
    async def ensure_user_exists(cls, user_id: str, platform: str = "unknown") -> None:
        """
        确保用户存在。
        如果缓存中没有，先在缓存中"乐观"注册，然后放入队列异步写入数据库。
        这消除了鉴权路径上的数据库 INSERT 操作。
        """
        if not AuthStateCache.is_user_exists(user_id):
            AuthStateCache.add_user_existence(user_id)
            await cls._user_queue.put((user_id, platform))

    @classmethod
    async def _user_creation_loop(cls):
        """后台任务：消费用户创建队列"""
        logger.info("启动异步用户注册服务...", LOG_CMD)
        while True:
            try:
                batch = {}
                user_id, platform = await cls._user_queue.get()
                batch[user_id] = platform

                try:
                    for _ in range(99):
                        user_id, platform = cls._user_queue.get_nowait()
                        batch[user_id] = platform
                except asyncio.QueueEmpty:
                    pass

                if batch:
                    await cls._batch_insert_users(batch)

            except Exception as e:
                logger.error(f"用户注册循环异常: {e}", LOG_CMD)
                await asyncio.sleep(1)

    @classmethod
    async def _batch_insert_users(cls, users: dict[str, str]):
        """执行批量插入"""
        try:
            last_user = await UserConsole.all().order_by("-uid").first()
            start_uid = (last_user.uid if last_user else 0) + 1

            user_objects = []
            for i, (user_id, platform) in enumerate(users.items()):
                user_objects.append(
                    UserConsole(user_id=user_id, uid=start_uid + i, platform=platform)
                )
            await UserConsole.bulk_create(user_objects, ignore_conflicts=True)

            dao = DataAccess(UserConsole)
            await dao._cache_items(user_objects)

            for uid in users.keys():
                AuthStateCache.remove_user_pending(uid)
            logger.debug(f"异步批量注册新用户: {len(user_objects)} 人", LOG_CMD)
        except Exception as e:
            logger.error(f"批量写入用户失败: {e}", LOG_CMD)
