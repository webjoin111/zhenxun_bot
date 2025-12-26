from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class GroupCacheItem:
    """群组缓存实体"""

    level: int = 5
    status: bool = True
    disabled_plugins: set[str] = field(default_factory=set)
    superuser_disabled_plugins: set[str] = field(default_factory=set)


@dataclass
class BotCacheItem:
    """Bot实例缓存实体"""

    status: bool = True
    disabled_plugins: set[str] = field(default_factory=set)


class AuthStateCache:
    """
    权限状态内存快照 (Pure Memory Store)

    设计原则:
    1. 只负责存取内存数据，不负责数据库 IO。
    2. 数据结构支持过期时间判断 (TTL)。
    3. 支持多维度索引。
    """

    _banned_users: ClassVar[dict[str, float]] = {}
    _banned_groups: ClassVar[dict[str, float]] = {}

    _global_disabled_plugins: ClassVar[set[str]] = set()

    _group_rules: ClassVar[dict[str, GroupCacheItem]] = {}

    _bot_rules: ClassVar[dict[str, BotCacheItem]] = {}

    _plugin_limits: ClassVar[dict[str, list]] = {}

    _plugin_info_map: ClassVar[dict[str, Any]] = {}

    _known_users: ClassVar[set[str]] = set()

    _pending_users: ClassVar[set[str]] = set()

    @classmethod
    def clear(cls):
        """清空所有缓存"""
        cls._banned_users.clear()
        cls._banned_groups.clear()
        cls._global_disabled_plugins.clear()
        cls._group_rules.clear()
        cls._bot_rules.clear()
        cls._plugin_limits.clear()
        cls._plugin_info_map.clear()
        cls._known_users.clear()
        cls._pending_users.clear()

    @classmethod
    def add_user_existence(cls, user_id: str):
        """标记用户已存在"""
        cls._known_users.add(str(user_id))

    @classmethod
    def is_user_exists(cls, user_id: str) -> bool:
        return str(user_id) in cls._known_users

    @classmethod
    def add_user_pending(cls, user_id: str):
        cls._pending_users.add(str(user_id))

    @classmethod
    def remove_user_pending(cls, user_id: str):
        cls._pending_users.discard(str(user_id))

    @classmethod
    def is_user_pending(cls, user_id: str) -> bool:
        return str(user_id) in cls._pending_users

    @classmethod
    def set_user_ban(cls, user_id: str, expire_time: float):
        """设置用户封禁 (-1 为永久)"""
        cls._banned_users[str(user_id)] = expire_time

    @classmethod
    def remove_user_ban(cls, user_id: str):
        uid = str(user_id)
        if uid in cls._banned_users:
            del cls._banned_users[uid]

    @classmethod
    def get_user_ban_expire(cls, user_id: str) -> float | None:
        """获取用户封禁过期时间，未被禁返回 None"""
        return cls._banned_users.get(str(user_id))

    @classmethod
    def set_group_ban(cls, group_id: str, expire_time: float):
        cls._banned_groups[str(group_id)] = expire_time

    @classmethod
    def remove_group_ban(cls, group_id: str):
        gid = str(group_id)
        if gid in cls._banned_groups:
            del cls._banned_groups[gid]

    @classmethod
    def get_group_ban_expire(cls, group_id: str) -> float | None:
        return cls._banned_groups.get(str(group_id))

    @classmethod
    def set_global_plugin_disabled(cls, module: str, disabled: bool):
        if disabled:
            cls._global_disabled_plugins.add(module)
        else:
            cls._global_disabled_plugins.discard(module)

    @classmethod
    def is_plugin_globally_disabled(cls, module: str) -> bool:
        return module in cls._global_disabled_plugins

    @classmethod
    def set_plugin_limits(cls, module: str, limits: list):
        cls._plugin_limits[module] = limits

    @classmethod
    def get_plugin_limits(cls, module: str) -> list:
        return cls._plugin_limits.get(module, [])

    @classmethod
    def set_plugin_info(cls, module: str, info: Any):
        cls._plugin_info_map[module] = info

    @classmethod
    def get_plugin_info(cls, module: str) -> Any | None:
        return cls._plugin_info_map.get(module)

    @classmethod
    def update_group_rule(
        cls,
        group_id: str,
        level: int | None = None,
        status: bool | None = None,
        disabled_plugins: set[str] | None = None,
        superuser_disabled_plugins: set[str] | None = None,
    ):
        gid = str(group_id)
        if gid not in cls._group_rules:
            cls._group_rules[gid] = GroupCacheItem()

        item = cls._group_rules[gid]
        if level is not None:
            item.level = level
        if status is not None:
            item.status = status
        if disabled_plugins is not None:
            item.disabled_plugins = disabled_plugins
        if superuser_disabled_plugins is not None:
            item.superuser_disabled_plugins = superuser_disabled_plugins

    @classmethod
    def get_group_rule(cls, group_id: str) -> GroupCacheItem | None:
        return cls._group_rules.get(str(group_id))

    @classmethod
    def remove_group_rule(cls, group_id: str):
        gid = str(group_id)
        if gid in cls._group_rules:
            del cls._group_rules[gid]

    @classmethod
    def update_bot_rule(
        cls,
        bot_id: str,
        status: bool | None = None,
        disabled_plugins: set[str] | None = None,
    ):
        bid = str(bot_id)
        if bid not in cls._bot_rules:
            cls._bot_rules[bid] = BotCacheItem()

        item = cls._bot_rules[bid]
        if status is not None:
            item.status = status
        if disabled_plugins is not None:
            item.disabled_plugins = disabled_plugins

    @classmethod
    def get_bot_rule(cls, bot_id: str) -> BotCacheItem | None:
        return cls._bot_rules.get(str(bot_id))

    @classmethod
    def remove_bot_rule(cls, bot_id: str):
        bid = str(bot_id)
        if bid in cls._bot_rules:
            del cls._bot_rules[bid]
