import aiofiles
import nonebot
from nonebot import get_loaded_plugins
from nonebot.drivers import Driver
from nonebot.plugin import Plugin, PluginMetadata
from ruamel.yaml import YAML
import ujson as json

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.configs.utils import PluginExtraData, PluginSetting
from zhenxun.models.group_console import GroupConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.plugin_limit import PluginLimit
from zhenxun.models.task_info import TaskInfo
from zhenxun.services.log import logger
from zhenxun.utils.enum import (
    BlockType,
    LimitCheckType,
    LimitWatchType,
    PluginLimitType,
    PluginType,
)

from .manager import manager

_yaml = YAML(pure=True)
_yaml.allow_unicode = True
_yaml.indent = 2

driver: Driver = nonebot.get_driver()


async def _handle_setting(
    plugin: Plugin,
    plugin_list: list[PluginInfo],
    limit_list: list[PluginLimit],
):
    """处理插件设置

    参数:
        plugin: Plugin
        plugin_list: 插件列表
        limit_list: 插件限制列表
    """
    metadata = plugin.metadata
    if not metadata:
        if not plugin.sub_plugins:
            return
        """父插件"""
        metadata = PluginMetadata(name=plugin.name, description="", usage="")
    extra = metadata.extra
    extra_data = PluginExtraData(**extra)
    logger.debug(f"{metadata.name}:{plugin.name} -> {extra}", "初始化插件数据")
    setting = extra_data.setting or PluginSetting()
    if metadata.type == "library":
        extra_data.plugin_type = PluginType.HIDDEN
    if extra_data.plugin_type == PluginType.HIDDEN:
        extra_data.menu_type = ""
    if plugin.sub_plugins:
        extra_data.plugin_type = PluginType.PARENT
    plugin_list.append(
        PluginInfo(
            module=plugin.name,
            module_path=plugin.module_name,
            name=metadata.name,
            author=extra_data.author,
            version=extra_data.version,
            level=setting.level,
            default_status=setting.default_status,
            limit_superuser=setting.limit_superuser,
            menu_type=extra_data.menu_type,
            cost_gold=setting.cost_gold,
            plugin_type=extra_data.plugin_type,
            admin_level=extra_data.admin_level,
            is_show=extra_data.is_show,
            ignore_prompt=extra_data.ignore_prompt,
            parent=(plugin.parent_plugin.module_name if plugin.parent_plugin else None),
            impression=setting.impression,
        )
    )
    if extra_data.limits:
        limit_list.extend(
            PluginLimit(
                module=plugin.name,
                module_path=plugin.module_name,
                limit_type=limit._type,
                watch_type=limit.watch_type,
                status=limit.status,
                check_type=limit.check_type,
                result=limit.result,
                cd=getattr(limit, "cd", None),
                max_count=getattr(limit, "max_count", None),
            )
            for limit in extra_data.limits
        )


async def fix_db_schema():
    """修复数据库架构问题"""
    from tortoise.connection import connections

    conn = connections.get("default")
    # 检查是否存在superuser列并处理
    try:
        await conn.execute_query("""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.columns
                        WHERE table_name='plugin_info' AND column_name='superuser') THEN
                    ALTER TABLE plugin_info DROP COLUMN superuser;
                END IF;
            END $$;
        """)
        logger.info("数据库架构检查完成")
    except Exception as e:
        logger.error(f"数据库架构修复失败: {e}")


@driver.on_startup
async def _():
    """
    初始化插件数据配置
    """
    # 修复数据库架构问题
    await fix_db_schema()

    plugin_list: list[PluginInfo] = []
    limit_list: list[PluginLimit] = []
    module2id = {}
    load_plugin = []
    if module_list := await PluginInfo.all().values("id", "module_path"):
        module2id = {m["module_path"]: m["id"] for m in module_list}
    for plugin in get_loaded_plugins():
        load_plugin.append(plugin.module_name)
        await _handle_setting(plugin, plugin_list, limit_list)
    create_list = []
    update_list = []
    for plugin in plugin_list:
        if plugin.module_path not in module2id:
            create_list.append(plugin)
        else:
            plugin.id = module2id[plugin.module_path]
            # 确保只更新应该来自插件元数据的字段
            await plugin.save(
                update_fields=[
                    "name",
                    "author",
                    "version",
                    "plugin_type",
                    "admin_level",
                    "is_show",
                    "ignore_prompt",
                    # 移除了 menu_type
                    # 确保 level, default_status, limit_superuser,
                    # cost_gold, impression, status, block_type 等用户配置不在此列表
                ]
            )

            # # 验证更新是否成功
            # updated_plugin = await PluginInfo.get(id=plugin.id)
            # if updated_plugin.menu_type != plugin.menu_type:
            #     logger.warning(
            #         f"插件 {plugin.name} 的menu_type更新失败: "
            #         f"期望值 '{plugin.menu_type}', "
            #         f"实际值 '{updated_plugin.menu_type}'"
            #     )
            #     # 尝试单独更新menu_type
            #     updated_plugin.menu_type = plugin.menu_type
            #     await updated_plugin.save(update_fields=["menu_type"])

            update_list.append(plugin)

    if create_list:
        await PluginInfo.bulk_create(create_list, 10)

    # 对于批量更新操作，逐个更新替代批量操作
    # 这里不使用被注释的批量更新代码，而是在上面的循环中已经处理

    await data_migration()
    await PluginInfo.filter(module_path__in=load_plugin).update(load_status=True)
    await PluginInfo.filter(module_path__not_in=load_plugin).update(load_status=False)
    manager.init()
    if limit_list:
        for limit in limit_list:
            if not manager.exists(limit.module, limit.limit_type):
                """不存在，添加"""
                manager.add(limit.module, limit)
    manager.save_file()
    await manager.load_to_db()


async def data_migration():
    # await limit_migration()
    await plugin_migration()
    await group_migration()


async def limit_migration():
    """插件限制迁移"""
    cd_file = DATA_PATH / "configs" / "plugins2cd.yaml"
    block_file = DATA_PATH / "configs" / "plugins2block.yaml"
    count_file = DATA_PATH / "configs" / "plugins2count.yaml"
    limit_data: dict[str, list[tuple[str, dict]]] = {}
    if cd_file.exists():
        async with aiofiles.open(cd_file, encoding="utf8") as f:
            if data := _yaml.load(await f.read()):
                for k in data["PluginCdLimit"]:
                    limit_data[k] = [("CD", data["PluginCdLimit"][k])]
        cd_file.unlink()
    if block_file.exists():
        async with aiofiles.open(block_file, encoding="utf8") as f:
            if data := _yaml.load(await f.read()):
                for k in data["PluginBlockLimit"]:
                    if k in limit_data:
                        limit_data[k].append(("BLOCK", data["PluginBlockLimit"][k]))
                    else:
                        limit_data[k] = [("BLOCK", data["PluginBlockLimit"][k])]
        block_file.unlink()
    if count_file.exists():
        async with aiofiles.open(count_file, encoding="utf8") as f:
            if data := _yaml.load(await f.read()):
                for k in data["PluginCountLimit"]:
                    if k in limit_data:
                        limit_data[k].append(("COUNT", data["PluginCountLimit"][k]))
                    else:
                        limit_data[k] = [("COUNT", data["PluginCountLimit"][k])]
        count_file.unlink()
    if limit_data:
        logger.info("开始迁移插件限制数据...")
        update_list = []
        create_list = []
        plugins = await PluginInfo.filter(module__in=limit_data.keys())
        for plugin in plugins:
            limits: list[PluginLimit] = await plugin.plugin_limit.all()  # type: ignore
            exits_limit = [x[0] for x in limit_data[plugin.module]]
            _not_create_type = []
            for limit in limits:
                if _limit_list := [
                    x[1]
                    for x in limit_data[plugin.module]
                    if x[0] == str(limit.limit_type)
                ]:
                    """修改"""
                    _not_create_type.append(str(limit.limit_type))
                    _limit = _limit_list[0]
                    watch_type = LimitWatchType.USER
                    if _limit.get("watch_type") == "group":
                        watch_type = LimitWatchType.GROUP
                    check_type = LimitCheckType.ALL
                    if _limit.get("check_type") == "private":
                        check_type = LimitCheckType.PRIVATE
                    elif _limit.get("check_type") == "group":
                        check_type = LimitCheckType.GROUP
                    limit.watch_type = watch_type
                    limit.result = _limit.get("rst", "")
                    limit.status = _limit.get("status", True)
                    if limit.watch_type != PluginLimitType.COUNT:
                        limit.check_type = check_type
                    if limit.watch_type == PluginLimitType.CD:
                        limit.cd = _limit["cd"]
                    if limit.watch_type == PluginLimitType.COUNT:
                        limit.max_count = _limit["count"]
                    # 改为逐个保存
                    await limit.save()
                    update_list.append(limit)
            for s in [e for e in exits_limit if e not in _not_create_type]:
                if _limit_list := [
                    x[1] for x in limit_data[plugin.module] if s == x[0]
                ]:
                    _limit = _limit_list[0]
                    limit_type = PluginLimitType.CD
                    if s == "BLOCK":
                        limit_type = PluginLimitType.BLOCK
                    elif s == "COUNT":
                        limit_type = PluginLimitType.COUNT
                    watch_type = LimitWatchType.USER
                    if _limit.get("watch_type") == "group":
                        watch_type = LimitWatchType.GROUP
                    check_type = LimitCheckType.ALL
                    if _limit.get("check_type") == "private":
                        check_type = LimitCheckType.PRIVATE
                    elif _limit.get("check_type") == "group":
                        check_type = LimitCheckType.GROUP
                    create_list.append(
                        PluginLimit(
                            module=plugin.module,
                            module_path=plugin.module_path,
                            plugin=plugin,
                            limit_type=limit_type,
                            watch_type=watch_type,
                            status=_limit.get("status", True),
                            check_type=check_type,
                            result=_limit.get("rst", ""),
                            cd=_limit.get("cd"),
                            max_count=_limit.get("max_count"),
                        )
                    )
        # 注释掉批量更新，使用单个保存方式
        # if update_list:
        #     await PluginLimit.bulk_update(
        #         update_list,
        #         [
        #             "watch_type",
        #             "status",
        #             "check_type",
        #             "result",
        #             "cd",
        #             "max_count",
        #         ],
        #         10,
        #     )
        if create_list:
            await PluginLimit.bulk_create(create_list, 10)
        logger.info("迁移插件限制数据完成!")


async def plugin_migration():
    """迁移插件数据"""
    setting_file = DATA_PATH / "configs" / "plugins2settings.yaml"
    plugin_file = DATA_PATH / "manager" / "plugins_manager.json"
    if setting_file.exists():
        async with aiofiles.open(setting_file, encoding="utf8") as f:
            if data := _yaml.load(await f.read()):
                logger.info("开始迁移插件setting数据...")
                data = data["PluginSettings"]
                plugins = await PluginInfo.filter(module__in=data.keys())
                for plugin in plugins:
                    if plugin_data_list := [
                        data[p] for p in data if p == plugin.module
                    ]:
                        plugin_data = plugin_data_list[0]
                        plugin.default_status = plugin_data.get("default_status", True)
                        plugin.level = plugin_data.get("level", 5)
                        plugin.limit_superuser = plugin_data.get(
                            "limit_superuser", False
                        )
                        plugin.menu_type = plugin_data.get("plugin_type", ["功能"])[0]
                        plugin.cost_gold = plugin_data.get("cost_gold", 0)
                        # 逐个保存替代批量更新
                        await plugin.save(
                            update_fields=[
                                "default_status",
                                "level",
                                "limit_superuser",
                                "menu_type",
                                "cost_gold",
                            ]
                        )
                # 注释掉批量更新，已在循环中处理
                # await PluginInfo.bulk_update(
                #     plugins,
                #     [
                #         "default_status",
                #         "level",
                #         "limit_superuser",
                #         "menu_type",
                #         "cost_gold",
                #     ],
                #     10,
                # )
        setting_file.unlink()
        logger.info("迁移插件setting数据完成!")
    if plugin_file.exists():
        async with aiofiles.open(plugin_file, encoding="utf8") as f:
            if data := json.loads(await f.read()):
                logger.info("开始迁移插件数据...")
                plugins = await PluginInfo.filter(module__in=data.keys())
                for plugin in plugins:
                    if plugin_data := data.get(plugin.module):
                        plugin.status = plugin_data.get("status", True)
                        block_type = None
                        get_block = plugin_data.get("block_type")
                        if get_block == "all":
                            block_type = BlockType.ALL
                        elif get_block == "private":
                            block_type = BlockType.PRIVATE
                        elif get_block == "group":
                            block_type = BlockType.GROUP
                        plugin.block_type = block_type
                        # 逐个保存替代批量更新
                        await plugin.save(update_fields=["status", "block_type"])
                # 注释掉批量更新，已在循环中处理
                # await PluginInfo.bulk_update(plugins, ["status", "block_type"], 10)
        plugin_file.unlink()
        logger.info("迁移插件数据完成!")


async def group_migration():
    """
    群组数据迁移
    """
    group_file = DATA_PATH / "manager" / "group_manager.json"
    if group_file.exists():
        async with aiofiles.open(group_file, encoding="utf8") as f:
            if data := json.loads(await f.read()):
                logger.info("开始迁移群组数据...")
                update_list = []
                create_list = []
                white_group = data["white_group"]
                old_group_list: dict = data["group_manager"]
                if close_task := data["close_task"]:
                    """全局被动关闭"""
                    await TaskInfo.filter(module__in=close_task).update(status=False)
                group_list = await GroupConsole.filter(
                    group_id__in=old_group_list.keys()
                )
                for old_group_id, old_group in old_group_list.items():
                    block_plugin = ""
                    block_task = ""
                    status = old_group.get("status", True)
                    level = old_group.get("level", 5)
                    if close_plugins := old_group.get("close_plugins"):
                        block_plugin = ",".join(close_plugins) + ","
                    if group_task_status := old_group.get("group_task_status"):
                        close_task = [
                            t for t in group_task_status if not group_task_status[t]
                        ]
                        block_task = ",".join(close_task) + ","
                    if group_ := [g for g in group_list if g.group_id == old_group_id]:
                        group = group_[0]
                        if group.group_id in white_group:
                            group.is_super = True
                        group.status = status
                        group.block_plugin = block_plugin
                        group.block_task = block_task
                        group.level = level
                        update_list.append(group)
                    else:
                        """添加"""
                        create_list.append(
                            GroupConsole(
                                group_id=old_group_id,
                                status=status,
                                level=level,
                                block_plugin=block_plugin,
                                block_task=block_task,
                                is_super=old_group_id in white_group,
                            )
                        )
                if update_list:
                    # 使用批量更新，因为这里的字段类型不会导致SQL错误
                    await GroupConsole.bulk_update(
                        update_list,
                        ["is_super", "status", "block_plugin", "block_task", "level"],
                        10,
                    )
                if create_list:
                    await GroupConsole.bulk_create(create_list, 10)
        group_file.unlink()
        logger.info("迁移群组数据完成!")
