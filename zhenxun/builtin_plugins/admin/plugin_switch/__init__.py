from nonebot.adapters import Bot, Event
from nonebot.exception import FinishedException
from nonebot.permission import SUPERUSER as SUPERUSER_PERM
from nonebot.plugin import PluginMetadata
from nonebot_plugin_alconna import AlconnaMatch, AlconnaQuery, Arparma, Match, Query
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import Config
from zhenxun.configs.utils import PluginExtraData, RegisterConfig
from zhenxun.services.log import logger
from zhenxun.services.tags import tag_manager
from zhenxun.utils.enum import BlockType, PluginType
from zhenxun.utils.message import MessageUtils
from zhenxun.utils.platform import PlatformUtils

from ._data_source import PluginManager, build_plugin, build_task
from .command import _group_status_matcher, _status_matcher

base_config = Config.get("plugin_switch")


__plugin_meta__ = PluginMetadata(
    name="功能开关",
    description="对群组内的功能限制，超级用户可以对群组以及全局的功能被动开关限制",
    usage="""### 功能开关管理

**基础用法 (群管理员/群主)**
- `开启/关闭 [功能名]` : 在当前群开关功能
- `开启/关闭群被动 [被动名]` : 在当前群开关被动技能
- `开启/关闭所有(插件|功能)` : 在当前群开关所有功能
- `醒来` / `休息吧` : 控制Bot在当前群的休眠状态

**高级用法 (超级用户)**
- **指定目标**:
  - `-g <群号...>` : 指定一个或多个群
  - `-t <标签>` : 指定标签下的所有群
  - `--all` : 所有已激活的群
- **白名单模式**:
  - `--only` : 仅在指定的目标群开启，其余群自动关闭

**全局控制 (超级用户)**
- `开启/关闭 [功能名] --scope [p/g/a]`
  - `p`: 私聊禁用, `g`: 群聊全局禁用, `a`: 完全禁用
- `开启/关闭所有(插件|功能)` (私聊中) : 全局开关所有插件

**默认状态管理 (超级用户)**
- `开启/关闭(插件|功能)df [功能名]` : 修改指定插件的进群默认状态
- `开启/关闭默认群被动 [被动名]` : 修改指定被动技能的进群默认状态
- `开启/关闭所有(插件|功能)df` : 修改所有插件的进群默认状态

**状态查询**
- `插件列表` : 查看所有插件状态
- `查看插件状态 [功能名]` : 查看指定功能在各群的开启情况
- `查看被动状态 [被动名]` : 查看指定被动技能在各群的开启情况
- `群被动状态` : 查看当前群被动技能状态

**示例**
- `关闭 签到 -t 游戏群` : 关闭所有"游戏群"标签下的签到
- `开启 色图 --only -g 123456` : 仅在群123456开启色图，其他群全部关闭
""".strip(),
    extra=PluginExtraData(
        author="HibiKier",
        version="0.1",
        plugin_type=PluginType.SUPER_AND_ADMIN,
        admin_level=base_config.get("CHANGE_GROUP_SWITCH_LEVEL", 2),
        configs=[
            RegisterConfig(
                key="CHANGE_GROUP_SWITCH_LEVEL",
                value=2,
                help="开关群功能权限",
                default_value=2,
                type=int,
            )
        ],
    ).to_dict(),
)


@_status_matcher.assign("$main")
async def _(
    bot: Bot,
    session: Uninfo,
    arparma: Arparma,
):
    if session.user.id in bot.config.superusers:
        image = await build_plugin()
        logger.info(
            "查看功能列表",
            arparma.header_result,
            session=session,
        )
        await MessageUtils.build_message(image).finish(reply_to=True)
    else:
        await MessageUtils.build_message("权限不足捏...").finish(reply_to=True)


async def get_target_groups(
    bot: Bot,
    event: Event,
    session: Uninfo,
    tag: str | None,
    groups: tuple[str, ...] | None,
    all_scope: bool,
) -> set[str] | None:
    """解析目标群组列表，包含标签、群号和全量选项。"""
    targets: set[str] = set()
    is_superuser = await SUPERUSER_PERM(bot, event)

    if (tag or groups or all_scope) and not is_superuser:
        await MessageUtils.build_message("权限不足，无法指定其他群组或标签").finish(
            reply_to=True
        )
        return None

    if groups:
        targets.update(str(group_id) for group_id in groups if group_id)

    if tag:
        tag_groups = await tag_manager.resolve_tag_to_group_ids(tag, bot=bot)
        targets.update(str(group_id) for group_id in tag_groups)

    if all_scope:
        all_groups, _ = await PlatformUtils.get_group_list(bot)
        targets.update(str(group.group_id) for group in all_groups if group.group_id)

    if not targets and session.group:
        targets.add(str(session.group.id))

    return targets


async def _handle_switch_command(
    status: bool,
    bot: Bot,
    event: Event,
    session: Uninfo,
    arparma: Arparma,
    plugin_name: Match[str],
    groups: Match[tuple[str, ...]] = AlconnaMatch("groups"),
    tag: Match[str] = AlconnaMatch("tag"),
    task: Query[bool] = AlconnaQuery("task.value", False),
    default_status: Query[bool] = AlconnaQuery("default.value", False),
    all_flag: Query[bool] = AlconnaQuery("all.value", False),
    force: Query[bool] = AlconnaQuery("force.value", False),
    only_flag: Query[bool] | None = None,
    block_type: Match[str] | None = None,
):
    if not all_flag.result and not plugin_name.available:
        await MessageUtils.build_message("请输入功能/被动名称").finish(reply_to=True)
        return

    name = plugin_name.result if plugin_name.available else ""
    is_superuser = await SUPERUSER_PERM(bot, event)
    only_flag_value = only_flag.result if only_flag else False
    is_force = force.result

    if is_superuser and plugin_name.available and default_status.result:
        result = await PluginManager.set_default_status(name, status)
        await MessageUtils.build_message(result).finish(reply_to=True)
        return

    if not status and block_type and block_type.available:
        if not is_superuser:
            await MessageUtils.build_message("权限不足，无法执行全局禁用").finish(
                reply_to=True
            )
            return
        _type = BlockType.ALL
        if block_type.result in ["p", "private"]:
            _type = BlockType.PRIVATE
        elif block_type.result in ["g", "group"]:
            _type = BlockType.GROUP
        result = await PluginManager.superuser_set_status(name, status, _type, None)
        await MessageUtils.build_message(result).finish(reply_to=True)
        return

    targets = await get_target_groups(
        bot,
        event,
        session,
        tag.result if tag and tag.available else None,
        groups.result if groups and groups.available else None,
        all_flag.result,
    )
    if targets is None:
        return

    if all_flag.result and not plugin_name.available:
        if targets:
            messages = []
            for gid in targets:
                messages.append(
                    await PluginManager.set_all_plugin_status(
                        status,
                        default_status.result if is_superuser else False,
                        gid,
                    )
                )
            await MessageUtils.build_message("\n".join(messages)).finish(reply_to=True)
            return
        if is_superuser and not session.group:
            result = await PluginManager.set_all_plugin_status(
                status, default_status.result, None
            )
            await MessageUtils.build_message(result).finish(reply_to=True)
            return
        await MessageUtils.build_message("请输入目标群组").finish(reply_to=True)
        return

    if not plugin_name.available:
        await MessageUtils.build_message("请输入功能/被动名称").finish(reply_to=True)
        return

    if not targets:
        if is_superuser and not session.group:
            target_block_type = BlockType.ALL if not status else None
            result = await PluginManager.superuser_set_status(
                name, status, target_block_type, None
            )
            await MessageUtils.build_message(result).finish(reply_to=True)
            return
        await MessageUtils.build_message("请选择一个目标群组").finish(reply_to=True)
        return

    msg = await PluginManager.batch_update_status(
        name,
        targets,
        status=status,
        is_task=task.result,
        is_superuser=is_superuser,
        is_whitelist_mode=only_flag_value,
        force=is_force,
        bot=bot,
    )
    action_name = "开启" if status else "关闭"
    logger.info(
        f"{action_name}操作: {name}, targets={targets}",
        arparma.header_result,
        session=session,
    )
    await MessageUtils.build_message(msg).finish(reply_to=True)


@_status_matcher.assign("open")
async def _(
    bot: Bot,
    event: Event,
    session: Uninfo,
    arparma: Arparma,
    plugin_name: Match[str],
    groups: Match[tuple[str, ...]] = AlconnaMatch("groups"),
    tag: Match[str] = AlconnaMatch("tag"),
    task: Query[bool] = AlconnaQuery("task.value", False),
    default_status: Query[bool] = AlconnaQuery("default.value", False),
    all_flag: Query[bool] = AlconnaQuery("all.value", False),
    only_flag: Query[bool] = AlconnaQuery("only.value", False),
    force: Query[bool] = AlconnaQuery("force.value", False),
):
    await _handle_switch_command(
        True,
        bot,
        event,
        session,
        arparma,
        plugin_name,
        groups,
        tag,
        task,
        default_status,
        all_flag,
        only_flag=only_flag,
        force=force,
    )


@_status_matcher.assign("close")
async def _(
    bot: Bot,
    event: Event,
    session: Uninfo,
    arparma: Arparma,
    plugin_name: Match[str],
    block_type: Match[str],
    groups: Match[tuple[str, ...]] = AlconnaMatch("groups"),
    tag: Match[str] = AlconnaMatch("tag"),
    task: Query[bool] = AlconnaQuery("task.value", False),
    default_status: Query[bool] = AlconnaQuery("default.value", False),
    all_flag: Query[bool] = AlconnaQuery("all.value", False),
    force: Query[bool] = AlconnaQuery("force.value", False),
):
    await _handle_switch_command(
        False,
        bot,
        event,
        session,
        arparma,
        plugin_name,
        groups,
        tag,
        task,
        default_status,
        all_flag,
        block_type=block_type,
        force=force,
    )


@_group_status_matcher.handle()
async def _(
    session: Uninfo,
    arparma: Arparma,
    status: str,
):
    if session.group:
        group_id = session.group.id
        is_wake = status == "wake"

        if is_wake and await PluginManager.is_wake(group_id):
            await MessageUtils.build_message("我还醒着呢！").finish()

        await PluginManager.set_group_active_status(group_id, is_wake)

        action = "醒来" if is_wake else "进行休眠"
        msg = "呜..醒来了..." if is_wake else "那我先睡觉了..."

        logger.info(action, arparma.header_result, session=session)
        await MessageUtils.build_message(msg).finish()
    return MessageUtils.build_message("群组id为空...").send()


@_status_matcher.assign("task")
async def _(
    session: Uninfo,
    arparma: Arparma,
):
    if arparma.find("check") or arparma.find("open") or arparma.find("close"):
        return

    image = await build_task(session.group.id if session.group else None)
    if image:
        logger.info("查看群被动列表", arparma.header_result, session=session)
        await MessageUtils.build_message(image).finish(reply_to=True)
    else:
        await MessageUtils.build_message("获取群被动任务失败...").finish(reply_to=True)


@_status_matcher.assign("check")
async def _(
    bot: Bot,
    event: Event,
    plugin_name: Match[str],
    task: Query[bool] = AlconnaQuery("task.value", False),
):
    if not await SUPERUSER_PERM(bot, event):
        await MessageUtils.build_message("只有超级用户可以查看全群状态统计").finish(
            reply_to=True
        )
        return

    name = plugin_name.result
    try:
        img = await PluginManager.render_global_status(
            name, is_task=task.result, bot=bot
        )
        await MessageUtils.build_message(img).finish(reply_to=True)
    except FinishedException:
        raise
    except ValueError as e:
        await MessageUtils.build_message(str(e)).finish(reply_to=True)
    except Exception as e:
        logger.error(f"渲染状态图表失败: {e}", e=e)
        await MessageUtils.build_message("生成状态报表失败，请检查日志").finish(
            reply_to=True
        )
