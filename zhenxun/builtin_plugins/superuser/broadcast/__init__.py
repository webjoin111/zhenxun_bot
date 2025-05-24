from arclet.alconna import AllParam
from nepattern import UnionPattern
from nonebot import get_driver
from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.rule import to_me
import nonebot_plugin_alconna as alc
from nonebot_plugin_alconna import (
    Alconna,
    Args,
    on_alconna,
)
from nonebot_plugin_alconna.uniseg.segment import (
    At,
    AtAll,
    Audio,
    Button,
    Emoji,
    File,
    Hyper,
    Image,
    Keyboard,
    Reference,
    Reply,
    Text,
    Video,
    Voice,
)
from nonebot_plugin_session import EventSession

from zhenxun.configs.utils import PluginExtraData, RegisterConfig, Task
from zhenxun.models.group_console import GroupConsole
from zhenxun.services.log import logger
from zhenxun.utils.enum import PluginType
from zhenxun.utils.message import MessageUtils

from .broadcast_manager import BroadcastManager
from .message_processor import (
    _extract_broadcast_content,
    get_broadcast_target_groups,
    send_broadcast_and_notify,
)
from .tag_manager import TagManager

BROADCAST_SEND_DELAY_RANGE = (1, 3)

__plugin_meta__ = PluginMetadata(
    name="广播",
    description="昭告天下！",
    usage="""
    广播 [消息内容]
    - 直接发送消息到除当前群组外的所有群组
    - 支持文本、图片、@、表情、视频等多种消息类型
    - 示例：广播 你们好！
    - 示例：广播 [图片] 新活动开始啦！

    广播 + 引用消息
    - 将引用的消息作为广播内容发送
    - 支持引用普通消息或合并转发消息
    - 示例：(引用一条消息) 广播

    广播撤回
    - 撤回最近一次由您触发的广播消息
    - 仅能撤回短时间内的消息
    - 示例：广播撤回

    群组标签管理:
    群组标签 -l                                 # 查看所有标签及其群组
    群组标签 <标签名>                            # 查看指定标签下的群组
    群组标签 <标签名> -g <群号1>,<群号2>         # 将群组添加到指定标签
    群组标签 <标签名> -r -g <群号1>,<群号2>      # 从指定标签移除群组
    群组标签 <标签名> -d                         # 删除指定标签

    广播到标签:
    广播 -t <标签名> [消息内容]
    - 向指定标签下的所有群组发送广播，强制推送。
    - 示例: 广播 -t 活跃群 大家晚上好！

    特性：
    - 在群组中使用广播时，不会将消息发送到当前群组
    - 在私聊中使用广播时，会发送到所有群组

    别名：
    - bc (广播的简写)
    - recall (广播撤回的别名)
    """.strip(),
    extra=PluginExtraData(
        author="HibiKier",
        version="1.3",
        plugin_type=PluginType.SUPERUSER,
        configs=[
            RegisterConfig(
                module="_task",
                key="DEFAULT_BROADCAST",
                value=True,
                help="被动 广播 进群默认开关状态",
                default_value=True,
                type=bool,
            )
        ],
        tasks=[Task(module="broadcast", name="广播")],
    ).to_dict(),
)

AnySeg = (
    UnionPattern(
        [
            Text,
            Image,
            At,
            AtAll,
            Audio,
            Video,
            File,
            Emoji,
            Reply,
            Reference,
            Hyper,
            Button,
            Keyboard,
            Voice,
        ]
    )
    @ "AnySeg"
)

_matcher = on_alconna(
    Alconna(
        "广播",
        Args["content?", AllParam],
        alc.Option(
            "-t|--tag", Args["tag_name_bc", str], help_text="向指定标签的群组广播"
        ),
    ),
    aliases={"bc"},
    priority=1,
    permission=SUPERUSER,
    block=True,
    rule=to_me(),
    use_origin=False,
)

_recall_matcher = on_alconna(
    Alconna("广播撤回"),
    aliases={"recall"},
    priority=1,
    permission=SUPERUSER,
    block=True,
    rule=to_me(),
)

tag_matcher = on_alconna(
    Alconna(
        "群组标签",
        Args["tag_name?", str],
        alc.Option(
            "-l|--list", action=alc.store_true, help_text="列出所有标签及其群组"
        ),
        alc.Option(
            "-g|--groups", Args["group_ids", str], help_text="指定群组ID，逗号分隔"
        ),
        alc.Option("-r|--remove", action=alc.store_true, help_text="从标签移除群组"),
        alc.Option("-d|--delete", action=alc.store_true, help_text="删除标签"),
    ),
    priority=1,
    permission=SUPERUSER,
    block=True,
)

driver = get_driver()


@driver.on_startup
async def _init_broadcast_tags():
    await TagManager.initialize()


@_matcher.handle()
async def handle_broadcast(
    bot: Bot,
    event: Event,
    session: EventSession,
    arp: alc.Arparma,
):
    broadcast_content_msg = await _extract_broadcast_content(bot, event, arp, session)
    if not broadcast_content_msg:
        return

    tag_name_to_broadcast = None
    force_send = False

    if "tag" in arp.options:
        tag_option_result: alc.OptionResult = arp.options["tag"]
        if (
            tag_option_result
            and tag_option_result.args
            and "tag_name_bc" in tag_option_result.args
        ):
            tag_name_to_broadcast = tag_option_result.args["tag_name_bc"]
            if isinstance(tag_name_to_broadcast, str) and tag_name_to_broadcast.strip():
                force_send = True
            else:
                tag_name_to_broadcast = None
                logger.warning(
                    f"广播的 -t 选项参数 tag_name_bc 无效: {tag_name_to_broadcast}",
                    "广播",
                )

    logger.debug(
        f"广播模式: {'强制发送到标签' if force_send else '普通发送'}, 标签名: {tag_name_to_broadcast}",
        "广播",
    )

    target_groups_console, groups_to_actually_send = await get_broadcast_target_groups(
        bot, session, tag_name_to_broadcast, force_send
    )

    if not target_groups_console:
        if tag_name_to_broadcast:
            await MessageUtils.build_message(
                f"标签 '{tag_name_to_broadcast}' 中没有群组或标签不存在。"
            ).send(reply_to=True)
        return

    if not groups_to_actually_send:
        if not force_send and target_groups_console:
            await MessageUtils.build_message(
                "没有启用了广播功能的目标群组可供立即发送。"
            ).send(reply_to=True)
        return

    try:
        await send_broadcast_and_notify(
            bot,
            event,
            broadcast_content_msg,
            groups_to_actually_send,
            target_groups_console,
            session,
            force_send,
        )
    except Exception as e:
        error_msg = "发送广播失败"
        BroadcastManager.log_error(error_msg, e, session)
        await MessageUtils.build_message(f"{error_msg}。").send(reply_to=True)


@_recall_matcher.handle()
async def handle_broadcast_recall(
    bot: Bot,
    event: Event,
    session: EventSession,
):
    """处理广播撤回命令"""
    await MessageUtils.build_message("正在尝试撤回最近一次广播...").send()

    try:
        success_count, error_count = await BroadcastManager.recall_last_broadcast(
            bot, session
        )

        user_id = str(event.get_user_id())
        if success_count == 0 and error_count == 0:
            await bot.send_private_msg(
                user_id=user_id,
                message="没有找到最近的广播消息记录，可能已经撤回或超过可撤回时间。",
            )
        else:
            result = f"广播撤回完成!\n成功撤回 {success_count} 条消息"
            if error_count:
                result += f"\n撤回失败 {error_count} 条消息 (可能已过期或无权限)"
            await bot.send_private_msg(user_id=user_id, message=result)
            BroadcastManager.log_info(
                f"广播撤回完成: 成功 {success_count}, 失败 {error_count}", session
            )
    except Exception as e:
        error_msg = "撤回广播消息失败"
        BroadcastManager.log_error(error_msg, e, session)
        user_id = str(event.get_user_id())
        await bot.send_private_msg(user_id=user_id, message=f"{error_msg}。")


@tag_matcher.handle()
async def handle_group_tags(
    bot: Bot,
    session: EventSession,
    arp: alc.Arparma,
    tag_name_arg: alc.Match[str] = alc.AlconnaMatch("tag_name"),
    remove_opt: alc.Match[bool] = alc.AlconnaMatch("remove.value", False),
):
    logger.debug(f"进入 handle_group_tags, arp: {arp}", "广播标签")

    if "list" in arp.options:
        logger.debug("处理 -l 选项 (通过 arp.options)", "广播标签")
        all_tags_info = await TagManager.get_groups_with_tag_info()
        if not all_tags_info:
            await MessageUtils.build_message("当前没有任何群组标签。").send(
                reply_to=True
            )
            return

        response_lines = ["已有的群组标签及其包含的群组："]
        for tag, groups_in_tag_ids in all_tags_info.items():
            if groups_in_tag_ids:
                group_consoles = await GroupConsole.filter(
                    group_id__in=groups_in_tag_ids
                )
                if group_consoles:
                    group_details = [
                        f"{gc.group_name}({gc.group_id})" for gc in group_consoles
                    ]
                    group_str = ", ".join(group_details)
                else:
                    group_str = "无有效群组 (可能群组已被解散或机器人不再其中)"
            else:
                group_str = "无群组"
            response_lines.append(f"- {tag}: {group_str}")
        await MessageUtils.build_message("\n".join(response_lines)).send(reply_to=True)
        return

    if not tag_name_arg.available:
        logger.debug("标签名参数 (tag_name_arg) 不可用 (非-l操作)", "广播标签")
        await MessageUtils.build_message("请输入要操作的标签名称。").send(reply_to=True)
        return

    tag_name = tag_name_arg.result.strip()
    if not tag_name:
        await MessageUtils.build_message("标签名称不能为空。").send(reply_to=True)
        return
    logger.debug(f"获取到标签名: '{tag_name}'", "广播标签")

    if "delete" in arp.options:
        logger.debug(
            f"处理 -d 选项 (通过 arp.options)，删除标签: {tag_name}", "广播标签"
        )
        if await TagManager.delete_tag(tag_name):
            await MessageUtils.build_message(f"标签 '{tag_name}' 已成功删除。").send(
                reply_to=True
            )
            BroadcastManager.log_info(f"标签 '{tag_name}' 已删除", session)
        else:
            await MessageUtils.build_message(f"标签 '{tag_name}' 不存在。").send(
                reply_to=True
            )
        return

    if "groups" in arp.options:
        logger.debug("处理 -g 选项 (直接从 arp.options 获取)", "广播标签")
        option_result: alc.OptionResult = arp.options["groups"]

        if option_result and option_result.args and "group_ids" in option_result.args:
            group_ids_str = option_result.args["group_ids"]
            if not isinstance(group_ids_str, str):
                logger.error(
                    "内部错误：-g 选项的 group_ids 参数不是字符串类型。", "广播标签"
                )
                await MessageUtils.build_message(
                    "内部错误：`-g` 选项的参数类型不正确。"
                ).send(reply_to=True)
                return

            group_ids = [
                gid.strip()
                for gid in group_ids_str.split(",")
                if gid.strip() and gid.strip().isdigit()
            ]
            if not group_ids:
                await MessageUtils.build_message(
                    "请为 `-g` 选项提供有效的群组ID (纯数字，逗号分隔)。"
                ).send(reply_to=True)
                return

            if remove_opt.available:
                logger.debug(f"从标签 '{tag_name}' 移除群组: {group_ids}", "广播标签")
                removed_count, not_in_tag_ids = await TagManager.remove_groups_from_tag(
                    tag_name, group_ids
                )
                msg = f"从标签 '{tag_name}' 中成功移除了 {removed_count} 个群组。"
                if not_in_tag_ids:
                    msg += (
                        f"\n以下群组不在标签中或输入无效: {', '.join(not_in_tag_ids)}"
                    )
                await MessageUtils.build_message(msg).send(reply_to=True)
                if removed_count > 0:
                    BroadcastManager.log_info(
                        f"从标签 '{tag_name}' 移除群组: {group_ids}", session
                    )
            else:
                logger.debug(f"向标签 '{tag_name}' 添加群组: {group_ids}", "广播标签")
                added_count, invalid_ids = await TagManager.add_groups_to_tag(
                    tag_name, group_ids
                )
                msg = f"向标签 '{tag_name}' 中成功添加了 {added_count} 个群组。"
                if invalid_ids:
                    msg += f"\n以下群组ID无效或已存在于标签中: {', '.join(invalid_ids)}"
                await MessageUtils.build_message(msg).send(reply_to=True)
                if added_count > 0:
                    BroadcastManager.log_info(
                        f"向标签 '{tag_name}' 添加群组: {group_ids}", session
                    )
            return
        else:
            logger.warning(
                "-g 选项存在于 arp.options，但无法提取 'group_ids' 参数或 option_result.args 为空。",
                "广播标签",
            )
            await MessageUtils.build_message("使用 `-g` 选项时，请提供群组ID。").send(
                reply_to=True
            )
            return

    logger.debug(f"执行默认操作：查看标签 '{tag_name}' 内容", "广播标签")
    groups_in_tag_ids = await TagManager.get_groups_by_tag(tag_name)
    if not groups_in_tag_ids:
        await MessageUtils.build_message(
            f"标签 '{tag_name}' 不存在或没有包含任何群组。"
        ).send(reply_to=True)
    else:
        group_consoles = await GroupConsole.filter(group_id__in=groups_in_tag_ids)
        if not group_consoles:
            await MessageUtils.build_message(
                f"标签 '{tag_name}' 中的群组当前无法获取信息 (可能已解散或机器人不在其中)。"
            ).send(reply_to=True)
            return

        group_id_to_name = {gc.group_id: gc.group_name for gc in group_consoles}
        valid_groups_in_tag = []
        for gid in groups_in_tag_ids:
            name = group_id_to_name.get(gid)
            if name is not None:
                valid_groups_in_tag.append(f"{name} ({gid})")

        if not valid_groups_in_tag:
            group_str = "无有效群组 (可能群组已被解散或机器人不再其中)"
        else:
            group_str = "\n - ".join(valid_groups_in_tag)

        await MessageUtils.build_message(
            f"标签 '{tag_name}' 包含以下群组：\n - {group_str}"
        ).send(reply_to=True)
