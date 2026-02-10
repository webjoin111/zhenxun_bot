from typing import Any, cast

from nonebot.adapters import Bot

from zhenxun import ui
from zhenxun.models.group_console import GroupConsole
from zhenxun.models.plugin_info import PluginInfo
from zhenxun.models.task_info import TaskInfo
from zhenxun.services.cache import CacheRoot
from zhenxun.ui.models import StatusBadgeCell, TextCell
from zhenxun.ui.models.core import LayoutData
from zhenxun.utils.common_utils import CommonUtils
from zhenxun.utils.enum import BlockType, CacheType, PluginType
from zhenxun.utils.exception import GroupInfoNotFound
from zhenxun.utils.platform import PlatformUtils


async def build_plugin() -> bytes:
    column_name = [
        "ID",
        "模块",
        "名称",
        "全局状态",
        "禁用类型",
        "加载状态",
        "菜单分类",
        "作者",
        "版本",
        "金币花费",
    ]
    plugin_list = await PluginInfo.filter(plugin_type__not=PluginType.HIDDEN).all()
    rows = []
    for plugin in plugin_list:
        status_cell = StatusBadgeCell(
            text="开启" if plugin.status else "关闭",
            status_type="ok" if plugin.status else "error",
        )
        load_cell = StatusBadgeCell(
            text="SUCCESS" if plugin.load_status else "ERROR",
            status_type="ok" if plugin.load_status else "error",
        )
        rows.append(
            [
                plugin.id,
                plugin.module,
                plugin.name,
                status_cell,
                plugin.block_type.value if plugin.block_type else "-",
                load_cell,
                plugin.menu_type or "-",
                plugin.author or "-",
                plugin.version or "-",
                plugin.cost_gold,
            ]
        )

    table = ui.table("Plugin List", "插件状态概览")
    table.set_headers(column_name)
    table.add_rows(rows)
    table.set_column_widths(
        [
            "60px",
            "150px",
            "150px",
            "80px",
            "100px",
            "100px",
            "100px",
            "100px",
            "80px",
            "80px",
        ]
    )
    return await ui.render(
        table,
        viewport={"width": 1400, "height": 10},
    )


async def build_task(group_id: str | None) -> bytes:
    """构造被动技能状态图片

    参数:
        group_id: 群组id

    异常:
        GroupInfoNotFound: 未找到群组

    返回:
        BuildImage: 被动技能状态图片
    """
    task_list = await TaskInfo.all()
    column_name = ["ID", "模块", "名称", "群组状态", "全局状态", "运行时间"]
    group = None
    if group_id:
        group = await GroupConsole.get_group_db(group_id=group_id)
        if not group:
            raise GroupInfoNotFound()
    else:
        column_name.remove("群组状态")
    rows = []
    for task in task_list:
        global_status_cell = StatusBadgeCell(
            text="开启" if task.status else "关闭",
            status_type="ok" if task.status else "error",
        )
        row = [task.id, task.module, task.name]
        if group:
            is_group_open = f"<{task.module}," not in group.block_task
            group_status_cell = StatusBadgeCell(
                text="开启" if is_group_open else "关闭",
                status_type="ok" if is_group_open else "error",
            )
            row.append(group_status_cell)
        row.extend([global_status_cell, task.run_time or "-"])
        rows.append(row)

    table = ui.table("Task List", "被动技能状态概览")
    table.set_headers(column_name)
    table.add_rows(rows)
    if group:
        table.set_column_widths(["60px", "150px", "150px", "100px", "100px", "auto"])
        viewport_width = 1200
    else:
        table.set_column_widths(["60px", "150px", "150px", "100px", "auto"])
        viewport_width = 1000

    return await ui.render(
        table,
        viewport={"width": viewport_width, "height": 10},
    )


class PluginManager:
    @staticmethod
    def _modify_block_string(current_str: str, module: str, add: bool) -> str:
        """辅助: 添加或移除禁用模块字符串"""
        items = CommonUtils.convert_module_format(current_str)
        if add:
            if module not in items:
                items.append(module)
        else:
            if module in items:
                items.remove(module)
        return CommonUtils.convert_module_format(items)

    @classmethod
    async def batch_update_status(
        cls,
        name: str,
        target_groups: set[str],
        status: bool,
        is_task: bool = False,
        is_superuser: bool = False,
        is_whitelist_mode: bool = False,
        bot: Bot | None = None,
        force: bool = False,
    ) -> str:
        """批量更新插件/被动状态"""
        module_name = name
        if is_task:
            task = await TaskInfo.get_or_none(name=name)
            if not task:
                return f"未找到被动技能: {name}"
            module_name = task.module
        else:
            plugin = None
            if name.isdigit():
                plugin = await PluginInfo.get_or_none(id=int(name))
            else:
                plugin = await PluginInfo.get_or_none(
                    name=name, load_status=True, plugin_type__not=PluginType.PARENT
                )
            if not plugin:
                return f"未找到插件: {name}"
            module_name = plugin.module

        use_superuser_field = is_superuser and not force
        if is_task:
            field_name = "superuser_block_task" if use_superuser_field else "block_task"
        else:
            field_name = (
                "superuser_block_plugin" if use_superuser_field else "block_plugin"
            )

        clean_targets = {str(group_id) for group_id in target_groups if group_id}

        groups_to_open = set()
        groups_to_close = set()

        if is_whitelist_mode and status:
            if bot:
                active_groups, _ = await PlatformUtils.get_group_list(
                    bot, only_group=True
                )
                all_group_set = {str(g.group_id) for g in active_groups if g.group_id}
            else:
                all_group_ids = await GroupConsole.all().values_list(
                    "group_id", flat=True
                )
                all_group_set = {str(group_id) for group_id in all_group_ids}
            groups_to_open = clean_targets
            groups_to_close = all_group_set - clean_targets
        else:
            if status:
                groups_to_open = clean_targets
            else:
                groups_to_close = clean_targets

        affected_ids = groups_to_open | groups_to_close
        if not affected_ids:
            return "没有目标群组需要操作。"

        groups_obj = await GroupConsole.filter(group_id__in=list(affected_ids)).all()
        update_list = []
        opened_groups: set[str] = set()
        closed_groups: set[str] = set()
        for group in groups_obj:
            gid = str(group.group_id)
            current_value = getattr(group, field_name)
            new_value = current_value
            is_changed = False
            change_type: str | None = None
            if gid in groups_to_open:
                new_value = cls._modify_block_string(current_value, module_name, False)
                if current_value != new_value:
                    is_changed = True
                    change_type = "open"
            elif gid in groups_to_close:
                new_value = cls._modify_block_string(current_value, module_name, True)
                if current_value != new_value:
                    is_changed = True
                    change_type = "close"
            if is_changed:
                setattr(group, field_name, new_value)
                update_list.append(group)
                if change_type == "open":
                    opened_groups.add(gid)
                elif change_type == "close":
                    closed_groups.add(gid)

        if update_list:
            await GroupConsole.bulk_update(update_list, [field_name], batch_size=500)
            await CacheRoot.clear(CacheType.GROUPS)

        action_str = "开启" if status else "关闭"
        item_str = "被动" if is_task else "插件"
        mode_str = "(白名单模式)" if is_whitelist_mode else ""
        if not update_list:
            return (
                f"目标群组的 {item_str} {name} 均已处于 {action_str} 状态，"
                "无需重复操作。"
            )

        opened_count = len(opened_groups)
        closed_count = len(closed_groups)

        if is_whitelist_mode:
            msg_parts = []
            if opened_count > 0:
                msg_parts.append(f"已开启 {opened_count} 个群组")
            if closed_count > 0:
                msg_parts.append(f"已关闭 {closed_count} 个群组")
            return f"{'，'.join(msg_parts)} 的 {item_str} {name} {mode_str}。"

        affected_count = len(opened_groups) if status else len(closed_groups)
        return f"已{action_str} {affected_count} 个群组的 {item_str} {name}。"

    @classmethod
    async def render_global_status(cls, name: str, is_task: bool, bot: Bot) -> bytes:
        """渲染全局状态报表，含差异化过滤和双栏展示"""
        if is_task:
            info = await TaskInfo.get_or_none(name=name)
            if not info:
                raise ValueError(f"未找到被动技能: {name}")
            module = info.module
            default_status = info.status
        else:
            info = (
                await PluginInfo.get_or_none(id=int(name))
                if name.isdigit()
                else await PluginInfo.get_or_none(
                    name=name, load_status=True, plugin_type__not=PluginType.PARENT
                )
            )
            if not info:
                raise ValueError(f"未找到插件: {name}")
            module = info.module
            default_status = info.status

        online_groups, _ = await PlatformUtils.get_group_list(bot)
        valid_keys = {(str(g.group_id), g.channel_id) for g in online_groups}

        all_db_groups = await GroupConsole.all()
        target_groups = [
            g for g in all_db_groups if (str(g.group_id), g.channel_id) in valid_keys
        ]

        total_count = len(target_groups)
        status_data = []
        for group in target_groups:
            gid = str(group.group_id)
            if is_task:
                is_su_blocked = await GroupConsole.is_superuser_block_task(gid, module)
                is_norm_blocked = await GroupConsole.is_block_task(gid, module)
                is_norm_blocked = f"<{module}," in group.block_task
            else:
                is_su_blocked = await GroupConsole.is_superuser_block_plugin(
                    gid, module
                )
                is_norm_blocked = await GroupConsole.is_normal_block_plugin(gid, module)

            is_open = bool(default_status) and not is_su_blocked and not is_norm_blocked

            if not default_status:
                status_text = "全局关闭"
                badge_color = "error"
            elif is_su_blocked:
                status_text = "系统禁用"
                badge_color = "error"
            elif is_norm_blocked:
                status_text = "群内关闭"
                badge_color = "warning"
            else:
                status_text = "开启"
                badge_color = "success"
            status_data.append(
                {
                    "id": str(group.group_id),
                    "name": group.group_name,
                    "status": is_open,
                    "status_text": status_text,
                    "badge_color": badge_color,
                }
            )

        open_list = [item for item in status_data if item["status"]]
        close_list = [item for item in status_data if not item["status"]]
        open_count = len(open_list)
        close_count = len(close_list)
        summary_tip = (
            f"总群数: {total_count} | 🟢 开启: {open_count} | 🔴 关闭: {close_count}"
        )

        open_rate = open_count / total_count if total_count > 0 else 0
        close_rate = close_count / total_count if total_count > 0 else 0

        kpi_row = LayoutData.row(gap="12px", align_items="stretch")

        card1_content = ui.vstack(
            [
                ui.text("总群数", font_size="13px", color="var(--color-text-muted)"),
                ui.text(
                    str(total_count),
                    font_size="24px",
                    bold=True,
                    color="var(--color-text-dark)",
                ),
            ],
            gap="2px",
            align_items="start",
            padding="0",
        )
        card1 = ui.card(card1_content).with_inline_style(
            {"--card-padding": "12px 16px"}
        )
        kpi_row.add_item(card1, metadata={"flex": True})

        card2_header = LayoutData.row(justify_content="space-between", width="100%")
        card2_header.add_item(
            ui.text("已开启", font_size="13px", color="var(--color-text-muted)")
        )
        card2_header.add_item(
            ui.text(
                f"{open_rate:.1%}",
                font_size="13px",
                bold=True,
                color="var(--color-accent-green)",
            )
        )

        card2_content = ui.vstack(
            [
                card2_header.build(),
                ui.text(
                    str(open_count),
                    font_size="24px",
                    bold=True,
                    color="var(--color-text-dark)",
                ),
            ],
            gap="2px",
            align_items="stretch",
            padding="0",
        )
        card2 = ui.card(card2_content).with_inline_style(
            {"--card-padding": "12px 16px"}
        )
        kpi_row.add_item(card2, metadata={"flex": True})

        card3_header = LayoutData.row(justify_content="space-between", width="100%")
        card3_header.add_item(
            ui.text("已关闭", font_size="13px", color="var(--color-text-muted)")
        )
        card3_header.add_item(
            ui.text(
                f"{close_rate:.1%}",
                font_size="13px",
                bold=True,
                color="var(--color-accent-red)",
            )
        )

        card3_content = ui.vstack(
            [
                card3_header.build(),
                ui.text(
                    str(close_count),
                    font_size="24px",
                    bold=True,
                    color="var(--color-text-dark)",
                ),
            ],
            gap="2px",
            align_items="stretch",
            padding="0",
        )
        card3 = ui.card(card3_content).with_inline_style(
            {"--card-padding": "12px 16px"}
        )
        kpi_row.add_item(card3, metadata={"flex": True})

        global_alert = None
        if not default_status:
            global_alert = ui.alert(
                "全局已禁用", f"功能 [{name}] 当前处于全局关闭状态。", type="error"
            )

        progress_section = ui.vstack(
            [
                ui.text(
                    f"功能 [{name}] 全局覆盖率",
                    font_size="14px",
                    color="var(--color-text-muted)",
                ),
                ui.progress_bar(
                    progress=open_rate * 100,
                    label=f"{open_count}/{total_count}",
                    color_scheme="primary",
                ),
            ],
            gap="8px",
        )

        display_list = []
        list_title = "群组状态详情"

        if total_count == 0:
            display_list = []
        elif not default_status:
            display_list = []
        elif open_rate > 0.9:
            display_list = close_list
            list_title = f"异常状态列表 (其余 {open_count} 个群均正常开启)"
        elif open_rate < 0.1:
            display_list = open_list
            list_title = f"异常状态列表 (其余 {close_count} 个群均已禁用)"
        else:
            display_list = sorted(status_data, key=lambda x: not x["status"])

        content_area = None

        if not display_list:
            if not default_status:
                content_area = ui.alert(
                    "全局关闭", "该功能已全局关闭，所有群组均不可用。", type="error"
                )
            else:
                content_area = ui.alert(
                    "状态完美", f"所有 {total_count} 个群组状态一致。", type="success"
                )
        elif len(display_list) <= 15:
            rows = []
            for item in display_list:
                status_cell = StatusBadgeCell(
                    text=item["status_text"],
                    status_type=item["badge_color"],
                )
                rows.append(
                    [
                        TextCell(content=str(item["id"])),
                        TextCell(content=str(item["name"])),
                        status_cell,
                    ]
                )

            content_area = ui.table(list_title, None) \
                .set_headers(["群号", "群名", "状态"]) \
                .set_column_widths(["160px", "auto", "100px"]) \
                .add_rows(rows)
        else:
            grid = LayoutData.grid(columns=3, gap="15px")

            MAX_SHOW = 60
            for item in display_list[:MAX_SHOW]:
                status_text = item["status_text"]
                badge_color = item["badge_color"]

                card_content = ui.vstack(
                    [
                        ui.text(str(item["name"]), bold=True, font_size="15px"),
                        LayoutData.row(justify_content="space-between", width="100%")
                        .add_item(
                            ui.text(str(item["id"]), font_size="12px", color="#999")
                        )
                        .add_item(ui.badge(status_text, color_scheme=badge_color)),
                    ],
                    gap="8px",
                    align_items="start",
                )

                grid.add_item(ui.card(card_content))

            content_area = LayoutData.column(gap="10px")
            content_area.add_item(grid.build())

            if len(display_list) > MAX_SHOW:
                content_area.add_item(
                    ui.text(
                        f"... 还有 {len(display_list) - MAX_SHOW} 个群组未显示 ...",
                        align="center",
                        color="#ccc",
                    )
                )
            content_area = content_area.build()

        main_layout = LayoutData.column(padding="40px", gap="30px")

        main_layout.add_item(
            ui.text(
                f"功能状态报告: {name}",
                font_size="32px",
                bold=True,
                align="center",
                color="var(--color-primary)",
            )
        )

        stats_items: list[Any] = []
        if global_alert:
            stats_items.append(global_alert)
        stats_items.extend(
            [
                kpi_row.build(),
                ui.divider(margin="15px 0"),
                progress_section,
                ui.text(
                    summary_tip,
                    font_size="13px",
                    color="var(--color-text-muted)",
                    align="center",
                ),
            ]
        )

        stats_panel = ui.card(ui.vstack(stats_items))
        main_layout.add_item(stats_panel)
        main_layout.add_item(content_area)

        return await ui.render(
            main_layout.build(),
            viewport={"width": 900, "height": 10},
        )

    @classmethod
    async def set_default_status(cls, plugin_name: str, status: bool) -> str:
        """设置插件进群默认状态

        参数:
            plugin_name: 插件名称
            status: 状态

        返回:
            str: 返回信息
        """
        if plugin_name.isdigit():
            plugin = await PluginInfo.get_or_none(id=int(plugin_name))
        else:
            plugin = await PluginInfo.get_or_none(
                name=plugin_name, load_status=True, plugin_type__not=PluginType.PARENT
            )
        if plugin:
            plugin.default_status = status
            await plugin.save(update_fields=["default_status"])
            status_text = "开启" if status else "关闭"
            return f"成功将 {plugin.name} 进群默认状态修改为: {status_text}"
        return "没有找到这个功能喔..."

    @classmethod
    async def set_all_plugin_status(
        cls, status: bool, is_default: bool = False, group_id: str | None = None
    ) -> str:
        """修改所有插件状态

        参数:
            status: 状态
            is_default: 是否进群默认.
            group_id: 指定群组id.

        返回:
            str: 返回信息
        """
        if is_default:
            await PluginInfo.filter(plugin_type=PluginType.NORMAL).update(
                default_status=status
            )
            return f"成功将所有功能进群默认状态修改为: {'开启' if status else '关闭'}"
        if group_id:
            if group := await GroupConsole.get_group_db(group_id=group_id):
                module_list = cast(
                    list[str],
                    await PluginInfo.filter(plugin_type=PluginType.NORMAL).values_list(
                        "module", flat=True
                    ),
                )
                if status:
                    group.block_plugin = ""
                else:
                    group.block_plugin = CommonUtils.convert_module_format(module_list)
                await group.save(update_fields=["block_plugin"])
                return f"成功将此群组所有功能状态修改为: {'开启' if status else '关闭'}"
            return "获取群组失败..."
        await PluginInfo.filter(plugin_type=PluginType.NORMAL).update(
            status=status, block_type=None if status else BlockType.ALL
        )
        await CacheRoot.invalidate_cache(CacheType.PLUGINS)
        return f"成功将所有功能全局状态修改为: {'开启' if status else '关闭'}"

    @classmethod
    async def is_wake(cls, group_id: str) -> bool:
        """是否醒来

        参数:
            group_id: 群组id

        返回:
            bool: 是否醒来
        """
        if c := await GroupConsole.get_group_db(group_id=group_id):
            return c.status
        return False

    @classmethod
    async def set_group_active_status(cls, group_id: str, status: bool):
        """设置群组激活状态 (休眠/醒来)

        参数:
            group_id: 群组id
            status: True为醒来，False为休眠
        """
        group, _ = await GroupConsole.get_or_create(
            group_id=group_id, channel_id__isnull=True
        )
        group.status = status
        await group.save(update_fields=["status"])

    @classmethod
    async def set_plugin_status(cls, module: str, status: bool):
        """设置插件状态

        参数:
            module: 模块名
            status: 状态
        """
        if plugin := await PluginInfo.get_plugin(module=module):
            plugin.status = status
            await plugin.save(update_fields=["status"])

    @classmethod
    async def set_global_all_task_status(cls, is_default: bool, status: bool) -> str:
        """设置所有被动技能全局状态

        参数:
            is_default: 是否为默认状态
            status: 开启/关闭

        返回:
            str: 返回信息
        """
        action = "开启" if status else "禁用"
        if is_default:
            await TaskInfo.all().update(default_status=status)
            return f"已{action}所有被动进群默认状态"
        else:
            await TaskInfo.all().update(status=status)
            return f"已全局{action}所有被动状态"

    @classmethod
    async def set_global_task_status(
        cls, name: str, status: bool, is_default: bool = False
    ) -> str:
        """设置单个被动技能全局状态

        参数:
            name: 被动技能名称
            status: 开启/关闭
            is_default: 是否为默认状态

        返回:
            str: 返回信息
        """
        action = "开启" if status else "禁用"
        if is_default:
            await TaskInfo.filter(name=name).update(default_status=status)
            return f"已{action}被动进群默认状态 {name}"
        else:
            await TaskInfo.filter(name=name).update(status=status)
            return f"已全局{action}被动状态 {name}"

    @classmethod
    async def superuser_task_handle(
        cls, task_name: str, group_id: str | None, status: bool
    ) -> str:
        """超级用户禁用被动技能

        参数:
            task_name: 被动技能名称
            group_id: 群组id
            status: 状态

        返回:
            str: 返回信息
        """
        if not (task := await TaskInfo.get_or_none(name=task_name)):
            return "没有找到这个功能喔..."
        if group_id:
            if status:
                await GroupConsole.set_unblock_task(group_id, task.module, True)
            else:
                await GroupConsole.set_block_task(group_id, task.module, True)
            status_str = "开启" if status else "关闭"
            return f"已成功将群组 {group_id} 被动技能 {task_name} {status_str}!"
        return "没有找到这个群组喔..."

    @classmethod
    async def superuser_set_status(
        cls,
        plugin_name: str,
        status: bool,
        block_type: BlockType | None,
        group_id: str | None,
    ) -> str:
        """超级用户设置插件状态（开启/禁用）

        参数:
            plugin_name: 插件名称
            status: True为开启，False为禁用
            block_type: 禁用类型
            group_id: 群组id

        返回:
            str: 返回信息
        """
        action_cn = "开启" if status else "关闭"
        if plugin_name.isdigit():
            plugin = await PluginInfo.get_or_none(id=int(plugin_name))
        else:
            plugin = await PluginInfo.get_or_none(
                name=plugin_name, load_status=True, plugin_type__not=PluginType.PARENT
            )
        if plugin:
            if group_id:
                is_blocked = await GroupConsole.is_superuser_block_plugin(
                    group_id, plugin.module
                )
                if status and is_blocked:
                    await GroupConsole.set_unblock_plugin(group_id, plugin.module, True)
                    return f"已成功{action_cn}群组 {group_id} 的 {plugin_name} 功能!"
                if not status and not is_blocked:
                    await GroupConsole.set_block_plugin(group_id, plugin.module, True)
                    return f"已成功{action_cn}群组 {group_id} 的 {plugin_name} 功能!"
                return f"此群组该功能已被超级用户{action_cn}，不要重复操作..."
            plugin.block_type = block_type
            plugin.status = not bool(block_type)
            await plugin.save(update_fields=["status", "block_type"])
            if not block_type:
                return f"已成功将 {plugin.name} 全局{action_cn}!"
            if block_type == BlockType.ALL:
                return f"已成功将 {plugin.name} 全局{action_cn}!"
            if block_type == BlockType.GROUP:
                return f"已成功将 {plugin.name} 全局群组{action_cn}!"
            if block_type == BlockType.PRIVATE:
                return f"已成功将 {plugin.name} 全局私聊{action_cn}!"
        return "没有找到这个功能喔..."
