import asyncio
from typing import ClassVar

import ujson as json

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.models.group_console import GroupConsole
from zhenxun.services.log import logger

TAG_FILE = DATA_PATH / "broadcast" / "tags.json"
TAG_FILE.parent.mkdir(parents=True, exist_ok=True)

_tags_data: ClassVar[dict[str, list[str]]] = {}
_lock = asyncio.Lock()


async def _load_tags():
    async with _lock:
        if TAG_FILE.exists():
            try:
                content = TAG_FILE.read_text(encoding="utf-8")
                if content:
                    _tags_data.clear()
                    _tags_data.update(json.loads(content))
                else:
                    _tags_data.clear()
            except json.JSONDecodeError:
                logger.error(
                    "加载群组标签失败: JSON解析错误，文件内容可能已损坏。", "广播标签"
                )
                _tags_data.clear()
            except Exception as e:
                logger.error(f"加载群组标签时发生未知错误: {e}", "广播标签")
                _tags_data.clear()
        else:
            _tags_data.clear()


async def _save_tags():
    async with _lock:
        try:
            TAG_FILE.write_text(
                json.dumps(_tags_data, ensure_ascii=False, indent=4), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"保存群组标签失败: {e}", "广播标签")


class TagManager:
    @classmethod
    async def initialize(cls):
        await _load_tags()

    @classmethod
    async def add_groups_to_tag(
        cls, tag_name: str, group_ids: list[str]
    ) -> tuple[int, list[str]]:
        """向标签添加群组，返回成功添加的数量和无效/重复的群组ID列表"""
        await _load_tags()
        if not group_ids:
            return 0, []

        valid_group_consoles = await GroupConsole.filter(group_id__in=group_ids)
        valid_group_ids_from_db = {str(g.group_id) for g in valid_group_consoles}

        logger.debug(f"输入群组ID: {group_ids}", "广播标签")
        logger.debug(f"数据库中有效的群组ID: {valid_group_ids_from_db}", "广播标签")

        if tag_name not in _tags_data:
            _tags_data[tag_name] = []
            logger.debug(f"标签 '{tag_name}' 不存在，已创建。", "广播标签")

        current_tag_groups = set(_tags_data[tag_name])
        added_count = 0
        processed_invalid_ids = []

        for gid_input in group_ids:
            gid_str = str(gid_input).strip()
            if not gid_str:
                continue

            if gid_str not in valid_group_ids_from_db:
                processed_invalid_ids.append(
                    f"{gid_str} (机器人未加入该群或群组记录不存在)"
                )
                logger.warning(
                    f"群组ID '{gid_str}' 无效或机器人未加入，无法添加到标签 '{tag_name}'。",
                    "广播标签",
                )
                continue

            if gid_str not in current_tag_groups:
                _tags_data[tag_name].append(gid_str)
                current_tag_groups.add(gid_str)
                added_count += 1
                logger.debug(
                    f"群组 '{gid_str}' 已添加到标签 '{tag_name}'。", "广播标签"
                )
            else:
                processed_invalid_ids.append(f"{gid_str} (已存在于标签中)")
                logger.debug(
                    f"群组 '{gid_str}' 已存在于标签 '{tag_name}'，跳过添加。",
                    "广播标签",
                )

        if added_count > 0:
            await _save_tags()
            logger.info(
                f"成功向标签 '{tag_name}' 添加了 {added_count} 个群组。", "广播标签"
            )

        return added_count, processed_invalid_ids

    @classmethod
    async def remove_groups_from_tag(
        cls, tag_name: str, group_ids: list[str]
    ) -> tuple[int, list[str]]:
        """从标签移除群组，返回成功移除的数量和不在标签中的群组ID列表"""
        await _load_tags()
        if tag_name not in _tags_data or not _tags_data[tag_name]:
            return 0, group_ids

        removed_count = 0
        not_in_tag_ids = []

        tag_groups_set = set(_tags_data[tag_name])
        groups_to_remove_set = set(map(str, group_ids))

        actually_removed_groups = list(
            tag_groups_set.intersection(groups_to_remove_set)
        )

        if actually_removed_groups:
            _tags_data[tag_name] = [
                gid
                for gid in _tags_data[tag_name]
                if gid not in actually_removed_groups
            ]
            if not _tags_data[tag_name]:
                del _tags_data[tag_name]
            await _save_tags()
            removed_count = len(actually_removed_groups)

        not_in_tag_ids = list(groups_to_remove_set.difference(tag_groups_set))

        return removed_count, not_in_tag_ids

    @classmethod
    async def get_groups_by_tag(cls, tag_name: str) -> list[str]:
        """根据标签名获取群组ID列表"""
        await _load_tags()
        return _tags_data.get(tag_name, [])[:]

    @classmethod
    async def delete_tag(cls, tag_name: str) -> bool:
        """删除标签"""
        await _load_tags()
        if tag_name in _tags_data:
            del _tags_data[tag_name]
            await _save_tags()
            return True
        return False

    @classmethod
    async def list_tags(cls) -> list[str]:
        """列出所有标签名"""
        await _load_tags()
        return list(_tags_data.keys())

    @classmethod
    async def get_groups_with_tag_info(cls) -> dict[str, list[str]]:
        """获取所有标签及其群组信息"""
        await _load_tags()
        return {tag: groups[:] for tag, groups in _tags_data.items()}
