from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from zhenxun.models.plugin_info import PluginInfo as DbPluginInfo
from zhenxun.services.log import logger
from zhenxun.utils.enum import BlockType, PluginType

from ....base_model import Result
from ....utils import authentication
from .data_source import ApiDataSource
from .model import (
    BatchUpdatePlugins,
    PluginCount,
    PluginDetail,
    PluginInfo,
    PluginSwitch,
    RenameMenuTypePayload,
    UpdatePlugin,
)

router = APIRouter(prefix="/plugin")


@router.get(
    "/get_plugin_list",
    dependencies=[authentication()],
    response_model=Result[list[PluginInfo]],
    response_class=JSONResponse,
    description="获取插件列表",  # type: ignore
)
async def _(
    plugin_type: list[PluginType] = Query(None), menu_type: str | None = None
) -> Result[list[PluginInfo]]:
    try:
        return Result.ok(
            await ApiDataSource.get_plugin_list(plugin_type, menu_type), "拿到信息啦!"
        )
    except Exception as e:
        logger.error(f"{router.prefix}/get_plugin_list 调用错误", "WebUi", e=e)
        return Result.fail(f"发生了一点错误捏 {type(e)}: {e}")


@router.get(
    "/get_plugin_count",
    dependencies=[authentication()],
    response_model=Result[PluginCount],
    response_class=JSONResponse,
    description="获取插件数量",  # type: ignore
)
async def _() -> Result[PluginCount]:
    try:
        plugin_count = PluginCount()
        plugin_count.normal = await DbPluginInfo.filter(
            plugin_type=PluginType.NORMAL, load_status=True
        ).count()
        plugin_count.admin = await DbPluginInfo.filter(
            plugin_type__in=[PluginType.ADMIN, PluginType.SUPER_AND_ADMIN],
            load_status=True,
        ).count()
        plugin_count.superuser = await DbPluginInfo.filter(
            plugin_type__in=[PluginType.SUPERUSER, PluginType.SUPER_AND_ADMIN],
            load_status=True,
        ).count()
        plugin_count.other = await DbPluginInfo.filter(
            plugin_type__in=[PluginType.HIDDEN, PluginType.DEPENDANT], load_status=True
        ).count()
        return Result.ok(plugin_count, "拿到信息啦!")
    except Exception as e:
        logger.error(f"{router.prefix}/get_plugin_count 调用错误", "WebUi", e=e)
        return Result.fail(f"发生了一点错误捏 {type(e)}: {e}")


@router.post(
    "/update_plugin",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
    description="更新插件参数",
)
async def _(param: UpdatePlugin) -> Result:
    try:
        await ApiDataSource.update_plugin(param)
        return Result.ok(info="已经帮你写好啦!")
    except (ValueError, KeyError):
        return Result.fail("插件数据不存在...")
    except Exception as e:
        logger.error(f"{router.prefix}/update_plugin 调用错误", "WebUi", e=e)
        return Result.fail(f"{type(e)}: {e}")


@router.post(
    "/change_switch",
    dependencies=[authentication()],
    response_model=Result,
    response_class=JSONResponse,
    description="开关插件",
)
async def _(param: PluginSwitch) -> Result:
    try:
        db_plugin = await DbPluginInfo.get_plugin(module=param.module)
        if not db_plugin:
            return Result.fail("插件不存在...")
        if not param.status:
            db_plugin.block_type = BlockType.ALL
            db_plugin.status = False
        else:
            db_plugin.block_type = None
            db_plugin.status = True
        await db_plugin.save()
        return Result.ok(info="成功改变了开关状态!")
    except Exception as e:
        logger.error(f"{router.prefix}/change_switch 调用错误", "WebUi", e=e)
        return Result.fail(f"{type(e)}: {e}")


@router.get(
    "/get_plugin_menu_type",
    dependencies=[authentication()],
    response_model=Result[list[str]],
    response_class=JSONResponse,
    description="获取插件类型",
)
async def _() -> Result[list[str]]:
    try:
        menu_type_list = []
        result = (
            await DbPluginInfo.filter(load_status=True)
            .annotate()
            .values_list("menu_type", flat=True)
        )
        for r in result:
            if r not in menu_type_list and r:
                menu_type_list.append(r)
        return Result.ok(menu_type_list)
    except Exception as e:
        logger.error(f"{router.prefix}/get_plugin_menu_type 调用错误", "WebUi", e=e)
        return Result.fail(f"{type(e)}: {e}")


@router.get(
    "/get_plugin",
    dependencies=[authentication()],
    response_model=Result[PluginDetail],
    response_class=JSONResponse,
    description="获取插件详情",
)
async def _(module: str) -> Result[PluginDetail]:
    try:
        return Result.ok(
            await ApiDataSource.get_plugin_detail(module), "已经帮你写好啦!"
        )
    except (ValueError, KeyError):
        return Result.fail("插件数据不存在...")
    except Exception as e:
        logger.error(f"{router.prefix}/get_plugin 调用错误", "WebUi", e=e)
        return Result.fail(f"{type(e)}: {e}")


@router.put("/plugins/batch_update", summary="批量更新插件配置")
async def batch_update_plugin_config_api(params: BatchUpdatePlugins):
    """批量更新插件配置，如开关、类型等"""
    result = await ApiDataSource.batch_update_plugins(params=params)
    if result["errors"]:
        # 可以根据需要返回更详细的错误信息，或者只返回一个笼统的失败信息
        # 这里我们返回包含错误详情的 200 OK，让前端处理
        # 或者可以抛出 HTTPException
        # from fastapi import HTTPException
        # raise HTTPException(
        #     status_code=400,
        #     detail={"message": "部分插件更新失败", "errors": result["errors"]}
        # )
        pass # 暂时只返回结果字典
    return result


# 新增：重命名菜单类型路由
@router.put(
    "/menu_type/rename",
    dependencies=[authentication()],
    response_model=Result,
    summary="重命名菜单类型"
)
async def rename_menu_type_api(payload: RenameMenuTypePayload) -> Result:
    try:
        result = await ApiDataSource.rename_menu_type(
            old_name=payload.old_name, new_name=payload.new_name)
        if result.get("success"):
            return Result.ok(info=result.get(
                "info",
                f"成功将 {result.get('updated_count', 0)} 个插件的菜单类型"
                f"从 '{payload.old_name}' 修改为 '{payload.new_name}'"
            ))
        else:
             # 这种情况理论上不会发生，因为 rename_menu_type 失败会抛异常
            return Result.fail(info=result.get("info", "重命名失败"))
    except ValueError as ve:
        return Result.fail(info=str(ve))
    except RuntimeError as re:
        logger.error(f"{router.prefix}/menu_type/rename 调用错误", "WebUi", e=re)
        return Result.fail(info=str(re))
    except Exception as e:
        logger.error(f"{router.prefix}/menu_type/rename 调用错误", "WebUi", e=e)
        return Result.fail(info=f"发生未知错误: {type(e).__name__}")
