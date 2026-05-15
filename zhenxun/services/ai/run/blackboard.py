import asyncio
import json
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError, create_model

from zhenxun.utils.pydantic_compat import (
    model_dump,
    model_fields,
    model_json_schema,
    model_validate,
)

T = TypeVar("T", bound=BaseModel)


class BlackboardManager(Generic[T]):
    """
    结构化黑板管理器，提供带锁的并发读写和 Schema 校验。
    解决多智能体 (Team) 并发状态竞争及无结构脏写的问题。
    """

    def __init__(self, schema: type[T], initial_state: T | None = None):
        self.schema = schema
        if initial_state is not None:
            self._state = initial_state
        else:
            try:
                self._state = schema()
            except ValidationError as e:
                raise ValueError(
                    f"初始化黑板状态失败，因为 Schema 存在无默认值的必填字段，请显式提供 initial_state: {e}"
                )

        self._lock = asyncio.Lock()

    async def read(self) -> str:
        """安全读取当前状态的 JSON 字符串表示"""
        async with self._lock:
            state_dict = model_dump(self._state)
            return json.dumps(state_dict, ensure_ascii=False)

    async def update(self, **kwargs: Any) -> str:
        """
        部分更新状态。
        将传入的 kwargs 与现有状态进行深度合并，并通过 Pydantic 进行严格校验。
        如果类型错误或非法，将抛出带有详细定位信息的 ValueError。
        """
        async with self._lock:
            current_dict = model_dump(self._state)

            def deep_update(d: dict, u: dict) -> dict:
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    elif isinstance(v, list) and k in d and isinstance(d[k], list):
                        d[k].extend(x for x in v if x not in d[k])
                    else:
                        d[k] = v
                return d

            merged_dict = deep_update(current_dict, kwargs)

            try:
                new_state = model_validate(self.schema, merged_dict)
                self._state = new_state
                return "全局黑板状态更新成功"
            except ValidationError as e:
                error_msgs = []
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    msg = err.get("msg", "")
                    error_msgs.append(f"字段 `{loc}`: {msg}")
                err_msg = "\n".join(error_msgs)
                raise ValueError(f"黑板状态更新失败，非法的数据格式:\n{err_msg}")

    def get_schema(self) -> dict[str, Any]:
        """获取结构化 Schema，用于提供给大模型参考或组装 Tool"""
        return model_json_schema(self.schema)


def create_blackboard_tools(manager: BlackboardManager) -> list[Any]:
    """
    工具工厂函数：根据 BlackboardManager 实例动态生成大模型可调用的读写工具。
    """
    from zhenxun.services.ai.core.exceptions import ToolRetryError
    from zhenxun.services.ai.tools.core.tool import FunctionTool
    from zhenxun.services.ai.tools.models import ToolOptions, ToolResult

    async def read_blackboard() -> ToolResult:
        """读取当前团队的全局共享黑板状态。"""
        content = await manager.read()
        return ToolResult(output=content)

    read_tool = FunctionTool(
        func=read_blackboard,
        name="read_blackboard",
        description="读取当前团队的全局共享黑板最新状态。当你需要获取最新数据或了解现状时请调用此工具。",
    )

    optional_fields = {}
    for field in model_fields(manager.schema):
        desc = (
            getattr(field.field_info, "description", None)
            if hasattr(field, "field_info")
            else None
        )
        optional_fields[field.name] = (
            Optional[field.annotation],
            Field(default=None, description=desc),
        )

    UpdateSchema = create_model(f"Update_{manager.schema.__name__}", **optional_fields)

    async def update_blackboard(**kwargs) -> ToolResult:
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not valid_kwargs:
            return ToolResult(
                output="警告：没有提供任何有效的更新字段，黑板状态未发生改变。"
            )
        try:
            res = await manager.update(**valid_kwargs)
            return ToolResult(output=res)
        except ValueError as e:
            raise ToolRetryError(str(e))

    update_tool = FunctionTool(
        func=update_blackboard,
        name="update_blackboard",
        description="局部更新全局黑板状态。请仅提供你需要修改的字段即可，不需要改变的字段请留空。",
        settings=ToolOptions(args_schema=UpdateSchema),
    )

    return [read_tool, update_tool]
