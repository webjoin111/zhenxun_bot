from collections.abc import Callable
from typing import Any


def AND(*triggers: str) -> dict[str, Any]:
    """逻辑与：所有前置方法都完成才执行"""
    return {"logic": "AND", "triggers": list(triggers)}


def OR(*triggers: str) -> dict[str, Any]:
    """逻辑或：任意前置方法完成即执行"""
    return {"logic": "OR", "triggers": list(triggers)}


def entry() -> Callable:
    """
    标记为工作流的入口节点。
    执行工作流时会自动作为第一批任务执行。
    """

    def decorator(func: Callable) -> Callable:
        setattr(
            func,
            "__workflow_meta__",
            {
                "type": "entry",
                "triggers": [],
                "logic": "OR",
                "paths": [],
            },
        )
        return func

    return decorator


def listen(condition: str | dict[str, Any]) -> Callable:
    """
    监听其他节点的完成状态。

    用法:
        @listen("step_a")
        @listen(AND("step_a", "step_b"))
    """

    def decorator(func: Callable) -> Callable:
        if isinstance(condition, str):
            meta = {
                "type": "listen",
                "triggers": [condition],
                "logic": "OR",
                "paths": [],
            }
        elif isinstance(condition, dict):
            meta = {
                "type": "listen",
                "triggers": condition["triggers"],
                "logic": condition.get("logic", "OR"),
                "paths": [],
            }
        else:
            raise TypeError("listen condition 必须是字符串或 AND/OR 函数的返回值")

        setattr(func, "__workflow_meta__", meta)
        return func

    return decorator


def router(
    condition: str | dict[str, Any] | None = None, paths: list[str] | None = None
) -> Callable:
    """
    标记为路由节点。执行此方法后，会根据返回值走向对应的 paths。
    """

    def decorator(func: Callable) -> Callable:
        meta = {"type": "router", "triggers": [], "logic": "OR", "paths": paths or []}
        if isinstance(condition, str):
            meta["triggers"] = [condition]
        elif isinstance(condition, dict):
            meta["triggers"] = condition["triggers"]
            meta["logic"] = condition.get("logic", "OR")

        if not condition:
            meta["type"] = "entry_router"

        setattr(func, "__workflow_meta__", meta)
        return func

    return decorator
