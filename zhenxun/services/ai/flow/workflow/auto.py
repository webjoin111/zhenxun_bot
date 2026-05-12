from typing import Any

from zhenxun.services.ai.flow.workflow.engine import Workflow
from zhenxun.services.ai.flow.workflow.nodes import Parallel, Router, Step
from zhenxun.services.log import logger


class AutoWorkflow(Workflow):
    """
    自动化声明式工作流 (Facade)。
    允许开发者通过 @entry, @listen 装饰器定义类方法，
    在初始化时，底层编译器会自动分析依赖并推导为原生的 Steps 和 Parallel 节点图。
    """

    def __init__(self, name: str | None = None, description: str = "", **kwargs: Any):
        workflow_name = name or self.__class__.__name__
        compiled_steps = self._compile_graph()

        super().__init__(
            name=workflow_name, steps=compiled_steps, description=description
        )

        self._auto_kwargs = kwargs

    def _compile_graph(self) -> list[Any]:
        """核心图推导编译器：支持 Router 嵌套与 AND/OR 拓扑排序"""
        methods_meta = {}
        router_paths = set()

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if hasattr(attr, "__workflow_meta__"):
                methods_meta[attr_name] = attr.__workflow_meta__
                if attr.__workflow_meta__.get("paths"):
                    router_paths.update(attr.__workflow_meta__["paths"])

        if not methods_meta:
            logger.warning(
                f"AutoWorkflow '{self.__class__.__name__}' 没有检测到任何被装饰的方法！"
            )
            return []

        top_level_methods = {
            k: v
            for k, v in methods_meta.items()
            if not any(t in router_paths for t in v.get("triggers", []))
        }
        branch_methods = {
            k: v
            for k, v in methods_meta.items()
            if any(t in router_paths for t in v.get("triggers", []))
        }

        layers: dict[str, int] = dict.fromkeys(top_level_methods, 0)

        changed = True
        loop_counter = 0
        while changed:
            changed = False
            loop_counter += 1
            if loop_counter > 100:
                raise RecursionError(
                    "AutoWorkflow 编译失败：检测到死循环依赖！请检查 @listen 中是否有环。"
                )

            for name, meta in top_level_methods.items():
                triggers = meta.get("triggers", [])
                if triggers:
                    max_trigger_layer = max(
                        (layers.get(t, -1) for t in triggers if t in layers), default=-1
                    )
                    if max_trigger_layer == -1:
                        continue

                    if max_trigger_layer + 1 > layers[name]:
                        layers[name] = max_trigger_layer + 1
                        changed = True

        layer_groups: dict[int, list[str]] = {}
        for name, layer_idx in layers.items():
            layer_groups.setdefault(layer_idx, []).append(name)

        workflow_steps = []
        for layer_idx in sorted(layer_groups.keys()):
            names = layer_groups[layer_idx]
            step_nodes = []
            for n in names:
                meta = top_level_methods[n]
                if meta["type"] in ("router", "entry_router"):
                    choices = []
                    for path in meta.get("paths", []):
                        branch_name = next(
                            (
                                bk
                                for bk, bv in branch_methods.items()
                                if path in bv.get("triggers", [])
                            ),
                            None,
                        )
                        if branch_name:
                            choices.append(
                                Step(name=path, executor=getattr(self, branch_name))
                            )

                    step_nodes.append(
                        Router(name=n, selector=getattr(self, n), choices=choices)
                    )
                else:
                    step_nodes.append(Step(name=n, executor=getattr(self, n)))

            if len(step_nodes) == 1:
                workflow_steps.append(step_nodes[0])
            else:
                workflow_steps.append(
                    Parallel(*step_nodes, name=f"Layer_{layer_idx}_Parallel")
                )

        return workflow_steps
