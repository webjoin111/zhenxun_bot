from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zhenxun.services.ai.flow.team.strategy import BaseTeamStrategy

from zhenxun.services.log import logger
from zhenxun.utils.utils import infer_plugin_namespace


class TeamStrategyRegistry:
    """
    多智能体协作策略注册中心。
    支持基于插件命名空间的隔离，防止不同插件之间的策略名称冲突。
    """

    _registry: dict[str, dict[str, type["BaseTeamStrategy"]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_cls: type["BaseTeamStrategy"],
        namespace: str | None = None,
    ) -> None:
        """
        注册一个团队策略。
        如果未显式提供 namespace，将自动推导当前调用者所在的插件命名空间。
        """
        ns = namespace or infer_plugin_namespace()
        if ns not in cls._registry:
            cls._registry[ns] = {}

        cls._registry[ns][name] = strategy_cls
        logger.debug(f"✅ 已注册 Team 协作策略: {ns}.{name}")

    @classmethod
    def get(
        cls, name: str, default_namespace: str | None = None
    ) -> type["BaseTeamStrategy"] | None:
        """
        获取策略类。
        解析顺序:
        1. 显式指定命名空间 (如 'plugin_a.debate') -> 精确匹配
        2. 未指定 -> 查 default_namespace -> 查 'global' -> 查 'builtin'
        """
        if "." in name:
            ns, s_name = name.split(".", 1)
            return cls._registry.get(ns, {}).get(s_name)

        ns = default_namespace or infer_plugin_namespace()

        if ns in cls._registry and name in cls._registry[ns]:
            return cls._registry[ns][name]

        if "global" in cls._registry and name in cls._registry["global"]:
            return cls._registry["global"][name]

        if "builtin" in cls._registry and name in cls._registry["builtin"]:
            return cls._registry["builtin"][name]

        return None


def team_strategy(name: str, namespace: str | None = None):
    """
    Team 协作策略注册装饰器。

    用法:
        @team_strategy("debate")
        class DebateStrategy(BaseTeamStrategy): ...
    """

    def decorator(cls):
        TeamStrategyRegistry.register(name, cls, namespace)
        return cls

    return decorator
