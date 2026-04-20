from abc import abstractmethod
import time
from typing import Any

from zhenxun.services.ai.sandbox.extension import (
    BaseSandboxPlugin,
    SandboxChannel,
)
from zhenxun.services.ai.types.sandbox import (
    SandboxRequirements,
    SandboxSecurityProfile,
)
from zhenxun.services.log import logger


class BaseSandboxDriver(SandboxChannel):
    """所有沙箱执行环境的底层驱动接口 (Stateful)"""

    def __init__(self):
        self.session_id: str | None = None
        self.last_active_time: float = time.time()
        self.loaded_skills: set[str] = set()
        self.installed_packages: set[str] = set()
        self._plugins: dict[str, BaseSandboxPlugin] = {}
        self._meta: dict[str, Any] = {}

    @property
    @abstractmethod
    def supports_state(self) -> bool:
        """当前驱动是否支持跨调用的持久化状态保留"""
        pass

    def get_meta(self, key: str, default: Any = None) -> Any:
        """【Channel 协议】获取底层驱动的元数据（如映射端口、Base_URL等）"""
        return self._meta.get(key, default)

    async def mount_plugin(self, plugin_name: str) -> None:
        """动态挂载一个高级能力插件到当前通道"""
        from zhenxun.services.ai.sandbox.extension import SandboxRegistry

        if plugin_name in self._plugins:
            return
        plugin_cls = SandboxRegistry.get_plugin_cls(plugin_name)
        if not plugin_cls:
            raise ValueError(f"Plugin '{plugin_name}' 未在 SandboxRegistry 注册。")
        plugin_instance = plugin_cls(self)
        await plugin_instance.on_mount()
        self._plugins[plugin_name] = plugin_instance

    def get_plugin(self, plugin_name: str) -> BaseSandboxPlugin:
        """获取已挂载的高级能力插件"""
        if plugin_name not in self._plugins:
            raise RuntimeError(f"Plugin '{plugin_name}' 尚未被挂载到此沙箱环境。")
        return self._plugins[plugin_name]

    def touch(self) -> None:
        """更新最后活跃时间，防止被 GC 清理"""
        self.last_active_time = time.time()

    async def is_alive(self) -> bool:
        """
        检测当前沙箱驱动是否仍然存活且可用。
        默认返回 True，要求各底层驱动根据自身特性实现具体探活逻辑。
        """
        return True

    async def install_dependencies(self, requirements: SandboxRequirements) -> bool:
        """通过注册的环境配置器(Provisioners)热安装依赖包"""
        from zhenxun.services.ai.sandbox.provisioner import ProvisionerRegistry

        self.touch()
        success = True

        for prov_name, pkgs in requirements.dependencies.items():
            if not pkgs:
                continue

            provisioner = ProvisionerRegistry.get(prov_name)
            if not provisioner:
                logger.warning(
                    f"[Sandbox] 未找到名为 '{prov_name}' 的环境配置器，跳过安装: {pkgs}"
                )
                continue

            pkgs_to_install = [
                p for p in pkgs if f"{prov_name}:{p}" not in self.installed_packages
            ]
            if not pkgs_to_install:
                continue

            res = await provisioner.install(self, pkgs_to_install)
            if res:
                self.installed_packages.update(
                    f"{prov_name}:{p}" for p in pkgs_to_install
                )
            else:
                success = False

        return success

    @abstractmethod
    async def start(
        self, session_id: str, profile: SandboxSecurityProfile | None = None
    ) -> None:
        """异步初始化并预热沙箱环境"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """异步清理并销毁沙箱环境"""
        pass
