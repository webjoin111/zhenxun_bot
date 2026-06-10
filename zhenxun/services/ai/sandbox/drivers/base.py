from abc import ABC, abstractmethod
import time
from typing import Any

from zhenxun.services.ai.sandbox.addons.base import BaseSandboxExtension
from zhenxun.services.ai.sandbox.models import (
    SandboxBlueprint,
    SandboxSessionState,
)
from zhenxun.services.ai.sandbox.protocols import (
    InteractiveTerminalSession,
    SandboxChannel,
)


class BaseSandboxSession(SandboxChannel):
    """沙箱会话接口，持有 Client 分配的具体资源，提供统一的标准操作"""

    def __init__(self, state: SandboxSessionState):
        self.state = state
        self.last_active_time: float = time.time()
        self.loaded_skills: set[str] = set()
        self.installed_packages: set[str] = set()
        self._extensions: dict[str, BaseSandboxExtension] = {}
        self._meta: dict[str, Any] = {}
        self.workspace_path: str = f"/workspace/{state.session_id}"

    @property
    def session_id(self) -> str:
        return self.state.session_id

    def get_meta(self, key: str, default: Any = None) -> Any:
        """获取底层驱动的元数据（如映射端口、Base_URL等）"""
        return self._meta.get(key, default)

    async def mount_extension(self, extension_name: str) -> None:
        """动态挂载一个高级能力扩展到当前通道"""
        from zhenxun.services.ai.sandbox.registry import SandboxRegistry

        extension_cls = SandboxRegistry.get_extension_cls(extension_name)
        if not extension_cls:
            raise ValueError(
                f"Extension '{extension_name}' 未在 SandboxRegistry 注册。"
            )
        extension_instance = extension_cls(self)
        await extension_instance.on_mount()
        self._extensions[extension_name] = extension_instance

    def get_extension(self, extension_name: str) -> BaseSandboxExtension:
        """获取已挂载的高级能力扩展"""
        if extension_name not in self._extensions:
            raise RuntimeError(f"Extension '{extension_name}' 尚未被挂载到此沙箱环境。")
        return self._extensions[extension_name]

    async def apply_blueprint(
        self, blueprint: SandboxBlueprint, base_path: str = "/workspace"
    ) -> None:
        """声明式应用初始化清单"""
        if blueprint.env:
            current_env = self.get_meta("env", {})
            current_env.update(blueprint.env)
            self._meta["env"] = current_env

        if blueprint.entries:
            for rel_path, entry in blueprint.entries.items():
                target_path = f"{base_path}/{rel_path}".replace("//", "/")
                await entry.apply(self, target_path)

    def touch(self) -> None:
        """更新最后活跃时间，防止被 GC 清理"""
        self.last_active_time = time.time()

    async def is_alive(self) -> bool:
        """
        检测当前沙箱驱动是否仍然存活且可用。
        默认返回 True，要求各底层驱动根据自身特性实现具体探活逻辑。
        """
        return True

    async def install_dependencies(self, blueprint: SandboxBlueprint) -> bool:
        """通过注册的环境配置器(Provisioners)热安装依赖包"""
        from zhenxun.services.ai.sandbox.environments import ProvisionerRegistry

        self.touch()
        success = True

        provisioner = ProvisionerRegistry.get("unified_manifest")
        if provisioner:
            res = await provisioner.install(self, blueprint)
            if res:
                self.installed_packages.add("unified_env_installed")
            else:
                success = False

        return success

    @abstractmethod
    async def create_pty_session(self) -> InteractiveTerminalSession:
        """创建一个交互式 PTY 终端会话"""
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def run_process(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        on_output: Any = None,
    ) -> Any:
        pass

    @abstractmethod
    async def read(self, path: str) -> bytes:
        pass

    @abstractmethod
    async def write(self, path: str, data: bytes) -> bool:
        pass

    @abstractmethod
    async def rm(self, path: str, recursive: bool = False) -> bool:
        pass

    @abstractmethod
    async def mkdir(self, path: str, parents: bool = False) -> bool:
        pass

    async def write_raw_file(self, path: str, content: str) -> bool:
        return await self.write(path, content.encode("utf-8"))

    async def read_raw_file(self, path: str) -> str:
        data = await self.read(path)
        return data.decode("utf-8", errors="replace")

    async def delete_raw_file(self, path: str) -> bool:
        return await self.rm(path)

    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool:
        return True


class BaseSandboxClient(ABC):
    """沙箱客户端接口，仅负责与底座通信及生命周期管理"""

    backend_id: str

    @abstractmethod
    async def create(
        self,
        session_id: str,
        blueprint: SandboxBlueprint | None = None,
    ) -> BaseSandboxSession:
        pass

    @abstractmethod
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession:
        pass

    @abstractmethod
    async def delete(self, session: BaseSandboxSession) -> None:
        pass


BaseSandboxDriver = BaseSandboxSession
