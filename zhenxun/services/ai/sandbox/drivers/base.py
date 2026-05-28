from abc import abstractmethod
import time
from typing import Any

from zhenxun.services.ai.sandbox.extension import (
    BaseSandboxExtension,
    SandboxChannel,
)
from zhenxun.services.ai.sandbox.models import (
    Manifest,
    SandboxCapabilities,
    SandboxRequirements,
    SandboxSecurityProfile,
)


class BaseSandboxDriver(SandboxChannel):
    """所有沙箱执行环境的底层驱动接口 (Stateful)"""

    def __init__(self):
        self.session_id: str | None = None
        self.last_active_time: float = time.time()
        self.loaded_skills: set[str] = set()
        self.installed_packages: set[str] = set()
        self._extensions: dict[str, BaseSandboxExtension] = {}
        self._meta: dict[str, Any] = {}

    @property
    @abstractmethod
    def supports_state(self) -> bool:
        """当前驱动是否支持跨调用的持久化状态保留"""
        pass

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> SandboxCapabilities:
        """声明该驱动的沙箱能力边界"""
        pass

    def get_meta(self, key: str, default: Any = None) -> Any:
        """获取底层驱动的元数据（如映射端口、Base_URL等）"""
        return self._meta.get(key, default)

    async def mount_extension(self, extension_name: str) -> None:
        """动态挂载一个高级能力扩展到当前通道"""
        from zhenxun.services.ai.sandbox.extension import SandboxRegistry

        if extension_name in self._extensions:
            return
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

        provisioner = ProvisionerRegistry.get("unified_manifest")
        if provisioner:
            res = await provisioner.install(self, requirements.env_setup)
            if res:
                self.installed_packages.add("unified_env_installed")
            else:
                success = False

        return success

    async def apply_manifest(
        self, manifest: Manifest, base_path: str = "/workspace"
    ) -> None:
        """声明式应用初始化清单"""
        import asyncio
        from typing import cast

        from zhenxun.services.ai.sandbox.extension import (
            SupportsCommandExecution,
            SupportsFileSystem,
        )
        from zhenxun.services.ai.sandbox.models import (
            DirEntry,
            FileEntry,
            GitRepoEntry,
            LocalFileEntry,
        )
        from zhenxun.services.log import logger

        if manifest.environment:
            current_env = self.get_meta("env", {})
            current_env.update(manifest.environment)
            self._meta["env"] = current_env

        if not manifest.entries:
            return

        if not isinstance(self, SupportsFileSystem):
            logger.warning(
                f"沙箱 {self.session_id} 不支持文件系统，无法应用 Manifest。"
            )
            return

        fs = cast(SupportsFileSystem, self)
        cmd = (
            cast(SupportsCommandExecution, self)
            if isinstance(self, SupportsCommandExecution)
            else None
        )

        async def _process_entries(entries: dict[str, Any], current_path: str):
            for name, entry in entries.items():
                target_path = f"{current_path}/{name}".replace("//", "/")
                if isinstance(entry, FileEntry):
                    content = (
                        entry.content.decode("utf-8")
                        if isinstance(entry.content, bytes)
                        else entry.content
                    )
                    await fs.write_raw_file(target_path, content)
                elif isinstance(entry, LocalFileEntry):
                    from pathlib import Path

                    local_path = Path(entry.src_path)
                    if await asyncio.to_thread(
                        local_path.exists
                    ) and await asyncio.to_thread(local_path.is_file):
                        content = await asyncio.to_thread(
                            local_path.read_text, encoding="utf-8", errors="ignore"
                        )
                        await fs.write_raw_file(target_path, content)
                elif isinstance(entry, DirEntry):
                    if cmd:
                        await cmd.execute_raw_command(f"mkdir -p {target_path}")
                    if entry.children:
                        await _process_entries(entry.children, target_path)
                elif isinstance(entry, GitRepoEntry):
                    if cmd:
                        command = f"git clone {entry.url} {target_path}"
                        if entry.ref:
                            command += (
                                f" && cd {target_path} && git checkout {entry.ref}"
                            )
                        await cmd.execute_raw_command(command)

        await _process_entries(manifest.entries, base_path)

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
