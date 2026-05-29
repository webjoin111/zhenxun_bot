from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxSession
    from zhenxun.services.ai.sandbox.models import SandboxBlueprint


class BaseProvisioner(ABC):
    """环境配置器基类，负责在沙箱内热安装依赖"""

    @property
    @abstractmethod
    def name(self) -> str:
        """配置器名称"""
        pass

    @abstractmethod
    async def install(self, session: "BaseSandboxSession", blueprint: Any) -> bool:
        """执行安装逻辑，成功返回 True"""
        pass

    @abstractmethod
    async def scan_and_setup_workspace(
        self, session: "BaseSandboxSession", workspace_dir: str
    ) -> bool:
        """扫描指定工作区（检测文件如 requirements.txt）并自动配置环境"""
        pass


class ProvisionerRegistry:
    """环境配置器注册中心"""

    _provisioners: ClassVar[dict[str, BaseProvisioner]] = {}

    @classmethod
    def register(cls, provisioner: BaseProvisioner) -> None:
        if provisioner.name in cls._provisioners:
            logger.warning(
                f"[ProvisionerRegistry] 覆盖已存在的配置器: {provisioner.name}"
            )
        cls._provisioners[provisioner.name] = provisioner
        logger.debug(f"[ProvisionerRegistry] 成功注册环境配置器: {provisioner.name}")

    @classmethod
    def get(cls, name: str) -> BaseProvisioner | None:
        return cls._provisioners.get(name)

    @classmethod
    def get_all(cls) -> dict[str, BaseProvisioner]:
        return cls._provisioners.copy()


class UnifiedManifestProvisioner(BaseProvisioner):
    """
    统一环境清单装配器。
    单向执行：系统依赖(apt) -> Python依赖(uv) -> 二进制检查(bins) -> 自定义脚本。
    """

    @property
    def name(self) -> str:
        return "unified_manifest"

    async def install(self, session: "BaseSandboxSession", blueprint: "SandboxBlueprint") -> bool:
        
        target_hash = blueprint.calculate_hash()
        hash_check = await session.exec("cat /workspace/.zx_env_hash")
        if hash_check.exit_code == 0 and hash_check.stdout.strip() == target_hash:
            logger.info(
                f"[Sandbox] ⚡ 环境指纹 ({target_hash[:8]}) "
                "匹配成功，跳过所有依赖安装环节！"
            )
            return True

        if blueprint.system_packages:
            pkg_str = " ".join(blueprint.system_packages)
            logger.info(
                f"[Sandbox] 正在为 '{session.session_id}' 安装系统级依赖: {pkg_str}"
            )
            await session.exec(
                "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                f"{pkg_str}",
                timeout=300,
            )

        if blueprint.python_packages:
            pkg_str = " ".join(blueprint.python_packages)

            check_uv = await session.exec("command -v uv")
            if check_uv.exit_code != 0:
                logger.info(
                    f"[Sandbox] '{session.session_id}' 未检测到 uv，正在极速下载 uv..."
                )
                await session.exec(
                    "pip install uv --disable-pip-version-check -q", timeout=60
                )

            logger.info(
                f"[Sandbox] 正在为 '{session.session_id}' 极速安装 "
                f"Python 依赖: {pkg_str}"
            )
            res = await session.exec(
                f"uv pip install --system {pkg_str}", timeout=120
            )

            if res.exit_code != 0:
                logger.warning(
                    "[Sandbox] uv 安装报错，可能遇到不兼容的原生包，"
                    f"尝试降级使用 pip: {res.stderr}"
                )
                await session.exec(
                    f"pip install {pkg_str} -q", timeout=180
                )

        if blueprint.node_packages:
            pkg_str = " ".join(blueprint.node_packages)
            logger.info(f"[Sandbox] 正在为 '{session.session_id}' 安装 Node 依赖: {pkg_str}")
            await session.exec(f"npm install -g {pkg_str}", timeout=180)

        if blueprint.install_scripts:
            for script in blueprint.install_scripts:
                logger.info(f"[Sandbox] 正在执行自定义装配脚本: {script}")
                await session.exec(script, timeout=300)

        await session.exec(
            f"echo '{target_hash}' > /workspace/.zx_env_hash"
        )
        logger.debug(f"[Sandbox] 环境装配完毕，已写入指纹快照: {target_hash[:8]}")

        return True

    async def scan_and_setup_workspace(
        self, session: "BaseSandboxSession", workspace_dir: str
    ) -> bool:

        check = await session.exec(
            f"test -f {workspace_dir}/requirements.txt"
        )
        if check.exit_code == 0:
            check_uv = await session.exec("command -v uv")
            if check_uv.exit_code == 0:
                await session.exec(
                    "uv pip install --system -r requirements.txt",
                    cwd=workspace_dir,
                    timeout=300,
                )
            else:
                await session.exec(
                    "pip install -q -r requirements.txt",
                    cwd=workspace_dir,
                    timeout=300,
                )

        check_npm = await session.exec(
            f"test -f {workspace_dir}/package.json"
        )
        if check_npm.exit_code == 0:
            logger.info(
                "[UnifiedProvisioner] 发现 package.json，正在安装 npm 依赖..."
            )
            await session.exec(
                "npm install --no-fund --no-audit --loglevel=error",
                cwd=workspace_dir,
                timeout=300,
            )
        return True


ProvisionerRegistry.register(UnifiedManifestProvisioner())
