from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

from zhenxun.services.log import logger
from zhenxun.services.ai.sandbox.extension import SupportsCommandExecution

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxDriver


class BaseProvisioner(ABC):
    """环境配置器基类，负责在沙箱内热安装依赖"""

    @property
    @abstractmethod
    def name(self) -> str:
        """配置器名称，例如 'python_pip', 'node_npm'"""
        pass

    @abstractmethod
    async def install(self, driver: "BaseSandboxDriver", packages: list[str]) -> bool:
        """执行安装逻辑，成功返回 True，失败返回 False"""
        pass

    @abstractmethod
    async def scan_and_setup_workspace(
        self, driver: "BaseSandboxDriver", workspace_dir: str
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


class PipProvisioner(BaseProvisioner):
    @property
    def name(self) -> str:
        return "python_pip"

    async def install(self, driver, packages: list[str]) -> bool:
        pkg_str = " ".join(packages)
        logger.info(
            f"[Sandbox] 正在为 Session '{driver.session_id}' 热安装 Python 依赖: {pkg_str}"
        )
        cmd_driver = cast(SupportsCommandExecution, driver)
        res = await cmd_driver.execute_raw_command(
            f"python3 -m pip install -q --disable-pip-version-check --no-warn-script-location {pkg_str} || true",
            timeout=60,
        )
        if res.stderr and "ERROR:" in res.stderr:
            logger.debug(
                f"[Sandbox] 依赖安装可能部分失败 (通常是本地模块引起的假阳性): {res.stderr.strip()}"
            )
        return True

    async def scan_and_setup_workspace(
        self, driver: "BaseSandboxDriver", workspace_dir: str
    ) -> bool:
        if not isinstance(driver, SupportsCommandExecution):
            return False
        cmd_driver = cast(SupportsCommandExecution, driver)

        check = await cmd_driver.execute_raw_command(f"test -f {workspace_dir}/requirements.txt")
        if check.exit_code == 0:
            logger.info(
                f"[PipProvisioner] 发现 requirements.txt，正在为 Session '{driver.session_id}' 安装依赖..."
            )
            res = await cmd_driver.execute_raw_command(
                "pip install -q --disable-pip-version-check --no-warn-script-location -r requirements.txt || true",
                cwd=workspace_dir,
                timeout=300,
            )
            if res.exit_code == 0:
                logger.info("[PipProvisioner] 已自动完成原生 pip install -r")
        return True


class NpmProvisioner(BaseProvisioner):
    @property
    def name(self) -> str:
        return "node_npm"

    async def install(self, driver, packages: list[str]) -> bool:
        pkg_str = " ".join(packages)
        logger.info(
            f"[Sandbox] 正在为 Session '{driver.session_id}' 热安装 Node 依赖: {pkg_str}"
        )
        cmd_driver = cast(SupportsCommandExecution, driver)
        res = await cmd_driver.execute_raw_command(
            f"npm install --no-fund --no-audit --loglevel=error {pkg_str}",
            timeout=180,
        )
        if res.exit_code != 0:
            logger.error(f"[Sandbox] Node 依赖安装失败: {res.stderr or res.stdout}")
            return False
        return True

    async def scan_and_setup_workspace(
        self, driver: "BaseSandboxDriver", workspace_dir: str
    ) -> bool:
        if not isinstance(driver, SupportsCommandExecution):
            return False
        cmd_driver = cast(SupportsCommandExecution, driver)
            
        check = await cmd_driver.execute_raw_command(f"test -f {workspace_dir}/package.json")
        if check.exit_code == 0:
            logger.info(
                f"[NpmProvisioner] 发现 package.json，正在为 Session '{driver.session_id}' 安装依赖..."
            )
            res = await cmd_driver.execute_raw_command(
                "npm install --no-fund --no-audit --loglevel=error",
                cwd=workspace_dir,
                timeout=300,
            )
            if res.exit_code == 0:
                logger.info("[NpmProvisioner] 已自动完成原生 npm install")
        return True


class AptProvisioner(BaseProvisioner):
    @property
    def name(self) -> str:
        return "system_apt"

    async def install(self, driver, packages: list[str]) -> bool:
        pkg_str = " ".join(packages)
        logger.info(
            f"[Sandbox] 正在为 Session '{driver.session_id}' 热安装系统依赖: {pkg_str}"
        )
        cmd_driver = cast(SupportsCommandExecution, driver)
        res = await cmd_driver.execute_raw_command(
            f"sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg_str}",
            timeout=300,
        )
        if res.exit_code != 0:
            logger.error(f"[Sandbox] 系统依赖安装失败: {res.stderr or res.stdout}")
            return False
        return True

    async def scan_and_setup_workspace(
        self, driver: "BaseSandboxDriver", workspace_dir: str
    ) -> bool:
        return True


ProvisionerRegistry.register(PipProvisioner())
ProvisionerRegistry.register(NpmProvisioner())
ProvisionerRegistry.register(AptProvisioner())
