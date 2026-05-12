from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

from zhenxun.services.ai.sandbox.extension import SupportsCommandExecution
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxDriver
    from zhenxun.services.ai.sandbox.models import EnvSetupConfig


class BaseProvisioner(ABC):
    """环境配置器基类，负责在沙箱内热安装依赖"""

    @property
    @abstractmethod
    def name(self) -> str:
        """配置器名称"""
        pass

    @abstractmethod
    async def install(self, driver: "BaseSandboxDriver", payload: Any) -> bool:
        """执行安装逻辑，成功返回 True"""
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


class UnifiedManifestProvisioner(BaseProvisioner):
    """
    统一环境清单装配器。
    单向执行：系统依赖(apt) -> Python依赖(uv) -> 二进制检查(bins) -> 自定义脚本。
    """

    @property
    def name(self) -> str:
        return "unified_manifest"

    async def install(self, driver: "BaseSandboxDriver", payload: Any) -> bool:
        from zhenxun.services.ai.sandbox.models import EnvSetupConfig
        env_setup = cast(EnvSetupConfig, payload)
        cmd_driver = cast(SupportsCommandExecution, driver)

        # 0. 检查环境 Hash 缓存，实现极速跳过
        target_hash = env_setup.calculate_hash()
        hash_check = await cmd_driver.execute_raw_command("cat /workspace/.zx_env_hash")
        if hash_check.exit_code == 0 and hash_check.stdout.strip() == target_hash:
            logger.info(f"[Sandbox] ⚡ 环境指纹 ({target_hash[:8]}) 匹配成功，跳过所有依赖安装环节！")
            return True

        if env_setup.system_packages:
            pkg_str = " ".join(env_setup.system_packages)
            logger.info(
                f"[Sandbox] 正在为 '{driver.session_id}' 安装系统级依赖: {pkg_str}"
            )
            await cmd_driver.execute_raw_command(
                f"sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {pkg_str}",
                timeout=300,
            )

        if env_setup.python_packages:
            pkg_str = " ".join(env_setup.python_packages)

            check_uv = await cmd_driver.execute_raw_command("command -v uv")
            if check_uv.exit_code != 0:
                logger.info(
                    f"[Sandbox] '{driver.session_id}' 未检测到 uv，正在极速下载 uv..."
                )
                await cmd_driver.execute_raw_command(
                    "pip install uv --disable-pip-version-check -q", timeout=60
                )

            logger.info(
                f"[Sandbox] 正在为 '{driver.session_id}' 极速安装 Python 依赖: {pkg_str}"
            )
            res = await cmd_driver.execute_raw_command(
                f"uv pip install --system {pkg_str}", timeout=120
            )

            if res.exit_code != 0:
                logger.warning(
                    f"[Sandbox] uv 安装报错，可能遇到不兼容的原生包，尝试降级使用 pip: {res.stderr}"
                )
                await cmd_driver.execute_raw_command(
                    f"pip install {pkg_str} -q", timeout=180
                )

        if env_setup.bins:
            for binary in env_setup.bins:
                res = await cmd_driver.execute_raw_command(f"command -v {binary}")
                if res.exit_code != 0:
                    logger.warning(
                        f"⚠️ [Sandbox] 警告：沙箱 '{driver.session_id}' 缺失声明的前置命令: {binary}"
                    )

        if env_setup.install_scripts:
            for script in env_setup.install_scripts:
                logger.info(f"[Sandbox] 正在执行自定义装配脚本: {script}")
                await cmd_driver.execute_raw_command(script, timeout=300)

        # 5. 写入最新环境指纹
        await cmd_driver.execute_raw_command(f"echo '{target_hash}' > /workspace/.zx_env_hash")
        logger.debug(f"[Sandbox] 环境装配完毕，已写入指纹快照: {target_hash[:8]}")

        return True

    async def scan_and_setup_workspace(
        self, driver: "BaseSandboxDriver", workspace_dir: str
    ) -> bool:
        if not isinstance(driver, SupportsCommandExecution):
            return False
        cmd_driver = cast(SupportsCommandExecution, driver)

        check = await cmd_driver.execute_raw_command(
            f"test -f {workspace_dir}/requirements.txt"
        )
        if check.exit_code == 0:
            check_uv = await cmd_driver.execute_raw_command("command -v uv")
            if check_uv.exit_code == 0:
                await cmd_driver.execute_raw_command(
                    "uv pip install --system -r requirements.txt",
                    cwd=workspace_dir,
                    timeout=300,
                )
            else:
                await cmd_driver.execute_raw_command(
                    "pip install -q -r requirements.txt", cwd=workspace_dir, timeout=300
                )
        
        # 统一接管原 NpmProvisioner 的工作区逻辑
        check_npm = await cmd_driver.execute_raw_command(f"test -f {workspace_dir}/package.json")
        if check_npm.exit_code == 0:
            logger.info(f"[UnifiedProvisioner] 发现 package.json，正在安装 npm 依赖...")
            await cmd_driver.execute_raw_command(
                "npm install --no-fund --no-audit --loglevel=error", cwd=workspace_dir, timeout=300
            )
        return True


ProvisionerRegistry.register(UnifiedManifestProvisioner())
