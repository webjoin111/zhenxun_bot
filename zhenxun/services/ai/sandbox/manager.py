import nonebot

from zhenxun.configs.config import Config
from zhenxun.services.ai.sandbox.models import (
    SandboxRequirements,
    SandboxSecurityProfile,
    SandboxTier,
)
from zhenxun.services.ai.utils.lifespan import ResourceLifespanMixin
from zhenxun.services.log import logger

from .drivers.base import BaseSandboxDriver
from .extension import SandboxRegistry


def register_sandbox_configs():
    """注册沙箱基础设施专属配置项"""
    Config.add_plugin_config(
        "sandbox",
        "SANDBOX_TYPE",
        "auto",
        help="沙箱底层驱动类型: auto (自动路由), docker, wasm (极速轻量)",
        type=str,
    )
    Config.add_plugin_config(
        "sandbox",
        "DOCKER_IMAGE",
        "zhenxun-sandbox:latest",
        help="Docker 沙箱使用的镜像名称 (自定义 Jupyter 增强版)",
        type=str,
    )
    Config.add_plugin_config(
        "sandbox",
        "WASM_IMAGE",
        "python",
        help="Wasmtime 沙箱使用的 Python 镜像名称 (不带后缀，默认 python，需置于 data/ai/sandbox/ 目录下)",
        type=str,
    )
    logger.info("沙箱(Sandbox) 基础设施配置项注册完成")


class _SandboxManager(ResourceLifespanMixin):
    """
    沙箱底层环境的全局调度中心。
    负责读取用户配置，并动态分发给对应的安全 Driver 执行。
    """

    def __init__(self):
        self.init_lifespan(ttl=1800)
        self._active_sandboxes: dict[str, BaseSandboxDriver] = {}

    def _create_driver(
        self,
        session_id: str,
        profile: SandboxSecurityProfile,
        requirements: SandboxRequirements | None,
    ) -> BaseSandboxDriver:
        global_type = Config.get_config("sandbox", "SANDBOX_TYPE", "auto")

        effective_type = (
            profile.sandbox_type if profile.sandbox_type != "auto" else global_type
        )

        providers = SandboxRegistry.get_all_providers()

        if effective_type != "auto":
            if effective_type not in providers:
                raise RuntimeError(f"未找到指定的沙箱提供者: {effective_type}")
            provider = providers[effective_type]
            if not provider.is_available():
                raise RuntimeError(
                    f"指定的沙箱提供者 {effective_type} 当前环境不可用(检查API Key或依赖包)"
                )
            logger.info(f"[SandboxManager] 精确路由：选择了 {effective_type} 提供者")
            return provider.create_driver(session_id)

        candidates = []
        for name, provider in providers.items():
            if not provider.is_available():
                continue
            score = provider.score(profile, requirements)
            if score >= 0:
                candidates.append((score, provider))

        if not candidates:
            logger.error(
                "⚠️ [SandboxManager] 无法满足沙箱意图！所有已注册的提供者均不可用或无法支持当前任务需求。"
            )
            raise RuntimeError(
                "未找到满足环境意图的安全沙箱环境。请检查是否已正确配置 API Key 或本地 Docker 环境。"
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_provider = candidates[0]

        logger.info(
            f"[SandboxManager] 智能路由策略生效：选择了 [{best_provider.get_name()}] (匹配得分: {best_score})"
        )
        return best_provider.create_driver(session_id)

    async def get_or_create_session(
        self,
        session_id: str,
        profile: SandboxSecurityProfile | None = None,
        requirements: SandboxRequirements | None = None,
    ) -> BaseSandboxDriver:
        """根据 Session ID 获取或创建持久化沙箱环境"""

        if not profile:
            profile = SandboxSecurityProfile()

        implied_tier = requirements.tier if requirements else SandboxTier.LIGHTWEIGHT
        if profile.sandbox_type == "auto":
            profile.needs_state = (
                profile.needs_state
                or getattr(profile, "keep_state", False)
                or implied_tier != SandboxTier.LIGHTWEIGHT
            )

        if requirements and (
            requirements.env_setup.python_packages
            or requirements.env_setup.system_packages
            or requirements.env_setup.install_scripts
        ):
            profile.enable_network = True

        plugins_to_mount = set()
        if profile and profile.required_plugins:
            plugins_to_mount.update(profile.required_plugins)
        if requirements and requirements.required_plugins:
            plugins_to_mount.update(requirements.required_plugins)

        self.touch(session_id)
        self._ensure_watchdog()

        if session_id in self._active_sandboxes:
            driver = self._active_sandboxes[session_id]

            is_alive = await driver.is_alive()
            if not is_alive:
                logger.warning(
                    f"⚠️ [SandboxManager] 检测到 Session '{session_id}' 的底层沙箱已失去响应，正在清理遗留资源并重建。"
                )
                await self.close_session(session_id)
            elif not driver.supports_state and profile.needs_state:
                logger.info(
                    f"🚀 [SandboxManager] 智能路由触发沙箱升维: Session '{session_id}' 需要持久化支持，丢弃旧无状态沙箱。"
                )
                await self.close_session(session_id)
            else:
                driver.touch()
                if requirements and (
                    requirements.env_setup.python_packages
                    or requirements.env_setup.system_packages
                    or requirements.env_setup.install_scripts
                ):
                    await driver.install_dependencies(requirements)
                for p in plugins_to_mount:
                    await driver.mount_plugin(p)
                return driver

        logger.info(
            f"[SandboxManager] 为 Session '{session_id}' 创建沙箱环境"
            f" (意图 needs_state={profile.needs_state})..."
        )
        driver = self._create_driver(session_id, profile, requirements)
        try:
            await driver.start(session_id, profile)
        except Exception as e:
            await driver.close()
            raise e

        if requirements and (
            requirements.env_setup.python_packages
            or requirements.env_setup.system_packages
            or requirements.env_setup.install_scripts
        ):
            await driver.install_dependencies(requirements)

        for p in plugins_to_mount:
            await driver.mount_plugin(p)

        self._active_sandboxes[session_id] = driver

        return driver

    async def setup_workspace_environment(
        self, session_id: str, workspace_dir: str
    ) -> bool:
        """统一扫描指定工作区，通过 Provisioner 体系完成环境装配"""
        if session_id not in self._active_sandboxes:
            logger.warning(
                f"[SandboxManager] 找不到活跃的 Session '{session_id}'，无法配置环境。"
            )
            return False

        driver = self._active_sandboxes[session_id]

        from zhenxun.services.ai.sandbox.provisioner import ProvisionerRegistry

        for prov in ProvisionerRegistry.get_all().values():
            await prov.scan_and_setup_workspace(driver, workspace_dir)
        return True

    async def release_resource(self, resource_id: str):
        if resource_id in self._active_sandboxes:
            driver = self._active_sandboxes.pop(resource_id)
            try:
                await driver.close()
            except Exception as e:
                logger.error(
                    f"[SandboxManager] 销毁沙箱环境失败 (Session: {resource_id}): {e}"
                )

    async def close_session(self, session_id: str) -> None:
        async with self._lifespan_lock:
            await self.release_resource(session_id)
            self._last_active_times.pop(session_id, None)

    async def shutdown_all(self) -> None:
        keys = list(self._active_sandboxes.keys())
        for sid in keys:
            await self.close_session(sid)
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()


sandbox_manager = _SandboxManager()


driver = nonebot.get_driver()


@driver.on_startup
async def _startup_sandboxes():
    providers = SandboxRegistry.get_all_providers()
    if "docker" in providers:
        import asyncio

        from .drivers.docker import DockerDriver

        async def _async_init_docker_sandbox():
            docker_provider = providers["docker"]
            try:
                is_alive = await asyncio.wait_for(
                    DockerDriver.check_engine_alive(), timeout=5.0
                )
            except asyncio.TimeoutError:
                is_alive = False
                logger.warning(
                    "[SandboxManager] Docker 引擎探活超时(5s)，可能处于假死状态，已自动禁用本地路由。"
                )
            except Exception:
                is_alive = False

            if hasattr(docker_provider, "set_engine_status"):
                getattr(docker_provider, "set_engine_status")(is_alive)

            if is_alive:
                await DockerDriver.prune_orphan_containers()
            else:
                logger.debug(
                    "[SandboxManager] 未检测到可用本地 Docker 引擎，Docker 沙箱路由已自动禁用。"
                )

        asyncio.create_task(_async_init_docker_sandbox())


@driver.on_shutdown
async def _shutdown_sandboxes():
    await sandbox_manager.shutdown_all()
    providers = SandboxRegistry.get_all_providers()
    if "docker" in providers and providers["docker"].is_available():
        from .drivers.docker import DockerDriver

        await DockerDriver.close_env()
