import nonebot

from zhenxun.configs.config import Config
from zhenxun.services.ai.sandbox.models import (
    SandboxBlueprint,
)
from zhenxun.services.ai.utils.lifespan import ResourceLifespanMixin
from zhenxun.services.log import logger

from .drivers.base import BaseSandboxClient, BaseSandboxSession
from .registry import SandboxRegistry

_startup_tasks = set()


def register_sandbox_configs():
    """注册沙箱基础设施专属配置项"""
    Config.add_plugin_config(
        "sandbox",
        "SANDBOX_TYPE",
        "docker",
        help="沙箱底层驱动类型: docker",
        type=str,
    )
    Config.add_plugin_config(
        "sandbox",
        "DOCKER_IMAGE",
        "zhenxun-sandbox:latest",
        help="Docker 沙箱使用的镜像名称 (自定义 Jupyter 增强版)",
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
        self._active_sessions: dict[str, BaseSandboxSession] = {}

    def _get_client(
        self,
        blueprint: SandboxBlueprint,
    ) -> BaseSandboxClient:
        global_type = Config.get_config("sandbox", "SANDBOX_TYPE", "docker")

        effective_type = (
            blueprint.sandbox_type
            if blueprint.sandbox_type and blueprint.sandbox_type != "auto"
            else global_type
        )

        clients = SandboxRegistry.get_all_clients()

        if effective_type not in clients:
            raise RuntimeError(f"未找到指定的沙箱客户端: {effective_type}")

        return clients[effective_type]()

    async def get_or_create_session(
        self,
        session_id: str,
        blueprint: SandboxBlueprint | None = None,
    ) -> BaseSandboxSession:
        """根据 Session ID 获取或创建持久化沙箱环境"""

        if not blueprint:
            blueprint = SandboxBlueprint()

        if (
            blueprint.python_packages
            or blueprint.system_packages
            or blueprint.node_packages
            or blueprint.install_scripts
        ):
            blueprint.enable_network = True

        self.touch(session_id)
        self._ensure_watchdog()

        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            is_alive = await session.is_alive()
            if not is_alive:
                logger.warning(
                    f"⚠️ [SandboxManager] Session '{session_id}' 已失去响应，重建中。"
                )
                await self.close_session(session_id)
            else:
                session.touch()
                if (
                    blueprint.python_packages
                    or blueprint.system_packages
                    or blueprint.node_packages
                    or blueprint.install_scripts
                ):
                    await session.install_dependencies(blueprint)
                await session.apply_blueprint(blueprint)
                for p in blueprint.required_extensions:
                    await session.mount_extension(p)
                return session

        logger.info(f"[SandboxManager] 为 Session '{session_id}' 创建沙箱环境...")
        client = self._get_client(blueprint)
        try:
            session = await client.create(session_id, blueprint)

            await session.apply_blueprint(blueprint)

            if (
                blueprint.python_packages
                or blueprint.system_packages
                or blueprint.node_packages
                or blueprint.install_scripts
            ):
                await session.install_dependencies(blueprint)

            for p in blueprint.required_extensions:
                await session.mount_extension(p)

            self._active_sessions[session_id] = session
            return session
        except Exception as e:
            logger.error(f"[SandboxManager] 创建 Session 失败: {e}")
            raise e

    async def setup_workspace_environment(
        self, session_id: str, workspace_dir: str
    ) -> bool:
        """统一扫描指定工作区，通过 Provisioner 体系完成环境装配"""
        if session_id not in self._active_sessions:
            logger.warning(
                f"[SandboxManager] 找不到活跃的 Session '{session_id}'，无法配置环境。"
            )
            return False

        session = self._active_sessions[session_id]

        from zhenxun.services.ai.sandbox.environments import ProvisionerRegistry

        for prov in ProvisionerRegistry.get_all().values():
            await prov.scan_and_setup_workspace(session, workspace_dir)
        return True

    async def release_resource(self, resource_id: str):
        if resource_id in self._active_sessions:
            session = self._active_sessions.pop(resource_id)
            try:
                client_cls = SandboxRegistry.get_client_cls(session.state.sandbox_type)
                client = client_cls()
                await client.delete(session)
            except Exception as e:
                logger.error(
                    f"[SandboxManager] 销毁沙箱环境失败 (Session: {resource_id}): {e}"
                )

    async def close_session(self, session_id: str) -> None:
        async with self._lifespan_lock:
            await self.release_resource(session_id)
            self._last_active_times.pop(session_id, None)

    async def shutdown_all(self) -> None:
        keys = list(self._active_sessions.keys())
        for sid in keys:
            await self.close_session(sid)
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()


sandbox_manager = _SandboxManager()


driver = nonebot.get_driver()


@driver.on_startup
async def _startup_sandboxes():
    from zhenxun.services.ai.sandbox.host_bridge import sandbox_rpc_server

    await sandbox_rpc_server.start()

    clients = SandboxRegistry.get_all_clients()
    if "docker" in clients:
        import asyncio

        from .drivers.docker import DockerSandboxClient

        async def _async_init_docker_sandbox():
            try:
                is_alive = await asyncio.wait_for(
                    DockerSandboxClient.check_engine_alive(), timeout=5.0
                )
            except asyncio.TimeoutError:
                is_alive = False
                logger.warning(
                    "[SandboxManager] Docker 引擎探活超时(5s)，"
                    "可能处于假死状态，已自动禁用本地路由。"
                )
            except Exception:
                is_alive = False

            DockerSandboxClient.set_engine_status(is_alive)

            if is_alive:
                await DockerSandboxClient.prune_orphan_containers()
            else:
                logger.debug(
                    "[SandboxManager] 未检测到可用本地 Docker 引擎，"
                    "Docker 沙箱路由已自动禁用。"
                )

        task = asyncio.create_task(_async_init_docker_sandbox())
        _startup_tasks.add(task)
        task.add_done_callback(_startup_tasks.discard)


@driver.on_shutdown
async def _shutdown_sandboxes():
    from zhenxun.services.ai.sandbox.host_bridge import sandbox_rpc_server

    await sandbox_rpc_server.stop()

    await sandbox_manager.shutdown_all()
    clients = SandboxRegistry.get_all_clients()
    if "docker" in clients and getattr(clients["docker"], "_engine_available", False):
        from .drivers.docker import DockerSandboxClient

        await DockerSandboxClient.close_env()
