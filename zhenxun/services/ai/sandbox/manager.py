from typing import Any, cast

import nonebot

from zhenxun.configs.config import Config
from zhenxun.services.ai.sandbox.models import (
    SandboxBlueprint,
)
from zhenxun.services.log import logger
from zhenxun.utils.lifespan import LifespanManager

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
    Config.add_plugin_config(
        "sandbox",
        "CLEANUP_TIMEOUT",
        1800,
        help="沙箱自动清理的闲置超时时间(秒)。0表示关闭，不自动清理",
        type=int,
    )
    Config.add_plugin_config(
        "sandbox",
        "ENABLE_VFS_HELPER",
        True,
        help="是否开启 VFS 路径逃逸防范探针，默认开启。遇到兼容性问题时可关闭",
        type=bool,
    )


class SandboxManager:
    """
    沙箱底层环境的全局调度中心。
    负责读取用户配置，并动态分发给对应的安全 Driver 执行。
    """

    def __init__(self):
        self._active_sessions: dict[str, BaseSandboxSession] = {}
        self.lifespan_manager = LifespanManager()

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

        return cast(BaseSandboxClient, clients[effective_type]())

    async def get_or_create_session(
        self,
        session_id: str,
        blueprint: SandboxBlueprint | None = None,
    ) -> BaseSandboxSession:
        """根据 Session ID 获取或创建持久化沙箱环境"""

        if not blueprint:
            blueprint = SandboxBlueprint()

        if blueprint.setup_steps:
            blueprint.enable_network = True

        cleanup_timeout = Config.get_config("sandbox", "CLEANUP_TIMEOUT", 1800)
        await self.lifespan_manager.register(
            session_id, ttl=float(cleanup_timeout), cleanup_callback=self.close_session
        )

        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            is_alive = await session.is_alive()
            if not is_alive:
                logger.warning(
                    f"⚠️ Session '{session_id}' 已失去响应，重建中。",
                    command="SandboxManager",
                )
                await self.close_session(session_id)
            else:
                session.touch()
                if blueprint.setup_steps:
                    await session.install_dependencies(blueprint)
                await session.apply_blueprint(blueprint)
                for p in blueprint.required_extensions:
                    await session.mount_extension(p)
                return session

        logger.info(
            f"为 Session '{session_id}' 创建沙箱环境...", command="SandboxManager"
        )
        client = self._get_client(blueprint)
        try:
            session = await client.create(session_id, blueprint)

            await session.apply_blueprint(blueprint)

            if blueprint.setup_steps:
                await session.install_dependencies(blueprint)

            for p in blueprint.required_extensions:
                await session.mount_extension(p)

            self._active_sessions[session_id] = session
            return session
        except Exception as e:
            logger.error(f"创建 Session 失败: {e}", command="SandboxManager")
            raise e

    async def setup_workspace_environment(
        self, session_id: str, workspace_dir: str
    ) -> bool:
        """统一扫描指定工作区，通过 Provisioner 体系完成环境装配"""
        if session_id not in self._active_sessions:
            logger.warning(
                f"找不到活跃的 Session '{session_id}'，无法配置环境。",
                command="SandboxManager",
            )
            return False

        session = self._active_sessions[session_id]

        from zhenxun.services.ai.sandbox.environments import ProvisionerRegistry

        for prov in ProvisionerRegistry.get_all().values():
            await prov.scan_and_setup_workspace(cast(Any, session), workspace_dir)
        return True

    async def release_resource(self, resource_id: str):
        if resource_id in self._active_sessions:
            session = self._active_sessions.pop(resource_id)
            try:
                client_cls = SandboxRegistry.get_client_cls(session.state.sandbox_type)
                client = client_cls()
                await client.delete(cast(Any, session))
            except Exception as e:
                logger.error(
                    f"销毁沙箱环境失败 (Session: {resource_id}): {e}",
                    command="SandboxManager",
                )

    async def close_session(self, session_id: str) -> None:
        await self.lifespan_manager.unregister(session_id)
        await self.release_resource(session_id)

    async def shutdown_all(self) -> None:
        keys = list(self._active_sessions.keys())
        for sid in keys:
            await self.close_session(sid)
        await self.lifespan_manager.stop()


sandbox_manager = SandboxManager()


driver = nonebot.get_driver()


@driver.on_startup
async def _startup_sandboxes():
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
                    "Docker 引擎探活超时(5s)，可能处于假死状态，已自动禁用本地路由。",
                    command="SandboxManager",
                )
            except Exception:
                is_alive = False

            DockerSandboxClient.set_engine_status(is_alive)

            if is_alive:
                await DockerSandboxClient.prune_orphan_containers()
            else:
                logger.debug(
                    "未检测到可用本地 Docker 引擎，Docker 沙箱路由已自动禁用。",
                    command="SandboxManager",
                )

        task = asyncio.create_task(_async_init_docker_sandbox())
        _startup_tasks.add(task)
        task.add_done_callback(_startup_tasks.discard)


@driver.on_shutdown
async def _shutdown_sandboxes():
    await sandbox_manager.shutdown_all()
    clients = SandboxRegistry.get_all_clients()
    if "docker" in clients and getattr(clients["docker"], "_engine_available", False):
        from .drivers.docker import DockerSandboxClient

        await DockerSandboxClient.close_env()
