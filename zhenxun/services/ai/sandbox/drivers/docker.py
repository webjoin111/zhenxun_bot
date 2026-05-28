import asyncio
import io
from pathlib import Path
import re
import socket
import tarfile

import aiodocker

from zhenxun.configs.config import Config
from zhenxun.services.ai.sandbox.extension import (
    InteractiveTerminalSession,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
    SupportsPortMapping,
)
from zhenxun.services.ai.sandbox.models import (
    SandboxCapabilities,
    SandboxExecutionResult,
    SandboxSecurityProfile,
)
from zhenxun.services.log import logger

from .base import BaseSandboxDriver


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DockerInteractiveTerminalSession(InteractiveTerminalSession):
    """原生 PTY 交互式会话：接管 Docker API Stream 流并清洗 ANSI 逃逸码"""

    def __init__(self, driver: "DockerDriver"):
        self.driver = driver
        self._lock = asyncio.Lock()
        self.stream = None
        self._read_task = None
        self.buffer = ""
        self.ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")

    async def start(self, cmd: str, env: dict[str, str] | None = None) -> None:
        async with self._lock:
            if not self.driver.container:
                raise RuntimeError("沙箱容器未启动")

            cmd_list = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else cmd
            env_list = [f"{k}={v}" for k, v in env.items()] if env else None

            exec_inst = await self.driver.container.exec(
                cmd=cmd_list,
                tty=True,
                stdin=True,
                stdout=True,
                stderr=True,
                workdir="/workspace",
                environment=env_list,
            )

            self.stream = exec_inst.start(detach=False)
            await self.stream.__aenter__()

            self._read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        try:
            while True:
                if not self.stream:
                    break
                msg = await self.stream.read_out()
                if msg is None:
                    break

                text = msg.data.decode("utf-8", errors="replace")
                clean_text = self.ansi_escape.sub("", text)
                self.buffer += clean_text

                if len(self.buffer) > 10000:
                    self.buffer = self.buffer[-10000:]
        except Exception as e:
            logger.debug(f"[PTY] 流读取正常结束或中断: {e}")

    async def send_input(self, text: str) -> None:
        async with self._lock:
            if self.stream:
                await self.stream.write_in(text.encode("utf-8"))

    async def read_output(self, timeout: int = 5) -> str:
        async with self._lock:
            lines = self.buffer.split("\n")
            return "\n".join(lines[-50:]).strip()

    async def interrupt(self) -> None:
        async with self._lock:
            if self.stream:
                await self.stream.write_in(b"\x03")

    async def close(self) -> None:
        async with self._lock:
            if self._read_task:
                self._read_task.cancel()
            if self.stream:
                await self.stream.close()
                self.stream = None


class DockerDriver(
    BaseSandboxDriver,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
    SupportsPortMapping,
):
    """
    Docker 驱动 (原生协议版)：
    完全抛弃寄生服务器，拥抱 aiodocker 原生 API 进行执行和文件读写。
    零依赖，支持任何第三方纯净镜像。
    """

    _global_docker_client = None
    _init_lock = asyncio.Lock()
    _engine_available: bool = False

    @classmethod
    def set_engine_status(cls, status: bool) -> None:
        """由框架启动钩子注入引擎存活状态"""
        cls._engine_available = status

    @classmethod
    def get_capabilities(cls) -> SandboxCapabilities:
        return SandboxCapabilities(
            supports_state=True,
            supported_capabilities=[
                "PythonExecutionCapability",
                "FileSystemCapability",
                "TerminalCapability",
                "SkillEnvironmentCapability",
            ],
            isolation_level=8,
            startup_latency=500,
        )

    async def execute_raw_command(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int = 30,
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        self.touch()
        if not self.container:
            return SandboxExecutionResult(exit_code=-1, error="容器未启动")

        if isinstance(command, str):
            cmd_list = ["/bin/sh", "-c", command]
        else:
            cmd_list = command

        env_list = [f"{k}={v}" for k, v in env.items()] if env else None

        try:
            exec_inst = await self.container.exec(
                cmd=cmd_list,
                stdout=True,
                stderr=True,
                workdir=cwd or "/workspace",
                environment=env_list,
            )

            stdout_buf = bytearray()
            stderr_buf = bytearray()
            is_timeout = False

            async def _read_stream():
                async with exec_inst.start(detach=False) as stream:
                    while True:
                        msg = await stream.read_out()
                        if msg is None:
                            break
                        if msg.stream == 1:
                            stdout_buf.extend(msg.data)
                        elif msg.stream == 2:
                            stderr_buf.extend(msg.data)

            try:
                await asyncio.wait_for(_read_stream(), timeout=timeout)
            except asyncio.TimeoutError:
                is_timeout = True

            info = await exec_inst.inspect()
            exit_code = info.get("ExitCode")
            if exit_code is None:
                exit_code = -1

            return SandboxExecutionResult(
                stdout=stdout_buf.decode("utf-8", errors="replace").strip(),
                stderr=stderr_buf.decode("utf-8", errors="replace").strip(),
                exit_code=exit_code,
                is_timeout=is_timeout,
            )
        except Exception as e:
            logger.error(f"[DockerDriver] 原生执行命令失败: {e}", e=e)
            return SandboxExecutionResult(exit_code=-1, error=str(e))

    async def create_pty_session(self) -> InteractiveTerminalSession:
        session = DockerInteractiveTerminalSession(self)
        return session

    async def write_raw_file(self, path: str, content: str) -> bool:
        self.touch()
        if not self.container:
            return False

        try:
            p = Path(path)
            dir_path = p.parent.as_posix()
            file_name = p.name

            await self.execute_raw_command(f"mkdir -p {dir_path}")

            import time

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=file_name)
                encoded = content.encode("utf-8")
                tarinfo.size = len(encoded)
                tarinfo.mtime = int(time.time())
                tar.addfile(tarinfo, io.BytesIO(encoded))

            tar_stream.seek(0)
            await self.container.put_archive(dir_path, tar_stream.read())
            return True
        except Exception as e:
            logger.debug(f"[DockerDriver] write_raw_file connection failed: {e}")
            return False

    async def read_raw_file(self, path: str) -> str:
        self.touch()
        if not self.container:
            return "Error: Container not running."

        try:
            tar_obj = await self.container.get_archive(path)

            import os

            filename = os.path.basename(path)
            member = None
            for m in tar_obj.getmembers():
                if m.name.endswith(filename):
                    member = m
                    break

            if not member:
                return f"Error: File {path} not found in archive."

            f = tar_obj.extractfile(member)
            if not f:
                return f"Error: Failed to extract {path}."

            return f.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Failed to read file connection error: {e}"

    async def delete_raw_file(self, path: str) -> bool:
        self.touch()
        if not self.container:
            return False
        try:
            res = await self.execute_raw_command(f"rm -rf {path}")
            return res.exit_code == 0
        except Exception:
            return False

    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool:
        self.touch()
        if not self.container:
            return False
        local_path = Path(local_dir_path)
        if not await asyncio.to_thread(local_path.is_dir):
            return False

        try:

            def _create_tar():
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                    tar.add(local_path, arcname=Path(sandbox_target_path).name)
                tar_stream.seek(0)
                return tar_stream.read()

            tar_bytes = await asyncio.to_thread(_create_tar)
            target_parent = Path(sandbox_target_path).parent.as_posix()

            await self.execute_raw_command(f"mkdir -p {target_parent}")
            await self.container.put_archive(target_parent, tar_bytes)
            return True
        except Exception as e:
            logger.error(f"[DockerDriver] upload_raw_dir failed: {e}")
            return False

    @property
    def supports_state(self) -> bool:
        return True

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def requires_local_docker(self) -> bool:
        return True

    def __init__(self):
        super().__init__()
        self.container = None
        self.kernel_id: str | None = None
        self._jupyter_port = -1

    async def is_alive(self) -> bool:
        """探活：直接通过 aiodocker 查验容器状态"""
        if not self.container:
            return False
        try:
            await self.container.show()
            return self.container._container.get("State", {}).get("Running", False)
        except Exception:
            return False

    @classmethod
    async def close_env(cls):
        """供全局进程关闭时调用，清理单例"""
        if cls._global_docker_client:
            await cls._global_docker_client.close()

    @classmethod
    async def check_engine_alive(cls) -> bool:
        """测试 Docker 引擎是否正在运行"""
        client = None
        try:
            client = aiodocker.Docker()
            await client.system.info()
            return True
        except Exception:
            return False
        finally:
            if client:
                await client.close()

    @classmethod
    async def prune_orphan_containers(cls):
        """清理遗留的沙箱容器 (Watchdog)"""
        need_close = False
        client = cls._global_docker_client
        if client is None:
            client = aiodocker.Docker()
            need_close = True

        try:
            containers = await client.containers.list(
                filters={"label": ["zhenxun_component=sandbox"]}
            )
            for c in containers:
                try:
                    await c.delete(force=True)
                    logger.info(f"[DockerDriver] 成功清理孤儿沙箱容器: {c.id[:12]}")
                except Exception as e:
                    logger.error(f"[DockerDriver] 清理孤儿沙箱失败 {c.id[:12]}: {e}")
        except Exception as e:
            logger.debug(
                f"[DockerDriver] 无法连接到 Docker 引擎，跳过孤儿容器清理: {e}"
            )
        finally:
            if need_close:
                await client.close()

    async def start(
        self, session_id: str, profile: SandboxSecurityProfile | None = None
    ) -> None:
        self.session_id = session_id
        self.profile = profile
        self.touch()

        if self._global_docker_client is None:
            async with self._init_lock:
                if self._global_docker_client is None:
                    self._global_docker_client = aiodocker.Docker()

        self._jupyter_port = _find_free_port()
        self._meta["jupyter_port"] = self._jupyter_port

        image = Config.get_config("sandbox", "DOCKER_IMAGE", "zhenxun-sandbox:latest")
        enable_network = profile.enable_network if profile else False

        logger.info(
            f"[DockerDriver] 正在启动纯净 Docker 沙箱"
            f" (Session: {session_id}, Image: {image}, "
            f"JupyterPort: {self._jupyter_port})..."
        )

        container_config = {
            "Image": image,
            "Entrypoint": [],
            "Cmd": [
                "/bin/sh",
                "-c",
                "mkdir -p /workspace && tail -f /dev/null",
            ],
            "ExposedPorts": {
                "8888/tcp": {},
            },
            "HostConfig": {
                "PortBindings": {
                    "8888/tcp": [{"HostPort": str(self._jupyter_port)}],
                },
                "Memory": 512 * 1024 * 1024,
                "MemorySwap": 512 * 1024 * 1024,
            },
            "Labels": {
                "zhenxun_component": "sandbox",
                "managed_by": "docker_driver",
            },
        }

        if "ExtraHosts" not in container_config["HostConfig"]:
            container_config["HostConfig"]["ExtraHosts"] = []
        container_config["HostConfig"]["ExtraHosts"].append(
            "host.docker.internal:host-gateway"
        )

        if not enable_network:
            container_config["HostConfig"]["Dns"] = ["0.0.0.0"]

        import uuid

        safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
        name = f"zx_sandbox_{safe_session_id}_{uuid.uuid4().hex[:8]}"
        self.container = await self._global_docker_client.containers.run(
            config=container_config, name=name
        )

        logger.info("[DockerDriver] 容器已秒级创建并就绪！")

    async def close(self) -> None:
        if self.container:
            logger.info(f"[DockerDriver] 正在销毁沙箱容器 (Session: {self.session_id})")
            try:
                await self.container.delete(force=True)
            except Exception as e:
                logger.error(f"[DockerDriver] 销毁容器失败: {e}")


from zhenxun.services.ai.sandbox.extension import SandboxRegistry

SandboxRegistry.register("docker", DockerDriver)
