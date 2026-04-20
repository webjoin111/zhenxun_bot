import asyncio
import base64
import io
from pathlib import Path
import re
import socket
import tarfile
from typing import TYPE_CHECKING, Any
import uuid

import aiohttp
import httpx

from zhenxun.configs.config import Config
from zhenxun.services.ai.sandbox.extension import (
    InteractiveTerminalSession,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
    SupportsPortMapping,
)
from zhenxun.services.ai.types.sandbox import SandboxExecutionResult, SandboxSecurityProfile
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump, model_validate

from .base import BaseSandboxDriver
from .ipc_models import (
    IpcCmdRunRequest,
    IpcCmdRunResponse,
    IpcFsDeleteRequest,
    IpcFsReadRequest,
    IpcFsReadResponse,
    IpcFsWriteRequest,
    IpcMcpStartRequest,
    IpcPtyCloseRequest,
    IpcPtyInputRequest,
    IpcPtyInterruptRequest,
    IpcPtyScreenRequest,
    IpcPtyScreenResponse,
    IpcPtyStartRequest,
    IpcPtyStartResponse,
)

try:
    import aiodocker as _aiodocker
    aiodocker: Any = _aiodocker
    DOCKER_AVAILABLE = True
except ImportError:
    aiodocker = None
    DOCKER_AVAILABLE = False

if TYPE_CHECKING:
    pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DockerInteractiveTerminalSession(InteractiveTerminalSession):
    """直连 OS Server 的 PTY 会话 (Track B)"""

    def __init__(self, driver: "DockerDriver"):
        self.driver = driver
        self._lock = asyncio.Lock()
        self.pty_id = None
        self.ws_headers = {}

    async def start(self, cmd: str, env: dict[str, str] | None = None) -> None:
        async with self._lock:
            req = IpcPtyStartRequest(command=cmd, env=env)
            resp = await self.driver._ipc_request(
                "POST", "/pty/start", json=model_dump(req), timeout=10
            )
            if resp.status_code == 200:
                parsed = model_validate(IpcPtyStartResponse, resp.json())
                self.pty_id = parsed.pty_id
            else:
                raise RuntimeError(f"Failed to start PTY: {resp.text}")

    async def send_input(self, text: str) -> None:
        async with self._lock:
            if not self.pty_id:
                return
            req = IpcPtyInputRequest(pty_id=self.pty_id, data=text)
            await self.driver._ipc_request(
                "POST", "/pty/input", json=model_dump(req), timeout=5
            )

    async def read_output(self, timeout: int = 5) -> str:
        async with self._lock:
            if not self.pty_id:
                return ""
            req = IpcPtyScreenRequest(pty_id=self.pty_id)
            resp = await self.driver._ipc_request(
                "POST", "/pty/screen", json=model_dump(req), timeout=timeout
            )
            if resp.status_code == 200:
                parsed = model_validate(IpcPtyScreenResponse, resp.json())
                return parsed.screen
            return ""

    async def interrupt(self) -> None:
        async with self._lock:
            if not self.pty_id:
                return
            req = IpcPtyInterruptRequest(pty_id=self.pty_id)
            await self.driver._ipc_request(
                "POST", "/pty/interrupt", json=model_dump(req), timeout=5
            )

    async def close(self) -> None:
        async with self._lock:
            if self.pty_id:
                req = IpcPtyCloseRequest(pty_id=self.pty_id)
                await self.driver._ipc_request(
                    "POST", "/pty/close", json=model_dump(req), timeout=5
                )
                self.pty_id = None


class DockerDriver(
    BaseSandboxDriver,
    SupportsCommandExecution,
    SupportsFileSystem,
    SupportsInteractivePTY,
    SupportsPortMapping,
):
    """
    Docker 驱动 (Jupyter 升维版)：启动带 Jupyter 的长连接有状态容器.
    - 通过 WebSocket 原生捕获图片 (matplotlib) 和执行流。
    - 变量状态可在多轮对话中持久保存。
    """

    _global_docker_client = None
    _global_http_session = None
    _init_lock = asyncio.Lock()

    async def _ipc_request(self, method: str, path: str, **kwargs):
        """自带容器死亡自愈机制的安全 IPC 通道"""

        if path == "/mcp/start" and "json" in kwargs:
            kwargs["json"] = model_dump(IpcMcpStartRequest(**kwargs["json"]))

        if (
            not hasattr(self, "_ipc_client")
            or self._ipc_client is None
            or self._ipc_client.is_closed
        ):
            self._ipc_client = httpx.AsyncClient(timeout=30)

        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._os_server_token}"

        url = f"http://127.0.0.1:{self._os_server_port}{path}"
        try:
            resp = await self._ipc_client.request(
                method, url, headers=headers, **kwargs
            )
            if resp.status_code in (502, 503, 504):
                raise ConnectionError(
                    f"HTTP {resp.status_code} - Sandbox OS Server not ready"
                )
            return resp
        except Exception as e:
            logger.warning(f"[DockerDriver] 内部服务请求失败，等待1秒后重试: {e}")
            import asyncio

            await asyncio.sleep(1)
            if (
                not hasattr(self, "_ipc_client")
                or self._ipc_client is None
                or self._ipc_client.is_closed
            ):
                self._ipc_client = httpx.AsyncClient(timeout=30)
            new_url = f"http://127.0.0.1:{self._os_server_port}{path}"
            return await self._ipc_client.request(
                method, new_url, headers=headers, **kwargs
            )

    async def execute_raw_command(
        self,
        command: str | list[str],
        cwd: str | None = None,
        timeout: int = 30,
        env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        self.touch()
        cmd_str = command if isinstance(command, str) else " ".join(command)
        req = IpcCmdRunRequest(command=cmd_str, cwd=cwd, env=env, timeout=timeout)
        try:
            resp = await self._ipc_request(
                "POST", "/cmd/run", json=model_dump(req), timeout=timeout + 5
            )
            parsed = model_validate(IpcCmdRunResponse, resp.json())
            return SandboxExecutionResult(
                stdout=parsed.stdout,
                stderr=parsed.stderr,
                exit_code=parsed.exit_code,
                is_timeout=parsed.is_timeout,
            )
        except Exception as e:
            return SandboxExecutionResult(
                exit_code=-1, error=f"Run Command IPC Failed: {e}"
            )

    async def create_pty_session(self) -> InteractiveTerminalSession:
        session = DockerInteractiveTerminalSession(self)
        session.ws_headers = {"Authorization": f"Bearer {self._os_server_token}"}
        return session

    async def write_raw_file(self, path: str, content: str) -> bool:
        self.touch()
        try:
            req = IpcFsWriteRequest(path=path, content=content)
            resp = await self._ipc_request(
                "POST", "/fs/write", json=model_dump(req), timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"[DockerDriver] write_raw_file connection failed: {e}")
            return False

    async def read_raw_file(self, path: str) -> str:
        self.touch()
        try:
            req = IpcFsReadRequest(path=path)
            resp = await self._ipc_request(
                "POST", "/fs/read", json=model_dump(req), timeout=10
            )
            if resp.status_code == 404:
                return f"Error: File {path} not found."
            parsed = model_validate(IpcFsReadResponse, resp.json())
            return parsed.content
        except Exception as e:
            return f"Failed to read file connection error: {e}"

    async def delete_raw_file(self, path: str) -> bool:
        self.touch()
        try:
            req = IpcFsDeleteRequest(path=path)
            resp = await self._ipc_request(
                "POST", "/fs/delete", json=model_dump(req), timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool:
        self.touch()
        local_path = Path(local_dir_path)
        if not local_path.is_dir():
            return False
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(local_path, arcname=Path(sandbox_target_path).name)
        headers = {"X-Target-Path": sandbox_target_path}
        try:
            resp = await self._ipc_request(
                "POST",
                "/fs/upload_dir",
                content=tar_stream.getvalue(),
                headers=headers,
                timeout=60,
            )
            return resp.status_code == 200
        except Exception:
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
        self.ws_url = ""
        self._os_server_port = -1
        self._browser_port = -1
        self._mcp_ports: list[int] = []
        self._ipc_client = None
        self._os_server_token: str = ""

    async def is_alive(self) -> bool:
        """主动探活：检查容器内的 Agent OS Server 是否存活"""
        if not self.container:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                headers = {"Authorization": f"Bearer {self._os_server_token}"}
                resp = await client.get(
                    f"http://127.0.0.1:{self._os_server_port}/alive", headers=headers
                )
                return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def close_env(cls):
        """供全局进程关闭时调用，清理单例"""
        if cls._global_http_session and not cls._global_http_session.closed:
            await cls._global_http_session.close()
        if cls._global_docker_client:
            await cls._global_docker_client.close()

    @classmethod
    async def check_engine_alive(cls) -> bool:
        """测试 Docker 引擎是否正在运行"""
        if not DOCKER_AVAILABLE:
            return False
        client = None
        try:
            if aiodocker is None:
                return False
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
        if not DOCKER_AVAILABLE:
            return

        need_close = False
        client = cls._global_docker_client
        if client is None:
            if aiodocker is None:
                return
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
        if not DOCKER_AVAILABLE:
            raise RuntimeError("aiodocker is not installed.")

        if self._global_docker_client is None or self._global_http_session is None:
            async with self._init_lock:
                if self._global_docker_client is None:
                    self._global_docker_client = aiodocker.Docker()
                if self._global_http_session is None:
                    self._global_http_session = aiohttp.ClientSession()

        port = _find_free_port()
        self._os_server_port = _find_free_port()
        self._browser_port = _find_free_port()
        self._mcp_ports = [_find_free_port() for _ in range(3)]
        self.base_url = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}"

        self._meta["os_server_port"] = self._os_server_port
        self._meta["browser_port"] = self._browser_port
        self._meta["base_url"] = self.base_url
        self._meta["ws_url"] = self.ws_url
        self._os_server_token = uuid.uuid4().hex

        image = Config.get_config("sandbox", "DOCKER_IMAGE", "zhenxun-sandbox:latest")

        enable_network = profile.enable_network if profile else False

        logger.info(
            f"[DockerDriver] 正在启动 Jupyter 容器"
            f" (OS_Port: {self._os_server_port}, Session: {session_id})..."
        )

        os_server_path = Path(__file__).parent / "agent_os.py"
        os_server_code = os_server_path.read_text(encoding="utf-8")
        os_server_b64 = base64.b64encode(os_server_code.encode("utf-8")).decode("utf-8")

        container_config = {
            "Image": image,
            "Entrypoint": [],
            "Cmd": [
                "/bin/bash",
                "-c",
                f"echo {os_server_b64} | base64 -d > /opt/agent_os.py && python3 /opt/agent_os.py & "
                f"if command -v jupyter-server &> /dev/null; then "
                f"exec jupyter-server --ServerApp.ip=0.0.0.0 --ServerApp.port=8888 --ServerApp.token= --ServerApp.password= --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin=* --ServerApp.allow_root=True --ServerApp.terminals_enabled=True; "
                f"else echo '[Init] Jupyter not found, running in lightweight mode.'; tail -f /dev/null; fi",
            ],
            "ExposedPorts": {
                "8888/tcp": {},
                f"{self._os_server_port}/tcp": {},
                f"{self._browser_port}/tcp": {},
            },
            "HostConfig": {
                "PortBindings": {
                    "8888/tcp": [{"HostPort": str(port)}],
                    f"{self._os_server_port}/tcp": [
                        {"HostPort": str(self._os_server_port)}
                    ],
                    f"{self._browser_port}/tcp": [
                        {"HostPort": str(self._browser_port)}
                    ],
                },
                "Memory": 512 * 1024 * 1024,
                "MemorySwap": 512 * 1024 * 1024,
            },
            "Labels": {
                "zhenxun_component": "sandbox",
                "managed_by": "docker_driver",
            },
            "Env": [
                f"OS_SERVER_PORT={self._os_server_port}",
                f"MCP_PORTS={','.join(map(str, self._mcp_ports))}",
                f"OS_SERVER_TOKEN={self._os_server_token}",
            ],
        }

        for p in self._mcp_ports:
            container_config["ExposedPorts"][f"{p}/tcp"] = {}
            container_config["HostConfig"]["PortBindings"][f"{p}/tcp"] = [
                {"HostPort": str(p)}
            ]

        if not enable_network:
            container_config["HostConfig"]["Dns"] = ["0.0.0.0"]

        safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id)
        name = f"zx_sandbox_{safe_session_id}_{uuid.uuid4().hex[:8]}"
        self.container = await self._global_docker_client.containers.run(
            config=container_config, name=name
        )

        logger.info("[DockerDriver] 容器已创建，正在等待内部 OS Server 就绪...")
        is_ready = False
        for _ in range(30):
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    headers = {"Authorization": f"Bearer {self._os_server_token}"}
                    resp = await client.get(
                        f"http://127.0.0.1:{self._os_server_port}/alive",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        is_ready = True
                        break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        if not is_ready:
            raise RuntimeError("沙箱内部 Agent OS Server 启动超时或崩溃！")
        logger.info("[DockerDriver] Agent OS Server 探针检测已就绪！")

        probe_res = await self.execute_raw_command("command -v jupyter-server")
        is_heavy_image = probe_res.exit_code == 0

        if profile and getattr(profile, "require_pty", False):
            logger.info(
                "[DockerDriver] 开发者意图声明强制 PTY 交互，主动降级并禁用 Jupyter..."
            )
            is_heavy_image = False

    async def close(self) -> None:
        if (
            hasattr(self, "_ipc_client")
            and self._ipc_client
            and not self._ipc_client.is_closed
        ):
            await self._ipc_client.aclose()
            self._ipc_client = None

        if self.container:
            logger.info(
                f"[DockerDriver] 正在销毁 Jupyter 容器 (Session: {self.session_id})"
            )
            try:
                await self.container.delete(force=True)
            except Exception as e:
                logger.error(f"[DockerDriver] 销毁容器失败: {e}")
