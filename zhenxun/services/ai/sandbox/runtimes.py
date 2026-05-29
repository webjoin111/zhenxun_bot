from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
import re
from typing import Any, ClassVar
import uuid

import aiohttp

from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxSession
from zhenxun.services.ai.sandbox.host_bridge import STUB_TEMPLATE, sandbox_rpc_server
from zhenxun.services.ai.sandbox.models import (
    CodeBlock,
    LanguageProfile,
    SandboxExecutionResult,
)
from zhenxun.services.log import logger
from zhenxun.utils.utils import infer_plugin_namespace


def parse_shebang(script_path: str | Path) -> str | None:
    path = Path(script_path)
    if not path.is_file():
        return None

    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        if not first_line.startswith("#!"):
            return None

        shebang = first_line[2:].strip()
        if not shebang:
            return None

        parts = shebang.split()
        if Path(parts[0]).name == "env":
            for part in parts[1:]:
                if not part.startswith("-"):
                    return part
            return None
        return Path(parts[0]).name
    except Exception:
        return None


def get_execution_command(
    script_path: str | Path, args: list[str] | None = None
) -> str:
    path = Path(script_path)
    interpreter = parse_shebang(path)
    if not interpreter:
        ext = path.suffix.lower()
        if ext == ".py":
            interpreter = "python3"
        elif ext in (".js", ".ts"):
            interpreter = "node"
        elif ext in (".sh", ".bash"):
            interpreter = "bash"
        else:
            interpreter = "sh"

    cmd_parts = [interpreter, path.as_posix()]
    if args:
        cmd_parts.extend(args)
    return " ".join(cmd_parts)


class JupyterKernelClient:
    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        base_url: str,
        ws_url: str,
        kernel_id: str,
    ):
        self.http_session = http_session
        self.base_url = base_url
        self.ws_url = ws_url
        self.kernel_id = kernel_id
        self.ws: aiohttp.ClientWebSocketResponse | None = None

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        if self.ws is None or self.ws.closed:
            self.ws = await self.http_session.ws_connect(
                f"{self.ws_url}/api/kernels/{self.kernel_id}/channels"
            )
        assert self.ws is not None
        return self.ws

    async def interrupt(self):
        try:
            async with self.http_session.post(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt"
            ):
                pass
            logger.debug(f"[JupyterClient] 成功向 Kernel {self.kernel_id} 发送中断信号")
        except Exception as e:
            logger.warning(f"[JupyterClient] 中断 Kernel 失败: {e}")

    async def execute(self, code: str, timeout: int = 30, on_output=None):
        background_tasks: set[asyncio.Task[None]] = set()

        try:
            ws = await self._connect_ws()
        except Exception as e:
            return SandboxExecutionResult(
                exit_code=-1, error=f"连接 Jupyter Kernel 失败: {e}"
            )

        msg_id = uuid.uuid4().hex
        req = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "channel": "shell",
            "content": {"code": code, "silent": False, "store_history": False},
        }

        try:
            if ws.closed:
                ws = await self._connect_ws()
            await ws.send_json(req)
        except Exception as e:
            logger.warning(f"[JupyterClient] WebSocket 发送失败，尝试重连: {e}")
            self.ws = None
            ws = await self._connect_ws()
            await ws.send_json(req)

        stdout_parts = []
        stderr_parts = []
        images = []
        exit_code = 0
        current_out_len = 0
        MAX_OUT_LEN = 50000

        async def _receive_loop():
            nonlocal exit_code, current_out_len
            while True:
                msg = await ws.receive_json()
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                def _append_text(text: str, is_stdout: bool = True) -> bool:
                    nonlocal current_out_len
                    if current_out_len + len(text) > MAX_OUT_LEN:
                        allowed_len = max(0, MAX_OUT_LEN - current_out_len)
                        truncated = (
                            text[:allowed_len]
                            + "\n...[系统警告：输出超长已被强行截断]..."
                        )
                        (stdout_parts if is_stdout else stderr_parts).append(truncated)
                        current_out_len += len(truncated)
                        return False
                    current_out_len += len(text)
                    (stdout_parts if is_stdout else stderr_parts).append(text)
                    if on_output:
                        asyncio.create_task(
                            on_output(
                                "stdout" if is_stdout else "stderr",
                                text.encode("utf-8"),
                            )
                        )
                    return True

                if msg_type == "stream":
                    if not _append_text(
                        content["text"], is_stdout=(content["name"] == "stdout")
                    ):
                        task = asyncio.create_task(self.interrupt())
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)
                        exit_code = -1
                        break
                elif msg_type == "error":
                    if not _append_text(
                        "\n".join(content["traceback"]), is_stdout=False
                    ):
                        break
                    exit_code = 1
                elif msg_type in ["display_data", "execute_result"]:
                    data = content.get("data", {})
                    if "text/plain" in data:
                        if not _append_text(data["text/plain"] + "\n", is_stdout=True):
                            break
                    if "image/png" in data:
                        images.append(data["image/png"])
                elif msg_type == "execute_reply":
                    if content.get("status") == "error":
                        exit_code = 1
                    break

        try:
            await asyncio.wait_for(_receive_loop(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            await self.interrupt()
            return SandboxExecutionResult(
                stdout="".join(stdout_parts),
                stderr="".join(stderr_parts),
                exit_code=-1,
                error=f"执行超时 ({timeout}s)",
            )
        except Exception as e:
            return SandboxExecutionResult(
                stdout="".join(stdout_parts),
                stderr="".join(stderr_parts),
                exit_code=-1,
                error=str(e),
            )

        return SandboxExecutionResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            exit_code=exit_code,
            images=images,
        )

    async def close(self):
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None


def extract_markdown_code_blocks(
    markdown_text: str, supported_languages: list[str] | None = None
) -> list["CodeBlock"]:
    """
    从 Markdown 文本中提取指定语言的代码块。
    支持如 ```python ... ``` 格式。
    """
    pattern = re.compile(
        r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(markdown_text)
    code_blocks: list[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        if supported_languages and language.lower() not in [
            l.lower() for l in supported_languages
        ]:
            continue
        code_blocks.append(CodeBlock(code=match[1], language=language))
    return code_blocks


class BaseCodeExecutor(ABC):
    """
    代码执行器抽象基类 (Autogen 范式)
    分离沙箱引擎和具体语言的执行逻辑。
    """

    def __init__(self, session: BaseSandboxSession):
        self.session = session

    async def _prepare_rpc_env(self) -> dict[str, str]:
        """将 stub 写入工作区，并返回需要注入的环境变量"""
        await self.session.write(
            "/workspace/zhenxun_stub.py", STUB_TEMPLATE.encode("utf-8")
        )

        is_docker = self.session.state.sandbox_type == "docker"
        host_ip = "host.docker.internal" if is_docker else "127.0.0.1"

        base_env = self.session.get_meta("env", {})
        base_env.update(
            {
                "ZHENXUN_RPC_URL": f"http://{host_ip}:{sandbox_rpc_server.port}/rpc",
                "ZHENXUN_SESSION_ID": self.session.session_id or "default",
            }
        )
        return base_env

    @abstractmethod
    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None,
        on_output: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> SandboxExecutionResult:
        pass


class GenericCLIExecutor(BaseCodeExecutor):
    """模板驱动的通用代码执行器"""

    def __init__(self, session, profile: "LanguageProfile"):
        super().__init__(session)
        self.profile = profile

    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None,
        on_output: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> SandboxExecutionResult:
        env = None
        if self.profile.inject_rpc_stub:
            if injected_code:
                await self.session.write(
                    "/workspace/zhenxun_host.py", injected_code.encode("utf-8")
                )
            env = await self._prepare_rpc_env()

        combined_code = "\n".join([b.code for b in code_blocks])
        script_path = f"/workspace/main{self.profile.source_ext}"
        await self.session.write(script_path, combined_code.encode("utf-8"))

        if self.profile.compile_cmd:
            compile_cmd = self.profile.compile_cmd.format(source_file=script_path)
            res = await self.session.exec(
                compile_cmd, timeout=timeout, env=env, on_output=on_output
            )
            if res.exit_code != 0:
                return res

        run_cmd = self.profile.run_cmd.format(source_file=script_path)
        return await self.session.exec(
            run_cmd, timeout=timeout, env=env, on_output=on_output
        )


class CodeExecutorRegistry:
    """多语言代码执行器动态注册中心"""

    _executors: ClassVar[dict[str, dict[str, dict[bool, type[BaseCodeExecutor]]]]] = {}
    _profiles: ClassVar[dict[str, "LanguageProfile"]] = {}

    _aliases: ClassVar[dict[str, str]] = {
        "py": "python",
        "js": "javascript",
        "sh": "bash",
        "shell": "bash",
        "ts": "typescript",
    }

    @classmethod
    def _normalize_lang(cls, language: str) -> str:
        lang_lower = language.lower().strip()
        return cls._aliases.get(lang_lower, lang_lower)

    @classmethod
    def register(
        cls,
        language: str,
        executor_cls: type[BaseCodeExecutor],
        is_stateful: bool = False,
        scope: str | None = None,
    ) -> None:
        ns = scope if scope is not None else infer_plugin_namespace()
        lang_norm = cls._normalize_lang(language)

        if ns not in cls._executors:
            cls._executors[ns] = {}
        if lang_norm not in cls._executors[ns]:
            cls._executors[ns][lang_norm] = {}

        cls._executors[ns][lang_norm][is_stateful] = executor_cls
        logger.debug(
            f"[CodeExecutorRegistry] 成功注册执行器: {ns} -> {lang_norm} (Stateful: {is_stateful}) -> {executor_cls.__name__}"
        )

    @classmethod
    def register_profile(cls, profile: "LanguageProfile") -> None:
        cls._profiles[profile.language.lower()] = profile
        for alias in profile.aliases:
            cls._profiles[alias.lower()] = profile
        logger.debug(f"[CodeExecutorRegistry] 成功注册语言配置模板: {profile.language}")

    @classmethod
    def create_executor(
        cls, language: str, needs_state: bool, session: Any, namespace: str = "global"
    ) -> BaseCodeExecutor:
        lang_norm = cls._normalize_lang(language)

        for target_ns in [namespace, "global"]:
            if target_ns in cls._executors and lang_norm in cls._executors[target_ns]:
                lang_executors = cls._executors[target_ns][lang_norm]
                if needs_state and True in lang_executors:
                    return lang_executors[True](session)
                if False in lang_executors:
                    return lang_executors[False](session)

        for ns_dict in cls._executors.values():
            if lang_norm in ns_dict:
                lang_executors = ns_dict[lang_norm]
                if needs_state and True in lang_executors:
                    return lang_executors[True](session)
                if False in lang_executors:
                    return lang_executors[False](session)

        if lang_norm in cls._profiles:
            return GenericCLIExecutor(session, cls._profiles[lang_norm])

        raise ValueError(
            f"当前沙箱生态未提供针对语言 '{language}' 的代码执行器。支持的语言有: {cls.get_supported_languages()}"
        )

    @classmethod
    def get_supported_languages(cls, namespace: str = "global") -> list[str]:
        langs = set(cls._executors.get("global", {}).keys())
        if namespace in cls._executors:
            langs.update(cls._executors[namespace].keys())
        langs.update(cls._profiles.keys())
        return list(langs)


class PythonJupyterExecutor(BaseCodeExecutor):
    """Python 有状态多模态执行器。动态在沙箱内按需启动 Jupyter Server。"""

    def __init__(self, session):
        super().__init__(session)
        self.jupyter_client = None
        self._http_session = None

    async def _ensure_jupyter_started(self):
        if self.jupyter_client:
            return

        jupyter_port = self.session.get_meta("jupyter_port")
        if not jupyter_port:
            raise RuntimeError("沙箱未映射 Jupyter 端口")

        check_jupyter = await self.session.exec("command -v jupyter-server")
        if check_jupyter.exit_code != 0:
            raise RuntimeError("沙箱内未安装 jupyter-server，无法启动高级环境")

        rpc_env = await self._prepare_rpc_env()
        env_str = " ".join([f"{k}={v}" for k, v in rpc_env.items()])

        start_cmd = (
            f"nohup env {env_str} jupyter-server "
            "--ServerApp.ip=0.0.0.0 --ServerApp.port=8888 "
            "--ServerApp.token='' --ServerApp.password='' "
            "--ServerApp.disable_check_xsrf=True "
            "--ServerApp.allow_origin='*' --ServerApp.allow_root=True "
            "> /workspace/jupyter.log 2>&1 &"
        )
        await self.session.exec(start_cmd)

        base_url = f"http://127.0.0.1:{jupyter_port}"
        ws_url = f"ws://127.0.0.1:{jupyter_port}"

        self._http_session = aiohttp.ClientSession()
        for _ in range(15):
            try:
                async with self._http_session.get(f"{base_url}/api/kernels") as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Jupyter 服务动态拉起超时")

        async with self._http_session.post(
            f"{base_url}/api/kernels", json={"name": "python3"}
        ) as resp:
            kernel_id = (await resp.json()).get("id")

        self.jupyter_client = JupyterKernelClient(
            self._http_session, base_url, ws_url, kernel_id
        )
        logger.info("[PythonJupyterExecutor] Jupyter 内核动态拉起成功！")

    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
        timeout: int = 30,
        injected_code: str | None = None,
        on_output: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> SandboxExecutionResult:
        await self._ensure_jupyter_started()

        if injected_code:
            await self.session.write(
                "/workspace/zhenxun_host.py", injected_code.encode("utf-8")
            )

        combined_code = "\n".join([b.code for b in code_blocks])
        result = await self.jupyter_client.execute(
            combined_code, timeout=timeout, on_output=on_output
        )
        return result

    async def close(self):
        if self.jupyter_client:
            await self.jupyter_client.close()
            self.jupyter_client = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None


CodeExecutorRegistry.register_profile(
    LanguageProfile(
        language="python",
        aliases=["py"],
        source_ext=".py",
        run_cmd="python3 {source_file}",
        inject_rpc_stub=True,
    )
)
CodeExecutorRegistry.register_profile(
    LanguageProfile(
        language="bash",
        aliases=["sh", "shell"],
        source_ext=".sh",
        run_cmd="bash {source_file}",
    )
)
CodeExecutorRegistry.register_profile(
    LanguageProfile(
        language="javascript",
        aliases=["js", "node"],
        source_ext=".js",
        run_cmd="node {source_file}",
    )
)
CodeExecutorRegistry.register(
    "python", PythonJupyterExecutor, is_stateful=True, scope="global"
)

__all__ = [
    "BaseCodeExecutor",
    "CodeExecutorRegistry",
    "GenericCLIExecutor",
    "PythonJupyterExecutor",
    "extract_markdown_code_blocks",
    "get_execution_command",
]
