import asyncio
from pathlib import Path
import sys
import uuid

import aiohttp

from zhenxun.services.log import logger

STDLIB_MODULES = getattr(
    sys,
    "stdlib_module_names",
    {
        "os",
        "sys",
        "re",
        "math",
        "time",
        "datetime",
        "json",
        "urllib",
        "random",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "base64",
        "hashlib",
        "csv",
        "threading",
        "multiprocessing",
        "asyncio",
        "typing",
        "subprocess",
        "logging",
        "sqlite3",
        "xml",
        "html",
        "socket",
        "io",
        "copy",
        "uuid",
    },
)

PACKAGE_ALIAS_MAP = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "dateutil": "python-dateutil",
    "git": "gitpython",
    "docx": "python-docx",
    "Crypto": "pycryptodome",
    "jwt": "pyjwt",
}


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

    async def _connect_ws(self):
        if self.ws is None or self.ws.closed:
            self.ws = await self.http_session.ws_connect(
                f"{self.ws_url}/api/kernels/{self.kernel_id}/channels"
            )

    async def interrupt(self):
        try:
            async with self.http_session.post(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt"
            ):
                pass
            logger.debug(f"[JupyterClient] 成功向 Kernel {self.kernel_id} 发送中断信号")
        except Exception as e:
            logger.warning(f"[JupyterClient] 中断 Kernel 失败: {e}")

    async def execute(self, code: str, timeout: int = 30):
        from zhenxun.services.ai.sandbox.models import SandboxExecutionResult

        try:
            await self._connect_ws()
        except Exception as e:
            return SandboxExecutionResult(
                exit_code=-1, error=f"连接 Jupyter Kernel 失败: {e}"
            )

        assert self.ws is not None
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
            if self.ws.closed:
                await self._connect_ws()
            await self.ws.send_json(req)
        except Exception as e:
            logger.warning(f"[JupyterClient] WebSocket 发送失败，尝试重连: {e}")
            self.ws = None
            await self._connect_ws()
            await self.ws.send_json(req)

        stdout_parts = []
        stderr_parts = []
        images = []
        exit_code = 0
        current_out_len = 0
        MAX_OUT_LEN = 50000

        async def _receive_loop():
            nonlocal exit_code, current_out_len
            while True:
                msg = await self.ws.receive_json()
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
                    return True

                if msg_type == "stream":
                    if not _append_text(
                        content["text"], is_stdout=(content["name"] == "stdout")
                    ):
                        asyncio.create_task(self.interrupt())
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
