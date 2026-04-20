import asyncio
import io
import os
from pathlib import Path
import shlex
import sys
import tarfile
import uuid

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import ptyprocess
from pydantic import VERSION as PYDANTIC_VERSION
from pydantic import BaseModel, Field
import pyte

IS_PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")


def model_dump(obj: BaseModel, **kwargs):
    """兼容 Pydantic V1/V2 的 dump 方法"""
    if IS_PYDANTIC_V2:
        return obj.model_dump(**kwargs)
    return obj.dict(**kwargs)


if "/workspace" not in sys.path:
    sys.path.append("/workspace")
if "/opt/zhenxun/extensions" not in sys.path:
    sys.path.append("/opt/zhenxun/extensions")

ACTIVE_PTY_SESSIONS = {}
OS_SERVER_TOKEN = os.environ.get("OS_SERVER_TOKEN")
MCP_SERVER_TASKS = set()
security_scheme = HTTPBearer(auto_error=False)


async def check_auth(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
):
    if OS_SERVER_TOKEN and OS_SERVER_TOKEN != "None":
        if not credentials or credentials.credentials != OS_SERVER_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")


class IpcStatusResponse(BaseModel):
    status: str = "ok"
    error: str | None = None


class IpcCmdRunRequest(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: int = 30


class IpcCmdRunResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    is_timeout: bool = False


class IpcFsWriteRequest(BaseModel):
    path: str
    content: str


class IpcFsReadRequest(BaseModel):
    path: str


class IpcFsReadResponse(BaseModel):
    content: str = ""
    error: str | None = None


class IpcFsDeleteRequest(BaseModel):
    path: str


class IpcPtyStartRequest(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None


class IpcPtyStartResponse(IpcStatusResponse):
    pty_id: str | None = None


class IpcMcpStartRequest(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None


class IpcMcpStartResponse(IpcStatusResponse):
    port: int | None = None


class IpcPtyInputRequest(BaseModel):
    pty_id: str
    data: str


class IpcPtyScreenRequest(BaseModel):
    pty_id: str


class IpcPtyScreenResponse(IpcStatusResponse):
    screen: str = ""


class IpcPtyInterruptRequest(BaseModel):
    pty_id: str


class IpcPtyCloseRequest(BaseModel):
    pty_id: str


class PtySession:
    """Track B: 驻留态虚拟屏幕会话"""

    def __init__(self, command: str, cwd=None, env=None, cols=120, rows=40):
        self.id = uuid.uuid4().hex
        args = shlex.split(command)
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        self.pty = ptyprocess.PtyProcessUnicode.spawn(
            args, cwd=cwd, env=run_env, dimensions=(rows, cols)
        )
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.active = True
        self.ws: WebSocket | None = None
        self._bg_tasks = set()
        self.task = asyncio.create_task(self._read_loop())

    async def _broadcast_screen(self, screen_data: str):
        if self.ws:
            try:
                await self.ws.send_json({"screen": screen_data})
            except Exception:
                pass

    async def _read_loop(self):
        while self.active and self.pty.isalive():
            try:
                data = await asyncio.to_thread(self.pty.read, 4096)
                if not data:
                    break
                self.stream.feed(data)
                t = asyncio.create_task(self._broadcast_screen(self.get_screen()))
                self._bg_tasks.add(t)
                t.add_done_callback(self._bg_tasks.discard)
            except EOFError:
                break
            except Exception as e:
                sys.stderr.write(f"PTY Read error: {e}\n")
                sys.stderr.flush()
                break
        self.active = False

    def write(self, data: str):
        if self.pty.isalive():
            self.pty.write(data)

    def get_screen(self) -> str:
        """返回滤除 ANSI 后的纯净文本矩阵"""
        lines = [line.rstrip() for line in self.screen.display]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def close(self):
        self.active = False
        if self.pty.isalive():
            self.pty.terminate(force=True)
        if self.task:
            self.task.cancel()


app = FastAPI(title="Zhenxun Agent OS Server", version="2.0.0")


@app.get("/alive", response_model=IpcStatusResponse, dependencies=[Depends(check_auth)])
async def handle_alive():
    return IpcStatusResponse()


@app.post(
    "/fs/write", response_model=IpcStatusResponse, dependencies=[Depends(check_auth)]
)
async def handle_fs_write(req: IpcFsWriteRequest):
    file_path = Path(req.path)
    await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(file_path.write_text, req.content, encoding="utf-8")
    return IpcStatusResponse()


@app.post(
    "/fs/read", response_model=IpcFsReadResponse, dependencies=[Depends(check_auth)]
)
async def handle_fs_read(req: IpcFsReadRequest):
    file_path = Path(req.path)
    if not file_path.exists():
        return JSONResponse(
            model_dump(IpcFsReadResponse(error="File not found")),
            status_code=404,
        )
    content = await asyncio.to_thread(
        file_path.read_text, encoding="utf-8", errors="replace"
    )
    return IpcFsReadResponse(content=content)


@app.post(
    "/fs/delete", response_model=IpcStatusResponse, dependencies=[Depends(check_auth)]
)
async def handle_fs_delete(req: IpcFsDeleteRequest):
    await asyncio.to_thread(Path(req.path).unlink, missing_ok=True)
    return IpcStatusResponse()


@app.post(
    "/fs/upload_dir",
    response_model=IpcStatusResponse,
    dependencies=[Depends(check_auth)],
)
async def handle_fs_upload_dir(request: Request):
    target_path = request.headers.get("X-Target-Path", "/workspace")
    post_data = await request.body()

    def _extract():
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(post_data), mode="r:*") as tar:
            tar.extractall(path=Path(target_path).parent)

    await asyncio.to_thread(_extract)
    return IpcStatusResponse()


@app.post(
    "/mcp/start", response_model=IpcMcpStartResponse, dependencies=[Depends(check_auth)]
)
async def handle_mcp_start(req: IpcMcpStartRequest):
    run_env = os.environ.copy()
    if req.env:
        run_env.update(req.env)

    available_ports = [int(p) for p in os.environ.get("MCP_PORTS", "").split(",") if p]
    if not available_ports:
        return JSONResponse(
            model_dump(
                IpcMcpStartResponse(status="error", error="No available MCP ports")
            ),
            status_code=500,
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            req.command,
            *req.args,
            env=run_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
        )
    except Exception as e:
        return JSONResponse(
            model_dump(
                IpcMcpStartResponse(
                    status="error",
                    error=f"Failed to execute command '{req.command}': {e}",
                )
            ),
            status_code=500,
        )

    async def tcp_handler(reader, writer):
        async def pipe(r, w):
            try:
                while True:
                    data = await r.read(4096)
                    if not data:
                        break
                    w.write(data)
                    await w.drain()
            except Exception:
                pass

        t1 = asyncio.create_task(pipe(reader, proc.stdin))
        t2 = asyncio.create_task(pipe(proc.stdout, writer))

        await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    server = None
    port = None
    for p in available_ports:
        try:
            server = await asyncio.start_server(tcp_handler, "0.0.0.0", p)
            port = p
            break
        except OSError:
            continue

    if not server:
        proc.kill()
        return JSONResponse(
            model_dump(
                IpcMcpStartResponse(status="error", error="All MCP ports are in use")
            ),
            status_code=500,
        )

    t = asyncio.create_task(server.serve_forever())
    MCP_SERVER_TASKS.add(t)
    t.add_done_callback(MCP_SERVER_TASKS.discard)

    return IpcMcpStartResponse(port=port)


@app.post(
    "/cmd/run", response_model=IpcCmdRunResponse, dependencies=[Depends(check_auth)]
)
async def handle_cmd_run(req: IpcCmdRunRequest):
    run_env = os.environ.copy()
    if req.env:
        run_env.update(req.env)

    script_path = f"/tmp/zx_cmd_{uuid.uuid4().hex}.sh"
    await asyncio.to_thread(
        Path(script_path).write_text, f"set -e\n{req.command}\n", encoding="utf-8"
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            script_path,
            cwd=req.cwd,
            env=run_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        is_timeout = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=req.timeout
            )
        except asyncio.TimeoutError:
            is_timeout = True
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()

        exit_code = proc.returncode if proc.returncode is not None else -1
        res = IpcCmdRunResponse(
            stdout=stdout_bytes.decode("utf-8", errors="replace").strip(),
            stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
            exit_code=exit_code if not is_timeout else -1,
            is_timeout=is_timeout,
        )
    except Exception as e:
        res = IpcCmdRunResponse(stderr=str(e), exit_code=-1, is_timeout=False)
    finally:
        try:
            os.remove(script_path)
        except Exception:
            pass

    return res


@app.post(
    "/pty/start", response_model=IpcPtyStartResponse, dependencies=[Depends(check_auth)]
)
async def handle_pty_start(req: IpcPtyStartRequest):
    try:
        session = PtySession(command=req.command, cwd=req.cwd, env=req.env)
        ACTIVE_PTY_SESSIONS[session.id] = session
        return IpcPtyStartResponse(pty_id=session.id)
    except Exception as e:
        return JSONResponse(
            model_dump(IpcPtyStartResponse(status="error", error=str(e))),
            status_code=500,
        )


@app.post(
    "/pty/input", response_model=IpcStatusResponse, dependencies=[Depends(check_auth)]
)
async def handle_pty_input(req: IpcPtyInputRequest):
    session = ACTIVE_PTY_SESSIONS.get(req.pty_id)
    if not session:
        return JSONResponse(
            model_dump(
                IpcStatusResponse(status="error", error="PTY Session Not Found")
            ),
            status_code=404,
        )
    session.write(req.data)
    return IpcStatusResponse()


@app.post(
    "/pty/screen",
    response_model=IpcPtyScreenResponse,
    dependencies=[Depends(check_auth)],
)
async def handle_pty_screen(req: IpcPtyScreenRequest):
    session = ACTIVE_PTY_SESSIONS.get(req.pty_id)
    if not session:
        return JSONResponse(
            model_dump(
                IpcPtyScreenResponse(status="error", error="PTY Session Not Found")
            ),
            status_code=404,
        )
    return IpcPtyScreenResponse(screen=session.get_screen())


@app.post(
    "/pty/interrupt",
    response_model=IpcStatusResponse,
    dependencies=[Depends(check_auth)],
)
async def handle_pty_interrupt(req: IpcPtyInterruptRequest):
    session = ACTIVE_PTY_SESSIONS.get(req.pty_id)
    if not session:
        return JSONResponse(
            model_dump(
                IpcStatusResponse(status="error", error="PTY Session Not Found")
            ),
            status_code=404,
        )
    session.write("\x03")
    return IpcStatusResponse()


@app.post(
    "/pty/close", response_model=IpcStatusResponse, dependencies=[Depends(check_auth)]
)
async def handle_pty_close(req: IpcPtyCloseRequest):
    session = ACTIVE_PTY_SESSIONS.pop(req.pty_id, None)
    if session:
        session.close()
    return IpcStatusResponse()


@app.websocket("/pty/ws/{pty_id}")
async def handle_pty_ws(websocket: WebSocket, pty_id: str):
    if OS_SERVER_TOKEN:
        auth = websocket.headers.get("Authorization")
        if auth != f"Bearer {OS_SERVER_TOKEN}":
            await websocket.close(code=1008, reason="Unauthorized")
            return
    await websocket.accept()
    session = ACTIVE_PTY_SESSIONS.get(pty_id)
    if not session:
        await websocket.close(code=1008, reason="PTY Session Not Found")
        return

    session.ws = websocket
    try:
        await websocket.send_json({"screen": session.get_screen()})
        while True:
            req = await websocket.receive_json()
            action = req.get("action")
            if action == "input":
                session.write(req.get("data", ""))
            elif action == "interrupt":
                session.write("\x03")
            elif action == "close":
                session.close()
                break
    except WebSocketDisconnect:
        pass
    finally:
        if session.ws == websocket:
            session.ws = None


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("OS_SERVER_PORT", 8000))
    sys.stdout.write(f"Agent OS Server listening on port {port}\n")
    sys.stdout.flush()
    uvicorn.run(app, host="0.0.0.0", port=port)
