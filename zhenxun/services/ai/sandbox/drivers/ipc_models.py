from pydantic import BaseModel, Field


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
