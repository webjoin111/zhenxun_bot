from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass
class SSEClientParams:
    """SSE 客户端连接参数"""

    url: str
    headers: dict[str, Any] | None = None
    timeout: float | None = 5
    sse_read_timeout: float | None = 60 * 5


@dataclass
class StreamableHTTPClientParams:
    """HTTP 客户端连接参数"""

    url: str
    headers: dict[str, Any] | None = None
    timeout: timedelta | None = timedelta(seconds=30)
    sse_read_timeout: timedelta | None = timedelta(seconds=60 * 5)
    terminate_on_close: bool | None = None
