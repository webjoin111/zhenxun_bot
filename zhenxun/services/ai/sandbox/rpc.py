from collections.abc import Callable

from aiohttp import web
from nonebot.utils import is_coroutine_callable

from zhenxun.services.log import logger

STUB_TEMPLATE = """
import os
import json
import urllib.request

def invoke_host(func_name, **kwargs):
    rpc_url = os.environ.get('ZHENXUN_RPC_URL')
    session_id = os.environ.get('ZHENXUN_SESSION_ID')

    if not rpc_url or not session_id:
        raise RuntimeError("未在沙箱环境变量中找到 RPC 链接信息")

    data = json.dumps({
        "session_id": session_id,
        "func_name": func_name,
        "kwargs": kwargs
    }).encode('utf-8')

    req = urllib.request.Request(
        rpc_url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        resp_dict = json.loads(response.read().decode('utf-8'))
        if "error" in resp_dict:
            raise RuntimeError(f"Host RPC Error: {resp_dict['error']}")
        return resp_dict.get("result")
"""


class SandboxRPCServer:
    """沙箱宿主 RPC 服务器，接受沙箱内 Python 脚本的 HTTP 调用。"""

    def __init__(self):
        self.app = web.Application()
        self.app.router.add_post("/rpc", self.handle_rpc)
        self.runner = None
        self.site = None
        self.port = 0
        self._routes: dict[str, dict[str, Callable]] = {}

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", 0)
        await self.site.start()
        server = getattr(self.site, "_server", None)
        sockets = getattr(server, "sockets", None)
        if sockets and len(sockets) > 0:
            self.port = sockets[0].getsockname()[1]
        else:
            logger.error("[SandboxRPC] 无法获取分配的端口！")
            self.port = 0
        logger.info(f"[SandboxRPC] 宿主 RPC 服务器已启动，监听端口: {self.port}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("[SandboxRPC] 宿主 RPC 服务器已关闭。")

    def register_route(self, session_id: str, func_name: str, func: Callable):
        if session_id not in self._routes:
            self._routes[session_id] = {}
        self._routes[session_id][func_name] = func

    def unregister_session(self, session_id: str):
        self._routes.pop(session_id, None)

    async def handle_rpc(self, request: web.Request):
        try:
            data = await request.json()
            session_id = data.get("session_id")
            func_name = data.get("func_name")
            kwargs = data.get("kwargs", {})

            func = self._routes.get(session_id, {}).get(func_name)
            if not func:
                return web.json_response(
                    {"error": f"找不到对应的授权函数: {func_name}"},
                    status=403,
                )

            if is_coroutine_callable(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            return web.json_response({"result": result})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


sandbox_rpc_server = SandboxRPCServer()

__all__ = ["STUB_TEMPLATE", "sandbox_rpc_server"]
