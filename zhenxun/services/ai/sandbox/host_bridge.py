from collections.abc import Callable
from dataclasses import dataclass
import inspect
import textwrap
from textwrap import indent

from aiohttp import web
from nonebot.utils import is_coroutine_callable

from zhenxun.services.log import logger


@dataclass(frozen=True)
class Alias:
    """导入别名结构，如 import pandas as pd"""

    name: str
    alias: str


@dataclass(frozen=True)
class ImportFromModule:
    """从模块中导入，如 from datetime import datetime"""

    module: str
    imports: list[str | Alias]


ImportType = str | Alias | ImportFromModule


def sandbox_function(
    python_packages: list[str] | None = None,
    global_imports: list[ImportType] | None = None,
    host_callable: bool = False,
):
    def decorator(func: Callable):
        setattr(func, "__sandbox_function__", True)
        setattr(func, "__sandbox_python_packages__", python_packages or [])
        setattr(func, "__sandbox_global_imports__", global_imports or [])
        setattr(func, "__sandbox_host_callable__", host_callable)
        return func

    return decorator


def _import_to_str(im: ImportType) -> str:
    if isinstance(im, str):
        if im.startswith("import ") or im.startswith("from "):
            return im
        return f"import {im}"
    elif isinstance(im, Alias):
        return f"import {im.name} as {im.alias}"
    elif isinstance(im, ImportFromModule):
        parts = []
        for i in im.imports:
            if isinstance(i, str):
                parts.append(i)
            else:
                parts.append(f"{i.name} as {i.alias}")
        return f"from {im.module} import {', '.join(parts)}"
    return ""


def to_stub(func: Callable) -> str:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func)
    stub = f"def {func.__name__}{sig}:\n"
    if doc:
        doc_str = indent(f'"""\n{doc}\n"""', "    ")
        stub += doc_str + "\n"
    stub += "    ...\n"
    return stub


def build_python_functions_file(funcs: list[Callable]) -> str:
    global_imports_map = {}
    code_blocks = []

    for func in funcs:
        if not getattr(func, "__sandbox_function__", False):
            continue

        imports = getattr(func, "__sandbox_global_imports__", [])
        for im in imports:
            im_str = _import_to_str(im)
            if im_str:
                global_imports_map[im_str] = None

        is_host_callable = getattr(func, "__sandbox_host_callable__", False)

        if is_host_callable:
            sig = inspect.signature(func)
            params = []
            for name, p in sig.parameters.items():
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    params.append(f"{name}={name}")
            kwargs_str = ", ".join(params)

            stub_code = f"def {func.__name__}{sig}:\n"
            if func.__doc__:
                doc_str = textwrap.indent(f'"""\n{func.__doc__}\n"""', "    ")
                stub_code += doc_str + "\n"

            stub_code += "    from zhenxun_stub import invoke_host\n"
            stub_code += f"    return invoke_host('{func.__name__}', {kwargs_str})\n"
            code_blocks.append(stub_code)
        else:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            lines = source.split("\n")
            while lines:
                if lines[0].strip().startswith("def ") or lines[0].strip().startswith(
                    "async def "
                ):
                    break
                lines.pop(0)
            code_blocks.append("\n".join(lines))

    content = ""
    if global_imports_map:
        content += "\n".join(global_imports_map.keys()) + "\n\n"

    content += "from zhenxun_stub import emit_event\n\n"
    content += "\n\n".join(code_blocks) + "\n"
    return content


STUB_TEMPLATE = """
import os
import json
import urllib.request

def emit_event(event_name, data):
    '''主动向宿主发射事件'''
    rpc_url = os.environ.get('ZHENXUN_RPC_URL')
    session_id = os.environ.get('ZHENXUN_SESSION_ID')
    if not rpc_url: return
    
    event_url = rpc_url.replace('/rpc', '/event')
    payload = json.dumps({"session_id": session_id, "event": event_name, "data": data}).encode('utf-8')
    req = urllib.request.Request(event_url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception:
        pass

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
    response = urllib.request.urlopen(req)
    
    # [Phase 3] 支持基于 NDJSON 的流式响应 (AsyncGenerator)
    if response.headers.get('Content-Type') == 'application/x-ndjson':
        def stream_generator():
            for line in response:
                if not line.strip(): continue
                item = json.loads(line.decode('utf-8'))
                if "error" in item: raise RuntimeError(item["error"])
                if item.get("done"): break
                yield item.get("yield")
        return stream_generator()
    else:
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
        self.app.router.add_post("/event", self.handle_event)
        self.runner = None
        self.site = None
        self.port = 0
        self._routes: dict[str, dict[str, Callable]] = {}
        self._event_handlers: dict[str, Callable] = {}

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

    def register_event_handler(self, session_id: str, handler: Callable):
        self._event_handlers[session_id] = handler

    def unregister_session(self, session_id: str):
        self._routes.pop(session_id, None)
        self._event_handlers.pop(session_id, None)

    async def handle_event(self, request: web.Request):
        try:
            data = await request.json()
            session_id = data.get("session_id")
            handler = self._event_handlers.get(session_id)
            if handler:
                import asyncio

                if asyncio.iscoroutinefunction(handler):
                    await handler(data.get("event"), data.get("data"))
                else:
                    handler(data.get("event"), data.get("data"))
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

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

            if inspect.isasyncgenfunction(func):
                import json

                response = web.StreamResponse(
                    headers={"Content-Type": "application/x-ndjson"}
                )
                await response.prepare(request)
                try:
                    async for item in func(**kwargs):
                        await response.write(
                            json.dumps({"yield": item}).encode("utf-8") + b"\n"
                        )
                    await response.write(
                        json.dumps({"done": True}).encode("utf-8") + b"\n"
                    )
                except Exception as e:
                    await response.write(
                        json.dumps({"error": str(e)}).encode("utf-8") + b"\n"
                    )
                return response

            if is_coroutine_callable(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            return web.json_response({"result": result})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


sandbox_rpc_server = SandboxRPCServer()

__all__ = [
    "STUB_TEMPLATE",
    "Alias",
    "ImportFromModule",
    "build_python_functions_file",
    "sandbox_function",
    "sandbox_rpc_server",
    "to_stub",
]
