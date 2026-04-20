import asyncio

import httpx

from zhenxun.services.ai.sandbox.extension import BaseSandboxPlugin
from zhenxun.services.log import logger

WORKER_CODE = """
import asyncio
import json
from aiohttp import web
from playwright.async_api import async_playwright

playwright_instance, browser, page = None, None, None

async def init_browser(app):
    global playwright_instance, browser, page
    playwright_instance = await async_playwright().start()
    browser = await playwright_instance.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
    page = await browser.new_page(viewport={'width': 1280, 'height': 800})

async def handle_ping(request):
    return web.json_response({"status": "pong"})

async def handle_goto(request):
    data = await request.json()
    try:
        await page.goto(data.get('url', 'about:blank'), wait_until="networkidle", timeout=15000)
    except Exception as e:
        pass
    return web.json_response({"status": "ok", "url": page.url})

async def handle_screenshot(request):
    data = await request.json()
    full_page = data.get('full_page', False)
    img_bytes = await page.screenshot(full_page=full_page, type='jpeg', quality=80)
    import base64
    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    return web.json_response({"status": "ok", "image_base64": b64_str})

async def handle_get_dom(request):
    title = await page.title()
    text = await page.evaluate("() => document.body.innerText")
    safe_text = text[:15000] if text else ""
    return web.json_response({"status": "ok", "title": title, "text": safe_text})

async def handle_click(request):
    data = await request.json()
    selector = data.get('selector')
    try:
        loc = page.locator(selector).first
        await loc.wait_for(state="visible", timeout=5000)
        await loc.click(timeout=3000)
        await page.wait_for_timeout(1000)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"status": "error", "msg": str(e)})

async def handle_type(request):
    data = await request.json()
    selector = data.get('selector')
    text = data.get('text', '')
    press_enter = data.get('press_enter', False)
    try:
        loc = page.locator(selector).first
        await loc.wait_for(state="visible", timeout=5000)
        await loc.fill(text, timeout=3000)
        if press_enter:
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(2000)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"status": "error", "msg": str(e)})

async def handle_scroll(request):
    data = await request.json()
    direction = data.get('direction', 'down')
    sign = 1 if direction == 'down' else -1
    await page.evaluate(f"window.scrollBy(0, window.innerHeight * {sign} * 0.8)")
    await page.wait_for_timeout(500)
    return web.json_response({"status": "ok"})

app = web.Application()
app.on_startup.append(init_browser)
app.router.add_get('/ping', handle_ping)
app.router.add_post('/goto', handle_goto)
app.router.add_post('/screenshot', handle_screenshot)
app.router.add_post('/get_dom', handle_get_dom)
app.router.add_post('/click', handle_click)
app.router.add_post('/type', handle_type)
app.router.add_post('/scroll', handle_scroll)

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
    web.run_app(app, host='0.0.0.0', port=port)
"""


class PlaywrightBrowserPlugin(BaseSandboxPlugin):
    """原生沙箱内置无头浏览器插件"""

    @property
    def plugin_name(self) -> str:
        return "playwright_browser"

    def __init__(self, channel):
        super().__init__(channel)
        self._http_client = httpx.AsyncClient(timeout=30)
        self.base_url = ""

    async def on_mount(self) -> None:
        from zhenxun.services.ai.sandbox.extension import (
            SupportsCommandExecution,
            SupportsFileSystem,
            SupportsPortMapping,
        )

        if (
            not isinstance(self.channel, SupportsPortMapping)
            or not isinstance(self.channel, SupportsCommandExecution)
            or not isinstance(self.channel, SupportsFileSystem)
        ):
            raise RuntimeError("底层沙箱缺少必要的能力协议，无法挂载浏览器插件。")

        await super().on_mount()

        browser_port = self.channel.get_meta("browser_port")
        if not browser_port:
            raise RuntimeError("底层沙箱未暴露浏览器端口，无法挂载此插件。")

        self.base_url = f"http://127.0.0.1:{browser_port}"

        logger.info("[BrowserPlugin] 正在检测沙箱浏览器依赖...")

        probe_pw = await self.channel.execute_raw_command(
            "python3 -c 'import playwright'"
        )
        probe_aio = await self.channel.execute_raw_command(
            "python3 -c 'import aiohttp'"
        )

        if probe_pw.exit_code != 0 or probe_aio.exit_code != 0:
            logger.info("[BrowserPlugin] 探测到缺少依赖，准备进行热修复...")

            if probe_aio.exit_code != 0:
                await self.channel.execute_raw_command(
                    "pip install aiohttp", timeout=120
                )

            if probe_pw.exit_code != 0:
                logger.info("[BrowserPlugin] 安装 Playwright 及 Chromium (耗时较长)...")
                pip_res = await self.channel.execute_raw_command(
                    "pip install playwright", timeout=300
                )
                if pip_res.exit_code != 0:
                    raise RuntimeError(f"安装 Playwright 失败: {pip_res.stderr}")

                pw_res = await self.channel.execute_raw_command(
                    "playwright install chromium --with-deps", timeout=600
                )
                if pw_res.exit_code != 0:
                    raise RuntimeError(f"安装 Chromium 内核失败: {pw_res.stderr}")

        worker_path = "/opt/zhenxun/browser_worker.py"
        await self.channel.write_raw_file(worker_path, WORKER_CODE)

        logger.info("[BrowserPlugin] 正在启动无头浏览器微服务...")
        await self.channel.execute_raw_command(
            f"nohup python3 {worker_path} {browser_port} > /opt/zhenxun/browser.log 2>&1 &"
        )

        for _ in range(20):
            try:
                resp = await self._http_client.get(f"{self.base_url}/ping", timeout=2)
                if resp.status_code == 200:
                    logger.info("[BrowserPlugin] 浏览器微服务启动成功并已连通！")
                    return
            except Exception:
                pass
            await asyncio.sleep(1)

        raise RuntimeError("沙箱内浏览器微服务启动超时！")

    async def on_unmount(self) -> None:
        await self._http_client.aclose()

    async def ping(self) -> bool:
        resp = await self._http_client.get(f"{self.base_url}/ping")
        return resp.status_code == 200

    async def goto(self, url: str) -> bool:
        resp = await self._http_client.post(f"{self.base_url}/goto", json={"url": url})
        return resp.status_code == 200

    async def screenshot(self, full_page: bool = False) -> str | None:
        resp = await self._http_client.post(
            f"{self.base_url}/screenshot", json={"full_page": full_page}, timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("image_base64")
        return None

    async def get_dom(self) -> str:
        resp = await self._http_client.post(f"{self.base_url}/get_dom")
        data = resp.json()
        return f"Title: {data.get('title')}\n\nContent:\n{data.get('text')}"

    async def click(self, selector: str) -> bool:
        resp = await self._http_client.post(
            f"{self.base_url}/click", json={"selector": selector}, timeout=10
        )
        data = resp.json()
        if data.get("status") != "ok":
            logger.warning(f"[BrowserPlugin] Click 失败: {data.get('msg')}")
            return False
        return True

    async def type_text(
        self, selector: str, text: str, press_enter: bool = False
    ) -> bool:
        payload = {"selector": selector, "text": text, "press_enter": press_enter}
        resp = await self._http_client.post(
            f"{self.base_url}/type", json=payload, timeout=15
        )
        return resp.json().get("status") == "ok"

    async def scroll(self, direction: str = "down") -> bool:
        resp = await self._http_client.post(
            f"{self.base_url}/scroll", json={"direction": direction}, timeout=10
        )
        return resp.json().get("status") == "ok"
