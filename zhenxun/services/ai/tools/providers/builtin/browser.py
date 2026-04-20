import base64
from typing import Any, cast

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.decorators import toolkit_tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.types.sandbox import SandboxSecurityProfile
from zhenxun.services.ai.types.tools import ToolResult


class WebBrowserToolkit(BaseToolkit):
    """
    网页浏览器工具箱。
    赋予大模型原生的网页浏览、点击、输入能力，每次操作自动回传最新的页面截图。
    """

    default_instructions = (
        "## 网页浏览器\n"
        "你拥有控制无头浏览器的能力，可以实时访问并驱动网页。工作流程如下：\n"
        "1. **导航**：使用 `open_url` 访问网页。你会收到页面的可见文本摘要和实时截图。\n"
        "2. **视觉决策**：仔细核对截图和文本。通过 `click_element` 进行交互。\n"
        "   - 关键选择器：务必使用文字选择器 (Text Selector)，如 `text='登录'` 或 `button:has-text('提交')`。\n"
        "   - 输入定位：输入框推荐使用 `input[type='search']` 或 `input[type='text']` 或者它的 placeholder 属性。\n"
        "3. **输入与回车**：使用 `type_text` 工具输入内容。如果这是一个搜索框，你可以直接设置 `press_enter=True` 来直接触发搜索，省去再点一次按钮的麻烦。\n"
        "4. **滚动翻页**：如果页面内容很长，需要查看下方内容，使用 `scroll_page` 工具。"
    )

    def __init__(self, profile: SandboxSecurityProfile | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.profile = profile or SandboxSecurityProfile(
            sandbox_type="docker",
            enable_network=True,
            needs_state=True,
            required_plugins=["playwright_browser"],
        )

    async def _get_browser(self, context: RunContext | None):
        from zhenxun.services.ai.sandbox.manager import sandbox_manager
        from zhenxun.services.ai.sandbox.plugins.browser import PlaywrightBrowserPlugin

        session_id = (context.session_id if context else None) or "default_browser_session"
        executor = await sandbox_manager.get_or_create_session(session_id, self.profile)
        return cast(PlaywrightBrowserPlugin, executor.get_plugin("playwright_browser"))

    async def _get_page_state(
        self, browser_plugin: Any, action_desc: str, is_error: bool = False
    ) -> ToolResult:
        """通用的状态获取方法，返回当前页面的 DOM 文本和截图给大模型"""
        dom_text = await browser_plugin.get_dom()
        b64_img = await browser_plugin.screenshot(full_page=False)

        img_bytes = base64.b64decode(b64_img) if b64_img else b""

        icon = "❌" if is_error else "✅"
        from zhenxun.services.ai.tools.core.response import ToolResponse
        res = ToolResponse.reply(
            output=f"{action_desc}\n\n当前提取的可见文本摘要:\n{dom_text}",
            display=f"{icon} {action_desc}。",
            image=img_bytes if img_bytes else None,
        )
        res.log_content = f"浏览器操作完成: {action_desc}"
        res.is_error = is_error
        return res

    @toolkit_tool(
        name="open_url",
        description="在浏览器中打开指定的 URL。返回加载后的页面文本和实时截图。",
    )
    async def open_url(self, url: str, context: RunContext) -> ToolResult:
        if not url.startswith("http"):
            url = f"https://{url}"

        await context.emit(f"正在启动无头浏览器并导航至 {url}...")
        browser = await self._get_browser(context)
        await browser.goto(url)
        await context.emit("页面加载完成，正在捕获视图快照...")
        return await self._get_page_state(browser, f"已访问 {url}")

    @toolkit_tool(
        name="click_element",
        description="点击页面上的元素。请基于你看到的图片内容，严格使用文本选择器，例如 `text='提交'`。",
    )
    async def click_element(self, selector: str, context: RunContext) -> ToolResult:
        await context.emit(f"尝试点击页面元素: {selector}...")
        browser = await self._get_browser(context)
        success = await browser.click(selector)
        if not success:
            return await self._get_page_state(
                browser,
                f"点击失败：无法定位可见元素 '{selector}'。请观察当前截图，检查是否出现了弹窗遮挡，或者元素需要滚动才能看见。",
                is_error=True,
            )
        return await self._get_page_state(browser, f"已点击 '{selector}'")

    @toolkit_tool(
        name="type_text",
        description="在指定的输入框中键入文本。如果 press_enter 为 true，输入完毕后会自动按下回车键。",
    )
    async def type_text(
        self,
        selector: str,
        text: str,
        press_enter: bool = False,
        context: RunContext | None = None,
    ) -> ToolResult:
        if context:
            await context.emit(f"尝试在 '{selector}' 处输入文本...")
        browser = await self._get_browser(context)
        success = await browser.type_text(selector, text, press_enter)
        if not success:
            return await self._get_page_state(
                browser,
                f"输入失败：无法定位可见元素 '{selector}'。请检查截图，页面是否已经变成移动版或者出现了 Cookie 弹窗遮挡？",
                is_error=True,
            )
        return await self._get_page_state(browser, f"已在 '{selector}' 处输入文本")

    @toolkit_tool(
        name="scroll_page",
        description="向下或向上滚动页面。direction 取值 'down' 或 'up'。",
    )
    async def scroll_page(
        self, direction: str = "down", context: RunContext | None = None
    ) -> ToolResult:
        if context:
            await context.emit(f"正在向{direction}滚动页面...")
        browser = await self._get_browser(context)
        success = await browser.scroll(direction)
        if not success:
            return ToolResult(output="滚动失败。", is_error=True)
        return await self._get_page_state(browser, f"已向{direction}滚动页面")

