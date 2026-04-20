import asyncio
from typing import Any, Protocol, cast

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.decorators import silent, toolkit_tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.types.sandbox import (
    SandboxExecutionResult,
    SandboxSecurityProfile,
)
from zhenxun.services.ai.types.tools import ToolResult
from zhenxun.services.log import logger


class PythonPluginProtocol(Protocol):
    @property
    def supports_state(self) -> bool: ...

    async def execute(self, code: str, timeout: int = 30) -> SandboxExecutionResult: ...


class SandboxToolkit(BaseToolkit):
    default_instructions = (
        "## 🖥️ 沙箱工作区与终端交互\n"
        "系统为你提供了物理隔离的沙箱环境，支持高级数据分析和交互式终端操作。"
        "请严格区分以下两种场景：\n"
        "1. **数据分析与纯计算**：使用 `execute_python_code`。"
        "该环境支持自动处理依赖并能拦截绘图(如 matplotlib)，"
        "但**绝对不支持 `input()` 交互**。\n"
        "2. **交互式程序与长效服务**：如果你的代码包含 `input()` "
        "或需要启动 Web Server，你**必须**遵循以下流程：\n"
        "   - **第一步**：使用 `write_sandbox_file` 将代码写入文件（如 `/workspace/game.py`）。\n"
        "   - **第二步**：使用 `execute_terminal_command` 并设置 `is_interactive=True` 运行该文件。\n"
        "   - **后续交互**：调用此命令后，系统会分配虚拟终端并返回初始屏幕画面。你需要通过观察屏幕来决定下一步。使用 `send_sandbox_input` 填入所需数据（别忘了换行符 `\\n`），并使用 `read_sandbox_screen` 随时刷新屏幕。\n"
        "   - **⚠️ 重要警告：终端占用**\n"
        "     同一个终端只能运行一个前台程序！如果程序陷入死循环或报错，你**必须首先**调用 `interrupt_sandbox` 发送 Ctrl+C 中断它，然后才能修改代码重新执行！\n"
        "3. **文件操作**：使用 `write_sandbox_file` 和 `read_sandbox_file` "
        "管理沙箱内的文件。"
    )

    def __init__(self, profile: SandboxSecurityProfile | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.profile = profile or SandboxSecurityProfile()
        if "universal_python" not in self.profile.required_plugins:
            self.profile.required_plugins.append("universal_python")
        self._executors = {}
        self._interactive_sessions = {}

    async def enter_session(self, session_id: str, context: RunContext) -> None:
        from zhenxun.services.ai.sandbox.manager import sandbox_manager

        logger.info(f"[SandboxToolkit] 预热沙箱环境 (Session: {session_id})")
        self._executors[session_id] = await sandbox_manager.get_or_create_session(
            session_id, self.profile
        )

    async def exit_session(self, session_id: str) -> None:

        if session_id in self._executors:
            logger.debug(
                f"[SandboxToolkit] 当前 Agent 交互轮次结束，沙箱驻留后台 (Session: {session_id})"
            )
            self._executors.pop(session_id, None)
        session = self._interactive_sessions.pop(session_id, None)
        if session:
            try:
                await session.close()
            except Exception:
                pass

    @toolkit_tool(
        name="execute_python_code",
        description=(
            "在沙箱环境中执行 Python 代码。\n"
            "默认超时时间为 45 秒。如果你的代码需要更长的时间或等待用户输入，"
            "它将被挂在后台运行。\n"
            "你将收到目前的屏幕输出，并可以决定后续操作。"
        ),
    )
    async def execute_python_code(self, code: str, context: RunContext) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        logger.info(
            f"大模型请求执行 Python 代码 (Session: {session_id}, "
            f"长度: {len(code)} 字符)"
        )

        from zhenxun.services.ai.sandbox.manager import sandbox_manager
        from zhenxun.services.ai.sandbox.utils import ASTAnalyzer

        reqs = ASTAnalyzer.analyze_code_requirements(code)
        deps = reqs.dependencies.get("python_pip", [])

        logger.info(f"沙箱路由感知: 判定为 {reqs.tier.value} 级别任务。")
        if deps:
            logger.info(f"沙箱热注入: 自动提取到待安装依赖 -> {deps}")

        await context.emit("正在分析代码依赖并分配沙箱环境...")
        executor = await sandbox_manager.get_or_create_session(
            session_id, self.profile, reqs
        )
        self._executors[session_id] = executor

        python_plugin = None
        for p_name in [
            "universal_python",
            "jupyter_python",
            "e2b_python",
            "wasm_python",
            "basic_python",
        ]:
            try:
                plugin = executor.get_plugin(p_name)
                python_plugin = cast(PythonPluginProtocol, plugin)
                break
            except RuntimeError:
                continue

        if not python_plugin:
            return ToolResult(
                output="当前沙箱环境尚未挂载任何 Python 执行插件。", is_error=True
            )

        await context.emit("沙箱已就绪，正在后台执行代码...")
        system_notice = None

        if not python_plugin.supports_state:
            system_notice = (
                "> [!NOTE] **环境降级提示**\n"
                "> 当前沙箱处于轻量级降级模式，不支持 matplotlib 绘图，"
                "且上下文变量无法在多次工具调用间保存。"
                "请确保你的代码每次都能独立运行并使用 print 输出结果。"
            )

        try:
            result = await python_plugin.execute(code, timeout=45)
        except Exception as e:
            logger.error(f"沙箱执行框架异常: {e}")
            return ToolResult(
                output=f"System Error: 容器执行环境不可用，异常信息: {e}",
                display=f"❌ 沙箱框架异常: {e}",
                is_error=True,
                terminate_run=True,
            )

        output_parts = []
        if result.exit_code != 0 or result.stderr:
            output_parts.append(f"Exit Code: {result.exit_code}")

        if result.stdout:
            out_str = result.stdout.strip()
            if len(out_str) > 2000:
                out_str = out_str[:2000] + "\n...[输出过长，已被系统安全截断]..."
            output_parts.append(f"Stdout:\n{out_str}")

        if result.stderr:
            err_str = result.stderr.strip()
            if len(err_str) > 2000:
                err_str = err_str[:2000] + "\n...[输出过长，已被系统安全截断]..."
            output_parts.append(f"Stderr:\n{err_str}")

        if result.error:
            output_parts.append(f"System Error:\n{result.error}")

        output_text = "\n".join(output_parts)

        if result.is_timeout:
            system_notice = (system_notice or "") + (
                "\n\n### ⚠️ 系统最高级警告：代码执行发生软超时！\n"
                "目前该 Python 进程**仍在沙箱后台挂起运行**！它可能处于以下状态：\n"
                "1. 卡在 `input()` 等待标准输入。\n"
                "2. 陷入了死循环 (如 `while True`)。\n\n"
                "#### 🚀 下一步行动指南\n"
                "- **交互场景**：若它在等待输入，"
                "请立即调用 `send_sandbox_input` 填入所需数据。\n"
                "- **异常场景**：若属于死循环或报错，"
                "请**务必首先**调用 `interrupt_sandbox` "
                "强制杀死该进程，然后再尝试修改并重新执行代码！"
            )
        elif result.exit_code != 0 or result.error or result.stderr:
            system_notice = (system_notice or "") + (
                "\n\n### 💡 影子反思系统引导\n"
                "检测到代码执行失败或产生错误输出！\n"
                "请仔细分析上述日志，定位 Python 逻辑漏洞，"
                "修正代码后重新调用本工具执行。"
            )

        image_bytes_list = []

        if getattr(result, "images", None):
            import base64

            for b64_str in result.images:
                try:
                    image_bytes_list.append(base64.b64decode(b64_str))
                except Exception:
                    pass

        if getattr(result, "artifacts", None):
            for filename, file_bytes in result.artifacts.items():
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    image_bytes_list.append(file_bytes)

        if result.exit_code != 0 and not python_plugin.supports_state:
            system_notice = (
                (system_notice or "")
                + "\n(提示：发生错误可能是因为轻量级环境缺少依赖，"
                "或你的代码没有正确导入模块)"
            )

        if result.stderr and "StdinNotImplementedError" in result.stderr:
            system_notice = (
                (system_notice or "") + "\n\n### 🚨 致命错误：环境限制\n"
                "当前高级环境不支持交互式输入 `input()`！\n"
                "请按照系统指南：先用 `write_sandbox_file` "
                "将你的代码保存为 `.py` 文件，"
                "然后使用 `execute_terminal_command` 在真实终端中运行它！"
            )

        from nonebot_plugin_alconna import Image as AlcImage
        from nonebot_plugin_alconna import UniMessage

        from zhenxun.services.ai.types.messages import (
            ImagePart,
            LLMContentPart,
            TextPart,
        )

        final_output: list[LLMContentPart] = [TextPart(text=output_text.strip())]
        display_msg = UniMessage()
        has_display = False

        for img_bytes in image_bytes_list:
            display_msg += AlcImage(raw=img_bytes)
            final_output.append(ImagePart(raw=img_bytes))
            has_display = True

        return ToolResult(
            output=final_output if len(final_output) > 1 else output_text.strip(),
            display=display_msg if has_display else None,
            is_error=False,
            system_prompt_append=system_notice,
        )

    @toolkit_tool(
        name="execute_terminal_command",
        description=(
            "在沙箱的终端中执行 Shell 命令"
            "（例如 `python3 script.py` 或 `npm start`）。\n"
            "如果只是执行普通的短时非交互脚本，保持 is_interactive=False 即可（执行速度极快且稳定）。\n"
            "如果程序包含 `input()` 或需要长期驻留（如 Server），请务必设置 is_interactive=True 开启虚拟屏幕模式！"
        ),
    )
    async def execute_terminal_command(
        self, command: str, context: RunContext, is_interactive: bool = False
    ) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )

        from zhenxun.services.ai.sandbox.manager import sandbox_manager

        await context.emit(f"正在虚拟终端执行命令: {command} ...")
        executor = await sandbox_manager.get_or_create_session(session_id, self.profile)

        if not is_interactive:
            try:
                res = await executor.execute_raw_command(command)
            except NotImplementedError:
                return ToolResult(
                    output="当前沙箱环境不支持终端执行能力。", is_error=True
                )

            if getattr(res, "is_timeout", False):
                return ToolResult(
                    output=f"⚠️ 警告：命令执行发生软超时（进程被挂起）！\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}\n\n"
                    f"[系统引导]：这通常是因为你的代码包含 `input()` 或启动了持久化服务导致进程阻塞。\n"
                    f"由于你使用了 is_interactive=False，系统无法与挂起的进程交互！\n"
                    f"👉 请务必设置 `is_interactive=True` 重新调用本工具执行！",
                    is_error=True,
                )

            return ToolResult(
                output=f"Exit Code: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )

        session = self._interactive_sessions.get(session_id)
        if session:
            await session.close()

        try:
            interactive_session = await executor.create_pty_session()
        except NotImplementedError:
            return ToolResult(
                output="当前沙箱环境不支持交互式 PTY 终端。", is_error=True
            )

        self._interactive_sessions[session_id] = interactive_session

        try:
            await interactive_session.start(command)
            await asyncio.sleep(1.5)
            screen = await interactive_session.read_output()
            return ToolResult(
                output=f"已成功在虚拟终端启动程序。\n"
                f"📺 初始屏幕快照如下:\n```text\n{screen}\n```\n\n"
                f"请仔细阅读屏幕快照，如果程序在等待输入，请调用 `send_sandbox_input` 发送按键（记得带换行符）。"
            )
        except Exception as e:
            return ToolResult(output=f"虚拟屏幕启动异常: {e}", is_error=True)

    @toolkit_tool(
        name="send_sandbox_input",
        description=(
            "向当前沙箱中正在挂起运行的后台进程"
            "（如等待 input() 的 Python 脚本）发送输入文本。\n"
            "注意：你需要自己在文本末尾加上换行符 \\n 来模拟回车键。"
        ),
    )
    async def send_sandbox_input(self, text: str, context: RunContext) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        interactive_session = self._interactive_sessions.get(session_id)
        if not interactive_session:
            return ToolResult(
                output="错误：当前会话没有处于运行中的交互式虚拟屏幕！请先使用 execute_terminal_command(is_interactive=True) 启动程序。",
                is_error=True,
            )

        text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")

        await interactive_session.send_input(text)

        await asyncio.sleep(1.5)
        output = await interactive_session.read_output(timeout=5)

        return ToolResult(
            output=f"已发送按键。📺 屏幕刷新后快照如下：\n```text\n{output}\n```",
            display="⌨️ 已向后台进程发送输入",
        )

    @toolkit_tool(
        name="read_sandbox_screen",
        description=(
            "主动窥探并读取当前虚拟屏幕的画面。当你觉得后台程序可能已经渲染出新内容时，可以使用此工具。"
        ),
    )
    async def read_sandbox_screen(self, context: RunContext) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        interactive_session = self._interactive_sessions.get(session_id)
        if not interactive_session:
            return ToolResult(output="没有运行中的虚拟屏幕。", is_error=True)

        output = await interactive_session.read_output()
        return ToolResult(output=f"📺 当前屏幕快照:\n```text\n{output}\n```")

    @toolkit_tool(
        name="interrupt_sandbox",
        description=(
            "向当前沙箱发送 Ctrl+C (SIGINT) 信号，强制中断正在死循环或挂起的后台进程。"
        ),
    )
    async def interrupt_sandbox(self, context: RunContext) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        interactive_session = self._interactive_sessions.get(session_id)
        if not interactive_session:
            return ToolResult(output="没有运行中的虚拟屏幕需要中断。", is_error=True)

        await interactive_session.interrupt()
        await asyncio.sleep(1)
        output = await interactive_session.read_output()
        return ToolResult(
            output=(
                f"✅ 成功发送 Ctrl+C 中断信号。📺 当前屏幕快照：\n```text\n{output}\n```"
            ),
            display="🛑 已强制中断后台进程",
        )

    @toolkit_tool(
        name="write_sandbox_file",
        description="将文本内容写入沙箱文件系统中，支持保存大块数据或配置，避免超过对话上下文。",
    )
    @silent()
    async def write_sandbox_file(
        self, path: str, content: str, context: RunContext
    ) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        executor = self._executors.get(session_id)
        if not executor:
            from zhenxun.services.ai.sandbox.manager import sandbox_manager

            executor = await sandbox_manager.get_or_create_session(
                session_id, self.profile
            )

        success = await executor.write_raw_file(path, content)
        if success:
            return ToolResult(
                output=f"成功将内容写入文件: {path}",
                log_content=f"📝 已向沙箱写入文件: {path}",
            )
        else:
            return ToolResult(
                output="写入文件失败 (当前沙箱环境失联或不支持持久化IO)",
                display="❌ 写入文件失败：沙箱失联",
                is_error=True,
                terminate_run=True,
            )

    @toolkit_tool(
        name="read_sandbox_file",
        description="从沙箱文件系统中读取指定文件的文本内容。",
    )
    @silent()
    async def read_sandbox_file(self, path: str, context: RunContext) -> ToolResult:
        session_id = (
            context.session_id if context.session_id else "default_sandbox_session"
        )
        executor = self._executors.get(session_id)
        if not executor:
            from zhenxun.services.ai.sandbox.manager import sandbox_manager

            executor = await sandbox_manager.get_or_create_session(
                session_id, self.profile
            )

        try:
            content = await executor.read_raw_file(path)
            if content.startswith("Error:") or content.startswith("Failed to"):
                return ToolResult(output=content, is_error=True)
            return ToolResult(
                output=content,
                log_content=f"已读取沙箱文件 {path} (共 {len(content)} 字符)",
            )
        except Exception as e:
            return ToolResult(
                output=f"读取文件发生框架级异常: {e}",
                is_error=True,
                terminate_run=True,
            )
