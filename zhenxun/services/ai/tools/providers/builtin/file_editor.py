import json
from typing import Any, Literal, cast

from pydantic import BaseModel

from zhenxun.services.ai.llm.api import generate_structured
from zhenxun.services.ai.run import Inject, RunContext
from zhenxun.services.ai.sandbox.models import SandboxSecurityProfile
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolErrorResult, ToolErrorType, ToolResult
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump


class FileEditorEngine:
    """
    纯逻辑 Diff 文件编辑引擎。
    完全无状态，操作字符串，负责提供安全的视图、替换和插入逻辑。
    """

    @staticmethod
    def view(content: str, start_line: int = 1, end_line: int = -1) -> str:
        """
        获取带有行号的文件视图。
        格式如:
          12 | def foo():
          13 |     pass
        """
        if not content:
            return "[文件为空]"

        lines = content.replace("\r\n", "\n").split("\n")
        total_lines = len(lines)

        start_line = max(1, start_line)
        end_line = total_lines if end_line == -1 else min(total_lines, end_line)

        if start_line > total_lines:
            return f"[起始行号 {start_line} 超出文件总行数 {total_lines}]"

        output = []
        for i in range(start_line - 1, end_line):
            output.append(f"{i + 1: >4} | {lines[i]}")

        return "\n".join(output)

    @staticmethod
    def replace(content: str, old_str: str, new_str: str) -> str:
        """
        安全替换文件内容。
        严格要求 old_str 必须在文本中 [唯一匹配]。
        """
        if not old_str:
            raise ValueError("替换失败：old_str 不能为空。")

        content = content.replace("\r\n", "\n")
        old_str = old_str.replace("\r\n", "\n")
        new_str = new_str.replace("\r\n", "\n")

        occurrences = content.count(old_str)

        if occurrences == 0:
            raise ValueError(
                "替换失败：在文件中未找到指定的 old_str。\n"
                "提示：请确保你提供的 old_str 的缩进、空格和换行与源文件 [完全一致]。"
            )
        elif occurrences > 1:
            raise ValueError(
                f"替换失败：在文件中找到了 {occurrences} 处匹配的 old_str。\n"
                "提示：请提供更多的上下文（包括前后的完整代码行），以确保只匹配到你要修改的那一处。"
            )

        return content.replace(old_str, new_str)

    @staticmethod
    def insert(content: str, line_number: int, new_str: str) -> str:
        """
        在指定的行号之后插入新内容。
        line_number = 0 表示插入在文件最开头。
        line_number = 1 表示插入在第1行之后。
        """
        content = content.replace("\r\n", "\n")
        lines = content.split("\n") if content else []
        total_lines = len(lines) if content else 0

        if line_number < 0 or line_number > total_lines:
            raise ValueError(
                f"插入失败：行号 {line_number} 超出文件范围 (0-{total_lines})。"
            )

        new_str_lines = new_str.replace("\r\n", "\n").split("\n")

        lines[line_number:line_number] = new_str_lines

        return "\n".join(lines)


class EditReflexion(BaseModel):
    command: Literal["view", "create", "replace", "insert", "undo"]
    path: str
    old_str: str = ""
    new_str: str = ""
    insert_line: int = -1
    start_line: int = 1
    end_line: int = -1


class FileEditorToolkit(BaseToolkit):
    """
    基于 Diff 的智能文件编辑器工具箱。
    赋予大模型局部修改大文件的能力，避免全量覆写导致的 Token 爆炸和幻觉错位。
    """

    default_instructions = (
        "## 智能文件编辑器\n"
        "你拥有高级的文件编辑权限。要求在修改文件时，**严禁**全量覆写。请遵循以下工作流：\n"
        "1. **预览**：使用 `view` 指令确认文件内容和准确行号。\n"
        "2. **精准替换**：使用 `replace` 命令。`old_str` 必须与原始内容完全一致（包括缩进和空格）。\n"  # noqa: E501
        "3. **行后插入**：使用 `insert` 命令在指定行号后新增内容。\n"
        "4. **自愈与撤销**：新文件使用 `create`，若修改引发语法错误，系统会自动回滚，请根据报错修正。"  # noqa: E501
    )

    def __init__(self, profile: SandboxSecurityProfile | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.profile = profile or SandboxSecurityProfile()
        self._history: dict[str, dict[str, str]] = {}

    async def _get_executor(self, context: RunContext):
        from zhenxun.services.ai.sandbox.manager import sandbox_manager

        session_id = context.session_id or "default_editor_session"
        return await sandbox_manager.get_or_create_session(session_id, self.profile)

    def _save_history(self, session_id: str, path: str, content: str):
        if session_id not in self._history:
            self._history[session_id] = {}
        self._history[session_id][path] = content

    async def _lint_and_rollback_if_failed(
        self, executor, path: str, old_content: str, command: str
    ) -> None:
        """对修改后的文件进行语法检查，如果失败则回滚并抛出异常供影子闭环反思"""
        if not path.endswith(".py"):
            return

        from zhenxun.services.ai.sandbox.extension import (
            SupportsCommandExecution,
            SupportsFileSystem,
        )

        cmd_exec = cast(SupportsCommandExecution, executor)
        fs_exec = cast(SupportsFileSystem, executor)

        try:
            check_res = await cmd_exec.execute_raw_command(
                f"python3 -m py_compile {path}"
            )
        except NotImplementedError:
            return

        if check_res.exit_code != 0:
            if command == "create":
                await fs_exec.delete_raw_file(path)
            else:
                await fs_exec.write_raw_file(path, old_content)
            error_msg = check_res.stderr or check_res.stdout
            raise ValueError(
                f"你提交的修改引入了致命的 Python 语法错误，系统已自动回滚了你的修改！\n请仔细阅读以下 Linter 错误信息，并在下一次工具调用中修正你的代码：\n{error_msg}"  # noqa: E501
            )

    @tool(
        name="edit_file",
        description="高级文件编辑工具。支持 view(查看), create(创建), replace(精准替换), insert(按行号插入), undo(撤销)。",  # noqa: E501
    )
    async def edit_file(
        self,
        command: Literal["view", "create", "replace", "insert", "undo"],
        path: str,
        context: RunContext,
        ui: Inject.UI,
        old_str: str = "",
        new_str: str = "",
        insert_line: int = -1,
        start_line: int = 1,
        end_line: int = -1,
    ) -> ToolResult:
        current_args = {
            "command": command,
            "path": path,
            "old_str": old_str,
            "new_str": new_str,
            "insert_line": insert_line,
            "start_line": start_line,
            "end_line": end_line,
        }

        await ui.send_text(f"📝 正在通过沙箱执行文件操作: {command} {path}...")
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return await self._edit_file_impl(context=context, **current_args)
            except ValueError as ve:
                if attempt == max_retries:
                    err_result = ToolErrorResult(
                        error_type=ToolErrorType.INVALID_ARGUMENTS,
                        message=str(ve),
                        is_retryable=True,
                    )
                    return (
                        ToolResult(
                            output=json.dumps(
                                model_dump(err_result), ensure_ascii=False
                            )
                        )
                        .show_to_user(f"❌ 编辑失败: {ve}")
                        .as_error()
                    )

                logger.info(f"📝 触发影子反思闭环 (第 {attempt + 1} 次重试): {ve}")

                prompt = (
                    f"你在尝试使用 `edit_file` 工具修改文件时发生了错误。\n"
                    f"【你传入的参数】\n"
                    f"{json.dumps(current_args, ensure_ascii=False, indent=2)}\n\n"
                    f"【系统抛出的错误】\n"
                    f"{ve}\n\n"
                    f"请深度反思这个错误。如果是 Linter 语法错误，请检查你的缩进、变量名和标点符号。\n"  # noqa: E501
                    f"如果是 replace 匹配失败，请注意 `old_str` 必须与源文件一字不差。\n"  # noqa: E501
                    f"⚠️ 警告：你必须且只能修正 `new_str` 等内容参数，你的 `command` 参数必须严格保持为 `{command}`，绝对不允许擅自改为 view 等其他指令！\n"  # noqa: E501
                    f"请重新输出修正后的完整参数。"
                )

                try:
                    fixed_args = await generate_structured(
                        prompt, response_model=EditReflexion
                    )
                    current_args = model_dump(fixed_args)
                except Exception as llm_e:
                    logger.error(f"影子反思 LLM 请求失败: {llm_e}")
                    return ToolResult(output=str(ve)).as_error()

        return ToolResult(output="未知错误").as_error()

    async def _edit_file_impl(
        self,
        command: str,
        path: str,
        context: RunContext,
        old_str: str = "",
        new_str: str = "",
        insert_line: int = -1,
        start_line: int = 1,
        end_line: int = -1,
    ) -> ToolResult:
        executor = await self._get_executor(context)
        session_id = context.session_id or "default_editor_session"

        from zhenxun.services.ai.sandbox.extension import SupportsFileSystem

        fs_executor = cast(SupportsFileSystem, executor)

        content = ""
        if command != "create":
            read_res = await fs_executor.read_raw_file(path)
            if read_res.startswith("Error: File") or read_res.startswith("Failed to"):
                if command == "view":
                    raise ValueError(f"文件不存在: {path}")
                elif command == "undo":
                    pass
                else:
                    raise ValueError(
                        f"操作失败：文件 {path} 不存在。请先使用 create 命令创建它。"
                    )
            else:
                content = read_res

        if command == "view":
            view_res = FileEditorEngine.view(content, start_line, end_line)
            return ToolResult(output=f"文件 {path} 的视图如下:\n{view_res}").with_log(
                f"👀 查看文件 {path}"
            )

        elif command == "create":
            check_res = await fs_executor.read_raw_file(path)
            if not (
                check_res.startswith("Error: File") or check_res.startswith("Failed to")
            ):
                raise ValueError(
                    f"创建失败：文件 {path} 已存在。如果你想修改它，请使用 replace 或 insert。"  # noqa: E501
                )

            if not new_str:
                raise ValueError("创建失败：new_str 不能为空。")

            await fs_executor.write_raw_file(path, new_str)
            self._save_history(session_id, path, "")

            await self._lint_and_rollback_if_failed(executor, path, "", "create")

            return ToolResult(output=f"✅ 文件 {path} 创建成功！").with_log(
                f"📝 创建文件 {path}"
            )

        elif command == "replace":
            if not old_str:
                raise ValueError("替换失败：old_str 不能为空。")

            new_content = FileEditorEngine.replace(content, old_str, new_str)

            self._save_history(session_id, path, content)
            await fs_executor.write_raw_file(path, new_content)

            await self._lint_and_rollback_if_failed(executor, path, content, "replace")

            return ToolResult(
                output=f"✅ 文件 {path} 局部替换成功！你可以使用 view 命令检查修改后的结果。"
            ).with_log(f"🔄 替换文件 {path} 内容")

        elif command == "insert":
            if insert_line < 0:
                raise ValueError("插入失败：请提供有效的 insert_line 行号。")
            if not new_str:
                raise ValueError("插入失败：new_str 不能为空。")

            new_content = FileEditorEngine.insert(content, insert_line, new_str)

            self._save_history(session_id, path, content)
            await fs_executor.write_raw_file(path, new_content)

            await self._lint_and_rollback_if_failed(executor, path, content, "insert")

            return ToolResult(
                output=f"✅ 成功在 {path} 的第 {insert_line} 行之后插入了代码！"
            ).with_log(f"➕ 在 {path} 插入内容")

        elif command == "undo":
            old_content = self._history.get(session_id, {}).get(path)
            if old_content is None:
                raise ValueError(f"撤销失败：找不到 {path} 的历史修改记录。")

            await fs_executor.write_raw_file(path, old_content)
            return ToolResult(
                output=f"✅ 文件 {path} 已成功回滚到上一次的状态！"
            ).with_log(f"⏪ 撤销文件 {path} 的修改")

        else:
            raise ValueError(f"未知指令: {command}")
