from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.core.exceptions import ToolRetryError
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.sandbox.models import SandboxSecurityProfile
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger


class PatchError(Exception):
    pass


class PatchOperation(BaseModel):
    op_type: Literal["update", "add"]
    path: str
    hunks: list[dict[str, list[str]]] = Field(default_factory=list)
    new_content: list[str] = Field(default_factory=list)


class PatchParser:
    """
    纯文本的 Patch 解析器，不依赖第三方库。
    支持解析类似 OpenAI Sandbox 的高宽容度 Diff 语法。
    """

    @classmethod
    def parse(cls, text: str) -> list[PatchOperation]:
        lines = text.splitlines()
        operations: list[PatchOperation] = []
        current_op: PatchOperation | None = None
        current_hunk: dict[str, list[str]] | None = None
        in_patch = False

        for line in lines:
            stripped = line.strip()
            if stripped == "*** Begin Patch":
                in_patch = True
                continue
            if stripped == "*** End Patch":
                in_patch = False
                if current_op:
                    if current_hunk and (
                        current_hunk["search"] or current_hunk["replace"]
                    ):
                        current_op.hunks.append(current_hunk)
                    operations.append(current_op)
                    current_op = None
                continue

            if not in_patch:
                continue

            if line.startswith("*** Update File: "):
                if current_op:
                    if current_hunk and (
                        current_hunk["search"] or current_hunk["replace"]
                    ):
                        current_op.hunks.append(current_hunk)
                    operations.append(current_op)
                current_op = PatchOperation(op_type="update", path=line[17:].strip())
                current_hunk = None
                continue

            if line.startswith("*** Add File: "):
                if current_op:
                    if current_hunk and (
                        current_hunk["search"] or current_hunk["replace"]
                    ):
                        current_op.hunks.append(current_hunk)
                    operations.append(current_op)
                current_op = PatchOperation(op_type="add", path=line[14:].strip())
                current_hunk = None
                continue

            if not current_op:
                continue

            if current_op.op_type == "add":
                if line.startswith("+"):
                    current_op.new_content.append(line[1:])
                continue

            if current_op.op_type == "update":
                if line.startswith("@@"):
                    if current_hunk and (
                        current_hunk["search"] or current_hunk["replace"]
                    ):
                        current_op.hunks.append(current_hunk)
                    current_hunk = {"search": [], "replace": []}
                    continue

                if current_hunk is not None:
                    if line.startswith("-"):
                        current_hunk["search"].append(line[1:])
                    elif line.startswith("+"):
                        current_hunk["replace"].append(line[1:])
                    elif line.startswith(" ") or line == "":
                        content = line[1:] if line.startswith(" ") else line
                        current_hunk["search"].append(content)
                        current_hunk["replace"].append(content)
                    else:
                        current_hunk["search"].append(line)
                        current_hunk["replace"].append(line)

        if current_op and current_op not in operations:
            if current_hunk and (current_hunk["search"] or current_hunk["replace"]):
                current_op.hunks.append(current_hunk)
            operations.append(current_op)

        return operations


class PatchApplier:
    """执行具体的补丁合并操作，带容错搜索"""

    @classmethod
    def apply_diff(
        cls, original: str, search_lines: list[str], replace_lines: list[str]
    ) -> str:
        orig_lines = original.splitlines()
        search_len = len(search_lines)

        if search_len == 0:
            raise PatchError("无效的 Hunk：没有提供需要匹配的上下文或删除行。")

        matches = []
        for i in range(len(orig_lines) - search_len + 1):
            if orig_lines[i : i + search_len] == search_lines:
                matches.append(i)

        if not matches:
            stripped_search = [s.rstrip() for s in search_lines]
            for i in range(len(orig_lines) - search_len + 1):
                if [
                    s.rstrip() for s in orig_lines[i : i + search_len]
                ] == stripped_search:
                    matches.append(i)

        if len(matches) == 0:
            err_context = "\n".join(search_lines)
            raise PatchError(
                f"匹配失败：无法在源文件中找到以下代码块。请确保你提供了充分的上下文行，且缩进一致：\n{err_context}"
            )
        if len(matches) > 1:
            raise PatchError(
                f"匹配歧义：找到了 {len(matches)} 处相同的代码块。请在 @@ Hunk 中提供更多的上下文行以便唯一定位。"
            )

        idx = matches[0]
        new_lines = orig_lines[:idx] + replace_lines + orig_lines[idx + search_len :]
        return "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")


class FileEditorToolkit(BaseToolkit):
    """
    智能文件编辑器工具箱。
    支持高级的、专为大模型设计的 Diff 语法，彻底消除全量覆写导致的代码幻觉。
    """

    default_instructions = (
        "## 智能文件编辑器 (apply_patch)\n"
        "你拥有修改沙箱文件的权限。当需要修改或创建文件时，你**必须**使用 `apply_patch` 工具。\n"
        "该工具使用一种简化、高宽容度的 Diff 语法。由于它是 FREEFORM 的字符串工具，"
        "请直接将补丁作为长字符串参数传入，不要额外包装在 JSON 键里。\n\n"
        "【语法规范】：\n"
        "*** Begin Patch\n"
        "*** Update File: src/main.py\n"
        "@@\n"
        " 保持不变的上下文行\n"
        "-需要删除的旧代码\n"
        "+需要插入的新代码\n"
        " 保持不变的上下文行\n"
        "*** End Patch\n\n"
        "【要求】：\n"
        "1. 支持的操作有 `*** Update File: <path>` 和 `*** Add File: <path>`。\n"
        "2. 对于 Update，必须提供至少 1-2 行的上下文（以空格开头），以便系统精确定位你要修改的位置。\n"
        "3. 新增行必须以 `+` 开头，删除行以 `-` 开头。\n"
        "4. 如果要创建新文件，使用 `*** Add File: <path>`，后面的每一行都以 `+` 开头。\n"
        "5. 一次调用可以包含多个 `*** Update File:` 或 `@@` 块。"
    )

    def __init__(self, profile: SandboxSecurityProfile | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.profile = profile or SandboxSecurityProfile()

    async def _get_executor(self, context: RunContext):
        from zhenxun.services.ai.sandbox.manager import sandbox_manager

        session_id = context.session_id or "default_editor_session"
        return await sandbox_manager.get_or_create_session(session_id, self.profile)

    @tool(
        name="apply_patch",
        description="基于简化 Diff 语法的多文件精确编辑器。传入复合语法的纯文本补丁包即可执行创建、修改等操作。",
    )
    async def apply_patch(
        self,
        patch_content: str,
        context: RunContext,
    ) -> ToolResult:
        executor = await self._get_executor(context)
        from zhenxun.services.ai.sandbox.extension import SupportsFileSystem

        fs_executor = cast(SupportsFileSystem, executor)

        try:
            ops = PatchParser.parse(patch_content)
            if not ops:
                raise ToolRetryError(
                    "解析补丁失败：未找到合法的 *** Begin Patch 和操作。请严格检查语法。"
                )

            reports = []
            for op in ops:
                if op.op_type == "add":
                    new_content = "\n".join(op.new_content)
                    await fs_executor.write_raw_file(op.path, new_content)
                    reports.append(f"成功创建文件: {op.path}")

                elif op.op_type == "update":
                    old_content = await fs_executor.read_raw_file(op.path)
                    if old_content.startswith("Error: File") or old_content.startswith(
                        "Failed to"
                    ):
                        raise ToolRetryError(
                            f"文件不存在，无法 Update：{op.path}。请先使用 *** Add File 创建。"
                        )

                    current_text = old_content
                    for hunk in op.hunks:
                        try:
                            current_text = PatchApplier.apply_diff(
                                current_text, hunk["search"], hunk["replace"]
                            )
                        except PatchError as e:
                            raise ToolRetryError(
                                f"在文件 {op.path} 中应用补丁失败：{e}"
                            )

                    await fs_executor.write_raw_file(op.path, current_text)
                    reports.append(f"成功更新文件: {op.path}")

            return ToolResult(output="\n".join(reports)).show_to_user(
                "📝 成功通过 apply_patch 修改了文件"
            )

        except ToolRetryError as e:
            raise e
        except Exception as e:
            logger.error(f"apply_patch 发生内部异常: {e}")
            raise ToolRetryError(f"应用补丁时发生致命异常: {e}")
