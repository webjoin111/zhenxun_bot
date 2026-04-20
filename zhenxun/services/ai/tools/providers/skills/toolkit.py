import ast
from collections.abc import Sequence
import os
import re
from typing import cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.sandbox.extension import (
    SupportsCommandExecution,
    SupportsFileSystem,
)
from zhenxun.services.ai.sandbox.manager import sandbox_manager
from zhenxun.services.ai.sandbox.utils import ASTAnalyzer, get_execution_command
from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.decorators import toolkit_tool, silent
from zhenxun.services.ai.tools.core.tool import BaseTool, FunctionTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
from zhenxun.services.ai.tools.providers.skills.models import Skill
from zhenxun.services.ai.types.sandbox import SandboxSecurityProfile
from zhenxun.services.ai.types.tools import ToolResult
from zhenxun.services.log import logger


class ScriptArgs(BaseModel):
    args: str = Field(default="", description="传递给脚本的命令行参数（字符串格式）")


class ReadFileArgs(BaseModel):
    file_path: str = Field(
        description="要读取的文件相对路径（例如 'references/a-stock-features.md'）"
    )


class SkillStaticToolkit(BaseToolkit):
    """轨道A：静态预装模式。将单个 Skill 的 scripts/ 目录直接映射为独立工具。"""

    def __init__(self, skill: Skill):
        self.skill = skill
        super().__init__()

    async def get_tools(self) -> Sequence[BaseTool]:
        tools = []
        scripts_dir = self.skill.path / "scripts"
        if not scripts_dir.is_dir():
            return tools

        for script_file in scripts_dir.iterdir():
            if script_file.is_file() and script_file.suffix == ".py":
                tool_name = (
                    f"run_{self.skill.name.replace('-', '_')}_{script_file.stem}"
                )
                desc = f"执行技能 {self.skill.name} 的脚本 {script_file.name}。"
                code_content = script_file.read_text(encoding="utf-8")
                try:
                    if doc := ast.get_docstring(ast.parse(code_content)):
                        desc += f"\n说明: {doc}"
                except Exception:
                    pass

                def make_func(filepath=script_file, code=code_content):
                    async def run_script(
                        args: str = "", context: RunContext | None = None, **kwargs
                    ) -> ToolResult:
                        session_id = (
                            context.session_id
                            if context and context.session_id
                            else f"skill_{self.skill.name}_session"
                        )
                        profile = SandboxSecurityProfile(
                            enable_network=self.skill.frontmatter.enable_network,
                            required_plugins=["universal_python"],
                        )

                        reqs = ASTAnalyzer.analyze_code_requirements(code)

                        executor = await sandbox_manager.get_or_create_session(
                            session_id, profile=profile, requirements=reqs
                        )
                        fs_executor = cast(SupportsFileSystem, executor)
                        cmd_executor = cast(SupportsCommandExecution, executor)

                        target_workspace = f"/workspace/{self.skill.name}"

                        if self.skill.name not in executor.loaded_skills:
                            await fs_executor.upload_raw_dir(
                                str(self.skill.path), target_workspace
                            )

                            metadata = self.skill.frontmatter.metadata or {}
                            openclaw_meta = metadata.get("openclaw")

                            try:
                                await sandbox_manager.setup_workspace_environment(
                                    session_id, target_workspace, openclaw_meta
                                )
                            except Exception as e:
                                logger.warning(f"设置沙箱工作区环境失败: {e}")
                            executor.loaded_skills.add(self.skill.name)

                        env_vars = {}
                        metadata = self.skill.frontmatter.metadata or {}
                        openclaw_meta = metadata.get("openclaw", {})
                        if (
                            "requires" in openclaw_meta
                            and "env" in openclaw_meta["requires"]
                        ):
                            for env_key in openclaw_meta["requires"]["env"]:
                                if env_key in kwargs:
                                    env_vars[env_key] = str(kwargs[env_key])
                                elif context and env_key in context.extra:
                                    env_vars[env_key] = str(context.extra[env_key])
                                elif env_key in os.environ:
                                    env_vars[env_key] = os.environ[env_key]

                        import shlex

                        arg_list = shlex.split(args)
                        cmd = get_execution_command(
                            f"scripts/{filepath.name}", arg_list
                        )

                        result = await cmd_executor.execute_raw_command(
                            cmd, cwd=target_workspace, timeout=60, env=env_vars
                        )

                        if result.exit_code != 0:
                            err_text = result.stderr or result.stdout

                            node_match = re.search(
                                r"Cannot find module '([^']+)'", err_text
                            )
                            py_match = re.search(r"No module named '([^']+)'", err_text)

                            if (node_match and node_match.group(1)) or (
                                py_match and py_match.group(1)
                            ):
                                is_node = bool(node_match)
                                pkg = (
                                    (node_match.group(1) if node_match else None)
                                    if node_match
                                    else (py_match.group(1) if py_match else None)
                                )
                                install_cmd = (
                                    f"npm install {pkg}"
                                    if is_node
                                    else f"pip install {pkg}"
                                )

                                logger.info(
                                    f"🔧 [自愈机制] 探测到沙箱依赖缺失 [{pkg}]，正在执行热修复: {install_cmd}"
                                )
                                await cmd_executor.execute_raw_command(
                                    install_cmd, cwd=target_workspace, timeout=120
                                )
                                result = await cmd_executor.execute_raw_command(
                                    cmd, cwd=target_workspace, timeout=60, env=env_vars
                                )

                        output = (
                            result.stdout
                            + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
                        ).strip()
                        return ToolResult(
                            output=output or "执行成功 (无输出)。",
                            is_error=result.exit_code != 0,
                        )

                    return run_script

                tools.append(
                    FunctionTool(
                        func=make_func(),
                        name=tool_name,
                        description=desc[:1024],
                    )
                )

        read_tool_name = f"read_{self.skill.name.replace('-', '_')}_file"

        def make_read_func():
            async def read_file(
                file_path: str, context: RunContext | None = None
            ) -> ToolResult:
                target_file = (self.skill.path / file_path).resolve()
                if not target_file.is_relative_to(self.skill.path.resolve()):
                    return ToolResult(
                        output="❌ 安全拦截：禁止读取技能目录之外的文件！",
                        is_error=True,
                    )
                if not target_file.is_file():
                    return ToolResult(
                        output=f"❌ 文件不存在: {file_path}", is_error=True
                    )
                content = target_file.read_text(encoding="utf-8")
                return ToolResult(
                    output=content,
                    log_content=f"已读取文件 {file_path} (共 {len(content)} 字符)",
                )

            return read_file

        tools.append(
            FunctionTool(
                func=make_read_func(),
                name=read_tool_name,
                description=f"读取技能 {self.skill.name} 的附加文件（如 references 或 templates 目录下的文件）。",
            )
        )

        return tools


class SkillMetaToolkit(BaseToolkit):
    """轨道B：动态发现模式。提供通用的元工具，供大模型按需加载和执行任意可用技能。"""

    @toolkit_tool(
        name="read_skill_instructions",
        description="加载指定技能的完整使用说明。当你需要使用某个技能时，请先调用此工具获取详情。",
    )
    @silent()
    async def read_skill_instructions(self, skill_name: str) -> ToolResult:
        skill = await skill_manager.get_skill_details(skill_name)
        if not skill:
            return ToolResult(output=f"未找到技能: {skill_name}", is_error=True)

        res = f"## 技能 {skill.name} 指南\n\n{skill.instructions}\n"
        if skill.scripts:
            res += "\n可用脚本 (使用 run_skill_script 执行):\n- " + "\n- ".join(
                skill.scripts
            )

        res += "\n\n**提示**: 你可以使用 `read_skill_file` 工具读取该技能目录下的任何附加文件（如 references/ 或 templates/ 下的文档）。"
        return ToolResult(output=res, log_content=f"已加载技能 {skill.name} 指南。")

    @toolkit_tool(
        name="run_skill_script",
        description="在安全沙箱中执行指定技能的内置 Python 脚本。",
    )
    async def run_skill_script(
        self,
        skill_name: str,
        script_name: str,
        args: str = "",
        context: RunContext | None = None,
        **kwargs,
    ) -> ToolResult:
        script_name = script_name.strip()
        if " " in script_name:
            parts = script_name.split(" ", 1)
            script_name = parts[0]
            args = f"{parts[1]} {args}".strip()

        skill = await skill_manager.get_skill_details(skill_name)
        if not skill or script_name not in skill.scripts:
            return ToolResult(
                output=f"技能 {skill_name} 不存在或不包含脚本 {script_name}",
                is_error=True,
            )

        script_file = skill.path / "scripts" / script_name
        code = script_file.read_text(encoding="utf-8")

        session_id = (
            context.session_id
            if context and context.session_id
            else "skill_meta_session"
        )

        reqs = ASTAnalyzer.analyze_code_requirements(code)

        profile = SandboxSecurityProfile(
            enable_network=skill.frontmatter.enable_network,
            required_plugins=["universal_python"],
        )
        executor = await sandbox_manager.get_or_create_session(
            session_id, profile=profile, requirements=reqs
        )
        fs_executor = cast(SupportsFileSystem, executor)
        cmd_executor = cast(SupportsCommandExecution, executor)

        target_workspace = f"/workspace/{skill.name}"

        if skill.name not in executor.loaded_skills:
            await fs_executor.upload_raw_dir(str(skill.path), target_workspace)

            metadata = skill.frontmatter.metadata or {}
            openclaw_meta = metadata.get("openclaw")

            try:
                await sandbox_manager.setup_workspace_environment(
                    session_id, target_workspace, openclaw_meta
                )
            except Exception as e:
                logger.warning(f"设置沙箱工作区环境失败: {e}")
            executor.loaded_skills.add(skill.name)

        env_vars = {}
        metadata = skill.frontmatter.metadata or {}
        openclaw_meta = metadata.get("openclaw", {})
        if "requires" in openclaw_meta and "env" in openclaw_meta["requires"]:
            for env_key in openclaw_meta["requires"]["env"]:
                if env_key in kwargs:
                    env_vars[env_key] = str(kwargs[env_key])
                elif context and env_key in context.extra:
                    env_vars[env_key] = str(context.extra[env_key])
                elif env_key in os.environ:
                    env_vars[env_key] = os.environ[env_key]

        import shlex

        arg_list = shlex.split(args)
        cmd = get_execution_command(f"scripts/{script_name}", arg_list)

        result = await cmd_executor.execute_raw_command(
            cmd, cwd=target_workspace, timeout=60, env=env_vars
        )

        if result.exit_code != 0:
            err_text = result.stderr or result.stdout

            node_match = re.search(r"Cannot find module '([^']+)'", err_text)
            py_match = re.search(r"No module named '([^']+)'", err_text)

            if (node_match and node_match.group(1)) or (py_match and py_match.group(1)):
                is_node = bool(node_match)
                pkg = (
                    (node_match.group(1) if node_match else None)
                    if node_match
                    else (py_match.group(1) if py_match else None)
                )
                install_cmd = f"npm install {pkg}" if is_node else f"pip install {pkg}"

                logger.info(
                    f"🔧 [自愈机制] 探测到沙箱依赖缺失 [{pkg}]，正在执行热修复: {install_cmd}"
                )
                await cmd_executor.execute_raw_command(
                    install_cmd, cwd=target_workspace, timeout=120
                )
                result = await cmd_executor.execute_raw_command(
                    cmd, cwd=target_workspace, timeout=60, env=env_vars
                )

        output = (
            result.stdout + ("\n[STDERR]\n" + result.stderr if result.stderr else "")
        ).strip()
        return ToolResult(
            output=output or "执行成功 (无输出)。",
            is_error=result.exit_code != 0,
            log_content=f"脚本 {script_name} 执行完毕, Exit Code: {result.exit_code}",
        )

    @toolkit_tool(
        name="read_skill_file",
        description="安全读取指定技能目录下的附加文件（如 references/ 里的参考文档或 scripts/ 里的代码）。",
    )
    @silent()
    async def read_skill_file(self, skill_name: str, file_path: str) -> ToolResult:
        skill = await skill_manager.get_skill_details(skill_name)
        if not skill:
            return ToolResult(output=f"未找到技能: {skill_name}", is_error=True)

        target_file = (skill.path / file_path).resolve()

        if not target_file.is_file():
            fallback_file = (skill.path / "scripts" / file_path).resolve()
            if fallback_file.is_file():
                target_file = fallback_file

        if not target_file.is_relative_to(skill.path.resolve()):
            return ToolResult(
                output="❌ 安全拦截：禁止读取技能目录之外的文件！", is_error=True
            )
        if not target_file.is_file():
            return ToolResult(output=f"❌ 找不到文件: {file_path}", is_error=True)

        content = target_file.read_text(encoding="utf-8")
        return ToolResult(
            output=content,
            log_content=f"已读取文件 {file_path} (共 {len(content)} 字符)。内容摘要: {content[:100]}...",
        )
