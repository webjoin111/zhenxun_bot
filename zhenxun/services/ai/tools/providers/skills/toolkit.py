import ast
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.sandbox.extension import (
    SupportsCommandExecution,
    SupportsFileSystem,
)
from zhenxun.services.ai.sandbox.manager import sandbox_manager
from zhenxun.services.ai.sandbox.models import SandboxSecurityProfile
from zhenxun.services.ai.sandbox.utils import get_execution_command
from zhenxun.services.ai.tools.core.decorators import silent, tool
from zhenxun.services.ai.tools.core.tool import FunctionTool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.ai.tools.providers.skills.manager import (
    skill_env_manager,
    skill_manager,
)
from zhenxun.services.ai.tools.providers.skills.models import Skill
from zhenxun.services.log import logger


class ScriptArgs(BaseModel):
    args: str = Field(default="")
    """传递给脚本的命令行参数（字符串格式）"""


class ReadFileArgs(BaseModel):
    file_path: str = Field(...)
    """要读取的文件相对路径（例如 'references/a-stock-features.md'）"""


class SkillSandboxExecutionMixin:
    """技能沙箱执行与自愈逻辑混入类"""

    async def _execute_skill_script_in_sandbox(
        self,
        skill: Skill,
        script_name: str,
        args: str,
        context: RunContext | None,
        **kwargs,
    ) -> ToolResult:

        session_id = (
            context.session_id
            if context and context.session_id
            else f"skill_{skill.id}_session"
        )
        profile = SandboxSecurityProfile(
            enable_network=skill.frontmatter.enable_network,
            required_plugins=["universal_python"],
        )

        from zhenxun.services.ai.sandbox.models import SandboxRequirements

        reqs = SandboxRequirements()
        reqs.env_setup = skill.frontmatter.env_setup

        executor = await sandbox_manager.get_or_create_session(
            session_id, profile=profile, requirements=reqs
        )
        fs_executor = cast(SupportsFileSystem, executor)
        cmd_executor = cast(SupportsCommandExecution, executor)

        target_workspace = f"/workspace/{skill.id}"

        if skill.id not in executor.loaded_skills:
            await fs_executor.upload_raw_dir(str(skill.path), target_workspace)
            try:
                await sandbox_manager.setup_workspace_environment(
                    session_id, target_workspace
                )
            except Exception as e:
                logger.warning(f"设置沙箱工作区环境失败: {e}")
            executor.loaded_skills.add(skill.id)

        configured_envs = skill_env_manager.get_envs_for_skill(skill.id)
        missing_keys = [
            k for k in skill.frontmatter.required_envs if not configured_envs.get(k)
        ]
        if missing_keys:
            return ToolResult(
                output=(
                    f"❌ 技能执行被系统拦截：缺少必需的全局环境变量 {missing_keys}。\n"
                    "💡 [智能体自愈引导]：当前技能的底层配置缺失，无法正常运行。"
                    "请你立即停止尝试，并向用户抱歉，提示用户（或 Bot 管理员）"
                    "在机器人后端的 `data/ai/skill_envs.json` 文件中为该技能配置"
                    "相应的环境变量（API Key 等），配置完成后方可使用。"
                ),
            ).as_error().with_log(
                f"技能 {skill.id} 因缺少环境变量 {missing_keys} 被拦截。"
            )

        env_vars = {}
        for k, v in configured_envs.items():
            if v:
                env_vars[k] = str(v)

        for k, v in kwargs.items():
            env_vars[k] = str(v)
        if context:
            for k, v in context.state.items():
                if isinstance(v, (str, int, float, bool)):
                    env_vars[k] = str(v)

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
                install_cmd = (
                    f"npm install {pkg}"
                    if is_node
                    else f"uv pip install --system {pkg} 或 pip install {pkg}"
                )

                final_output = f"""❌ 脚本执行失败 (Exit Code: {result.exit_code})。
输出日志:
{err_text}

💡 [智能体自愈引导]：当前沙箱环境缺失依赖包 [{pkg}]！
这通常是因为技能作者未能正确在 SKILL.md 中声明该依赖。
**解决方案**：请你立即调用 `execute_skill_command` 工具执行 `{install_cmd}`，
等待安装成功后，再重新调用 `run_skill_script` 执行当前脚本！"""
                return ToolResult(
                    output=final_output,
                ).as_error().with_log(
                    f"脚本因缺失依赖 {pkg} 失败，已引导 Agent 自愈。"
                )

        output = (
            result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
        ).strip()

        if output:
            logger.debug(f"Console Output from {script_name}:\n{output}")

        if getattr(result, "is_timeout", False) or result.exit_code == -1:
            final_output = f"""🚨 脚本执行发生严重系统异常或超时被强杀
(Exit Code: {result.exit_code})！
这通常意味着网络不通、下载数据过大耗时太长，或沙箱环境崩溃。输出为空。"""
        elif result.exit_code != 0:
            final_output = f"""❌ 脚本执行失败 (Exit Code: {result.exit_code})。
输出日志:
{output or "无日志输出"}"""
        else:
            final_output = output or "✅ 执行成功 (无控制台输出)。"

        tool_result = ToolResult(output=final_output).with_log(
            f"脚本 {script_name} 执行完毕, Exit Code: {result.exit_code}"
        )
        if result.exit_code != 0:
            tool_result = tool_result.as_error()
        return tool_result

    async def _execute_skill_command_in_sandbox(
        self,
        skill: Skill,
        command: str,
        context: RunContext | None,
        **kwargs,
    ) -> ToolResult:
        session_id = (
            context.session_id
            if context and context.session_id
            else f"skill_{skill.id}_session"
        )
        profile = SandboxSecurityProfile(
            enable_network=skill.frontmatter.enable_network,
            required_plugins=["universal_python"],
        )

        from zhenxun.services.ai.sandbox.models import SandboxRequirements

        reqs = SandboxRequirements()
        reqs.env_setup = skill.frontmatter.env_setup

        executor = await sandbox_manager.get_or_create_session(
            session_id, profile=profile, requirements=reqs
        )
        fs_executor = cast(SupportsFileSystem, executor)
        cmd_executor = cast(SupportsCommandExecution, executor)

        target_workspace = f"/workspace/{skill.id}"

        if skill.id not in executor.loaded_skills:
            await fs_executor.upload_raw_dir(str(skill.path), target_workspace)
            try:
                await sandbox_manager.setup_workspace_environment(
                    session_id, target_workspace
                )
            except Exception as e:
                logger.warning(f"设置沙箱工作区环境失败: {e}")
            executor.loaded_skills.add(skill.id)

        configured_envs = skill_env_manager.get_envs_for_skill(skill.id)
        missing_keys = [
            k for k in skill.frontmatter.required_envs if not configured_envs.get(k)
        ]
        if missing_keys:
            return ToolResult(
                output=(
                    f"❌ 技能执行被系统拦截：缺少必需的全局环境变量 {missing_keys}。\n"
                    "💡 [智能体自愈引导]：当前技能的底层配置缺失，无法正常运行。"
                    "请你立即停止尝试，并向用户抱歉，提示用户（或 Bot 管理员）"
                    "在机器人后端的 `data/ai/skill_envs.json` 文件中为该技能配置"
                    "相应的环境变量（API Key 等），配置完成后方可使用。"
                ),
            ).as_error().with_log(
                f"技能 {skill.id} 因缺少环境变量 {missing_keys} 被拦截。"
            )

        env_vars = {}
        for k, v in configured_envs.items():
            if v:
                env_vars[k] = str(v)

        for k, v in kwargs.items():
            env_vars[k] = str(v)
        if context:
            for k, v in context.state.items():
                if isinstance(v, (str, int, float, bool)):
                    env_vars[k] = str(v)

        result = await cmd_executor.execute_raw_command(
            command, cwd=target_workspace, timeout=180, env=env_vars
        )

        output = (
            result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
        ).strip()

        if getattr(result, "is_timeout", False) or result.exit_code == -1:
            final_output = f"""🚨 终端命令执行发生严重系统异常或超时被强杀
(Exit Code: {result.exit_code})！
这通常意味着网络不通、下载数据过大耗时太长，或沙箱环境崩溃。输出为空。"""
        elif result.exit_code == 127 or "command not found" in output.lower():
            final_output = f"""❌ 终端命令执行失败
(Exit Code: {result.exit_code})。提示“命令未找到”。
输出日志:
{output}

💡 [系统自愈引导]：当前沙箱环境缺少该命令对应的依赖程序！
请查阅你刚刚阅读的技能指南中的 Quick Start 或 Prerequisites 部分，
找到对应的安装命令（例如 `npm install -g xxx`, `npx ...` 或
`pip install xxx`）。
请先调用 `execute_skill_command` 执行这些安装命令，等待安装成功后，
再重新执行你原本的任务命令！"""
        elif result.exit_code != 0:
            final_output = f"""❌ 终端命令执行失败 (Exit Code: {result.exit_code})。
输出日志:
{output or "无日志输出"}"""
        else:
            final_output = output or "✅ 执行成功 (无控制台输出)。"

        tool_result = ToolResult(output=final_output).with_log(
            f"终端命令 {command} 执行完毕, Exit Code: {result.exit_code}"
        )
        if result.exit_code != 0:
            tool_result = tool_result.as_error()
        return tool_result


class SkillStaticToolkit(BaseToolkit, SkillSandboxExecutionMixin):
    """轨道A：静态预装模式。将单个 Skill 的 scripts/ 目录直接映射为独立工具。"""

    def __init__(self, skill: Skill):
        self.skill = skill
        super().__init__()

    async def get_tools(self, context: RunContext | None = None) -> dict[str, Any]:
        tools = []
        scripts_dir = self.skill.path / "scripts"
        if not scripts_dir.is_dir():
            return {}

        for script_file in scripts_dir.iterdir():
            if script_file.is_file() and script_file.suffix == ".py":
                tool_name = f"run_{self.skill.id.replace('-', '_')}_{script_file.stem}"
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
                        return await self._execute_skill_script_in_sandbox(
                            self.skill, filepath.name, args, context, **kwargs
                        )

                    return run_script

                tools.append(
                    FunctionTool(
                        func=make_func(),
                        name=tool_name,
                        description=desc[:1024],
                    )
                )

        read_tool_name = f"read_{self.skill.id.replace('-', '_')}_file"

        def make_read_func():
            async def read_file(
                file_path: str, context: RunContext | None = None
            ) -> ToolResult:
                target_file = (self.skill.path / file_path).resolve()
                if not target_file.is_relative_to(self.skill.path.resolve()):
                    return ToolResult(
                        output="❌ 安全拦截：禁止读取技能目录之外的文件！"
                    ).as_error()
                if not target_file.is_file():
                    return ToolResult(output=f"❌ 文件不存在: {file_path}").as_error()
                content = target_file.read_text(encoding="utf-8")
                return ToolResult(output=content).with_log(
                    f"已读取文件 {file_path} (共 {len(content)} 字符)"
                )

            return read_file

        tools.append(
            FunctionTool(
                func=make_read_func(),
                name=read_tool_name,
                description=(
                    f"读取技能 {self.skill.name} 的附加文件"
                    "（如 references 或 templates 目录下的文件）。"
                ),
            )
        )

        tools_dict = {}
        for t in tools:
            t.parent_toolkit = self
            tools_dict[t.name] = t
        return tools_dict


class SkillMetaToolkit(BaseToolkit, SkillSandboxExecutionMixin):
    """轨道B：动态发现模式。提供通用的元工具，供大模型按需加载和执行任意可用技能。"""

    def __init__(self, allowed_skills: list[Skill] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._allowed_skills = allowed_skills
        if self._allowed_skills is not None:
            self._instance_instructions = (
                (self._instance_instructions or self.default_instructions)
                + "\n\n🚨 **[系统安全提示]**：当前运行在严格沙盒隔离模式，"
                "你只能访问特定被授权的技能。"
            )

    async def _get_skill(self, skill_name: str) -> Skill | None:
        """按需从隔离白名单或全局管理器中获取技能"""
        if self._allowed_skills is not None:
            for s in self._allowed_skills:
                if s.id == skill_name or s.name == skill_name:
                    return s
            return None
        return await skill_manager.get_skill_details(skill_name)

    def get_instructions(self) -> str | None:
        base_inst = super().get_instructions() or ""
        if self._allowed_skills is not None:
            catalog_parts = []
            for skill in self._allowed_skills:
                catalog_parts.append(
                    f"  <skill>\n    <name>{skill.id}</name>\n"
                    f"    <description>{skill.description}</description>\n  </skill>"
                )
            if catalog_parts:
                catalog_xml = (
                    "<available_skills>\n"
                    + "\n".join(catalog_parts)
                    + "\n</available_skills>"
                )
                tag_name = (
                    f"{self.config.prefix}{self.__class__.__name__}_Instructions"
                    if self.config.prefix
                    else f"{self.__class__.__name__}_Instructions"
                )
                closing_tag = f"</{tag_name}>"
                if base_inst.endswith(closing_tag):
                    base_inst = (
                        base_inst[: -len(closing_tag)]
                        + f"\n--- 当前受限的可用技能库 ---\n\n{catalog_xml}\n"
                        + closing_tag
                    )
                else:
                    base_inst += f"\n\n--- 当前受限的可用技能库 ---\n\n{catalog_xml}"
        return base_inst

    default_instructions = (
        "## 技能元工具系统\n"
        "你可以通过此工具箱动态加载和执行外部技能。工作流如下：\n"
        "1. 使用 `read_skill_instructions` 传入技能名称，获取该技能的完整指南。\n"
        "2. 系统将返回严谨的 `<skill>` XML 节点树，"
        "请仔细阅读 `<instructions>` 了解业务规则。\n"
        "3. **[重点] 环境装配与执行规范**：\n"
        "   - **环境预检**：如果指南中明确要求安装全局依赖"
        "（如 `npm install -g`, `npx ...`, `pip install`），"
        "你必须**优先**调用 `execute_skill_command` 执行安装。\n"
        "   - **脚本执行**：若指南中要求执行物理存在的脚本文件"
        "（如 `<available_scripts>` 节点列出的文件），"
        "请调用 `run_skill_script`。\n"
        "   - **终端执行**：若指南中提供的是纯命令行终端指令"
        "（例如 `curl`, `infsh`, `gh` 等），请调用 "
        "`execute_skill_command` 在技能专属沙箱中直接执行该命令。\n"
        "   - **智能自愈**：如果执行脚本时报错 `ModuleNotFoundError` 等依赖缺失错误，"
        "你必须调用 `execute_skill_command` 手动执行 "
        "`uv pip install ...` 或 `npm install ...` 补齐依赖后再重试。\n"
        "4. 如需阅读参考文档，请参考 `<available_references>` 节点并调用 "
        "`read_skill_file`。"
    )

    @tool(
        name="read_skill_instructions",
        description="加载指定技能的完整使用说明与可用资源清单。返回值为 XML 结构。",
    )
    @silent()
    async def read_skill_instructions(self, skill_name: str) -> ToolResult:
        skill = await self._get_skill(skill_name)
        if not skill:
            return ToolResult(output=f"越权操作或未找到技能: {skill_name}").as_error()

        res = skill.to_xml()
        res += (
            "\n\n**提示**: 你可以使用 `read_skill_file` 工具读取该技能"
            "目录下的任何附加文件（如 references/ 或 templates/ 下的文档）。"
        )
        return ToolResult(output=res).with_log(f"已加载技能 {skill.name} 指南。")

    @tool(
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

        skill = await self._get_skill(skill_name)
        if not skill or script_name not in skill.scripts:
            return ToolResult(
                output=f"越权操作/未找到技能 {skill_name} 或不包含脚本 {script_name}",
            ).as_error()

        return await self._execute_skill_script_in_sandbox(
            skill, script_name, args, context, **kwargs
        )

    @tool(
        name="execute_skill_command",
        description=(
            "在技能专属的安全沙箱内执行任意终端命令行指令"
            "（如 curl, npm, gh 等）。仅当技能指南中提供的是终端命令"
            "而非特定脚本文件时使用。"
        ),
    )
    async def execute_skill_command(
        self,
        skill_name: str,
        command: str,
        context: RunContext | None = None,
        **kwargs,
    ) -> ToolResult:
        skill = await self._get_skill(skill_name)
        if not skill:
            return ToolResult(
                output=f"越权操作或未找到技能: {skill_name}",
            ).as_error()

        return await self._execute_skill_command_in_sandbox(
            skill, command, context, **kwargs
        )

    @tool(
        name="read_skill_file",
        description=(
            "安全读取指定技能目录下的附加文件"
            "（如 references/ 里的参考文档或 scripts/ 里的代码）。"
        ),
    )
    @silent()
    async def read_skill_file(
        self, skill_name: str, file_path: str, context: RunContext | None = None
    ) -> ToolResult:
        skill = await self._get_skill(skill_name)
        if not skill:
            return ToolResult(output=f"越权操作或未找到技能: {skill_name}").as_error()

        content = await skill_manager.read_skill_resource(skill, file_path)

        if content is not None:
            return ToolResult(
                output=content,
            ).with_log(
                (
                    f"已从本地读取文件 {file_path} (共 {len(content)} 字符)。"
                    f"内容摘要: {content[:100]}..."
                )
            )

        session_id = (
            context.session_id
            if context and context.session_id
            else f"skill_{skill.id}_session"
        )
        profile = SandboxSecurityProfile(
            enable_network=skill.frontmatter.enable_network,
            required_plugins=["universal_python"],
        )
        try:
            executor = await sandbox_manager.get_or_create_session(
                session_id, profile=profile
            )
            fs_executor = cast(SupportsFileSystem, executor)

            clean_file_path = file_path.lstrip("/")
            sandbox_target_path = f"/workspace/{skill.id}/{clean_file_path}"

            content = await fs_executor.read_raw_file(sandbox_target_path)

            if content.startswith("Error: File") or content.startswith("Failed to"):
                return ToolResult(
                    output=(
                        f"❌ 找不到文件: {file_path} "
                        f"(Provider {skill.source} 与沙箱中均未找到)"
                    ),
                ).as_error()

            return ToolResult(
                output=content,
            ).with_log(
                (
                    "已从沙箱读取动态生成的文件 "
                    f"{sandbox_target_path} (共 {len(content)} 字符)。"
                )
            )
        except Exception as e:
            return ToolResult(output=f"❌ 尝试读取沙箱文件时发生异常: {e}").as_error()
