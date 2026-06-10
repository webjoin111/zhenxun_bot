from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zhenxun.services.ai.sandbox.models import SandboxBlueprint

DisclosureLevel = Annotated[
    Literal[1, 2, 3], "Progressive disclosure levels for skill loading."
]
METADATA: DisclosureLevel = 1
INSTRUCTIONS: DisclosureLevel = 2
RESOURCES: DisclosureLevel = 3


class SkillEnvConfig(BaseModel):
    """全量技能环境变量配置根节点"""

    envs: dict[str, dict[str, str]] = Field(default_factory=dict)


class SkillFrontmatter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(...)
    """技能名称"""
    description: str = Field(...)
    """技能描述"""
    compatibility: str | None = Field(default=None)
    """环境要求"""
    metadata: dict[str, Any] | None = Field(default=None)
    """自定义元数据"""
    allowed_tools: list[str] | None = Field(default=None, alias="allowed-tools")
    """允许的工具"""
    permissions: dict[str, Any] | None = Field(default=None)
    """沙箱权限声明 (如 network: true/false)"""
    blueprint: SandboxBlueprint = Field(default_factory=SandboxBlueprint)
    """统一环境装配蓝图声明 (根据 metadata 等自动推导)"""
    required_envs: list[str] = Field(default_factory=list)
    """声明该技能必需的全局环境变量 Key"""

    @model_validator(mode="before")
    @classmethod
    def parse_blueprint(cls, values: dict[str, Any]) -> dict[str, Any]:
        env_setup_data = values.get("env_setup", {})
        blueprint = values.get("blueprint")
        if isinstance(blueprint, SandboxBlueprint):
            return values

        python_packages = env_setup_data.get("python_packages", [])
        system_packages = env_setup_data.get("system_packages", [])
        bins = env_setup_data.get("bins", [])
        install_scripts = env_setup_data.get("install_scripts", [])
        required_envs = values.get("required_envs", [])

        metadata = values.get("metadata", {})
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
                values["metadata"] = metadata
            except Exception:
                metadata = {}

        if isinstance(metadata, dict):
            for provider in ["openclaw", "clawbot"]:
                if provider in metadata and isinstance(metadata[provider], dict):
                    data = metadata[provider]
                    requires = data.get("requires", {})
                    if isinstance(requires, dict):
                        bins.extend(requires.get("bins", []))
                        envs = requires.get("env", [])
                        if isinstance(envs, list):
                            required_envs.extend(envs)

                    installs = data.get("install", [])
                    if isinstance(installs, list):
                        for inst in installs:
                            if isinstance(inst, dict):
                                kind = inst.get("kind")
                                if kind in ("python", "pip"):
                                    pkg = inst.get("package", "")
                                    if isinstance(pkg, str):
                                        python_packages.extend(pkg.split())
                                elif kind in ("node", "npm"):
                                    pkg = inst.get("package", "")
                                    if isinstance(pkg, str) and pkg:
                                        install_scripts.append(f"npm install -g {pkg}")
                                elif kind == "brew":
                                    formula = inst.get("formula", "")
                                    if isinstance(formula, str) and formula:
                                        system_packages.extend(formula.split())

        key = "allowed-tools"
        alt_key = "allowed_tools"
        raw_tools = values.get(key) or values.get(alt_key)

        tools_list = []
        if isinstance(raw_tools, str):
            tools_list = raw_tools.split()
        elif isinstance(raw_tools, list):
            tools_list = raw_tools

        import re

        for tool in tools_list:
            if isinstance(tool, str) and tool.startswith("Bash("):
                match = re.search(r"Bash\((.*?)\)", tool)
                if match:
                    inner = match.group(1)
                    cmd = inner.split(":")[0]
                    if cmd not in bins:
                        bins.append(cmd)

        from zhenxun.services.ai.sandbox.models import AptSetup, PythonSetup, ShellSetup

        steps = []
        if system_packages:
            steps.append(AptSetup(packages=list(dict.fromkeys(system_packages))))
        if python_packages:
            steps.append(PythonSetup(packages=list(dict.fromkeys(python_packages))))
        if install_scripts:
            steps.append(ShellSetup(scripts=install_scripts))

        values["blueprint"] = {"setup_steps": steps}
        values["required_envs"] = list(dict.fromkeys(required_envs))
        return values

    @model_validator(mode="before")
    @classmethod
    def parse_allowed_tools(cls, values: dict[str, Any]) -> dict[str, Any]:
        """将字符串格式的 allowed-tools 按空格切分为列表"""
        key = "allowed-tools"
        alt_key = "allowed_tools"
        raw = values.get(key) or values.get(alt_key)
        if isinstance(raw, str):
            values[key] = raw.split()
        return values

    @property
    def enable_network(self) -> bool:
        """网络权限：默认开启（高信任静态资产），允许在 YAML 中通过 permissions.network 显式关闭"""  # noqa: E501
        if self.permissions and isinstance(self.permissions, dict):
            return self.permissions.get("network", True)
        return True


class Skill(BaseModel):
    frontmatter: SkillFrontmatter
    """解析后的YAML元数据"""
    instructions: str | None = Field(default=None)
    """SKILL.md正文指令，在INSTRUCTIONS级别填充"""
    path: Path
    """技能所在目录"""
    disclosure_level: DisclosureLevel = Field(default=METADATA)
    """当前披露等级"""
    scripts: list[str] = Field(default_factory=list)
    """脚本文件列表"""
    references: list[str] = Field(default_factory=list)
    """参考文档列表"""
    source: str = Field(default="local")
    """技能来源提供者标识"""

    def with_disclosure_level(
        self,
        level: DisclosureLevel,
        instructions: str | None = None,
        scripts: list[str] | None = None,
        references: list[str] | None = None,
    ) -> "Skill":
        """创建一个提升了披露等级的全新 Skill 实例"""
        return Skill(
            frontmatter=self.frontmatter,
            instructions=instructions
            if instructions is not None
            else self.instructions,
            path=self.path,
            disclosure_level=level,
            scripts=scripts if scripts is not None else self.scripts,
            references=references if references is not None else self.references,
            source=self.source,
        )

    @property
    def id(self) -> str:
        """技能的唯一标识符（强制使用所在文件夹的名称）"""
        return self.path.name

    def to_xml(self) -> str:
        """将技能转化为结构化的 XML 格式，供大模型友好读取"""
        xml = f"<skill>\n  <name>{self.id}</name>\n  <description>{self.description}</description>\n"  # noqa: E501
        if self.instructions:
            xml += f"  <instructions>\n{self.instructions}\n  </instructions>\n"
        if self.scripts:
            xml += (
                "  <available_scripts>\n"
                + "\n".join(f"    <script>{s}</script>" for s in self.scripts)
                + "\n  </available_scripts>\n"
            )
        if self.references:
            xml += (
                "  <available_references>\n"
                + "\n".join(f"    <reference>{r}</reference>" for r in self.references)
                + "\n  </available_references>\n"
            )
        xml += "</skill>"
        return xml

    @property
    def name(self) -> str:
        """技能的显示名称（来自 YAML，可能不符合规范）"""
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description


class SkillMount(BaseModel):
    """
    局部私有技能挂载器。
    允许将本地特定目录作为私有技能注入，避免污染全局
    """

    path: Path = Field(description="技能所在目录的物理路径")

    async def resolve(self, context: Any | None = None) -> Any:
        from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
        from zhenxun.services.ai.tools.providers.skills.toolkit import SkillMetaToolkit

        skill = skill_manager.load_local_skill(self.path)
        toolkit = SkillMetaToolkit(allowed_skills=[skill])

        return await toolkit.resolve(context)
