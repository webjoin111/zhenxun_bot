from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DisclosureLevel = Annotated[
    Literal[1, 2, 3], "Progressive disclosure levels for skill loading."
]
METADATA: DisclosureLevel = 1
INSTRUCTIONS: DisclosureLevel = 2
RESOURCES: DisclosureLevel = 3


class SkillFrontmatter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述")
    license: str | None = Field(default=None, description="许可证")
    compatibility: str | None = Field(default=None, description="环境要求")
    metadata: dict[str, Any] | None = Field(default=None, description="自定义元数据")

    allowed_tools: list[str] | None = Field(
        default=None, alias="allowed-tools", description="允许的工具"
    )
    permissions: dict[str, Any] | None = Field(
        default=None, description="沙箱权限声明 (如 network: true/false)"
    )

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
        """网络权限：默认开启（高信任静态资产），允许在 YAML 中通过 permissions.network 显式关闭"""
        if self.permissions and isinstance(self.permissions, dict):
            return self.permissions.get("network", True)
        return True


class Skill(BaseModel):
    frontmatter: SkillFrontmatter = Field(description="解析后的YAML元数据")
    instructions: str | None = Field(
        default=None, description="SKILL.md正文指令，在INSTRUCTIONS级别填充"
    )
    path: Path = Field(description="技能所在目录")
    disclosure_level: DisclosureLevel = Field(
        default=METADATA, description="当前披露等级"
    )
    scripts: list[str] = Field(default_factory=list, description="脚本文件列表")
    references: list[str] = Field(default_factory=list, description="参考文档列表")

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
        )

    @property
    def id(self) -> str:
        """技能的唯一标识符（强制使用所在文件夹的名称）"""
        return self.path.name

    @property
    def name(self) -> str:
        """技能的显示名称（来自 YAML，可能不符合规范）"""
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description
