import asyncio
from pathlib import Path
import re
from typing import cast
from typing_extensions import Self

import yaml

from zhenxun.services.log import logger

from .models import INSTRUCTIONS, METADATA, RESOURCES, Skill, SkillFrontmatter


class SkillManager:
    """全局技能注册与发现中心 (极简文件系统加载模式)"""

    _instance: "SkillManager | None" = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Self, cls._instance)

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._skills: dict[str, Skill] = {}
        self._scan_dirs: list[Path] = []
        self._initialized = True
        self._discovery_lock = asyncio.Lock()

    def add_scan_dir(self, path: str | Path):
        """添加技能扫描目录"""
        p = Path(path).resolve()
        if p not in self._scan_dirs:
            self._scan_dirs.append(p)

    async def discover_skills(self) -> dict[str, Skill]:
        """执行目录扫描，发现并解析所有合法技能"""
        async with self._discovery_lock:
            if self._skills:
                return self._skills

            for scan_dir in self._scan_dirs:
                if not scan_dir.exists() or not scan_dir.is_dir():
                    continue
                for child in scan_dir.iterdir():
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    try:
                        skill = self._parse_skill_metadata(child)
                        if skill:
                            if skill.id in self._skills:
                                logger.warning(
                                    f"Skill 名称冲突: {skill.id}，新加载的将覆盖旧的。"
                                )
                            self._skills[skill.id] = skill
                            logger.debug(f"成功发现 Skill Metadata: {skill.id}")
                    except Exception as e:
                        logger.error(f"解析技能 {child.name} 失败: {e}", e=e)

            logger.info(f"技能扫描完成，共加载 {len(self._skills)} 个技能。")
            return self._skills

    async def get_skill_details(self, name: str) -> Skill | None:
        """获取技能并强制提升至 RESOURCES(3) 级别"""
        skills = await self.discover_skills()
        skill = skills.get(name)
        if not skill:
            return None

        if skill.disclosure_level < RESOURCES:
            async with self._discovery_lock:
                skill = self._load_skill_resources(skill)
                self._skills[name] = skill
        return skill

    async def clear_cache(self):
        """清空缓存以重新扫描(方便测试热更)"""
        async with self._discovery_lock:
            self._skills.clear()

    def _parse_frontmatter_and_body(self, skill_dir: Path) -> tuple[dict | None, str]:
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            return None, ""
        content = skill_md_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None, ""
        match = re.search(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
        if not match:
            return None, ""
        yaml_content, body = match.groups()
        try:
            return yaml.safe_load(yaml_content), body.strip()
        except yaml.YAMLError as e:
            logger.warning(f"Skill {skill_dir.name} YAML 解析失败: {e}")
            return None, ""

    def _parse_skill_metadata(self, skill_dir: Path) -> Skill | None:
        """[等级1] 解析 YAML 元数据"""
        fm_dict, _ = self._parse_frontmatter_and_body(skill_dir)
        if (
            not isinstance(fm_dict, dict)
            or "name" not in fm_dict
            or "description" not in fm_dict
        ):
            return None
        return Skill(
            frontmatter=SkillFrontmatter(**fm_dict),
            path=skill_dir,
            disclosure_level=METADATA,
        )

    def _load_skill_instructions(self, skill: Skill) -> Skill:
        """[等级2] 加载 Markdown 正文"""
        if skill.disclosure_level >= INSTRUCTIONS:
            return skill
        _, body = self._parse_frontmatter_and_body(skill.path)
        return skill.with_disclosure_level(level=INSTRUCTIONS, instructions=body)

    def _load_skill_resources(self, skill: Skill) -> Skill:
        """[等级3] 扫描资源目录 (scripts, references)"""
        if skill.disclosure_level >= RESOURCES:
            return skill
        if skill.disclosure_level < INSTRUCTIONS:
            skill = self._load_skill_instructions(skill)

        scripts = (
            [
                f.name
                for f in (skill.path / "scripts").iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
            if (skill.path / "scripts").is_dir()
            else []
        )
        refs = (
            [
                f.name
                for f in (skill.path / "references").iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
            if (skill.path / "references").is_dir()
            else []
        )
        return skill.with_disclosure_level(
            level=RESOURCES, scripts=scripts, references=refs
        )


skill_manager = SkillManager()
