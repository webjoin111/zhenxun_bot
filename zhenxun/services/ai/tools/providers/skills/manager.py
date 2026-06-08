import asyncio
import json
from pathlib import Path
import re
from typing import cast
from typing_extensions import Self

import yaml

from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.tools.providers.skills.models import (
    INSTRUCTIONS,
    METADATA,
    RESOURCES,
    Skill,
    SkillEnvConfig,
    SkillFrontmatter,
)
from zhenxun.services.log import logger


class SkillConfigManager:
    """基于 Skill ID 隔离的持久化配置金库管理器"""

    def __init__(self):
        self.config_path = DATA_PATH / "ai" / "skill_envs.json"
        self.config = SkillEnvConfig()
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    self.config = SkillEnvConfig.model_validate(json.load(f))
            except Exception as e:
                logger.error(f"加载 skill_envs.json 失败: {e}")

    def save(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存 skill_envs.json 失败: {e}")

    def get_envs_for_skill(self, skill_id: str) -> dict[str, str]:
        return self.config.envs.get(skill_id, {})

    def ensure_template(self, skill_id: str, required_envs: list[str]):
        """确保配置文件中存在所需的模板，缺失则补充为空字符串"""
        if not required_envs:
            return

        changed = False
        skill_envs = self.config.envs.setdefault(skill_id, {})
        for env_key in required_envs:
            if env_key not in skill_envs:
                skill_envs[env_key] = ""
                changed = True

        if changed:
            self.save()


skill_env_manager = SkillConfigManager()


class SkillManager:
    """全局技能注册与发现中心 (本地文件系统模式)"""

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
        p = Path(path).resolve()
        if p not in self._scan_dirs:
            self._scan_dirs.append(p)

    def load_local_skill(self, path: str | Path) -> Skill:
        """
        [无痕加载] 读取指定路径的技能并返回独立的 Skill 实例。
        该技能不会被注册到全局 manager 中，专门用于局部按需挂载。
        """
        skill = self._parse_skill_from_path(path)
        if not skill:
            raise ValueError(
                f"无法从路径加载技能，请检查目录与 SKILL.md 是否合法: {path}"
            )
        return skill

    async def discover_skills(self) -> dict[str, Skill]:
        """遍历所有扫描目录，发现并合并所有合法技能"""
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
                            self._skills[skill.id] = skill
                    except Exception as e:
                        logger.error(f"解析技能 {child.name} 失败: {e}", e=e)

            logger.info(f"技能扫描完成，共加载 {len(self._skills)} 个技能。")
            return self._skills

    async def get_skill_details(self, name: str) -> Skill | None:
        """获取技能详细信息，并在内存中缓存补全后的资源"""
        skills = await self.discover_skills()
        skill = skills.get(name)
        if not skill:
            return None
        if skill.disclosure_level < RESOURCES:
            skill = self._load_skill_resources(skill)
            async with self._discovery_lock:
                self._skills[name] = skill
        return skill

    async def read_skill_resource(self, skill: Skill, file_path: str) -> str | None:
        """提供统一的资源文件（如 markdown 参考、脚本等）物理读取接口"""
        target_file = (skill.path / file_path).resolve()
        if not target_file.is_file():
            fallback_file = (skill.path / "scripts" / file_path).resolve()
            if fallback_file.is_file():
                target_file = fallback_file

        if target_file.is_file() and target_file.is_relative_to(skill.path.resolve()):
            return target_file.read_text(encoding="utf-8")
        return None

    async def clear_cache(self):
        """清空缓存以重新扫描(方便测试热更)"""
        async with self._discovery_lock:
            self._skills.clear()

    def _parse_skill_from_path(self, path: str | Path) -> Skill | None:
        """按径解析技能，返回提权到最高级别的孤立技能实例，不污染全局状态"""
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            logger.warning(f"[SkillManager] 路径不存在或不是目录: {p}")
            return None
        try:
            skill = self._parse_skill_metadata(p)
            if skill:
                skill = self._load_skill_resources(skill)
            return skill
        except Exception as e:
            logger.error(f"解析孤立技能 {p.name} 失败: {e}", e=e)
            return None

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
        fm_dict, _ = self._parse_frontmatter_and_body(skill_dir)
        if (
            not isinstance(fm_dict, dict)
            or "name" not in fm_dict
            or "description" not in fm_dict
        ):
            return None

        skill = Skill(
            frontmatter=SkillFrontmatter(**fm_dict),
            path=skill_dir,
            disclosure_level=METADATA,
            source="local",
        )

        if skill.frontmatter.required_envs:
            skill_env_manager.ensure_template(skill.id, skill.frontmatter.required_envs)

        return skill

    def _load_skill_instructions(self, skill: Skill) -> Skill:
        if skill.disclosure_level >= INSTRUCTIONS:
            return skill
        _, body = self._parse_frontmatter_and_body(skill.path)
        return skill.with_disclosure_level(level=INSTRUCTIONS, instructions=body)

    def _load_skill_resources(self, skill: Skill) -> Skill:
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
