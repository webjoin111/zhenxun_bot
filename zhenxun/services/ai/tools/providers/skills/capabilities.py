from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zhenxun.services.ai.capabilities import AbstractCapability
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
from zhenxun.services.ai.tools.providers.skills.models import Skill, SkillSource
from zhenxun.services.ai.tools.providers.skills.toolkit import SkillMetaToolkit
from zhenxun.utils.utils import infer_plugin_namespace


class SkillCapability(AbstractCapability):
    """技能库挂载能力组件"""

    def __init__(
        self,
        skills: Sequence[str | Path | Skill | SkillSource] | None = None,
        namespace: str | None = None,
    ):
        self.skills = skills or []
        self.namespace = namespace or infer_plugin_namespace()

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []

        if self.skills:
            resolved_skills = await skill_manager.resolve_mixed_skills(
                self.skills, namespace=self.namespace
            )
            tools.append(SkillMetaToolkit(allowed_skills=resolved_skills))

        return tools
