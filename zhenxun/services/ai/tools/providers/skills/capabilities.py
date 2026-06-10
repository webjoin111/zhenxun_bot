from typing import Any

from zhenxun.services.ai.protocols.capabilities import AbstractCapability
from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.providers.skills.manager import skill_manager
from zhenxun.services.ai.tools.providers.skills.toolkit import SkillMetaToolkit
from zhenxun.services.log import logger


class SkillCapability(AbstractCapability):
    """技能库挂载能力组件"""

    def __init__(self):
        self.skills: list[str] = []
        self.available_skills: list[str] = []

    async def get_system_prompts(self, context: RunContext) -> list[str]:
        prompts = []

        if self.skills:
            skill_parts = []
            for skill_name in self.skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill:
                    skill_parts.append(
                        f"## Skill: {skill.name}\n\n{skill.instructions}"
                    )
                else:
                    logger.warning(
                        "SkillCapability 请求挂载的技能 "
                        f"'{skill_name}' 不存在，已跳过。"
                    )

            if skill_parts:
                prompts.append(
                    "\n\n--- 挂载的专用技能手册 ---\n\n"
                    + "\n\n".join(skill_parts)
                )

        if self.available_skills:
            catalog_parts = []
            for skill_name in self.available_skills:
                skill = await skill_manager.get_skill_details(skill_name)
                if skill:
                    catalog_parts.append(
                        f"  <skill>\n    <name>{skill.id}</name>\n"
                        f"    <description>{skill.description}</description>\n"
                        "  </skill>"
                    )

            if catalog_parts:
                catalog_xml = (
                    "<available_skills>\n"
                    + "\n".join(catalog_parts)
                    + "\n</available_skills>"
                )
                instruction = (
                    "### 🛠️ [外部技能调用规范]\n"
                    "以下是系统外挂技能目录。**禁止**臆造参数或直接推测调用。\n"
                    "**标准操作程序 (SOP)：**\n"
                    "1. **查阅指南**：必须首先调用 "
                    "`read_skill_instructions` 获取该技能的详细指南。\n"
                    "2. **精准执行**：阅读指南后，严格按照规范使用 "
                    "`run_skill_script` 执行。\n"
                    "3. **严禁盲测**：严禁在未读取指南的情况下尝试猜测参数。"
                )
                prompts.append(
                    f"\n\n--- 可选技能库 ---\n\n{instruction}\n{catalog_xml}"
                )

        return prompts

    async def get_tools(self, context: RunContext) -> list[Any]:
        tools = []

        if self.skills or self.available_skills:
            tools.append(SkillMetaToolkit())

        return tools
