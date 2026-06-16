from typing import Protocol, runtime_checkable

from zhenxun.services.ai.context.rag.models import BaseRecord, ConsolidationPlan
from zhenxun.services.ai.llm.api import generate_structured
from zhenxun.services.log import logger


@runtime_checkable
class Consolidator(Protocol):
    """数据融合器协议"""

    async def consolidate(
        self, new_content: str, existing_records: list[BaseRecord]
    ) -> ConsolidationPlan: ...


class NullConsolidator(Consolidator):
    """无作为的融合器：永远直接插入"""

    async def consolidate(
        self, new_content: str, existing_records: list[BaseRecord]
    ) -> ConsolidationPlan:
        return ConsolidationPlan(actions=[], insert_new=True)


class LLMConsolidator(Consolidator):
    """基于大模型的智能数据融合器"""

    default_prompt_template = """你是一个高级知识库的记忆整合引擎。
最新输入的内容是：
<new_memory>
{new_content}
</new_memory>

以下是数据库中检索到的、与新内容语义相似的现有记录：
<existing_records>
{records_str}
</existing_records>

你的任务是：
1. 评估新内容与现有记录的关系。
2. 决定对每一条现有记录采取什么动作（keep, update, delete）。
   - 如果新旧内容互补但应该合并为一条，对旧记录使用 update，并在 new_content 字段中填入合并后的完整文本。
   - 如果新内容证明旧记录已过期/完全错误，对旧记录使用 delete。
   - 如果新内容与旧内容不冲突，对旧记录使用 keep。
3. 决定是否将新内容作为全新的一条记录插入（insert_new）。
   - 如果你已经将新信息合并到了某个旧记录的 update 动作中，或者旧记录已经包含了新信息，请务必设置 insert_new=False。
   - 如果这是一条完全独立的新信息，请设置 insert_new=True。"""  # noqa: E501

    def __init__(
        self,
        model_name: str | None = None,
        prompt_template: str | None = None,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template or self.default_prompt_template

    async def consolidate(
        self, new_content: str, existing_records: list[BaseRecord]
    ) -> ConsolidationPlan:
        if not existing_records:
            return ConsolidationPlan(actions=[], insert_new=True)

        records_str = ""
        for r in existing_records:
            records_str += f"- [ID: {r.id}] 内容: {r.content}\n"

        prompt = self.prompt_template.format(
            new_content=new_content,
            records_str=records_str,
        )

        from zhenxun.services.ai.llm.manager import get_default_model

        model_to_use = self.model_name or get_default_model("chat")

        try:
            logger.debug("🧠 正在启动 RAG 数据融合分析 (Consolidation)...")
            plan = await generate_structured(
                message=prompt,
                response_model=ConsolidationPlan,
                model=model_to_use,
                instruction="严格按照逻辑判断，不要遗失关键细节。",
            )
            return plan
        except Exception as e:
            logger.warning(f"RAG 融合分析失败，将降级为直接追加模式: {e}")
            return ConsolidationPlan(actions=[], insert_new=True)
