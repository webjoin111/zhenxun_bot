from zhenxun.services.ai.llm.api import generate_structured
from zhenxun.services.ai.memory.interfaces import MemoryConsolidator
from zhenxun.services.ai.memory.models import ConsolidationPlan, MemoryRecord
from zhenxun.services.log import logger


class LLMMemoryConsolidator(MemoryConsolidator):
    """
    基于大模型的智能记忆整合器 (The AI Brain)。
    在保存新记忆前，通过大模型分析新记忆与现有相似记忆的关系，
    智能决策合并、更新、删除或直接追加，从根本上解决知识库的冗余和冲突问题。
    """

    def __init__(self, model_name: str | None = None):
        """
        :param model_name: 用于执行记忆 analysis 的 LLM 模型名。如果为空则使用系统默认模型。
        """
        self.model_name = model_name

    async def consolidate(
        self, new_content: str, existing_records: list[MemoryRecord]
    ) -> ConsolidationPlan:
        if not existing_records:
            return ConsolidationPlan(actions=[], insert_new=True)

        records_str = ""
        for r in existing_records:
            records_str += f"- [ID: {r.id}] 内容: {r.content}\n"

        prompt = f"""你是一个高级知识库的记忆整合引擎。
用户的最新记忆输入是：
<new_memory>
{new_content}
</new_memory>

以下是数据库中检索到的、与新记忆语义相似的现有记录：
<existing_records>
{records_str}
</existing_records>

你的任务是：
1. 评估新记忆与现有记录的关系。
2. 决定对每一条现有记录采取什么动作（keep, update, delete）。
   - 如果新旧记忆互补但应该合并为一条，对旧记录使用 update，并在 new_content 字段中填入合并后的完整文本。
   - 如果新记忆证明旧记录已过期/完全错误，对旧记录使用 delete。
   - 如果新记忆与旧记忆不冲突（比如只是恰好提到了相似的实体），对旧记录使用 keep。
3. 决定是否将新记忆作为全新的一条记录插入（insert_new）。
   - 如果你已经将新信息合并到了某个旧记录的 update 动作中，或者旧记录已经包含了新信息，请务必设置 insert_new=False。
   - 如果这是一条完全独立的新信息，请设置 insert_new=True。"""  # noqa: E501

        try:
            logger.debug("🧠 正在启动 LLM 记忆整合分析...")
            plan = await generate_structured(
                message=prompt,
                response_model=ConsolidationPlan,
                model=self.model_name,
                instruction="严格按照逻辑判断，不要遗失关键细节。",
            )
            return plan
        except Exception as e:
            logger.warning(f"LLM 记忆整合分析失败，将降级为直接追加模式: {e}")
            return ConsolidationPlan(actions=[], insert_new=True)
