from typing import Annotated, Any

from pydantic import Field

from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger

from .models import TaskBoardState, TaskNodeStatus


class TaskPlanningToolkit(BaseToolkit):
    """
    任务规划工具箱 (Planner Toolkit)。
    大模型专用的黑板操作工具。大模型被剥夺了执行权，仅能拆解、指派和总结任务。
    """

    default_prefix = ""

    default_instructions = """<instructions>
## 🛠️ 任务规划工作流指南
你现在的角色是**项目经理 (Planner)**。你的唯一职责是拆解任务、分配人员并监控看板状态，**系统底层会自动拉起专家执行任务**。
1. **规划**：使用 `create_task` 拆解任务，设定 `assignee`（专家名称）和 `depends_on`（依赖的其它任务ID）。
2. **等待与监控**：每次你创建或更新任务后，请立刻停止工具调用，系统引擎会自动并发执行 pending 任务并再次唤醒你。
3. **🩹 智能自愈与重试**：如果你被唤醒后，看到看板上有任务处于 `failed` 状态，
请仔细阅读失败结果 (result)。你可以通过 `update_task_status` 将该任务的状态重新修改为 `pending`
以触发重新执行（可以附带修改建议在 result 里），或者创建新任务替代它。
4. **终结**：当你确认所有目标已达成时，调用 `mark_all_complete` 附上最终总结，正式结束整个流水线。
⚠️ 警告：你没有任何执行具体业务代码或查询的工具，你只能操作任务看板！
</instructions>"""  # noqa: E501

    def __init__(self, members: list[BaseRunnable], **kwargs):
        super().__init__(**kwargs)
        self.members = members

    def _get_board(self, context: RunContext) -> TaskBoardState:
        """从运行上下文中安全的获取或初始化任务看板状态"""
        if "__task_board__" not in context.session.shared_state:
            context.session.shared_state["__task_board__"] = TaskBoardState()
        return context.session.shared_state["__task_board__"]

    @tool(description="创建一个新任务并加入看板。")
    async def create_task(
        self,
        title: Annotated[str, Field(description="任务的简短、可行动的标题")],
        description: Annotated[
            str, Field(description="详细的任务说明，告诉执行者需要做什么以及期望的产出")
        ],
        assignee: Annotated[
            str,
            Field(
                description=(
                    "负责执行此任务的专家名称，必须完全匹配 <team_members> "
                    "中提供的 id，严禁捏造"
                )
            ),
        ],
        context: RunContext,
        depends_on: Annotated[
            list[str] | None,
            Field(
                description=(
                    "该任务依赖的前置任务 ID 列表。如果该任务可独立执行，请留空数组 []"
                ),
            ),
        ] = None,
        metadata: Annotated[
            dict[str, Any] | None,
            Field(
                description="可选的附加字典，用于向执行专家传递额外的结构化约束或参数"
            ),
        ] = None,
    ) -> ToolResult:
        board = self._get_board(context)

        valid_member_names = [m.name for m in self.members]
        if assignee not in valid_member_names:
            return ToolResult(
                output=(
                    f"❌ 创建失败：未找到名为 '{assignee}' 的专家。"
                    f"可用专家: {valid_member_names}"
                )
            ).as_error()

        task = board.create_task(
            title=title,
            description=description,
            assignee=assignee,
            dependencies=depends_on,
            metadata=metadata,
        )

        logger.debug(f"  🆕 [新建任务] `{task.title}` -> 👨💼{task.assignee}")

        board_str = board.render_board_to_string()
        return ToolResult(
            output=(
                f"✅ 任务创建成功！任务 ID: [{task.id}]，"
                f"状态: {task.status.value}\n\n{board_str}"
            )
        )

    @tool(
        description=(
            "手动强制更新任务的状态（仅在特殊情况下使用，"
            "因为 execute_task 会自动更新状态）。"
        )
    )
    async def update_task_status(
        self,
        task_id: Annotated[str, Field(description="要更新的任务的唯一 ID")],
        status: Annotated[
            TaskNodeStatus,
            Field(
                description=(
                    "新的任务状态，支持: pending(用于重试), completed, failed 等"
                )
            ),
        ],
        context: RunContext,
        result: Annotated[
            str,
            Field(
                description=(
                    "提供结果、失败原因，或在设为 pending 重试时给执行专家的建议"
                )
            ),
        ] = "",
    ) -> ToolResult:
        board = self._get_board(context)
        updated = board.update_task_status(task_id, status, result if result else None)
        if not updated:
            return ToolResult(output=f"❌ 找不到 ID 为 '{task_id}' 的任务。").as_error()

        task_obj = board.get_task(task_id)
        task_title = task_obj.title if task_obj else "Unknown"
        logger.debug(f"  🔄 [任务状态变更] `{task_title}` -> {status.value}")

        board_str = board.render_board_to_string()
        return ToolResult(
            output=f"✅ 任务 [{task_id}] 已更新为 {status.value}。\n\n{board_str}"
        )

    @tool(
        description=(
            "声明整体目标已完成。在调用此工具后，"
            "大模型将被立刻中断并直接将 summary 返回给用户。"
        )
    )
    async def mark_all_complete(
        self,
        summary: Annotated[
            str, Field(description="对于整个流程执行结果的最终中文战报/总结")
        ],
        context: RunContext,
    ) -> ToolResult:
        board = self._get_board(context)
        board.is_goal_complete = True
        board.final_summary = summary

        from zhenxun.services.ai.tools.models import EndRunResult
        return EndRunResult(output=summary)
