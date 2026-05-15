import asyncio
from typing import Annotated

from pydantic import Field

from zhenxun.services.ai.core.exceptions import ControlFlowException, EndRunException
from zhenxun.services.ai.core.stream_events import ToolCallStart, ToolStreamChunk
from zhenxun.services.ai.flow.base import BaseRunnable
from zhenxun.services.ai.run.context import RunContext
from zhenxun.services.ai.run.models import AgentRunEnd
from zhenxun.services.ai.tools.core.decorators import tool
from zhenxun.services.ai.tools.core.toolkit import BaseToolkit
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger

from .task_board import TaskBoardState, TaskNodeStatus


class TaskManagementToolkit(BaseToolkit):
    """
    自主任务管理工具箱。
    提供给 Team Leader 用于操作黑板状态并调度 Member Agent。
    """

    default_instructions = (
        "<instructions>\n"
        "## 🛠️ 自主任务工作流指南\n"
        "你目前处于自主任务模式。你需要将用户的庞大目标拆解为具体的任务，并指派给专家。\n"
        "1. **规划**：使用 `create_task` 拆解任务，设定 `assignee`（专家名称）和 `depends_on`（依赖的其它任务ID）。\n"
        "2. **执行**：使用 `execute_task`（单个）或 `execute_tasks_parallel`（并发）驱动专家执行没有被阻塞的 pending 任务。\n"
        "3. **看板**：每次调用工具后，你会收到最新的看板快照，请以此决定下一步行动。\n"
        "4. **终结**：当你认为所有任务都已经完成，或目标已达成时，调用 `mark_all_complete` 并附上最终总结。\n"
        "</instructions>"
    )

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
                description="负责执行此任务的专家名称，必须完全匹配 <team_members> 中提供的 id，严禁捏造"
            ),
        ],
        depends_on: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="该任务依赖的前置任务 ID 列表。如果该任务可独立执行，请留空数组 []",
            ),
        ],
        context: RunContext,
    ) -> ToolResult:
        board = self._get_board(context)

        valid_member_names = [m.name for m in self.members]
        if assignee not in valid_member_names:
            return ToolResult(
                output=f"❌ 创建失败：未找到名为 '{assignee}' 的专家。可用专家: {valid_member_names}"
            ).as_error()

        task = board.create_task(
            title=title,
            description=description,
            assignee=assignee,
            dependencies=depends_on,
        )
        board_str = board.render_board_to_string()
        return ToolResult(
            output=f"✅ 任务创建成功！任务 ID: [{task.id}]，状态: {task.status.value}\n\n{board_str}"
        )

    @tool(
        description="手动强制更新任务的状态（仅在特殊情况下使用，因为 execute_task 会自动更新状态）。"
    )
    async def update_task_status(
        self,
        task_id: Annotated[str, Field(description="要更新的任务的唯一 ID")],
        status: Annotated[
            TaskNodeStatus, Field(description="新的任务状态，如 completed, failed")
        ],
        context: RunContext,
        result: Annotated[
            str, Field(description="如果任务已完成或失败，提供对应的结果或原因说明")
        ] = "",
    ) -> ToolResult:
        board = self._get_board(context)
        updated = board.update_task_status(task_id, status, result if result else None)
        if not updated:
            return ToolResult(output=f"❌ 找不到 ID 为 '{task_id}' 的任务。").as_error()
        board_str = board.render_board_to_string()
        return ToolResult(
            output=f"✅ 任务 [{task_id}] 已更新为 {status.value}。\n\n{board_str}"
        )

    @tool(
        description="声明整体目标已完成。在调用此工具后，大模型将被立刻中断并直接将 summary 返回给用户。"
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

        raise EndRunException(result_output=summary, display=None)

    @tool(
        description="执行一个处于 pending 状态的任务。系统会自动调用对应的专家来完成它。"
    )
    async def execute_task(
        self,
        task_id: Annotated[str, Field(description="要执行的任务 ID")],
        context: RunContext,
    ) -> ToolResult:
        board = self._get_board(context)
        task = board.get_task(task_id)

        if not task:
            return ToolResult(output=f"❌ 找不到任务 '{task_id}'。").as_error()
        if task.status != TaskNodeStatus.pending:
            return ToolResult(
                output=f"❌ 任务 [{task_id}] 的状态为 {task.status.value}，不可执行。必须为 pending。"
            ).as_error()
        if not task.assignee:
            return ToolResult(
                output=f"❌ 任务 [{task_id}] 尚未指派 assignee。"
            ).as_error()

        member_agent = next((m for m in self.members if m.name == task.assignee), None)
        if not member_agent:
            return ToolResult(
                output=f"❌ 找不到分配的专家 '{task.assignee}'。"
            ).as_error()

        board.update_task_status(task_id, TaskNodeStatus.in_progress)
        streamer = context.run.streamer
        if streamer:
            await streamer.send(
                ToolStreamChunk(
                    tool_name="TaskManager",
                    content=f"🚀 正在委派 [{task.assignee}] 执行任务: {task.title}...",
                )
            )

        logger.info(
            f"🔄 [ExecuteTask] 正在启动子专家 {member_agent.name} 处理任务 {task_id}"
        )
        sub_context = context.clone_for_member(member_agent.name)

        try:
            task_prompt = f"### 🎯 你被指派的任务目标：\n{task.description}"

            if task.dependencies:
                dep_results = []
                for dep_id in task.dependencies:
                    dep_task = board.get_task(dep_id)
                    if dep_task and dep_task.result:
                        dep_results.append(
                            f"【前置任务 [{dep_task.title}] 的产出】:\n{dep_task.result}"
                        )
                if dep_results:
                    task_prompt += (
                        "\n\n### 📦 你的任务依赖以下前置结果，请基于此进行处理：\n"
                        + "\n\n".join(dep_results)
                    )

            response = None
            async with member_agent.run_stream(
                prompt=task_prompt, context=sub_context
            ) as stream_result:
                async for event in stream_result.stream_events():

                    if isinstance(event, AgentRunEnd):
                        response = event.result
                    elif streamer:
                        if isinstance(event, ToolStreamChunk):
                            await streamer.send(event)
                        elif isinstance(event, ToolCallStart):
                            await streamer.send(
                                ToolStreamChunk(
                                    tool_name=member_agent.name,
                                    content=f"🔁 正在调用其专属工具: {event.tool_name}...",
                                )
                            )

            if response is None:
                raise RuntimeError(f"专家 {member_agent.name} 未返回任何响应。")

            final_output = str(response.output)

            board.update_task_status(task_id, TaskNodeStatus.completed, final_output)
            board_str = board.render_board_to_string()

            if streamer:
                await streamer.send(
                    ToolStreamChunk(
                        tool_name="TaskManager",
                        content=f"✅ [{task.assignee}] 已完成任务: {task.title}！",
                    )
                )

            return ToolResult(
                output=f"✅ 任务执行成功。专家返回结果:\n{final_output}\n\n{board_str}"
            )

        except Exception as e:
            if isinstance(e, ControlFlowException):
                raise e
            logger.error(f"执行子任务失败: {e}", e=e)
            board.update_task_status(task_id, TaskNodeStatus.failed, f"执行异常: {e}")
            board_str = board.render_board_to_string()
            if streamer:
                await streamer.send(
                    ToolStreamChunk(
                        tool_name="TaskManager",
                        content=f"❌ [{task.assignee}] 执行任务 {task.title} 失败！",
                    )
                )
            return ToolResult(
                output=f"❌ 任务执行失败，异常信息: {e}\n\n{board_str}"
            ).as_error()

    @tool(description="并行执行多个互不依赖的 pending 任务。利用并发最大化执行效率。")
    async def execute_tasks_parallel(
        self,
        task_ids: Annotated[list[str], Field(description="要并行执行的任务 ID 列表")],
        context: RunContext,
    ) -> ToolResult:
        board = self._get_board(context)

        tasks_to_run = []
        for tid in task_ids:
            task = board.get_task(tid)
            if not task:
                return ToolResult(output=f"❌ 找不到任务 '{tid}'。").as_error()
            if task.status != TaskNodeStatus.pending:
                return ToolResult(
                    output=f"❌ 任务 [{tid}] 状态为 {task.status.value}，不可执行。必须为 pending。"
                ).as_error()
            if not task.assignee:
                return ToolResult(
                    output=f"❌ 任务 [{tid}] 尚未指派 assignee。"
                ).as_error()
            member_agent = next(
                (m for m in self.members if m.name == task.assignee), None
            )
            if not member_agent:
                return ToolResult(
                    output=f"❌ 找不到分配的专家 '{task.assignee}'。"
                ).as_error()
            tasks_to_run.append((task, member_agent))

        if not tasks_to_run:
            return ToolResult(output="❌ 提供的任务列表均不可执行。").as_error()

        streamer = context.run.streamer
        if streamer:
            await streamer.send(
                ToolStreamChunk(
                    tool_name="TaskManager",
                    content=f"🚀 正在并发委派执行 {len(tasks_to_run)} 个任务...",
                )
            )

        for task, _ in tasks_to_run:
            board.update_task_status(task.id, TaskNodeStatus.in_progress)

        async def _run_single(task, member_agent):
            logger.info(
                f"🔄 [Parallel] 启动专家 {member_agent.name} 处理任务 {task.id}"
            )
            sub_context = context.clone_for_member(member_agent.name)
            task_prompt = f"### 🎯 你被指派的任务目标：\n{task.description}"

            if task.dependencies:
                dep_results = []
                for dep_id in task.dependencies:
                    dep_task = board.get_task(dep_id)
                    if dep_task and dep_task.result:
                        dep_results.append(
                            f"【前置任务 [{dep_task.title}] 的产出】:\n{dep_task.result}"
                        )
                if dep_results:
                    task_prompt += (
                        "\n\n### 📦 你的任务依赖以下前置结果，请基于此进行处理：\n"
                        + "\n\n".join(dep_results)
                    )

            try:
                response = None
                async with member_agent.run_stream(
                    prompt=task_prompt, context=sub_context
                ) as stream_result:
                    async for event in stream_result.stream_events():

                        if isinstance(event, AgentRunEnd):
                            response = event.result
                        elif streamer:
                            if isinstance(event, ToolStreamChunk):
                                await streamer.send(event)
                            elif isinstance(event, ToolCallStart):
                                await streamer.send(
                                    ToolStreamChunk(
                                        tool_name=member_agent.name,
                                        content=f"🔁 正在调用其专属工具: {event.tool_name}...",
                                    )
                                )
                if response is None:
                    raise RuntimeError(f"专家 {member_agent.name} 未返回任何响应。")
                return (task.id, True, str(response.output))
            except Exception as e:
                if isinstance(e, ControlFlowException):
                    raise e
                logger.error(f"并行任务 {task.id} 失败: {e}", e=e)
                return (task.id, False, str(e))

        results = await asyncio.gather(*[_run_single(t, m) for t, m in tasks_to_run])

        result_outputs = []
        for tid, success, res_text in results:
            if success:
                board.update_task_status(tid, TaskNodeStatus.completed, res_text)
                result_outputs.append(f"✅ 任务 [{tid}] 成功: {res_text}")
            else:
                board.update_task_status(
                    tid, TaskNodeStatus.failed, f"执行异常: {res_text}"
                )
                result_outputs.append(f"❌ 任务 [{tid}] 失败: {res_text}")

        if streamer:
            await streamer.send(
                ToolStreamChunk(
                    tool_name="TaskManager",
                    content=f"🏁 并发执行完毕，成功 {sum(1 for _, s, _ in results if s)} 个，失败 {sum(1 for _, s, _ in results if not s)} 个。",
                )
            )

        board_str = board.render_board_to_string()
        final_output_str = "\n".join(result_outputs)
        return ToolResult(
            output=f"并行执行结束。结果摘要:\n{final_output_str}\n\n{board_str}"
        )
