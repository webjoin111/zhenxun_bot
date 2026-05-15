from enum import Enum
import uuid

from pydantic import BaseModel, Field


class TaskNodeStatus(str, Enum):
    """团队自主任务节点状态枚举"""

    pending = "pending"
    """待处理：所有前置依赖已完成，等待分配执行"""
    in_progress = "in_progress"
    """进行中：正在被 Member Agent 执行"""
    completed = "completed"
    """已完成：执行成功"""
    failed = "failed"
    """已失败：执行报错或由于前置依赖失败而自动失败"""
    blocked = "blocked"
    """阻塞中：有前置依赖任务尚未完成"""


class SubTaskRecord(BaseModel):
    """单条子任务（工单）数据契约"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    assignee: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    status: TaskNodeStatus = TaskNodeStatus.pending
    result: str | None = None
    notes: list[str] = Field(default_factory=list)


class TaskBoardState(BaseModel):
    """
    Team 自主任务模式下的全局共享黑板状态 (Task Board)。
    提供任务的 CRUD、依赖拓扑计算和格式化渲染功能。
    """

    tasks: list[SubTaskRecord] = Field(default_factory=list)
    is_goal_complete: bool = False
    final_summary: str | None = None

    def create_task(
        self,
        title: str,
        description: str = "",
        assignee: str | None = None,
        dependencies: list[str] | None = None,
    ) -> SubTaskRecord:
        task = SubTaskRecord(
            title=title,
            description=description,
            assignee=assignee,
            dependencies=dependencies or [],
        )
        self.tasks.append(task)
        self._update_blocked_statuses()
        return task

    def get_task(self, task_id: str) -> SubTaskRecord | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def update_task_status(
        self, task_id: str, status: TaskNodeStatus, result: str | None = None
    ) -> SubTaskRecord | None:
        task = self.get_task(task_id)
        if not task:
            return None
        task.status = status
        if result is not None:
            task.result = result
        self._update_blocked_statuses()
        return task

    def _is_blocked(self, task: SubTaskRecord) -> bool:
        """检查该任务是否有尚未完成的前置依赖"""
        if not task.dependencies:
            return False
        for dep_id in task.dependencies:
            dep = self.get_task(dep_id)
            if dep is None:
                return True
            if dep.status != TaskNodeStatus.completed:
                return True
        return False

    def _has_failed_dependency(self, task: SubTaskRecord) -> bool:
        """检查该任务是否有已经失败的前置依赖"""
        if not task.dependencies:
            return False
        for dep_id in task.dependencies:
            dep = self.get_task(dep_id)
            if dep is not None and dep.status == TaskNodeStatus.failed:
                return True
        return False

    def _update_blocked_statuses(self) -> None:
        """重新计算所有未终结任务的阻塞状态 (基于拓扑依赖)"""
        for task in self.tasks:
            if task.status == TaskNodeStatus.blocked:
                if self._has_failed_dependency(task):
                    task.status = TaskNodeStatus.failed
                    task.result = "自动标记失败: 前置依赖任务已失败。"
                elif not self._is_blocked(task):
                    task.status = TaskNodeStatus.pending
            elif task.status == TaskNodeStatus.pending:
                if self._has_failed_dependency(task):
                    task.status = TaskNodeStatus.failed
                    task.result = "自动标记失败: 前置依赖任务已失败。"
                elif self._is_blocked(task):
                    task.status = TaskNodeStatus.blocked

    def get_available_tasks(self, for_assignee: str | None = None) -> list[SubTaskRecord]:
        """获取所有当前无依赖阻塞、可立即执行的 Pending 任务"""
        available = []
        for task in self.tasks:
            if task.status != TaskNodeStatus.pending:
                continue
            if self._is_blocked(task):
                continue
            if for_assignee and task.assignee and task.assignee != for_assignee:
                continue
            available.append(task)
        return available

    def all_terminal(self) -> bool:
        """判断是否所有的任务都已经进入了终结状态（完成或失败）"""
        if not self.tasks:
            return False
        return all(
            t.status in (TaskNodeStatus.completed, TaskNodeStatus.failed)
            for t in self.tasks
        )

    def render_board_to_string(self) -> str:
        """渲染供 LLM 阅读的 Markdown 看板战报"""
        if not self.tasks:
            return "目前尚未创建任何任务。"

        counts: dict[str, int] = {}
        for t in self.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1

        parts = [f"{v} {k}" for k, v in counts.items()]
        header = f"### 📋 任务状态总览 (共计 {len(self.tasks)} 个任务: {', '.join(parts)}):"

        lines = [header]
        for t in self.tasks:
            status_str = t.status.value.upper()
            assignee_str = f" (指派给: {t.assignee})" if t.assignee else " (尚未指派)"
            lines.append(f"  [{t.id}] {t.title} - {status_str}{assignee_str}")
            if t.dependencies:
                lines.append(f"      依赖于: {t.dependencies}")
            if t.result:
                result_preview = (
                    t.result[:200] + "..." if len(t.result) > 200 else t.result
                )
                lines.append(f"      结果: {result_preview}")
            if t.notes:
                for note in t.notes[-3:]:
                    lines.append(f"      附注: {note}")

        if self.is_goal_complete and self.final_summary:
            lines.append(f"\n✅ 终极目标已标记完成: {self.final_summary}")

        return "<current_task_state>\n" + "\n".join(lines) + "\n</current_task_state>"
