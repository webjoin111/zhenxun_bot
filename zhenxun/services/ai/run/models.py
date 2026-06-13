"""
运行时（Run）相关核心类型定义
"""

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any, Generic, cast
from typing_extensions import TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from zhenxun.services.ai.core.messages import LLMMessage, UsageInfo
from zhenxun.services.ai.core.stream_events import AgentStreamEvent
from zhenxun.utils.pydantic_compat import model_dump


class ChatSummary(BaseModel):
    total: int = 0
    """大模型调用总次数"""
    total_latency_ms: float = 0.0
    """大模型调用总耗时（毫秒）"""
    by_stop_reason: dict[str, int] = Field(default_factory=dict)
    """按停止原因分类的大模型调用计数"""


class ToolSummary(BaseModel):
    total: int = 0
    """工具执行总次数"""
    ok: int = 0
    """工具成功执行次数"""
    error: int = 0
    """工具执行失败次数"""
    total_latency_ms: float = 0.0
    """工具执行总耗时（毫秒）"""
    by_name: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """按工具名称细分的执行状态统计"""


class AgentRunSummary(BaseModel):
    """Agent 单次运行的全局可观测性遥测摘要"""
    chats: ChatSummary = Field(default_factory=ChatSummary)
    """大模型调用遥测摘要"""
    tools: ToolSummary = Field(default_factory=ToolSummary)
    """工具执行遥测摘要"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """Token 消耗总计"""
    total_latency_ms: float = 0.0
    """智能体运行总耗时（毫秒）"""


class CancellationToken:
    """全局取消令牌，用于在异步链路中传递中止信号"""

    def __init__(self):
        self._cancelled = False
        self._futures: list[asyncio.Future] = []

    def cancel(self) -> None:
        self._cancelled = True
        for f in self._futures:
            if not f.done():
                f.cancel()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError(
                "任务已被主动取消 (CancellationToken triggered)"
            )

    def link_future(self, future: asyncio.Future) -> None:
        if self._cancelled:
            future.cancel()
        else:
            self._futures.append(future)


class HandoffPayload(BaseModel):
    """移交信息载荷"""
    target: str
    reason: str = ""
    context_data: Any = ""


OutputDataT = TypeVar("OutputDataT", default=str)


class AgentRunResult(BaseModel, Generic[OutputDataT]):
    """Agent 单次无状态运行的结果"""

    output: OutputDataT
    """最终输出数据"""
    messages: list[LLMMessage] = Field(default_factory=list)
    """本次运行产生/更新的历史消息"""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    """本次运行的Token消耗总计"""
    structured_data: Any | None = None
    """拦截到的结构化结果字典"""
    telemetry: AgentRunSummary | None = None
    """单次运行的完整可观测性遥测摘要"""
    handoff: HandoffPayload | None = None
    """向外抛出的软移交载荷（存在时说明Agent发起了移交请求）"""

    class Config:
        arbitrary_types_allowed = True


class AgentRunStart(AgentStreamEvent):
    """智能体运行开始"""

    agent_name: str


class AgentRunError(AgentStreamEvent):
    """智能体运行发生异常"""

    error: BaseException


class AgentRunEnd(AgentStreamEvent):
    """智能体运行完全结束"""

    result: AgentRunResult[Any]


class StreamedRunResult(Generic[OutputDataT]):
    """
    智能体流式运行的结果代理对象。
    提供高度解耦的方法来消费底层事件流，支持获取纯净文本或全部事件。
    """

    def __init__(self, streamer: Any):
        self._streamer = streamer
        self.is_complete: bool = False
        self._result: AgentRunResult[OutputDataT] | None = None

    async def stream_events(self) -> AsyncIterator[Any]:
        """获取底层的所有原始事件（包含工具调用过程等）"""
        from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunError

        async for event in self._streamer:
            if isinstance(event, AgentRunEnd):
                self._result = cast(AgentRunResult[OutputDataT], event.result)
                self.is_complete = True
            elif isinstance(event, AgentRunError):
                raise event.error.with_traceback(None) from None
            yield event

    async def stream_text(self, delta: bool = False) -> AsyncIterator[str]:
        """
        过滤大模型的输出文本。
        """
        full_text = ""
        async for _ in self.stream_events():
            pass

        if self._result is not None:
            output = self._result.output
            if isinstance(output, str):
                full_text = output
            else:
                if isinstance(output, BaseModel):
                    full_text = json.dumps(model_dump(output), ensure_ascii=False)
                else:
                    full_text = str(output)

            if delta:
                yield full_text
            else:
                yield full_text

    async def get_output(self) -> OutputDataT:
        """阻塞并等待整个 Agent 执行完毕，返回最终的解析输出数据"""
        if self._result is not None:
            return self._result.output

        async for _ in self.stream_events():
            pass

        if self._result is None:
            raise RuntimeError("Agent 运行异常结束，未产生最终结果。")

        return self._result.output

    async def get_run_result(self) -> AgentRunResult[OutputDataT]:
        """获取完整的运行结果对象（包含 Token 消耗等）"""
        if self._result is not None:
            return self._result

        await self.get_output()

        return cast(AgentRunResult[OutputDataT], self._result)


class TaskResult(BaseModel):
    """单个数据契约任务的执行结果"""

    task_id: str
    """关联的任务唯一 ID"""

    output: Any
    """任务的实际产出（如果是结构化任务，则为解析后的 Pydantic 实例；否则为纯文本）"""

    raw_response: str | None = None
    """大模型返回的原始纯文本内容"""

    usage: UsageInfo = Field(default_factory=UsageInfo)
    """该任务执行期间的 Token 消耗统计"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Task(BaseModel):
    """标准化数据契约（意图载体 Payload），定义大模型需要做什么及产出什么格式"""

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    """任务的唯一标识符"""

    name: str | None = None
    """任务的简短名称"""

    description: str
    """详细的任务指令（告诉大模型具体需要做什么）"""

    expected_output: str
    """预期输出的自然语言描述（指导大模型如何组织最终答案）"""

    response_model: Any | None = None
    """强制要求返回的强类型结构 (Pydantic Model) 或
    OutputDefinition，为空则返回普通文本"""

    tools: list[str | Any] | None = None
    """针对此特定任务动态追加或覆盖的工具列表"""

    guardrails: list[Any] | None = None
    """护栏验证列表。支持传入函数、BaseGuardrail 实例，
    或直接传入自然语言字符串规则（自动转为 LLM 裁判）"""

    _parsed_guardrails: list[Any] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _parse_and_set_guardrails(self) -> "Task":
        from zhenxun.services.ai.core.guardrails import parse_guardrails

        self._parsed_guardrails = parse_guardrails(self.guardrails)
        return self


__all__ = [
    "AgentRunResult",
    "CancellationToken",
    "OutputDataT",
    "StreamedRunResult",
    "Task",
    "TaskResult",
]
