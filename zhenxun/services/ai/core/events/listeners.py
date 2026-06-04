import json

from zhenxun.services.ai.core.events import EventCenter
from zhenxun.services.ai.core.events.event_types import (
    AgentEndEvent,
    AgentStartEvent,
    ModelEndEvent,
    ModelStartEvent,
    SandboxExecutionCompletedEvent,
    SandboxExecutionStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from zhenxun.services.log import logger


@EventCenter.subscribe(AgentStartEvent, priority=100)
async def log_agent_start(event: AgentStartEvent):
    logger.debug(f"🚀 [智能体开始] {event.agent_name} 发起运行流程")


@EventCenter.subscribe(ModelStartEvent, priority=100)
async def log_model_start(event: ModelStartEvent):
    logger.debug(f"🧠 [模型调用] 使用 <u><y>{event.model_name}</y></u>")


@EventCenter.subscribe(ModelEndEvent, priority=100)
async def log_model_end(event: ModelEndEvent):
    text_summary = (
        f"'{event.response.text[:50]}...'" if event.response.text else "无文本"
    )
    tool_calls_summary = (
        f", 请求了 {len(event.response.tool_calls)} 个工具调用。"
        if event.response.tool_calls
        else ""
    )
    logger.debug(
        f"✅ [模型结束] 耗时 {event.duration_ms:.2f}ms. "
        f"响应: {text_summary}{tool_calls_summary}"
    )


@EventCenter.subscribe(ToolCallEvent, priority=100)
async def log_tool_start(event: ToolCallEvent):
    args_str = json.dumps(event.arguments, ensure_ascii=False)
    logger.debug(
        f"🛠️ [工具调用] <u><c>{event.tool_name}</c></u> 参数: {args_str}"
    )


@EventCenter.subscribe(ToolResultEvent, priority=100)
async def log_tool_end(event: ToolResultEvent):
    if event.error:
        from zhenxun.services.ai.core.exceptions import ControlFlowException

        if isinstance(event.error, ControlFlowException):
            return
        logger.error(
            f"❌ [工具错误] <u><c>{event.tool_name}</c></u> 失败，"
            f"耗时 {event.duration_ms:.2f}ms. 错误: <r>{event.error}</r>"
        )
    elif event.result:
        display = getattr(event.result, "display_content", None)
        if display is None:
            display = getattr(event.result, "output", event.result)
        display_str = str(display)
        logger.debug(
            f"✅ [工具结束] <u><c>{event.tool_name}</c></u> 完成，"
            f"耗时 {event.duration_ms:.2f}ms. 结果: '{display_str[:100]}...'"
        )


@EventCenter.subscribe(AgentEndEvent, priority=100)
async def log_agent_end(event: AgentEndEvent):
    logger.debug(
        f"🏁 [智能体结束] {event.agent_name} "
        f"总执行时间: {event.duration_ms:.2f}ms"
    )


@EventCenter.subscribe(SandboxExecutionStartedEvent)
async def log_sandbox_start(event: SandboxExecutionStartedEvent):
    code_preview = event.code.strip()[:50].replace("\n", "\\n")
    logger.debug(
        f"🐳 [沙箱开始] 会话 {event.session_id} "
        f"正在执行代码: '{code_preview}...'"
    )


@EventCenter.subscribe(SandboxExecutionCompletedEvent)
async def log_sandbox_end(event: SandboxExecutionCompletedEvent):
    status = (
        "✅ 成功" if event.exit_code == 0
        else f"❌ 失败(码:{event.exit_code})"
    )
    logger.debug(f"{status} [沙箱结束] 耗时: {event.duration_ms:.2f}ms")
