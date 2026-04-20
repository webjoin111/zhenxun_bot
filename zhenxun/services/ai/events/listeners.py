import hashlib
import json

from zhenxun.services.ai.events import EventCenter
from zhenxun.services.ai.events.event_types import (
    AgentEndEvent,
    AgentStartEvent,
    ModelEndEvent,
    ModelStartEvent,
    SandboxExecutionCompletedEvent,
    SandboxExecutionStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from zhenxun.services.ai.types.messages import ToolCallPart
from zhenxun.services.ai.types.exceptions import LLMErrorCode, LLMException
from zhenxun.services.log import logger


@EventCenter.subscribe(ModelStartEvent, priority=1)
async def stuck_detection_listener(event: ModelStartEvent):
    max_repeated_errors = 3
    action_hashes = []
    messages = list(event.messages)
    idx = len(messages) - 1

    while idx >= 0:
        msg = messages[idx]
        if msg.role == "tool":
            batch_tool_contents = []
            while idx >= 0 and messages[idx].role == "tool":
                for tr in messages[idx].tool_returns:
                    batch_tool_contents.append(f"{tr.tool_name}:{tr.output}")
                idx -= 1

            if (
                idx >= 0
                and messages[idx].role == "assistant"
                and messages[idx].tool_calls
            ):
                assistant_msg = messages[idx]
                batch_tool_calls = []
                for tc in assistant_msg.tool_calls:
                    if isinstance(tc, ToolCallPart):
                        args_str = (
                            tc.args
                            if isinstance(tc.args, str)
                            else json.dumps(tc.args, ensure_ascii=False)
                        )
                        batch_tool_calls.append(
                            f"{tc.tool_name}:{args_str}"
                        )

                batch_tool_calls.sort()
                batch_tool_contents.sort()

                state_str = (
                    "|".join(batch_tool_calls) + "||" + "|".join(batch_tool_contents)
                )
                state_hash = hashlib.md5(state_str.encode("utf-8")).hexdigest()
                action_hashes.append(state_hash)
                idx -= 1
            else:
                break
        elif msg.role == "assistant":
            idx -= 1
        else:
            break

    if len(action_hashes) >= max_repeated_errors:
        recent_hashes = action_hashes[:max_repeated_errors]
        if len(set(recent_hashes)) == 1:
            logger.warning(
                f"[StuckDetection] 拦截到死循环：连续 {max_repeated_errors} 次产生完全相同的状态哈希碰撞。"
            )
            raise LLMException(
                message=f"Agent 触发终极防呆机制：连续 {max_repeated_errors} 次产生完全相同的无效工具调用状态，已物理阻断以节省 Token。",
                code=LLMErrorCode.GENERATION_FAILED,
            )


@EventCenter.subscribe(AgentStartEvent, priority=100)
async def log_agent_start(event: AgentStartEvent):
    logger.debug("🚀 [智能体开始] 发起运行流程")


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
        f"✅ [模型结束] 耗时 {event.duration_ms:.2f}ms. 响应: {text_summary}{tool_calls_summary}"
    )


@EventCenter.subscribe(ToolCallEvent, priority=100)
async def log_tool_start(event: ToolCallEvent):
    args_str = json.dumps(event.arguments, ensure_ascii=False)
    logger.debug(f"🛠️ [工具调用] <u><c>{event.tool_name}</c></u> 参数: {args_str}")


@EventCenter.subscribe(ToolResultEvent, priority=100)
async def log_tool_end(event: ToolResultEvent):
    if event.error:
        logger.error(
            f"❌ [工具错误] <u><c>{event.tool_name}</c></u> 失败，耗时 {event.duration_ms:.2f}ms. 错误: <r>{event.error}</r>"
        )
    elif event.result:
        display = getattr(event.result, "display_content", None)
        if display is None:
            display = getattr(event.result, "output", event.result)
        display_str = str(display)
        logger.debug(
            f"✅ [工具结束] <u><c>{event.tool_name}</c></u> 完成，耗时 {event.duration_ms:.2f}ms. 结果: '{display_str[:100]}...'"
        )


@EventCenter.subscribe(AgentEndEvent, priority=100)
async def log_agent_end(event: AgentEndEvent):
    logger.debug(f"🏁 [智能体结束] 总执行时间: {event.duration_ms:.2f}ms")


@EventCenter.subscribe(SandboxExecutionStartedEvent)
async def log_sandbox_start(event: SandboxExecutionStartedEvent):
    code_preview = event.code.strip()[:50].replace("\n", "\\n")
    logger.debug(
        f"🐳 [沙箱开始] 会话 {event.session_id} 正在执行代码: '{code_preview}...'"
    )


@EventCenter.subscribe(SandboxExecutionCompletedEvent)
async def log_sandbox_end(event: SandboxExecutionCompletedEvent):
    status = "✅ 成功" if event.exit_code == 0 else f"❌ 失败(码:{event.exit_code})"
    logger.debug(f"{status} [沙箱结束] 耗时: {event.duration_ms:.2f}ms")
