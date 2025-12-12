import json
import time
from typing import Any

from nonebot.matcher import Matcher

from zhenxun.services.llm.types import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolResult,
)
from zhenxun.services.llm.types.protocols import (
    BaseCallbackHandler,
    ToolCallData,
)
from zhenxun.services.log import logger


class LoggingCallbackHandler(BaseCallbackHandler):
    """一个将 Agent 执行步骤记录到日志的回调处理器。"""

    def __init__(self, indent_char: str = "  "):
        self.indent_char = indent_char
        self.depth = 0
        self.start_times: dict[str, float] = {}

    def _log(self, message: str, level: str = "debug"):
        log_func = getattr(logger, level)
        log_func(f"{self.indent_char * self.depth}{message}", "AgentExecutor")

    async def on_agent_start(self, messages: list[LLMMessage], **kwargs: Any) -> None:
        self.start_times["agent"] = time.monotonic()
        self._log("🚀 [智能体开始]")
        self.depth += 1

    async def on_model_start(
        self, model_name: str, messages: list[LLMMessage], **kwargs: Any
    ) -> None:
        self.start_times["model"] = time.monotonic()
        self._log(f"🧠 [模型调用] 使用 <u><y>{model_name}</y></u>")

    async def on_model_end(
        self, response: LLMResponse, duration: float, **kwargs: Any
    ) -> None:
        duration_ms = duration * 1000
        text_summary = f"'{response.text[:50]}...'" if response.text else "无文本"
        tool_calls_summary = (
            f", 请求了 {len(response.tool_calls)} 个工具调用。"
            if response.tool_calls
            else ""
        )
        self._log(
            f"✅ [模型结束] 耗时 {duration_ms:.2f}ms. "
            f"响应: {text_summary}{tool_calls_summary}"
        )

    async def on_tool_start(
        self, tool_call: LLMToolCall, data: ToolCallData, **kwargs: Any
    ) -> None:
        self.start_times[tool_call.id] = time.monotonic()
        self.depth += 1
        args_str = json.dumps(data.tool_args, ensure_ascii=False, indent=2)
        indent_str = self.indent_char * self.depth
        self._log(
            f"🛠️ [工具调用] <u><c>{data.tool_name}</c></u> 参数:\n{indent_str}{args_str}"
        )

    async def on_tool_end(
        self,
        result: ToolResult | None,
        error: Exception | None,
        tool_call: LLMToolCall,
        duration: float,
        **kwargs: Any,
    ) -> None:
        duration_ms = duration * 1000
        tool_name = tool_call.function.name
        if error:
            self._log(
                f"❌ [工具错误] <u><c>{tool_name}</c></u> "
                f"失败，耗时 {duration_ms:.2f}ms. 错误: <r>{error}</r>"
            )
        elif result:
            display = result.display_content or str(result.output)
            self._log(
                f"✅ [工具结束] <u><c>{tool_name}</c></u> "
                f"完成，耗时 {duration_ms:.2f}ms. 结果: '{display[:100]}...'"
            )
        self.depth -= 1

    async def on_agent_end(
        self, final_history: list[LLMMessage], duration: float, **kwargs: Any
    ) -> None:
        duration_ms = duration * 1000
        self.depth -= 1
        self._log(f"🏁 [智能体结束] 总执行时间: {duration_ms:.2f}ms")


class InteractiveCallbackHandler(LoggingCallbackHandler):
    """
    一个实现了与用户交互的回调处理器，使用 nonebot-plugin-waiter。
    """

    def __init__(self, matcher: Matcher, indent_char: str = "  "):
        super().__init__(indent_char)
        self.matcher = matcher
