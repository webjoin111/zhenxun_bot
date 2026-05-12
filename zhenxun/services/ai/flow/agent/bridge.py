from typing import Any, Generic, cast
from typing_extensions import TypeVar

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage

from zhenxun.services.ai.core.exceptions import ControlFlowException
from zhenxun.services.ai.core.messages import UsageInfo
from zhenxun.services.ai.core.stream_events import ToolStreamChunk
from zhenxun.services.ai.run import AgentRunResult, RunContext
from zhenxun.services.ai.run.models import AgentRunEnd, AgentRunError
from zhenxun.services.log import logger
from zhenxun.utils.message import MessageUtils

T_Deps = TypeVar("T_Deps", default=Any)
T_Out = TypeVar("T_Out", default=str)


class AgentRunner(Generic[T_Out]):
    """
    智能体运行器。
    负责将大模型的纯净数据流包装为平台交互动作（发消息、UI渲染）。
    自带 ContextVars 隐式上下文提取魔法。
    """

    def __init__(
        self,
        runnable: Any,
        context: RunContext | None = None,
        **kwargs: Any,
    ):
        self.runnable = runnable
        self.context = context or RunContext(**kwargs)

        is_stateless = (
            getattr(self.runnable.runtime_config, "stateless", True)
            if hasattr(self.runnable, "runtime_config")
            else True
        )
        if is_stateless and self.context.session_id:
            if not self.context.session_id.startswith("stateless_"):
                import uuid

                self.context.session_id = (
                    f"stateless_{self.context.session_id}_{uuid.uuid4().hex[:8]}"
                )
                if self.context.session:
                    self.context.session.session_id = self.context.session_id

    @property
    def _bot(self) -> Bot | None:
        return getattr(self.context.deps, "bot", None) if self.context.deps else None

    @property
    def _event(self) -> Event | None:
        return getattr(self.context.deps, "event", None) if self.context.deps else None

    async def reply(
        self, prompt: Any = None, reply_to: bool = False, **kwargs: Any
    ) -> AgentRunResult[T_Out]:
        """交互式执行：将 Agent 运行过程中的工具调用状态和最终结果自动发送给用户。"""
        ui_ctx = None
        ui_streamer = (
            getattr(self.runnable.runtime_config, "ui_streamer", None)
            if hasattr(self.runnable, "runtime_config")
            else None
        )

        if ui_streamer:
            from zhenxun.services.ai.core.ui.context import UIStreamerContext

            ui_ctx = UIStreamerContext(
                self.context.session_id or "default_session",
                streamer_type=ui_streamer,
            )
            await ui_ctx.__aenter__()

        final_result = None

        try:
            async with self.runnable.run_stream(
                prompt=prompt,
                context=self.context,
                **kwargs,
            ) as stream_result:
                async for stream_event in stream_result.stream_events():
                    if (
                        isinstance(stream_event, ToolStreamChunk)
                        and self._bot
                        and self._event
                    ):
                        display_msg = (
                            stream_event.metadata.get("display")
                            if stream_event.metadata
                            else None
                        )
                        if display_msg:
                            if isinstance(display_msg, UniMessage):
                                await display_msg.send(
                                    self._event, bot=self._bot, reply_to=reply_to
                                )
                            else:
                                await self._bot.send(self._event, display_msg)
                        elif stream_event.content:
                            await self._bot.send(self._event, stream_event.content)

                    elif isinstance(stream_event, AgentRunEnd):
                        final_result = stream_event.result

                    elif isinstance(stream_event, AgentRunError):
                        raise stream_event.error

        except ControlFlowException as e:
            logger.debug(
                f"{self.runnable.name} 控制流正常中断: {type(e).__name__} - {e}"
            )
            display_msg = getattr(e, "display", None)
            if display_msg and self._bot and self._event:
                if isinstance(display_msg, UniMessage):
                    await display_msg.send(
                        self._event, bot=self._bot, reply_to=reply_to
                    )
                else:
                    await self._bot.send(self._event, str(display_msg))
            output_val = getattr(e, "result_output", None) or str(e)
            return cast(
                AgentRunResult[T_Out],
                AgentRunResult(output=output_val, usage=UsageInfo()),
            )

        except Exception as e:
            logger.error(f"{self.runnable.name} 运行失败: {e}", e=e)
            if self._bot and self._event:
                await MessageUtils.build_message(f"❌ 运行发生错误: {e}").send()
            raise e

        finally:
            if ui_ctx:
                await ui_ctx.__aexit__(None, None, None)

        if final_result and final_result.output and self._bot and self._event:
            ui_report = ui_ctx.render() if ui_ctx else ""
            if isinstance(final_result.output, UniMessage):
                if ui_report:
                    await MessageUtils.build_message(ui_report).send()
                await final_result.output.send(
                    self._event, bot=self._bot, reply_to=reply_to
                )
                final_result.output = final_result.output.extract_plain_text()
            else:
                final_msg = str(final_result.output)
                if ui_report:
                    final_msg = f"{ui_report}\n\n🤖 最终总结:\n{final_msg}"
                await MessageUtils.build_message(final_msg).send()

        if final_result is None:
            raise RuntimeError("智能体运行流异常结束：未返回最终结果。")

        return cast(AgentRunResult[T_Out], final_result)
