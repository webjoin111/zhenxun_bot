from abc import ABC, abstractmethod
import inspect
from typing import TYPE_CHECKING, Any

from nonebot.adapters import Message as PlatformMessage

from zhenxun.services.ai.core.exceptions import ToolFatalError
from zhenxun.services.ai.run.di import DependencyInjector
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.run import RunContext
    from zhenxun.services.ai.tools.core.tool import BaseTool


class ToolRunner(ABC):
    """
    工具运行器基类协议。
    负责将参数请求物理落实为目标执行。
    """

    @abstractmethod
    async def run(
        self, tool: "BaseTool", context: "RunContext", **kwargs: Any
    ) -> ToolResult:
        pass


class NativeToolRunner(ToolRunner):
    """
    原生 Python 函数工具运行器。
    负责处理依赖注入 (DI)、异步包装、生成器流式收集以及框架级的多模态消息转换。
    """

    async def run(
        self, tool: "BaseTool", context: "RunContext", **kwargs: Any
    ) -> ToolResult:
        target_func = tool.get_execute_target()
        signature_target = tool.get_signature_target()

        if not target_func:
            return ToolResult(output="Error: 未找到有效的执行目标(run 方法)").as_error()

        call_kwargs = dict(kwargs)

        try:
            target_call_kwargs = await DependencyInjector.resolve_all(
                sig=inspect.signature(signature_target),
                call_kwargs=dict(call_kwargs),
                context=context,
            )
        except ValueError as e:
            logger.error(f"工具 {tool.name} 依赖注入失败: {e}", e=e)
            raise ToolFatalError(f"框架依赖注入失败: {e}")

        is_async_gen = getattr(
            target_func, "_is_async_gen", False
        ) or inspect.isasyncgenfunction(target_func)
        if is_async_gen:
            res = None
            async for chunk in target_func(**target_call_kwargs):
                if isinstance(chunk, ToolResult):
                    res = chunk
                else:
                    from zhenxun.services.ai.core.stream_events import ToolStreamChunk
                    from zhenxun.services.ai.tools.models import ToolResultChunk

                    chunk_obj = (
                        chunk
                        if isinstance(chunk, ToolResultChunk)
                        else ToolResultChunk(content=str(chunk))
                    )
                    if context.run.streamer:
                        await context.run.streamer.send(
                            ToolStreamChunk(
                                tool_name=tool.name,
                                content=chunk_obj.content,
                                metadata=chunk_obj.metadata,
                            )
                        )
            if res is None:
                res = ToolResult(output="Stream finished successfully.")
        else:
            res = await target_func(**target_call_kwargs)

        if isinstance(res, ToolResult):
            final_result = res
        else:
            if str(type(res)).find("Message") != -1:
                from zhenxun.services.ai.message_builder import MessageBuilder

                uni_msg = (
                    MessageBuilder.message_to_unimessage(res)
                    if isinstance(res, PlatformMessage)
                    else res
                )
                parts = await MessageBuilder.unimsg_to_llm_parts(uni_msg)

                final_result = ToolResult(output=parts).show_to_user(uni_msg)
            else:
                final_result = ToolResult(output=res)

        return final_result
