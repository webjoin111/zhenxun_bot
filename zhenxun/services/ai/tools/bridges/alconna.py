from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
import re
from typing import Any, cast

from arclet.alconna import Alconna, Arparma, OptionResult
from arclet.alconna.model import HeadResult
from nonebot.exception import FinishedException
from nonebot.internal.matcher import current_bot, current_event, current_matcher
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_alconna.consts import ALCONNA_RESULT
from nonebot_plugin_alconna.matcher import AlconnaMatcher, on_alconna
from nonebot_plugin_alconna.model import CommandResult
from pydantic import BaseModel

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.schema import build_tool_model
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.engine.registry import tool_provider_manager
from zhenxun.services.ai.types.tools import ToolDefinition, ToolResult

_hijack_buffer: ContextVar[list[str] | None] = ContextVar(
    "llm_hijack_buffer", default=None
)
_original_alc_send = AlconnaMatcher.send.__func__


async def _hijacked_alc_send(cls, message, *args, **kwargs):
    buffer = _hijack_buffer.get()
    if buffer is not None:
        try:
            if isinstance(message, UniMessage):
                buffer.append(message.extract_plain_text())
            else:
                buffer.append(str(message))
        except Exception:
            buffer.append(str(message))
        return None

    return await _original_alc_send(cls, message, *args, **kwargs)


AlconnaMatcher.send = cast(Any, classmethod(_hijacked_alc_send))


class AlconnaBridgeHelper:
    """
    Alconna 与 LLM Tool Calling 的桥接核心辅助类
    """

    @staticmethod
    def generate_schema(
        alc: Alconna, params_model: type[BaseModel] | None = None
    ) -> dict[str, Any]:
        """
        [阶段一]：生成 JSON Schema
        如果提供了 Pydantic 模型，则优先使用模型生成严谨的 Schema。
        否则降级，尝试通过反射 Alconna 的节点树生成基础 Schema。
        """
        if params_model:
            return params_model.model_json_schema(mode="serialization")

        properties = {}
        required = []

        for arg in alc.args:
            properties[arg.name] = {
                "type": "string",
                "description": arg.notice or arg.name,
            }
            if not arg.optional:
                required.append(arg.name)

        for opt in alc.options:
            properties[opt.name] = {
                "type": "boolean" if opt.args.empty else "string",
                "description": opt.help_text or opt.name,
            }

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def spoof_state(alc: Alconna, llm_args: dict[str, Any], state: dict) -> dict:
        """
        [阶段二]：状态欺骗与注入
        将大模型返回的 JSON 参数，无缝塞进 NoneBot/Alconna 的依赖注入体系中。
        """
        arp = Arparma(
            _id=alc._hash,
            origin="LLM_SPOOFED_INPUT",
            matched=True,
            header_match=HeadResult(matched=True),
        )

        for key, value in llm_args.items():
            is_option = False
            for opt in alc.options:
                if opt.name == key or key in opt.aliases:
                    arp.options[opt.name] = OptionResult(value=value)
                    is_option = True
                    break
            if not is_option:
                arp.main_args[key] = value

        state[ALCONNA_RESULT] = CommandResult(result=arp, output=None)
        return state

    @staticmethod
    @asynccontextmanager
    async def llm_hijack_context():
        """
        [阶段三]：执行域隔离包装器
        """
        buffer: list[str] = []
        token = _hijack_buffer.set(buffer)
        try:
            yield buffer
        except FinishedException:
            pass
        finally:
            _hijack_buffer.reset(token)


def on_llm_alconna(
    command: Alconna | str,
    *args,
    params_model: type[BaseModel] | None = None,
    tool_name: str | None = None,
    tool_description: str | None = None,
    **kwargs,
) -> type[AlconnaMatcher]:
    """
    提供统一装饰器，创建 Alconna 命令同时将其注册为 LLM 工具。
    支持第三方开发者传递 Pydantic 模型以生成极其严谨的 Schema。
    """
    matcher_cls = on_alconna(command, *args, **kwargs)

    if isinstance(command, str):
        alc = matcher_cls.command()
    else:
        alc = command

    schema = AlconnaBridgeHelper.generate_schema(alc, params_model)

    raw_name = tool_name or alc.command
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
    desc = tool_description or alc.meta.description or f"Command {name}"

    class AlconnaTool(BaseTool):
        _dynamic_def: Any = None

        def __init__(self):
            super().__init__(name=name, description=desc)

        async def get_definition(self, context: RunContext | None = None) -> ToolDefinition | None:
            if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
                return self._dynamic_def
            tool_def = ToolDefinition(
                name=self.name,
                description=self.description,
                parameters=schema,
            )
            if context and self.settings.prepare:
                from nonebot.utils import is_coroutine_callable
                if is_coroutine_callable(self.settings.prepare):
                    tool_def = await self.settings.prepare(context, tool_def)
                else:
                    tool_def = self.settings.prepare(context, tool_def)
            return tool_def

        async def execute(
            self, context: RunContext | None = None, **tool_kwargs
        ) -> ToolResult:
            if not context:
                return ToolResult(output="Context is missing", is_error=True)

            bot = context.bot
            event = context.event

            if not bot or not event:
                return ToolResult(
                    output="Bot or Event is missing in context", is_error=True
                )

            state = {}
            AlconnaBridgeHelper.spoof_state(alc, tool_kwargs, state)
            matcher_instance = matcher_cls()
            matcher_instance.state.update(state)

            async with AlconnaBridgeHelper.llm_hijack_context() as buffer:
                b_token = current_bot.set(bot)
                e_token = current_event.set(event)
                m_token = current_matcher.set(matcher_instance)

                try:
                    async with AsyncExitStack() as stack:
                        for handler in matcher_instance.handlers:
                            await handler(
                                matcher=matcher_instance,
                                bot=bot,
                                event=event,
                                state=matcher_instance.state,
                                stack=stack,
                                dependency_cache={},
                            )
                except FinishedException:
                    pass
                except Exception as e:
                    return ToolResult(
                        output=f"Tool Execution Error: {e}", is_error=True
                    )
                finally:
                    current_bot.reset(b_token)
                    current_event.reset(e_token)
                    current_matcher.reset(m_token)

            output_str = "\n".join(buffer) if buffer else "命令执行成功 (无返回文本)"
            return ToolResult(output=output_str, display=output_str)

    tool_provider_manager.register_tool(AlconnaTool())

    return matcher_cls

