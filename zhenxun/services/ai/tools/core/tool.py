from collections.abc import Callable
import hashlib
import inspect
import json
from typing import Any, Awaitable, cast

from nonebot.adapters import Message as PlatformMessage
from nonebot.utils import is_coroutine_callable
from pydantic import ValidationError

from zhenxun.services.ai.protocols.tool import ToolExecutable
from zhenxun.services.ai.protocols.tool import (
    ToolMiddleware,
    ToolNextCall,
)
from zhenxun.services.ai.types.exceptions import (
    NeedsInputException,
    ToolFatalError,
    ToolRetryError,
)
from zhenxun.services.ai.types.tools import (
    ToolDefinition,
    ToolOptions,
    ToolResult,
)
from zhenxun.services.log import logger
from zhenxun.utils.pydantic_compat import model_dump, model_validate

from .context import RunContext
from .di import DependencyInjector
from .schema import _parse_docstring, build_tool_model


class LazyToolProxy:
    """延迟实例化的工具代理类。"""

    _dynamic_def: Any = None

    def __init__(
        self, name: str, description: str, factory: Callable[[], ToolExecutable]
    ):
        self.name = name
        self.description = description
        self._factory = factory
        self._instance: ToolExecutable | None = None

    async def _get_instance(self) -> ToolExecutable:
        if self._instance is None:
            logger.debug(f"JIT 懒加载实例化工具: {self.name}")
            from nonebot.utils import is_coroutine_callable

            res = self._factory()
            if is_coroutine_callable(self._factory):
                res = await cast(Awaitable[ToolExecutable], res)
            self._instance = res
        return self._instance

    def clone_with_options(self, override: Any) -> "LazyToolProxy":
        original_factory = self._factory

        def new_factory() -> ToolExecutable:
            from nonebot.utils import is_coroutine_callable

            res = original_factory()
            if is_coroutine_callable(original_factory):

                async def async_wrapper():
                    async_res = await cast(Awaitable[Any], res)
                    if hasattr(async_res, "clone_with_options"):
                        return async_res.clone_with_options(override)
                    return async_res

                return cast(ToolExecutable, async_wrapper())
            if hasattr(res, "clone_with_options"):
                return cast(Any, res).clone_with_options(override)
            return cast(ToolExecutable, res)

        new_name = (
            getattr(override, "new_name", None)
            or getattr(override, "name", self.name)
            or self.name
        )
        new_desc = getattr(override, "description", None) or self.description
        return LazyToolProxy(name=new_name, description=new_desc, factory=new_factory)

    async def get_definition(self, context: Any | None = None) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        instance = await self._get_instance()
        return await instance.get_definition(context)

    async def execute(self, context: Any | None = None, **kwargs: Any) -> ToolResult:
        instance = await self._get_instance()
        return await instance.execute(context=context, **kwargs)

    async def should_confirm(self, **kwargs: Any) -> str | None:
        instance = await self._get_instance()
        if hasattr(instance, "should_confirm"):
            return await instance.should_confirm(**kwargs)
        return None


class BaseTool:
    """
    面向对象的工具基类。支持多模态契约与纯 Pydantic V2 参数验证。

    参数:
        name: 工具名称。大模型将基于此识别并调用。如果不提供，默认使用类名。
        description: 工具的说明文档。大模型会根据此描述理解工具用途。
        settings: 配置选项（包含缓存、拦截、中间件等）。
    """

    name: str
    description: str
    settings: ToolOptions
    current_usage_count: int = 0
    _dynamic_def: Any = None
    args_schema: Any = None
    _base_schema: dict[str, Any] | None = None
    _param_model: Any = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.settings.metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any]):
        self.settings.metadata = value

    @property
    def prepare(self) -> Any:
        return self.settings.prepare

    @prepare.setter
    def prepare(self, value: Any):
        self.settings.prepare = value

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        settings: ToolOptions | None = None,
    ):
        self.name = name or getattr(self, "name", self.__class__.__name__)
        self.description = description or getattr(
            self, "description", self.__doc__ or "未提供描述"
        )
        self.settings = settings or getattr(self, "settings", ToolOptions())
        self.current_usage_count = 0

        class_meta = getattr(self.__class__, "metadata", {})
        class_prepare = getattr(self.__class__, "prepare", None)
        if class_meta and not isinstance(class_meta, property):
            self.settings.metadata.update(class_meta)
        if class_prepare and not isinstance(class_prepare, property):
            self.settings.prepare = class_prepare

    def clone_with_options(self, override: Any) -> "BaseTool":
        """创建当前工具的浅拷贝，并覆盖 ToolOptions 和基本属性"""
        import copy

        new_tool = copy.copy(self)

        if hasattr(override, "to_tool_options"):
            override_settings = override.to_tool_options()
        else:
            override_settings = getattr(
                override, "options", getattr(override, "settings", None)
            )

        if override_settings:
            new_tool.settings = self.settings.merge(override_settings)

        new_name = getattr(override, "new_name", None)
        if new_name:
            new_tool.name = new_name

        new_desc = getattr(override, "description", None)
        if new_desc:
            new_tool.description = new_desc

        return new_tool

    async def should_confirm(self, **kwargs: Any) -> str | None:
        """拦截高危工具执行"""
        if self.settings.require_approval:
            args_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            return f"即将在本地执行高危工具 [{self.name}]\n参数：\n{args_str}"
        return None

    async def _handle_validation_error(self, e: ValidationError, kwargs: dict) -> None:
        """
        将 Pydantic 参数校验失败转化为 ToolRetryError 或 NeedsInputException
        """
        error_msgs = [
            f"[{'.'.join(str(x) for x in err['loc'])}]: {err['msg']}"
            for err in e.errors()
        ]
        err_msg = "; ".join(error_msgs)

        if self.settings.interactive:
            first_error = e.errors()[0]
            field_name = (
                str(first_error["loc"][-1]) if first_error.get("loc") else "unknown"
            )
            field_desc = f"参数验证失败: {first_error['msg']}"

            raise NeedsInputException(field_name, field_desc, kwargs)

        raise ToolRetryError(f"参数格式不符合 Schema 规范: {err_msg}")

    async def get_definition(
        self, context: RunContext | None = None
    ) -> ToolDefinition | None:
        """获取工具的 JSON Schema 定义。支持动态修改 Schema 甚至隐藏工具。"""
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def

        if hasattr(self, "args_schema") and self.args_schema:
            from zhenxun.utils.pydantic_compat import model_json_schema

            schema = model_json_schema(self.args_schema)
        elif hasattr(self, "_base_schema"):
            schema = self._base_schema or {"type": "object", "properties": {}}
        else:
            schema = {"type": "object", "properties": {}}

        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
            metadata=self.settings.metadata.copy(),
        )

        if context and self.settings.prepare:
            from nonebot.utils import is_coroutine_callable

            if is_coroutine_callable(self.settings.prepare):
                tool_def = await self.settings.prepare(context, tool_def)
            else:
                tool_def = self.settings.prepare(context, tool_def)

        return tool_def

    def _generate_cache_key(self, arguments: dict[str, Any]) -> str:
        """
        根据工具名称和参数生成绝对唯一的缓存 Key。
        通过对字典 key 进行排序并转化为 json 字符串，保证参数顺序不同但内容相同时 Hash 结果一致。
        """
        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        key_str = f"{self.name}:{args_str}"
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        """工具执行的总入口，负责组装和调度中间件管道"""
        context_to_pass = context or RunContext()

        async def core_handler(kw: dict[str, Any], ctx: RunContext) -> ToolResult:
            return await self._core_execution(ctx, **kw)

        from zhenxun.services.ai.tools.engine.middlewares import GLOBAL_MIDDLEWARES

        all_middlewares = GLOBAL_MIDDLEWARES + self.settings.middlewares
        handler: ToolNextCall = core_handler
        for middleware in reversed(all_middlewares):

            def _wrap(mw: ToolMiddleware, nxt: ToolNextCall) -> ToolNextCall:
                async def _wrapped_handler(
                    kw: dict[str, Any], c: RunContext
                ) -> ToolResult:
                    return await mw(self, kw, c, nxt)

                return _wrapped_handler

            handler = _wrap(middleware, handler)

        try:
            return await handler(kwargs, context_to_pass)
        except Exception as e:
            logger.error(f"工具 {self.name} 执行抛出异常，将交由底层引擎处理: {e}")
            raise

    async def _core_execution(self, context: RunContext, **kwargs: Any) -> ToolResult:
        """核心执行流水线 (Core Execution Pipeline)"""
        retry_key = f"__tool_retries_{self.name}"
        retries = context.extra.get(retry_key, 0)

        if (
            self.settings.max_usage_count is not None
            and self.current_usage_count >= self.settings.max_usage_count
        ):
            raise ToolFatalError(
                f"工具 '{self.name}' 已达到最大调用次数上限 ({self.settings.max_usage_count})"
            )
        self.current_usage_count += 1

        target_func = getattr(self, "_arun", getattr(self, "_run", None))

        if not target_func:
            return ToolResult(output="Error: 未实现 _run 或 _arun", is_error=True)

        call_kwargs = dict(kwargs)

        if hasattr(self, "_param_model") and self._param_model:
            try:
                validated_model = model_validate(self._param_model, call_kwargs)
                call_kwargs = model_dump(validated_model, exclude_unset=True)
            except ValidationError as e:
                await self._handle_validation_error(e, kwargs)

        from nonebot.utils import is_coroutine_callable

        if self.settings.pre_hook:
            if is_coroutine_callable(self.settings.pre_hook):
                await self.settings.pre_hook(context, call_kwargs)
            else:
                self.settings.pre_hook(context, call_kwargs)

        available_injects = {}
        if context.bot:
            available_injects["bot"] = context.bot
        if context.event:
            available_injects["event"] = context.event
        if context.matcher:
            available_injects["matcher"] = context.matcher
            available_injects["state"] = context.matcher.state
        available_injects.update(context.extra)

        inject_kwargs = dict(call_kwargs)
        for k, v in available_injects.items():
            if k not in inject_kwargs:
                inject_kwargs[k] = v

        try:
            call_kwargs, inject_kwargs = await DependencyInjector.resolve_all(
                sig=inspect.signature(target_func),
                call_kwargs=call_kwargs,
                inject_kwargs=inject_kwargs,
                context=context,
                available_injects=available_injects,
            )
        except ValueError as e:
            logger.error(f"工具 {self.name} 依赖注入失败: {e}", e=e)
            raise ToolFatalError(f"框架依赖注入失败: {e}")

        if self.settings.args_validator:
            try:
                if is_coroutine_callable(self.settings.args_validator):
                    await self.settings.args_validator(context, call_kwargs)
                else:
                    self.settings.args_validator(context, call_kwargs)
            except Exception as val_err:
                logger.warning(f"🔧 工具 [{self.name}] 业务参数校验拦截: {val_err}")
                raise ToolRetryError(f"业务逻辑验证拒绝了你的参数：{val_err}")

        if inspect.isasyncgenfunction(target_func):
            res = None
            async for chunk in target_func(**call_kwargs):
                if isinstance(chunk, ToolResult):
                    res = chunk
                else:
                    from zhenxun.services.ai.events import (
                        EventCenter,
                        ToolStreamEvent,
                    )
                    from zhenxun.services.ai.types.tools import ToolResultChunk

                    chunk_obj = (
                        chunk
                        if isinstance(chunk, ToolResultChunk)
                        else ToolResultChunk(content=str(chunk))
                    )
                    await EventCenter.publish(
                        ToolStreamEvent(
                            tool_call_id="unknown",
                            tool_name=self.name,
                            chunk=chunk_obj,
                            session_id=context.session_id,
                        )
                    )
            if res is None:
                res = ToolResult(output="Stream finished successfully.")
        elif is_coroutine_callable(target_func):
            res = await target_func(**call_kwargs)
        else:
            import asyncio

            res = await asyncio.to_thread(target_func, **call_kwargs)

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

                final_result = ToolResult(
                    output=parts,
                    display=uni_msg,
                )
            else:
                final_result = ToolResult(output=res)

        if self.settings.result_as_answer or self.settings.direct_reply:
            final_result.terminate_run = True
            if self.settings.direct_reply and not final_result.display:
                final_result.display = final_result.output
        if self.settings.silent:
            final_result.display = None

        if self.settings.post_hook:
            if is_coroutine_callable(self.settings.post_hook):
                await self.settings.post_hook(context, final_result)
            else:
                self.settings.post_hook(context, final_result)

        if not final_result.is_error:
            context.extra[retry_key] = 0

        return final_result


class FunctionTool(BaseTool):
    """
    将普通 Python 函数包装为 BaseTool 的实现。

    参数:
        func: 实际执行的 Python 异步或同步函数。
        name: 覆盖函数名的工具名称。
        description: 覆盖函数 docstring 的工具描述。
        settings: 配置选项。
    """

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        settings: ToolOptions | None = None,
    ):
        super().__init__(name=name, description=description, settings=settings)

        if is_coroutine_callable(func):
            self._arun = func
        else:
            self._run = func

        schema, model = build_tool_model(func, strict=self.settings.strict)
        self._base_schema = schema
        self._param_model = model

        main_desc, _ = _parse_docstring(func.__doc__)
        if not description or description == "未提供描述":
            self.description = main_desc or "未提供描述"
