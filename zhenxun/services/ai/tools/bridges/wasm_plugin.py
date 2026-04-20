import json
from pathlib import Path
from typing import Any

from zhenxun.services.ai.tools.core.context import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.types.tools import ToolDefinition, ToolResult
from zhenxun.services.log import logger


class WasmPluginTool(BaseTool):
    """
    将任意跨语言编译的 Wasm 模块包装为大模型工具。
    标准通信协议：参数经 JSON 序列化输入 Wasm 模块的 stdin；
    Wasm 模块的 stdout 输出 JSON 字符串作为执行结果。
    """
    _dynamic_def: Any = None

    def __init__(
        self,
        name: str,
        description: str,
        wasm_path: Path,
        parameters_schema: dict,
        plugin_args: list[str] | None = None,
    ):
        super().__init__(name=name, description=description)
        self.wasm_path = wasm_path
        self.parameters_schema = parameters_schema
        self.plugin_args = plugin_args or []

    async def get_definition(self, context: RunContext | None = None) -> ToolDefinition | None:
        if hasattr(self, "_dynamic_def") and self._dynamic_def is not None:
            return self._dynamic_def
        tool_def = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
            metadata=self.metadata or {},
        )
        if context and self.settings.prepare:
            from nonebot.utils import is_coroutine_callable
            if is_coroutine_callable(self.settings.prepare):
                tool_def = await self.settings.prepare(context, tool_def)
            else:
                tool_def = self.settings.prepare(context, tool_def)
        return tool_def

    async def execute(
        self, context: RunContext | None = None, **kwargs: Any
    ) -> ToolResult:
        from zhenxun.services.ai.sandbox.drivers.wasm import WasmtimeCoreEngine

        stdin_data = json.dumps(kwargs, ensure_ascii=False)
        logger.info(
            f"🧩 [WasmPlugin] 正在执行跨语言插件: {self.name} (载荷: {stdin_data})"
        )

        res = await WasmtimeCoreEngine.run_wasm_plugin(
            self.wasm_path, stdin_data, self.plugin_args
        )

        if res["exit_code"] != 0:
            return ToolResult(
                output=f"Plugin Execution Failed: {res['stderr']}", is_error=True
            )

        try:
            parsed_out = json.loads(res["stdout"])
            return ToolResult(output=parsed_out)
        except json.JSONDecodeError:
            return ToolResult(output=res["stdout"])
