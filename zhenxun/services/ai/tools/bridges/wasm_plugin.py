import json
from pathlib import Path
from typing import Any

from zhenxun.services.ai.run import RunContext
from zhenxun.services.ai.tools.core.tool import BaseTool
from zhenxun.services.ai.tools.models import ToolResult
from zhenxun.services.log import logger


class WasmPluginTool(BaseTool):
    """
    将任意跨语言编译的 Wasm 模块包装为大模型工具。
    标准通信协议：参数经 JSON 序列化输入 Wasm 模块的 stdin；
    Wasm 模块的 stdout 输出 JSON 字符串作为执行结果。
    """

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
        self._base_schema = parameters_schema
        self.plugin_args = plugin_args or []

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
                output=f"Plugin Execution Failed: {res['stderr']}"
            ).as_error()

        try:
            parsed_out = json.loads(res["stdout"])
            return ToolResult(output=parsed_out)
        except json.JSONDecodeError:
            return ToolResult(output=res["stdout"])
