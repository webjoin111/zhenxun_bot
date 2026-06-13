from typing import Any

from zhenxun.services.ai.tools.core.tool import ServerSideTool


class WebSearchTool(ServerSideTool):
    """原生网页搜索工具"""

    type_id = "web_search"

    def __init__(
        self,
        name: str = "google_search",
        description: str = "Use Google Search to find real-time information.",
        dynamic_threshold: float | None = None,
        domain_filters: list[str] | None = None,
    ):
        super().__init__(name=name, description=description)
        self.dynamic_threshold = dynamic_threshold
        self.domain_filters = domain_filters

    def to_gemini_payload(self) -> dict[str, Any]:
        return {"googleSearch": {}}

    def to_openai_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "web_search"}
        if self.domain_filters:
            payload["filters"] = self.domain_filters
        return payload


class CodeExecutionTool(ServerSideTool):
    """原生代码沙箱执行工具"""

    type_id = "code_execution"

    def __init__(self, name: str = "code_execution", timeout: int | None = None):
        super().__init__(
            name=name, description="Execute Python code to solve math or data problems."
        )
        self.timeout = timeout

    def to_gemini_payload(self) -> dict[str, Any]:
        return {"codeExecution": {}}

    def to_openai_payload(self) -> dict[str, Any]:
        return {"type": "code_interpreter"}


class ComputerUseTool(ServerSideTool):
    """原生的桌面环境控制工具"""

    type_id = "computer_use"

    def __init__(self, display_width_px: int = 1024, display_height_px: int = 768):
        super().__init__(name="computer_use", description="Control computer UI")
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px

    def to_openai_payload(self) -> dict[str, Any]:
        return {
            "type": "computer_use",
            "display_width_px": self.display_width_px,
            "display_height_px": self.display_height_px,
        }


class FileSearchTool(ServerSideTool):
    """原生文件检索工具"""

    type_id = "file_search"

    def __init__(self, name: str = "file_search"):
        super().__init__(name=name, description="Search indexed files.")

    def to_gemini_payload(self) -> dict[str, Any]:
        return {"fileSearch": {}}

    def to_openai_payload(self) -> dict[str, Any]:
        return {"type": "file_search"}


class GoogleMapsTool(ServerSideTool):
    """原生谷歌地图检索工具"""

    type_id = "google_map"

    def __init__(self):
        super().__init__(name="google_map", description="Search Google Maps data.")

    def to_gemini_payload(self) -> dict[str, Any]:
        return {"googleMaps": {}}


class UrlContextTool(ServerSideTool):
    """原生 URL 上下文拉取工具"""

    type_id = "url_context"

    def __init__(self):
        super().__init__(name="url_context", description="Fetch context from URL.")

    def to_gemini_payload(self) -> dict[str, Any]:
        return {"urlContext": {}}
