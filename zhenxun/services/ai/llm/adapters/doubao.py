from __future__ import annotations

from typing import TYPE_CHECKING

from .handlers.openai_handlers import (
    OpenAITextHandler,
)
from .openai import OpenAICompatAdapter

if TYPE_CHECKING:
    from zhenxun.services.ai.llm.service import LLMModel


class DoubaoAdapter(OpenAICompatAdapter):
    """火山方舟 (Doubao) API 适配器"""

    def __init__(self):
        """初始化 Doubao 适配器并复用 OpenAI 兼容处理器。"""
        super().__init__()
        self.text_handler = OpenAITextHandler(api_type=self.api_type)

    @property
    def api_type(self) -> str:
        """适配器主类型标识。"""
        return "doubao"

    @property
    def supported_api_types(self) -> list[str]:
        """当前适配器支持的 API 类型列表。"""
        return ["doubao"]

    def get_chat_endpoint(self, model: "LLMModel") -> str:
        """返回对话端点，优先使用模型级自定义端点。"""
        if model.model_detail.endpoint:
            return model.model_detail.endpoint
        return "/v3/chat/completions"
