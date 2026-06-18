from zhenxun.services.ai.core.protocols.llm import LLMModelBase
from zhenxun.services.ai.llm.adapters.handlers.mimo_handlers import (
    MiMoAudioHandler,
    MiMoTextHandler,
)
from zhenxun.services.ai.llm.adapters.handlers.openai_handlers import (
    OpenAIImageHandler,
)
from zhenxun.services.ai.llm.adapters.openai import OpenAICompatAdapter


class MiMoAdapter(OpenAICompatAdapter):
    """小米 MiMo 大模型专有适配器"""

    def __init__(self):
        super().__init__()
        self.text_handler = MiMoTextHandler(api_type=self.api_type)
        self.image_handler = OpenAIImageHandler()
        self.audio_handler = MiMoAudioHandler()

    @property
    def api_type(self) -> str:
        return "mimo"

    @property
    def supported_api_types(self) -> list[str]:
        return ["mimo"]

    def get_chat_endpoint(self, model: LLMModelBase) -> str:
        if model.model_detail.endpoint:
            return model.model_detail.endpoint
        return "/v1/chat/completions"
