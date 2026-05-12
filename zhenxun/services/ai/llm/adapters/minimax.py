from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from zhenxun.services.ai.core.configs import GenerationConfig
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.ai.core.models import ModelCapabilities, ModelDetail
from zhenxun.services.ai.llm.adapters.base import RequestData
from zhenxun.services.ai.llm.adapters.handlers.base import BaseAudioHandler

from .handlers.openai_handlers import (
    OpenAIConfigMapper,
    OpenAITextHandler,
)
from .openai import OpenAICompatAdapter

if TYPE_CHECKING:
    from zhenxun.services.ai.core.configs import TTSConfig
    from zhenxun.services.ai.core.messages import AudioResponse
    from zhenxun.services.ai.llm.adapters.base import BaseAdapter
    from zhenxun.services.ai.llm.service import LLMModel


class MiniMaxAudioHandler(BaseAudioHandler):
    """MiniMax 专有文本转语音处理器"""

    def prepare_speech_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        input_text: str,
        voice: str,
        config: "TTSConfig",
    ) -> RequestData:
        endpoint = "/v1/t2a_v2"
        base_url = (
            model.api_base.rstrip("/") if model.api_base else "https://api.minimaxi.com"
        )
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        url = f"{base_url}{endpoint}"

        headers = adapter.get_base_headers(api_key)

        voice_setting: dict[str, Any] = {"voice_id": voice}
        if config.speed != 1.0:
            voice_setting["speed"] = config.speed
        if config.minimax_options.vol is not None:
            voice_setting["vol"] = config.minimax_options.vol
        if config.minimax_options.pitch is not None:
            voice_setting["pitch"] = config.minimax_options.pitch
        if config.minimax_options.emotion is not None:
            voice_setting["emotion"] = config.minimax_options.emotion

        target_format = config.response_format
        if target_format not in ("mp3", "pcm", "flac", "wav"):
            target_format = "mp3"

        audio_setting = {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": target_format,
            "channel": 1,
        }

        body: dict[str, Any] = {
            "model": model.model_name,
            "text": input_text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
        }

        if config.minimax_options.timbre_weights:
            body["timbre_weights"] = config.minimax_options.timbre_weights
            body["voice_setting"]["voice_id"] = ""

        if config.minimax_options.pronunciation_dict:
            body["pronunciation_dict"] = config.minimax_options.pronunciation_dict

        return RequestData(url=url, headers=headers, body=body)

    async def parse_speech_response(
        self, adapter: "BaseAdapter", model: "LLMModel", raw_response: Any
    ) -> "AudioResponse":
        resp_bytes = await raw_response.aread()
        data = json.loads(resp_bytes)

        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise LLMException(
                f"MiniMax 语音合成失败: {base_resp.get('status_msg')}",
                code=LLMErrorCode.API_RESPONSE_INVALID,
                details=data,
            )

        try:
            audio_hex = data["data"]["audio"]
            audio_bytes = bytes.fromhex(audio_hex)
        except (KeyError, ValueError) as e:
            raise LLMException(f"解析 MiniMax 语音 Hex 数据失败: {e}", details=data)

        extra_info = data.get("extra_info", {})
        audio_format = extra_info.get("audio_format", "mp3")
        usage_chars = extra_info.get("usage_characters", 0)

        from zhenxun.services.ai.core.messages import AudioResponse, UsageInfo

        usage = UsageInfo()
        usage.prompt_tokens = usage_chars

        return AudioResponse(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            usage=usage,
            model_name=model.model_name,
            raw_response=data,
        )


class MiniMaxConfigMapper(OpenAIConfigMapper):
    """MiniMax 专属配置映射器"""

    def map_config(
        self,
        config: GenerationConfig,
        model_detail: ModelDetail | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> dict[str, Any]:
        params = super().map_config(config, model_detail, capabilities)
        params["reasoning_split"] = True
        return params


class MiniMaxTextHandler(OpenAITextHandler):
    """MiniMax 复合文本处理器"""

    def __init__(self, api_type: str = "minimax"):
        super().__init__(api_type=api_type)
        self.mapper = MiniMaxConfigMapper(api_type=api_type)


class MiniMaxAdapter(OpenAICompatAdapter):
    """
    MiniMax API 适配器。
    """

    def __init__(self):
        """初始化 MiniMax 适配器并挂载专属处理器。"""
        super().__init__()
        self.text_handler = MiniMaxTextHandler(api_type=self.api_type)
        self.audio_handler = MiniMaxAudioHandler()

    @property
    def api_type(self) -> str:
        """适配器主类型标识。"""
        return "minimax"

    @property
    def supported_api_types(self) -> list[str]:
        """当前适配器支持的 API 类型列表。"""
        return ["minimax"]

    def get_chat_endpoint(self, model: "LLMModel") -> str:
        """根据官方兼容要求，重写获取端点，允许自定义覆盖"""
        if model.model_detail.endpoint:
            return model.model_detail.endpoint
        return "/v1/chat/completions"

    def _get_base_url(self, model: "LLMModel") -> str:
        base_url = (
            model.api_base.rstrip("/") if model.api_base else "https://api.minimaxi.com"
        )
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url
