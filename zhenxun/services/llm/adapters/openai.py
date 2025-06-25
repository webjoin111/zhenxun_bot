"""
OpenAI API 适配器

支持 OpenAI、DeepSeek、智谱AI 和其他 OpenAI 兼容的 API 服务。
"""

from typing import TYPE_CHECKING, Any

from .base import OpenAICompatAdapter, RequestData

if TYPE_CHECKING:
    from ..service import LLMModel
    from ..types.enums import EmbeddingTaskType


class OpenAIAdapter(OpenAICompatAdapter):
    """OpenAI兼容API适配器"""

    @property
    def api_type(self) -> str:
        return "openai"

    @property
    def supported_api_types(self) -> list[str]:
        return ["openai", "deepseek", "zhipu", "general_openai_compat", "ark"]

    def get_chat_endpoint(self, model: "LLMModel") -> str:
        """返回聊天完成端点"""
        if model.api_type == "ark":
            return "/api/v3/chat/completions"
        if model.api_type == "zhipu":
            return "/api/paas/v4/chat/completions"
        return "/v1/chat/completions"

    def get_embedding_endpoint(self) -> str:
        """返回嵌入端点"""
        return "/v1/embeddings"

    def prepare_embedding_request(
        self,
        model: "LLMModel",
        api_key: str,
        texts: list[str],
        task_type: "EmbeddingTaskType | str",
        **kwargs: Any,
    ) -> RequestData:
        """准备嵌入请求 - OpenAI兼容格式"""
        _ = task_type

        # 根据 api_type 动态选择端点
        if model.api_type == "zhipu":
            endpoint = "/v4/embeddings"
        else:
            endpoint = self.get_embedding_endpoint()

        url = self.get_api_url(model, endpoint)
        headers = self.get_base_headers(api_key)

        body = {
            "model": model.model_name,
            "input": texts,
        }

        if kwargs:
            body.update(kwargs)

        return RequestData(url=url, headers=headers, body=body)
