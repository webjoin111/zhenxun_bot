from typing import TYPE_CHECKING

from zhenxun.services.ai.llm.adapters.base import RequestData
from zhenxun.services.ai.llm.adapters.handlers.openai_handlers import (
    OpenAIEmbeddingHandler,
    OpenAIRerankHandler,
    OpenAITextHandler,
)
from zhenxun.services.ai.llm.adapters.openai import OpenAICompatAdapter

if TYPE_CHECKING:
    from zhenxun.services.ai.llm.adapters.base import BaseAdapter
    from zhenxun.services.ai.llm.service import LLMModel


class GLMRerankHandler(OpenAIRerankHandler):
    """GLM 专有的重排处理器（重写了端点构建逻辑）"""

    def prepare_rerank_request(
        self,
        adapter: "BaseAdapter",
        model: "LLMModel",
        api_key: str,
        query: str,
        documents: list[str | dict[str, str]],
        top_n: int,
    ) -> RequestData:
        """构建 GLM 重排请求，统一将文档归一化为字符串列表。"""
        endpoint = "/api/paas/v4/rerank"
        url = adapter.get_api_url(model, endpoint)
        headers = adapter.get_base_headers(api_key)

        safe_documents = []
        for doc in documents:
            if isinstance(doc, dict):
                safe_documents.append(doc.get("text", str(doc)))
            else:
                safe_documents.append(str(doc))

        body = {
            "model": model.model_name,
            "query": query,
            "documents": safe_documents,
            "top_n": top_n,
        }
        return RequestData(url=url, headers=headers, body=body)


class GLMAdapter(OpenAICompatAdapter):
    """GLM (智谱) 大模型专有适配器 (继承 OpenAI 兼容协议处理标准聊天)"""

    def __init__(self):
        """初始化 GLM 适配器并装配专有图像/重排处理器。"""
        super().__init__()
        self.text_handler = OpenAITextHandler(api_type=self.api_type)
        self.embedding_handler = OpenAIEmbeddingHandler()
        self.rerank_handler = GLMRerankHandler()

    @property
    def api_type(self) -> str:
        """适配器主类型标识。"""
        return "glm"

    @property
    def supported_api_types(self) -> list[str]:
        """当前适配器支持的 API 类型列表。"""
        return ["glm"]

    def get_chat_endpoint(self, model: "LLMModel") -> str:
        """返回对话端点，优先使用模型级自定义端点。"""
        if model.model_detail.endpoint:
            return model.model_detail.endpoint
        return "/api/paas/v4/chat/completions"

    def get_embedding_endpoint(self, model: "LLMModel") -> str:
        """返回嵌入端点。"""
        return "/api/paas/v4/embeddings"
