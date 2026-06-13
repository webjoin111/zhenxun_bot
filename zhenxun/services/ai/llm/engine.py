from abc import ABC, abstractmethod
import asyncio
import json
from typing import Any

import httpx

from zhenxun.services.ai.llm.adapters.base import RequestData
from zhenxun.services.ai.llm.core import LLMHttpClient
from zhenxun.services.ai.protocols.middleware import LLMContext


class BaseEngine(ABC):
    """
    底层模型执行引擎协议。
    彻底解耦 HTTP 与本地显存模型。
    """

    @abstractmethod
    async def execute(
        self, context: LLMContext, payload: RequestData
    ) -> httpx.Response | Any:
        """
        执行底层调用。

        参数:
            context: 执行上下文
            payload: 由 Adapter 构建的底层所需数据

        返回:
            httpx.Response | Any: 原始执行结果，通常为 HTTP 响应对象
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """释放引擎相关资源"""
        pass


class HttpEngine(BaseEngine):
    """基于 HTTP 的执行引擎"""

    def __init__(self, client: LLMHttpClient):
        self.client = client

    async def execute(
        self, context: LLMContext, payload: RequestData
    ) -> httpx.Response:
        if not isinstance(payload, RequestData):
            raise ValueError("HttpEngine 仅支持 RequestData 类型的 Payload")

        method = getattr(payload, "method", "POST").upper()
        req_kwargs = {"headers": payload.headers, "timeout": context.timeout}

        if method in ("POST", "PUT", "PATCH"):
            if payload.files:
                req_kwargs["data"] = payload.body
                req_kwargs["files"] = payload.files
            else:
                req_kwargs["content"] = json.dumps(payload.body, ensure_ascii=False)
        elif method == "GET" and payload.body:
            req_kwargs["params"] = payload.body

        post_task = asyncio.create_task(
            self.client.request(method, payload.url, **req_kwargs)
        )

        if context.cancellation_token:
            context.cancellation_token.link_future(post_task)

        return await post_task

    async def close(self) -> None:
        pass
