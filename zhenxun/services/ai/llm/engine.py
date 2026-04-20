from abc import ABC, abstractmethod
import asyncio
import json
from typing import Any

from zhenxun.services.ai.llm.adapters.base import RequestData
from zhenxun.services.ai.llm.core import LLMHttpClient
from zhenxun.services.ai.protocols.middleware import LLMContext


class BaseEngine(ABC):
    """
    底层模型执行引擎协议。
    彻底解耦 HTTP 与本地显存模型。
    """

    @abstractmethod
    async def execute(self, context: LLMContext, payload: Any) -> Any:
        """
        执行底层调用。

        参数:
            context: 执行上下文
            payload: 由 Adapter 构建的底层所需数据 (如 HTTP 的 RequestData)

        返回:
            Any: 原始执行结果
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

    async def execute(self, context: LLMContext, payload: Any) -> Any:
        if not isinstance(payload, RequestData):
            raise ValueError("HttpEngine 仅支持 RequestData 类型的 Payload")

        post_task = asyncio.create_task(
            self.client.post(
                payload.url,
                headers=payload.headers,
                content=json.dumps(payload.body, ensure_ascii=False)
                if not payload.files
                else None,
                data=payload.body if payload.files else None,
                files=payload.files,
                timeout=context.timeout,
            )
        )

        if context.cancellation_token:
            context.cancellation_token.link_future(post_task)

        return await post_task

    async def close(self) -> None:
        pass
