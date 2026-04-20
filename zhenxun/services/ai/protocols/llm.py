from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from zhenxun.services.ai.types.configs import LLMEmbeddingConfig, LLMGenerationConfig
from zhenxun.services.ai.types.messages import LLMContentPart, LLMMessage, LLMResponse
from zhenxun.services.ai.types.models import ModelName
from zhenxun.services.ai.types.tools import ToolChoice

T = TypeVar("T", bound=BaseModel)


class LLMInterface(Protocol):
    """
    一个协议，定义了工具或智能体在执行期间可以安全调用的 LLM 能力防腐层。
    这是对完整 LLM 服务的一个受限、安全的子集。
    """

    async def chat(
        self,
        message: str | LLMMessage | list[LLMContentPart],
        *,
        model: ModelName = None,
        tools: list[dict[str, Any] | str] | None = None,
    ) -> LLMResponse:
        """执行一次无状态的、一次性的聊天调用。"""
        ...

    async def generate_structured(
        self,
        message: str | LLMMessage | list[LLMContentPart],
        response_model: type[T],
        *,
        model: ModelName = None,
        tools: list[dict[str, Any] | str] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        instruction: str | None = None,
    ) -> T:
        """执行一次无状态的、一次性的结构化内容生成。"""
        ...

    async def generate_internal(
        self,
        messages: list[LLMMessage],
        *,
        model: ModelName = None,
        config: Any | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        timeout: float | None = None,
        model_instance: Any = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any = None,
    ) -> LLMResponse:
        """[内部] 执行生成任务（供执行引擎调用）。"""
        ...


class LLMModelBase(ABC):
    """底层 LLM 模型抽象基类（约束 Service 实现）"""

    @abstractmethod
    async def generate_response(
        self,
        messages: list[LLMMessage],
        config: LLMGenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | ToolChoice | None = None,
        timeout: float | None = None,
        extra: dict[str, Any] | None = None,
        cancellation_token: Any | None = None,
    ) -> LLMResponse:
        """生成高级响应"""
        pass

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: list[str],
        config: LLMEmbeddingConfig,
    ) -> list[list[float]]:
        """生成文本嵌入向量"""
        pass
