from typing import Any

from zhenxun.services.ai.protocols.resource import PromptProvider, ResourceProvider
from zhenxun.services.log import logger


class ContextResourceManager:
    """全局上下文资源(Prompt/Resource)管理器"""

    _instance: "ContextResourceManager | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._prompt_providers: list[PromptProvider] = []
        self._resource_providers: list[ResourceProvider] = []
        self._initialized = True

    def register_prompt_provider(self, provider: PromptProvider) -> None:
        if provider not in self._prompt_providers:
            self._prompt_providers.append(provider)

    def register_resource_provider(self, provider: ResourceProvider) -> None:
        if provider not in self._resource_providers:
            self._resource_providers.append(provider)

    async def fetch_prompt(self, name: str, **kwargs: Any) -> str | None:
        for provider in self._prompt_providers:
            try:
                result = await provider.get_prompt(name, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    f"从 {provider.__class__.__name__} 获取 Prompt '{name}' 失败: {e}"
                )
        return None

    async def fetch_resource(self, uri: str, **kwargs: Any) -> str | None:
        for provider in self._resource_providers:
            try:
                result = await provider.read_resource(uri, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    f"从 {provider.__class__.__name__} 读取 Resource '{uri}' 失败: {e}"
                )
        return None


context_resource_manager = ContextResourceManager()


class BuiltinContextProvider:
    """用于测试和内置资源的 Context Provider"""

    async def get_prompt(self, name: str, **kwargs: Any) -> str | None:
        if name == "test_prompt":
            return (
                "这是一个测试Prompt：[系统当前处于Debug模式，"
                "请在回复时带上'DEBUG_MODE_ACTIVE'标记]"
            )
        return None

    async def read_resource(self, uri: str, **kwargs: Any) -> str | None:
        if uri == "test://resource":
            return "这是一段测试资源内容：[资源数据：1024x768, 校验码: 0xDEADBEEF]"
        return None


_builtin_provider = BuiltinContextProvider()
context_resource_manager.register_prompt_provider(_builtin_provider)
context_resource_manager.register_resource_provider(_builtin_provider)
