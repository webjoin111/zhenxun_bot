from typing import Any, Protocol


class PromptProvider(Protocol):
    """Prompt 源协议。用于动态获取指定的系统提示词模板。"""

    async def get_prompt(self, name: str, **kwargs: Any) -> str | None: ...


class ResourceProvider(Protocol):
    """资源源协议。用于读取外部的静态或动态资源。"""

    async def read_resource(self, uri: str, **kwargs: Any) -> str | None: ...
