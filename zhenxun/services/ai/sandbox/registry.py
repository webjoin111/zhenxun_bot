from typing import TYPE_CHECKING, ClassVar

from zhenxun.services.log import logger

if TYPE_CHECKING:
    from zhenxun.services.ai.sandbox.addons.base import BaseSandboxExtension
    from zhenxun.services.ai.sandbox.drivers.base import BaseSandboxClient


class SandboxRegistry:
    _clients: ClassVar[dict[str, type["BaseSandboxClient"]]] = {}
    _extensions: ClassVar[dict[str, type["BaseSandboxExtension"]]] = {}

    @classmethod
    def register_client(cls, name: str, client_cls: type["BaseSandboxClient"]) -> None:
        if name in cls._clients:
            logger.warning(f"[SandboxRegistry] 覆盖已存在的沙箱客户端: {name}")
        cls._clients[name] = client_cls
        logger.info(f"[SandboxRegistry] 成功注册沙箱客户端: {name}")

    @classmethod
    def get_client_cls(cls, name: str) -> type["BaseSandboxClient"]:
        if name not in cls._clients:
            raise ValueError(f"未找到名为 '{name}' 的沙箱客户端。")
        return cls._clients[name]

    @classmethod
    def get_all_clients(cls) -> dict[str, type["BaseSandboxClient"]]:
        return cls._clients.copy()

    @classmethod
    def register_extension(cls, extension_cls: type["BaseSandboxExtension"]) -> None:
        if (
            isinstance(getattr(extension_cls, "extension_name", None), property)
            and extension_cls.extension_name.fget
        ):
            name = extension_cls.extension_name.fget(None)
        else:
            name = getattr(extension_cls, "extension_name", str(extension_cls))
        cls._extensions[name] = extension_cls
        logger.debug(f"[SandboxRegistry] 成功注册沙箱扩展: {name}")

    @classmethod
    def get_extension_cls(cls, name: str) -> type["BaseSandboxExtension"] | None:
        return cls._extensions.get(name)
