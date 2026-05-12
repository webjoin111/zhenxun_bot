from abc import ABC, abstractmethod

from zhenxun.services.ai.sandbox.models import (
    SandboxCapabilities,
    SandboxRequirements,
    SandboxSecurityProfile,
)

from ..drivers.base import BaseSandboxDriver


class BaseSandboxProvider(ABC):
    """沙箱提供者抽象基类 (遵循 Provider Pattern)"""

    @abstractmethod
    def get_name(self) -> str:
        """返回提供者唯一名称，如 'docker', 'e2b', 'wasm'"""
        pass

    @abstractmethod
    def get_capabilities(self) -> SandboxCapabilities:
        """声明该提供者的沙箱能力边界"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查当前环境该提供者是否可用（如依赖包是否安装，API Key是否配置）"""
        pass

    @abstractmethod
    def score(
        self, profile: SandboxSecurityProfile, requirements: SandboxRequirements | None
    ) -> int:
        """核心路由打分算法。返回当前请求的匹配得分，返回 -1 表示无法支持该任务。"""
        pass

    @abstractmethod
    def create_driver(self, session_id: str) -> BaseSandboxDriver:
        """实例化并返回沙箱执行环境的底层驱动"""
        pass

