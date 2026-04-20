"""
沙箱相关核心类型定义
"""

from enum import Enum

from pydantic import BaseModel, Field


class SandboxTier(str, Enum):
    """沙箱层级枚举"""

    LIGHTWEIGHT = "LIGHTWEIGHT"
    STANDARD = "STANDARD"
    HEAVY = "HEAVY"


class SandboxRequirements(BaseModel):
    """代码执行对沙箱的需求评估"""

    tier: SandboxTier = Field(
        default=SandboxTier.LIGHTWEIGHT, description="最低需求层级"
    )
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict, description="环境依赖字典，Key为Provisioner名称"
    )
    required_plugins: list[str] = Field(
        default_factory=list, description="需要自动挂载的插件列表"
    )

    def merge(self, other: "SandboxRequirements") -> "SandboxRequirements":
        """合并另一个依赖需求对象"""
        if not other:
            return self

        for prov_name, pkgs in other.dependencies.items():
            if prov_name not in self.dependencies:
                self.dependencies[prov_name] = []
            self.dependencies[prov_name] = list(
                set(self.dependencies[prov_name] + pkgs)
            )

        self.required_plugins = list(set(self.required_plugins + other.required_plugins))

        tier_order = {
            SandboxTier.LIGHTWEIGHT: 1,
            SandboxTier.STANDARD: 2,
            SandboxTier.HEAVY: 3,
        }
        if tier_order[other.tier] > tier_order[self.tier]:
            self.tier = other.tier
        return self


class SandboxExecutionResult(BaseModel):
    """沙箱统一的执行结果数据结构"""

    stdout: str = Field(default="", description="标准输出流")
    stderr: str = Field(default="", description="标准错误流")
    exit_code: int = Field(default=0, description="进程退出码")
    error: str | None = Field(default=None, description="框架执行错误信息")
    is_timeout: bool = Field(
        default=False, description="进程是否因超时被挂起(仍在运行)"
    )
    images: list[str] = Field(default_factory=list, description="Base64图片列表")
    artifacts: dict[str, bytes] = Field(
        default_factory=dict, description="生成的文件工件"
    )

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and self.error is None


class SandboxCapabilities(BaseModel):
    """沙箱能力声明"""

    supports_state: bool = Field(default=False)
    supported_capabilities: list[str] = Field(
        default_factory=list, description="该沙箱提供者支持的Capability类名列表"
    )
    isolation_level: int = Field(default=1, description="隔离级别 (1-10)")
    startup_latency: int = Field(default=1000, description="启动延迟估算(ms)")


class SandboxSecurityProfile(BaseModel):
    """沙箱安全策略配置"""

    enable_network: bool = Field(default=False, description="是否允许访问外网")
    sandbox_type: str = Field(default="auto", description="强制驱动类型")
    needs_state: bool = Field(default=True, description="是否需要持久化状态")
    require_gpu: bool = Field(default=False, description="是否需要 GPU")
    require_pty: bool = Field(
        default=False,
        description="是否强制需要全功能 PTY 交互终端 (将禁用高级 Jupyter 绘图特性)",
    )
    required_plugins: list[str] = Field(
        default_factory=list, description="强制挂载的插件列表"
    )
