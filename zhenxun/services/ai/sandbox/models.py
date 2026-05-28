"""
沙箱相关核心类型定义
"""

from enum import Enum
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SandboxTier(str, Enum):
    """沙箱层级枚举"""

    LIGHTWEIGHT = "LIGHTWEIGHT"
    STANDARD = "STANDARD"
    HEAVY = "HEAVY"


class EnvSetupConfig(BaseModel):
    """统一环境装配声明模型"""

    python_packages: list[str] = Field(default_factory=list)
    """Python 依赖包列表 (将通过 uv 安装)"""
    system_packages: list[str] = Field(default_factory=list)
    """系统级依赖包列表 (如 apt-get install)"""
    bins: list[str] = Field(default_factory=list)
    """前置环境检查的二进制命令 (如 node, git)"""
    install_scripts: list[str] = Field(default_factory=list)
    """自定义安装 Shell 脚本列表"""

    def calculate_hash(self) -> str:
        """计算当前环境清单的 MD5 指纹，用于沙箱缓存命中"""
        data_to_hash = {
            "py": sorted(self.python_packages),
            "sys": sorted(self.system_packages),
            "bin": sorted(self.bins),
            "scripts": self.install_scripts,
        }
        json_str = json.dumps(data_to_hash, separators=(",", ":"))
        return hashlib.md5(json_str.encode("utf-8")).hexdigest()


class SandboxRequirements(BaseModel):
    """代码执行对沙箱的需求评估"""

    tier: SandboxTier = Field(default=SandboxTier.LIGHTWEIGHT)
    """最低需求层级"""
    env_setup: EnvSetupConfig = Field(default_factory=EnvSetupConfig)
    """沙箱环境装配配置"""
    required_extensions: list[str] = Field(default_factory=list)
    """需要自动挂载的扩展列表"""

    def merge(self, other: "SandboxRequirements") -> "SandboxRequirements":
        """合并另一个依赖需求对象"""
        if not other:
            return self

        self.env_setup.python_packages = list(
            dict.fromkeys(
                self.env_setup.python_packages + other.env_setup.python_packages
            )
        )
        self.env_setup.system_packages = list(
            dict.fromkeys(
                self.env_setup.system_packages + other.env_setup.system_packages
            )
        )
        self.env_setup.bins = list(
            dict.fromkeys(self.env_setup.bins + other.env_setup.bins)
        )
        self.env_setup.install_scripts.extend(other.env_setup.install_scripts)

        self.required_extensions = list(
            set(self.required_extensions + other.required_extensions)
        )

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
    sandbox_type: str = Field(default="docker", description="强制驱动类型")
    needs_state: bool = Field(default=True, description="是否需要持久化状态")
    require_gpu: bool = Field(default=False, description="是否需要 GPU")
    required_extensions: list[str] = Field(
        default_factory=list, description="强制挂载的扩展列表"
    )


class CodeBlock(BaseModel):
    """大模型生成的、将被沙箱执行的代码块抽象结构"""

    code: str = Field(description="具体的代码片段内容")
    language: str = Field(description="代码的编程语言，如 python, sh, bash 等")


class BaseEntry(BaseModel):
    pass


class FileEntry(BaseEntry):
    type: Literal["file"] = "file"
    content: str | bytes


class LocalFileEntry(BaseEntry):
    type: Literal["local_file"] = "local_file"
    src_path: str


class DirEntry(BaseEntry):
    type: Literal["dir"] = "dir"
    children: dict[str, "EntryUnion"] = Field(default_factory=dict)


class GitRepoEntry(BaseEntry):
    type: Literal["git_repo"] = "git_repo"
    url: str
    ref: str | None = None


EntryUnion = Annotated[
    FileEntry | LocalFileEntry | DirEntry | GitRepoEntry, Field(discriminator="type")
]
DirEntry.model_rebuild()


class Manifest(BaseModel):
    """声明式沙箱工作区初始化清单"""

    entries: dict[str, EntryUnion] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
