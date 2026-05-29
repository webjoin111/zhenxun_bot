"""
沙箱相关核心类型定义
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class LanguageProfile(BaseModel):
    """语言执行配置模板"""

    language: str
    """语言名称"""
    aliases: list[str] = Field(default_factory=list)
    """语言别名列表"""
    source_ext: str
    """源文件后缀名"""
    compile_cmd: str | None = None
    """编译命令"""
    run_cmd: str
    """运行命令"""
    deps_install_cmd: str | None = None
    """依赖安装命令"""
    inject_rpc_stub: bool = False
    """是否注入 RPC 存根"""


class BlueprintFile(BaseModel):
    """蓝图预置文件配置"""

    path: str
    """沙箱内文件路径"""
    content: str | bytes | None = None
    """文件内容，支持文本或二进制"""
    local_path: str | None = None
    """宿主机本地文件路径"""


class SandboxBlueprint(BaseModel):
    """沙箱大一统声明式蓝图配置"""

    sandbox_type: str = Field(default="auto")
    """强制驱动类型，如 docker 或 local"""
    enable_network: bool = Field(default=False)
    """是否允许访问外网"""
    needs_state: bool = Field(default=False)
    """是否需要持久化状态"""

    python_packages: list[str] = Field(default_factory=list)
    """Python 依赖包"""
    system_packages: list[str] = Field(default_factory=list)
    """系统级依赖包 (apt)"""
    node_packages: list[str] = Field(default_factory=list)
    """Node 依赖包 (npm)"""
    install_scripts: list[str] = Field(default_factory=list)
    """自定义 Shell 脚本"""

    files: list[BlueprintFile] = Field(default_factory=list)
    """预置文件列表"""
    env: dict[str, str] = Field(default_factory=dict)
    """环境变量"""

    required_extensions: list[str] = Field(default_factory=list)
    """需要自动挂载的扩展"""

    def with_sandbox_type(self, sandbox_type: str) -> "SandboxBlueprint":
        """设置沙箱驱动类型"""
        self.sandbox_type = sandbox_type
        return self

    def with_network(self, enable: bool = True) -> "SandboxBlueprint":
        """设置是否启用网络访问"""
        self.enable_network = enable
        return self

    def with_state(self, enable: bool = True) -> "SandboxBlueprint":
        """设置是否启用状态持久化"""
        self.needs_state = enable
        return self

    def with_python_packages(self, packages: list[str]) -> "SandboxBlueprint":
        """追加预置 Python 依赖包"""
        self.python_packages.extend(packages)
        self.python_packages = list(dict.fromkeys(self.python_packages))
        return self

    def with_system_packages(self, packages: list[str]) -> "SandboxBlueprint":
        """追加预置系统包依赖"""
        self.system_packages.extend(packages)
        self.system_packages = list(dict.fromkeys(self.system_packages))
        return self

    def with_node_packages(self, packages: list[str]) -> "SandboxBlueprint":
        """追加预置 Node 依赖包"""
        self.node_packages.extend(packages)
        self.node_packages = list(dict.fromkeys(self.node_packages))
        return self

    def with_install_scripts(self, scripts: list[str]) -> "SandboxBlueprint":
        """追加自定义安装脚本"""
        self.install_scripts.extend(scripts)
        return self

    def with_file(self, path: str, content: str | bytes) -> "SandboxBlueprint":
        """预置内存字符串或二进制文件"""
        self.files.append(BlueprintFile(path=path, content=content))
        return self

    def with_local_file(self, path: str, local_path: str) -> "SandboxBlueprint":
        """预置本地宿主机文件挂载"""
        self.files.append(BlueprintFile(path=path, local_path=local_path))
        return self

    def with_env(self, key: str, value: str) -> "SandboxBlueprint":
        """设置环境变量"""
        self.env[key] = value
        return self

    def with_extension(self, extension: str) -> "SandboxBlueprint":
        """声明需要挂载的扩展"""
        if extension not in self.required_extensions:
            self.required_extensions.append(extension)
        return self

    def calculate_hash(self) -> str:
        """计算当前环境配置的 MD5 指纹，用于缓存命中"""
        data_to_hash = {
            "py": sorted(self.python_packages),
            "sys": sorted(self.system_packages),
            "node": sorted(self.node_packages),
            "scripts": self.install_scripts,
        }
        json_str = json.dumps(data_to_hash, separators=(",", ":"))
        return hashlib.md5(json_str.encode("utf-8")).hexdigest()

    def merge(self, other: "SandboxBlueprint") -> "SandboxBlueprint":
        """合并另一个蓝图配置"""
        if not other:
            return self
        self.enable_network = self.enable_network or other.enable_network
        self.needs_state = self.needs_state or other.needs_state
        self.sandbox_type = (
            other.sandbox_type if other.sandbox_type != "auto" else self.sandbox_type
        )

        self.python_packages = list(
            dict.fromkeys(self.python_packages + other.python_packages)
        )
        self.system_packages = list(
            dict.fromkeys(self.system_packages + other.system_packages)
        )
        self.node_packages = list(
            dict.fromkeys(self.node_packages + other.node_packages)
        )
        self.install_scripts.extend(other.install_scripts)
        self.files.extend(other.files)
        self.env.update(other.env)
        self.required_extensions = list(
            dict.fromkeys(self.required_extensions + other.required_extensions)
        )
        return self


class SandboxExecutionResult(BaseModel):
    """沙箱统一的执行结果数据结构"""

    stdout: str = Field(default="")
    """标准输出流"""
    stderr: str = Field(default="")
    """标准错误流"""
    exit_code: int = Field(default=0)
    """进程退出码"""
    error: str | None = Field(default=None)
    """框架执行错误信息"""
    is_timeout: bool = Field(default=False)
    """进程是否因超时被挂起(仍在运行)"""
    images: list[str] = Field(default_factory=list)
    """Base64 图片列表"""
    artifacts: dict[str, bytes] = Field(default_factory=dict)
    """生成的文件工件"""

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and self.error is None


class SandboxSessionState(BaseModel):
    """沙箱会话的序列化状态，用于无状态恢复"""

    session_id: str
    """会话唯一标识符"""
    backend_id: str
    """后端驱动分配的容器/沙箱 ID"""
    sandbox_type: str = "docker"
    """沙箱驱动类型"""
    workspace_root_ready: bool = False
    """工作区目录是否已初始化完成"""
    extra_data: dict[str, Any] = Field(default_factory=dict)
    """会话的其它序列化元数据"""


class CodeBlock(BaseModel):
    """大模型生成的、将被沙箱执行的代码块抽象结构"""

    code: str
    """具体的代码片段内容"""
    language: str
    """代码的编程语言，如 python, sh, bash 等"""
