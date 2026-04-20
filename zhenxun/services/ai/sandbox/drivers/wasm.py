import asyncio
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import aiofiles

from zhenxun.configs.config import Config
from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.types.sandbox import (
    SandboxExecutionResult,
    SandboxSecurityProfile,
)
from zhenxun.services.log import logger

from .base import BaseSandboxDriver
from ..extension import SupportsFileSystem

try:
    import wasmtime

    WASMTIME_AVAILABLE = True
except ImportError:
    wasmtime = None
    WASMTIME_AVAILABLE = False


def get_wasm_python_path() -> Path:
    """动态获取配置的 wasm 镜像路径"""
    image_name = Config.get_config("sandbox", "WASM_IMAGE", "python")
    return DATA_PATH / "ai" / "sandbox" / f"{image_name}.wasm"


class WasmtimeCoreEngine:
    """
    WebAssembly (Wasmtime) 底层沙箱执行引擎。
    仅负责纯计算与字符串代码的物理隔离执行，不涉及高层 Agent 路由逻辑。
    """

    @classmethod
    def is_available(cls) -> bool:
        return WASMTIME_AVAILABLE

    @classmethod
    def check_wasm_file(cls) -> bool:
        return get_wasm_python_path().exists()

    @classmethod
    async def run_wasm_plugin(
        cls,
        wasm_path: Path,
        stdin_data: str,
        args: list[str] | None = None,
        fuel: int = 2_000_000_000,
    ) -> dict[str, Any]:
        """
        执行通用的跨语言 Wasm 插件 (Rust/Go/C++ 等编译的 WASI 组件)。
        通过标准输入 (stdin) 传递 JSON 参数，通过标准输出 (stdout) 接收结果。
        """
        return await asyncio.to_thread(
            cls._run_wasm_plugin_sync, wasm_path, stdin_data, args or [], fuel
        )

    @classmethod
    async def run_code(
        cls, code: str, fuel: int = 200_000_000, workspace_dir: str | None = None
    ) -> dict[str, Any]:
        """
        利用 asyncio.to_thread 异步执行同步的 wasmtime 引擎，防止阻塞主线程。
        """
        return await asyncio.to_thread(cls._run_code_sync, code, fuel, workspace_dir)

    @classmethod
    def _run_wasm_plugin_sync(
        cls, wasm_path: Path, stdin_data: str, args: list[str], fuel: int
    ) -> dict[str, Any]:
        if not WASMTIME_AVAILABLE:
            return {"exit_code": -1, "stdout": "", "stderr": "wasmtime 库未安装。"}
        if not wasm_path.exists():
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"找不到插件文件: {wasm_path}",
            }

        wt: Any = wasmtime

        stdin_fd, stdin_path = tempfile.mkstemp(suffix=".in")
        with open(stdin_fd, "w", encoding="utf-8") as f:
            f.write(stdin_data)

        stdout_fd, stdout_path = tempfile.mkstemp(suffix=".out")
        stderr_fd, stderr_path = tempfile.mkstemp(suffix=".err")
        os.close(stdout_fd)
        os.close(stderr_fd)

        exit_code = 0
        try:
            config = wt.Config()
            config.consume_fuel = True
            config.cache = True

            engine = wt.Engine(config)
            store = wt.Store(engine)

            if hasattr(store, "set_fuel"):
                store.set_fuel(fuel)
            elif hasattr(store, "add_fuel"):
                store.add_fuel(fuel)

            wasi = wt.WasiConfig()
            wasi.argv = [wasm_path.name] + args
            wasi.stdin_file = stdin_path
            wasi.stdout_file = stdout_path
            wasi.stderr_file = stderr_path
            store.set_wasi(wasi)

            linker = wt.Linker(engine)
            linker.define_wasi()
            module = wt.Module.from_file(engine, str(wasm_path))
            instance = linker.instantiate(store, module)
            start_func = instance.exports(store).get("_start")
            start_func(store)

        except Exception as e:
            with open(stderr_path, "a", encoding="utf-8") as err_f:
                err_f.write(f"\n[Execution Error]: {e}")
            exit_code = -1
        finally:
            with open(stdout_path, encoding="utf-8") as f:
                stdout_str = f.read()
            with open(stderr_path, encoding="utf-8") as f:
                stderr_str = f.read()
            for p in [stdin_path, stdout_path, stderr_path]:
                Path(p).unlink(missing_ok=True)

        return {
            "exit_code": exit_code,
            "stdout": stdout_str.strip(),
            "stderr": stderr_str.strip(),
        }

    @classmethod
    def _run_code_sync(
        cls, code: str, fuel: int, workspace_dir: str | None
    ) -> dict[str, Any]:
        if not WASMTIME_AVAILABLE:
            return {"exit_code": -1, "stdout": "", "stderr": "wasmtime 库未安装。"}

        wt: Any = wasmtime
        wasm_path = get_wasm_python_path()
        if not wasm_path.exists():
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"缺少底层沙箱镜像文件: {wasm_path}",
            }

        stdout_fd, stdout_path = tempfile.mkstemp(suffix=".out")
        stderr_fd, stderr_path = tempfile.mkstemp(suffix=".err")
        os.close(stdout_fd)
        os.close(stderr_fd)

        exit_code = 0
        try:
            config = wt.Config()
            config.consume_fuel = True
            config.cache = True

            engine = wt.Engine(config)
            store = wt.Store(engine)

            if hasattr(store, "set_fuel"):
                store.set_fuel(fuel)
            elif hasattr(store, "add_fuel"):
                store.add_fuel(fuel)

            wasi = wt.WasiConfig()
            wasi.argv = ["python", "-c", code]
            wasi.stdout_file = stdout_path
            wasi.stderr_file = stderr_path

            if workspace_dir and Path(workspace_dir).exists():
                wasi.preopen_dir(workspace_dir, "/workspace")

            store.set_wasi(wasi)

            linker = wt.Linker(engine)
            linker.define_wasi()

            module = wt.Module.from_file(engine, str(wasm_path))
            instance = linker.instantiate(store, module)
            start_func = instance.exports(store).get("_start")

            start_func(store)

        except wt.ExitTrap as e:
            exit_code = e.code
        except wt.Trap as e:
            with open(stderr_path, "a", encoding="utf-8") as err_f:
                err_f.write(f"\n[Wasmtime Trap 拦截]: 执行被强制中止，原因 -> {e}")
            exit_code = -1
        except Exception as e:
            with open(stderr_path, "a", encoding="utf-8") as err_f:
                err_f.write(f"\n[System Error]: {e}")
            exit_code = -1
        finally:
            with open(stdout_path, encoding="utf-8") as f:
                stdout_str = f.read()
            with open(stderr_path, encoding="utf-8") as f:
                stderr_str = f.read()

            Path(stdout_path).unlink(missing_ok=True)
            Path(stderr_path).unlink(missing_ok=True)

        return {
            "exit_code": exit_code,
            "stdout": stdout_str.strip(),
            "stderr": stderr_str.strip(),
        }


class WasmDriver(BaseSandboxDriver, SupportsFileSystem):
    """Wasmtime 沙箱的标准驱动实现"""

    def _get_real_path(self, sandbox_path: str) -> Path:
        rel_path = sandbox_path.replace("/workspace/", "").lstrip("/")
        return Path(self.workspace) / rel_path

    async def write_raw_file(self, path: str, content: str) -> bool:
        self.touch()
        real_path = self._get_real_path(path)
        real_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(real_path, "w", encoding="utf-8") as f:
            await f.write(content)
        return True

    async def read_raw_file(self, path: str) -> str:
        self.touch()
        real_path = self._get_real_path(path)
        if not real_path.exists():
            return f"Error: File {path} not found."
        async with aiofiles.open(real_path, encoding="utf-8") as f:
            return await f.read()

    async def delete_raw_file(self, path: str) -> bool:
        self.touch()
        try:
            self._get_real_path(path).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    async def upload_raw_dir(
        self, local_dir_path: str, sandbox_target_path: str
    ) -> bool:
        self.touch()
        try:
            await asyncio.to_thread(
                shutil.copytree,
                Path(local_dir_path),
                self._get_real_path(sandbox_target_path),
                dirs_exist_ok=True,
            )
            return True
        except Exception as e:
            logger.error(f"[WasmDriver] upload_dir 失败: {e}")
            return False

    @property
    def supports_state(self) -> bool:
        return False

    async def start(
        self, session_id: str, profile: SandboxSecurityProfile | None = None
    ) -> None:
        self.session_id = session_id
        self.profile = profile

        self.workspace = tempfile.mkdtemp(prefix=f"wasm_workspace_{session_id}_")

        self.touch()
        logger.info(f"[WasmDriver] 极速沙箱已分配 (Session: {session_id})")

    async def close(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)
        logger.info(f"[WasmDriver] 极速沙箱已销毁回收 (Session: {self.session_id})")
