from .base import BaseCodeExecutor
from .python_cli import PythonCLIExecutor
from .python_jupyter import PythonJupyterExecutor
from .registry import CodeExecutorRegistry

CodeExecutorRegistry.register(
    "python", PythonCLIExecutor, is_stateful=False, scope="global"
)
CodeExecutorRegistry.register(
    "python", PythonJupyterExecutor, is_stateful=True, scope="global"
)

__all__ = [
    "BaseCodeExecutor",
    "CodeExecutorRegistry",
    "PythonCLIExecutor",
    "PythonJupyterExecutor",
]
