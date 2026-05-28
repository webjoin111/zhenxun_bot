from collections.abc import Callable
from dataclasses import dataclass
import inspect
import textwrap
from textwrap import indent


@dataclass(frozen=True)
class Alias:
    """导入别名结构，如 import pandas as pd"""

    name: str
    alias: str


@dataclass(frozen=True)
class ImportFromModule:
    """从模块中导入，如 from datetime import datetime"""

    module: str
    imports: list[str | Alias]


ImportType = str | Alias | ImportFromModule


def sandbox_function(
    python_packages: list[str] | None = None,
    global_imports: list[ImportType] | None = None,
    host_callable: bool = False,
):
    """
    极其强大的宿主函数注入装饰器。
    将普通的 Python 宿主函数标记为“沙箱可用函数”，
    供后续将其源码和依赖直接投递到沙箱内部。

    Args:
        python_packages: 该函数运行所需的 pip 包（如 ["pandas>=1.0.0", "numpy"]）。
        global_imports: 该函数内部需要的导入声明。
        host_callable: 是否为宿主代理函数。如果是，则不拷贝代码，
            而是由沙箱发起 RPC 调用宿主执行。
    """

    def decorator(func: Callable):
        setattr(func, "__sandbox_function__", True)
        setattr(func, "__sandbox_python_packages__", python_packages or [])
        setattr(func, "__sandbox_global_imports__", global_imports or [])
        setattr(func, "__sandbox_host_callable__", host_callable)
        return func

    return decorator


def _import_to_str(im: ImportType) -> str:
    """内部辅助方法：将 ImportRequirement 转为合法 Python import 字符串"""
    if isinstance(im, str):
        if im.startswith("import ") or im.startswith("from "):
            return im
        return f"import {im}"
    elif isinstance(im, Alias):
        return f"import {im.name} as {im.alias}"
    elif isinstance(im, ImportFromModule):
        parts = []
        for i in im.imports:
            if isinstance(i, str):
                parts.append(i)
            else:
                parts.append(f"{i.name} as {i.alias}")
        return f"from {im.module} import {', '.join(parts)}"
    return ""


def to_stub(func: Callable) -> str:
    """
    抽取函数签名与注释，生成给大模型阅读的存根 (Stub)。
    大模型据此可知沙箱内有哪些 API 可用。
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func)
    stub = f"def {func.__name__}{sig}:\n"
    if doc:
        doc_str = indent(f'"""\n{doc}\n"""', "    ")
        stub += doc_str + "\n"
    stub += "    ...\n"
    return stub


def build_python_functions_file(funcs: list[Callable]) -> str:
    """
    将多个宿主函数融合成一段合法的、即将投递给沙箱物理执行的 Python 脚本。
    """
    global_imports_map = {}
    code_blocks = []

    for func in funcs:
        if not getattr(func, "__sandbox_function__", False):
            continue

        imports = getattr(func, "__sandbox_global_imports__", [])
        for im in imports:
            im_str = _import_to_str(im)
            if im_str:
                global_imports_map[im_str] = None

        is_host_callable = getattr(func, "__sandbox_host_callable__", False)

        if is_host_callable:
            sig = inspect.signature(func)
            params = []
            for name, p in sig.parameters.items():
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    params.append(f"{name}={name}")
            kwargs_str = ", ".join(params)

            stub_code = f"def {func.__name__}{sig}:\n"
            if func.__doc__:
                doc_str = textwrap.indent(f'"""\n{func.__doc__}\n"""', "    ")
                stub_code += doc_str + "\n"

            stub_code += "    from zhenxun_stub import invoke_host\n"
            stub_code += (
                f"    return invoke_host('{func.__name__}', {kwargs_str})\n"
            )
            code_blocks.append(stub_code)
        else:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            lines = source.split("\n")
            while lines:
                if (
                    lines[0].strip().startswith("def ")
                    or lines[0].strip().startswith("async def ")
                ):
                    break
                lines.pop(0)
            code_blocks.append("\n".join(lines))

    content = ""
    if global_imports_map:
        content += "\n".join(global_imports_map.keys()) + "\n\n"

    content += "\n\n".join(code_blocks) + "\n"
    return content
