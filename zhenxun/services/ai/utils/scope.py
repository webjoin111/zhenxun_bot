import re


def normalize_scope_path(path: str) -> str:
    """标准化作用域路径，消除多余的斜杠并确保以 / 开头"""
    if not path or path == "/":
        return "/"
    path = re.sub(r"/+", "/", path)
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def join_scope_paths(root: str | None, inner: str | None) -> str:
    """拼接根作用域和内部作用域"""
    root = root.rstrip("/") if root else ""
    inner = inner.strip("/") if inner else ""
    if root and inner:
        result = f"{root}/{inner}"
    elif root:
        result = root
    elif inner:
        result = f"/{inner}"
    else:
        result = "/"
    return normalize_scope_path(result)
