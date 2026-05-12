import re
import time

import numpy as np

from zhenxun.services.ai.memory.models import MemoryConfig, MemoryRecord


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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """使用 numpy 计算两组向量的余弦相似度"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_composite_score(
    record: MemoryRecord, semantic_score: float, config: MemoryConfig
) -> tuple[float, list[str]]:
    """计算复合相关性得分：结合语义相似度、时间衰减与重要性权重"""
    age_seconds = time.time() - record.created_at
    age_days = max(age_seconds / 86400.0, 0.0)
    decay = 0.5 ** (age_days / config.recency_half_life_days)

    composite = (
        config.semantic_weight * semantic_score
        + config.recency_weight * decay
        + config.importance_weight * record.importance
    )

    reasons: list[str] = ["semantic"]
    if decay > 0.5:
        reasons.append("recency")
    if record.importance > 0.5:
        reasons.append("importance")

    return composite, reasons

