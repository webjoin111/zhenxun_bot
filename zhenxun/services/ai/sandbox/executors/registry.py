from typing import ClassVar

from zhenxun.services.log import logger
from zhenxun.utils.utils import infer_plugin_namespace

from .base import BaseCodeExecutor


class CodeExecutorRegistry:
    """多语言代码执行器动态注册中心"""

    _executors: ClassVar[dict[str, dict[str, dict[bool, type[BaseCodeExecutor]]]]] = {}

    _aliases: ClassVar[dict[str, str]] = {
        "py": "python",
        "js": "javascript",
        "sh": "bash",
        "shell": "bash",
        "ts": "typescript",
    }

    @classmethod
    def _normalize_lang(cls, language: str) -> str:
        lang_lower = language.lower().strip()
        return cls._aliases.get(lang_lower, lang_lower)

    @classmethod
    def register(
        cls,
        language: str,
        executor_cls: type[BaseCodeExecutor],
        is_stateful: bool = False,
        scope: str | None = None,
    ) -> None:
        ns = scope if scope is not None else infer_plugin_namespace()
        lang_norm = cls._normalize_lang(language)

        if ns not in cls._executors:
            cls._executors[ns] = {}
        if lang_norm not in cls._executors[ns]:
            cls._executors[ns][lang_norm] = {}

        cls._executors[ns][lang_norm][is_stateful] = executor_cls
        logger.debug(
            f"[CodeExecutorRegistry] 成功注册执行器: {ns} -> {lang_norm} (Stateful: {is_stateful}) -> {executor_cls.__name__}"
        )

    @classmethod
    def get_executor_cls(
        cls, language: str, needs_state: bool, namespace: str = "global"
    ) -> type[BaseCodeExecutor]:
        lang_norm = cls._normalize_lang(language)

        for target_ns in [namespace, "global"]:
            if target_ns in cls._executors and lang_norm in cls._executors[target_ns]:
                lang_executors = cls._executors[target_ns][lang_norm]
                if needs_state and True in lang_executors:
                    return lang_executors[True]
                if False in lang_executors:
                    return lang_executors[False]

        # 3. 兜底回退：如果专属和全局都没找到，遍历全系统所有命名空间寻找该语言支持
        for ns_dict in cls._executors.values():
            if lang_norm in ns_dict:
                lang_executors = ns_dict[lang_norm]
                if needs_state and True in lang_executors:
                    return lang_executors[True]
                if False in lang_executors:
                    return lang_executors[False]

        raise ValueError(
            f"当前沙箱生态未提供针对语言 '{language}' 的代码执行器。支持的语言有: {cls.get_supported_languages()}"
        )

    @classmethod
    def get_supported_languages(cls, namespace: str = "global") -> list[str]:
        langs = set(cls._executors.get("global", {}).keys())
        if namespace in cls._executors:
            langs.update(cls._executors[namespace].keys())
        return list(langs)
