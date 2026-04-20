from zhenxun.services.ai.protocols.hooks import AfterLLMCallHook, BeforeLLMCallHook

_GLOBAL_BEFORE_HOOKS: list[BeforeLLMCallHook] = []
_GLOBAL_AFTER_HOOKS: list[AfterLLMCallHook] = []


def register_before_llm_hook(hook: BeforeLLMCallHook) -> None:
    """注册全局 LLM 调用前拦截器 (可修改 Prompt)"""
    _GLOBAL_BEFORE_HOOKS.append(hook)


def register_after_llm_hook(hook: AfterLLMCallHook) -> None:
    """注册全局 LLM 调用后拦截器 (可修改返回文本)"""
    _GLOBAL_AFTER_HOOKS.append(hook)
