from typing import Any

from zhenxun.services.ai.memory.interfaces import BaseChatContext
from zhenxun.services.ai.memory.long_term_memory import MemoryScope
from zhenxun.services.ai.memory.models import MemoryConfig
from zhenxun.services.ai.memory.working_memory import InMemoryChatContext
from zhenxun.services.log import logger


class GlobalMemoryManager:
    """
    全局记忆大管家 (IoC 容器)。
    负责管理短/长期记忆引擎的注册与分配，解耦 Agent 配置与底层数据库实例。
    """

    def __init__(self):
        self._chat_backends: dict[str, BaseChatContext] = {}
        self._ltm_backends: dict[str, MemoryScope] = {}
        
        self._default_chat_backend_name: str = "default"
        self._default_ltm_backend_name: str = "default"
        
        # 默认挂载一个内存级历史管理器兜底
        self.register_chat_backend("default", InMemoryChatContext())

    def register_chat_backend(self, name: str, backend: BaseChatContext) -> None:
        """注册一个短期对话历史存储引擎"""
        self._chat_backends[name] = backend
        logger.debug(f"已注册短期记忆存储后端: '{name}'")

    def register_ltm_backend(self, name: str, backend: MemoryScope) -> None:
        """注册一个长期向量检索记忆引擎"""
        self._ltm_backends[name] = backend
        logger.debug(f"已注册长期记忆检索后端: '{name}'")

    def set_default_chat_backend(self, name: str) -> None:
        """设置全局默认的短期记忆引擎标识符"""
        self._default_chat_backend_name = name

    def set_default_ltm_backend(self, name: str) -> None:
        """设置全局默认的长期记忆引擎标识符"""
        self._default_ltm_backend_name = name

    def get_chat_context(self, config: MemoryConfig | None) -> BaseChatContext | None:
        """根据配置分配对应的短期对话历史实例"""
        if not config or not config.enable_short_term:
            return None
            
        backend_name = config.chat_backend or self._default_chat_backend_name
        backend = self._chat_backends.get(backend_name)
        
        if backend is None:
            logger.warning(f"请求的短期记忆后端 '{backend_name}' 未注册，将回退到默认引擎。")
            return self._chat_backends.get(self._default_chat_backend_name)
        return backend

    def get_long_term_memory(self, config: MemoryConfig | None) -> MemoryScope | None:
        """根据配置分配对应的长期向量记忆实例"""
        if not config or not config.long_term_scope:
            return None
            
        backend_name = config.ltm_backend or self._default_ltm_backend_name
        return self._ltm_backends.get(backend_name)


# 全局单例
memory_manager = GlobalMemoryManager()
