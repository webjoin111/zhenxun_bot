import asyncio
from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.engine.token_estimator import global_estimator
from zhenxun.services.ai.llm.capabilities import get_model_capabilities
from zhenxun.services.ai.llm.manager import get_global_default_model_name
from zhenxun.services.ai.memory.working_memory import (
    CondenserPipeline,
    CondenserRegistry,
    _get_default_memory,
)
from zhenxun.services.ai.protocols.memory import BaseWorkingMemory, SessionMetadata
from zhenxun.services.log import logger

class AsyncMemoryCondenser:
    """
    负责在后台异步压缩记忆，提供并发锁保证无损合并。
    专注于底层记忆 management 优化。
    """
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self.memory: BaseWorkingMemory = _get_default_memory()
        self._compressing_tasks: dict[str, asyncio.Task] = {}

    def _get_lock(self, scope: str) -> asyncio.Lock:
        if scope not in self._locks:
            self._locks[scope] = asyncio.Lock()
        return self._locks[scope]

    def trigger_compression(self, scope: str):
        config = get_llm_config().context_settings
        if not config.enabled:
            return

        if scope not in self._compressing_tasks:
            task = asyncio.create_task(self._background_condense(scope))
            self._compressing_tasks[scope] = task

            def _on_done(t):
                self._compressing_tasks.pop(scope, None)

            task.add_done_callback(_on_done)

    async def _background_condense(self, scope: str):
        session = SessionMetadata(session_id=scope)
        async with self._get_lock(scope):
            current_history = await self.memory.get_history(session)
            original_len = len(current_history)

        model_name = get_global_default_model_name() or "Gemini/gemini-2.0-flash"
        caps = get_model_capabilities(model_name)
        config = get_llm_config().context_settings
        threshold_tokens = int(caps.max_input_tokens * config.trigger_threshold)

        current_tokens = global_estimator.estimate_context(current_history, model_name)
        if current_tokens <= threshold_tokens:
            return

        logger.info(
            f"[AsyncMemoryCondenser] Scope {scope} 触发后台上下文压缩 ({current_tokens}/{threshold_tokens})"
        )

        reducers = [CondenserRegistry.get("tool_compactor")]
        if config.enable_summarization:
            reducers.append(CondenserRegistry.get("llm_summarizer"))
        reducers.append(CondenserRegistry.get("message_dropper"))

        pipeline = CondenserPipeline(reducers)

        try:
            compressed_history = await pipeline.run(
                current_history, target_tokens=threshold_tokens, model_name=model_name
            )
        except Exception as e:
            logger.error(f"[AsyncMemoryCondenser] 后台压缩失败: {e}", e=e)
            return

        async with self._get_lock(scope):
            latest_history = await self.memory.get_history(session)
            new_msgs = latest_history[original_len:]
            final_history = compressed_history + new_msgs
            await self.memory.set_history(session, final_history)
            logger.info(
                f"[AsyncMemoryCondenser] Scope {scope} 后台压缩完成，已无损合并 {len(new_msgs)} 条并发新消息。"
            )

async_memory_condenser = AsyncMemoryCondenser()
