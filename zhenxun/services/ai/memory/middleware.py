from collections.abc import Callable

from zhenxun.services.ai.memory.scope import MemoryScope
from zhenxun.services.ai.protocols.memory import (
    BaseWorkingMemory,
    SessionMetadata,
)
from zhenxun.services.ai.protocols.middleware import (
    BaseLLMMiddleware,
    LLMContext,
    NextCall,
)
from zhenxun.services.ai.types.messages import LLMMessage, LLMResponse
from zhenxun.services.log import logger


class MemoryMiddleware(BaseLLMMiddleware):
    """
    记忆中间件：接管大模型调用的上下文加载与保存。
    使得 LLM Client 本身完全无状态。
    """

    def __init__(
        self,
        session_meta: SessionMetadata,
        working_memory: BaseWorkingMemory | None = None,
        long_term_memory: MemoryScope | None = None,
        sanitizer: Callable[[LLMMessage], LLMMessage] | None = None,
    ):
        self.wm = working_memory
        self.ltm = long_term_memory
        self.session_meta = session_meta
        self.session_id = session_meta.session_id
        self.sanitizer = sanitizer

    async def __call__(self, context: LLMContext, next_call: NextCall) -> LLMResponse:
        if self.ltm and context.messages:
            last_content = str(context.messages[-1].content)
            matches = await self.ltm.recall(last_content)
            if matches:
                fact_str = "\n".join(
                    f"- {m.record.content} (相关性: {m.score:.2f})" for m in matches
                )
                sys_msg = LLMMessage.system(
                    f"[系统补充：有关用户的长期记忆设定]\n{fact_str}"
                )
                context.messages.insert(0, sys_msg)
                logger.debug(f"已动态注入 {len(matches)} 条长期记忆。")

        if self.wm:
            history = await self.wm.get_history(self.session_meta)
            context.messages = history + context.messages

        response = await next_call(context)

        if self.wm and context.messages:
            user_msg = context.messages[-1]
            if self.sanitizer:
                user_msg = self.sanitizer(user_msg)

            msgs_to_save = [user_msg]
            if response.content_parts:
                ast_msg = LLMMessage(
                    role="assistant",
                    content=response.content_parts,
                )
                msgs_to_save.append(ast_msg)

            await self.wm.add_messages(self.session_meta, msgs_to_save)

        return response
