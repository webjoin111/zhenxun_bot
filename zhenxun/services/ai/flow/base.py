from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import contextlib
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from zhenxun.services.ai.run.models import StreamedRunResult

T_RunResult = TypeVar("T_RunResult")


class BaseRunnable(ABC, Generic[T_RunResult]):
    """
    所有可执行 AI 编排实体的统一基类 (Composite Pattern)。
    统一了 Agent, Team, Workflow 的核心契约，支持物理上的任意嵌套。
    """

    name: str
    """可执行实体的名称标识"""

    description: str
    """可执行实体的详细描述。用于外部路由(Router)或上层智能体(DelegateTool)决定是否调用它"""

    persona: Any | None = None
    """(可选) 实体的角色设定 (Persona)。包含 role 和 goal，在多智能体路由移交时优先级最高"""

    runtime_config: Any
    """运行时配置，如是否无状态、UI输出模式等"""

    @abstractmethod
    def bind(self, **kwargs: Any) -> Any:
        """DI 注入语法糖：返回 Depends，自动绑定当前上下文"""
        pass

    @abstractmethod
    async def reply(
        self,
        prompt: Any = None,
        reply_to: bool = False,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> T_RunResult:
        """交互执行语法糖，自动渲染流式进度并最终将结果回复给终端用户"""
        pass

    @abstractmethod
    async def run(
        self, prompt: Any = None, *, context: Any = None, **kwargs: Any
    ) -> T_RunResult:
        """阻塞式核心运行入口，等待整个实体执行完毕并返回最终结果"""
        pass

    @abstractmethod
    @contextlib.asynccontextmanager
    async def run_stream(
        self, prompt: Any = None, *, context: Any = None, **kwargs: Any
    ) -> "AsyncIterator[StreamedRunResult[Any]]":
        """流式运行入口，返回上下文管理器，用于消费底层执行流事件 (StreamedRunResult)"""
        yield cast(Any, None)
