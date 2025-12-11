from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from typing import Any


@dataclass
class LimitResult:
    """
    限制器计算结果数据类。

    属性:
        passed: 是否通过限制。
        new_state: 更新后的状态数据 (用于写回存储)。
        retry_after: 若未通过，需等待的秒数。
        expire_at: 状态的绝对过期时间戳。
    """

    passed: bool
    """是否通过限制"""
    new_state: dict[str, Any]
    """更新后的状态数据 (用于写回存储)"""
    retry_after: float = 0.0
    """若未通过，需等待的秒数"""
    expire_at: float = 0.0
    """状态的绝对过期时间戳"""


class BaseLimiter(ABC):
    """
    限制器抽象基类 (无状态)。
    """

    @abstractmethod
    def check(self, state: dict[str, Any] | None, **kwargs) -> LimitResult:
        """
        检查限制状态。

        参数:
            state: 当前存储的状态数据。
            kwargs: 动态参数。

        返回:
            LimitResult: 限制检查结果。
        """
        ...


class FreqLimiter(BaseLimiter):
    """
    频率限制器 (固定窗口冷却)。
    """

    def __init__(self, default_cd_seconds: int):
        """
        初始化频率限制器。

        参数:
            default_cd_seconds: 默认冷却时间 (秒)。
        """
        self.default_cd = default_cd_seconds
        self._legacy_cache: dict[Any, float] = {}

    def check(
        self,
        state: dict[str, Any] | Any | None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        检查是否处于冷却中。

        支持新版状态字典检查和旧版基于key的检查。

        参数:
            state: 状态字典 {'last_trigger': float} 或旧版 key。
            duration: 本次冷却时间，覆盖默认值。
            kwargs: 额外参数。

        返回:
            Any: LimitResult (新版) 或 bool (旧版)。
        """
        now = time.time()
        cd = duration if duration is not None else self.default_cd

        if not isinstance(state, dict) and state is not None:
            key = state
            last_trigger = self._legacy_cache.get(key, 0.0)
            if now < last_trigger + cd:
                return False
            return True

        state = state or {}
        last_trigger = state.get("last_trigger", 0.0)

        if now < last_trigger + cd:
            retry_after = (last_trigger + cd) - now
            return LimitResult(
                passed=False,
                new_state=state,
                retry_after=retry_after,
                expire_at=last_trigger + cd,
            )

        new_state = {"last_trigger": now}
        return LimitResult(
            passed=True,
            new_state=new_state,
            retry_after=0.0,
            expire_at=now + cd,
        )

    def start_cd(self, key: Any, duration: float | None = None):
        """
        [兼容旧版] 手动开始冷却。

        参数:
            key: 标识键。
            duration: 冷却时长 (可选，兼容参数)。
        """
        self._legacy_cache[key] = time.time()

    def left_time(self, key: Any) -> float:
        """
        [兼容旧版] 获取剩余冷却时间。

        参数:
            key: 标识键。

        返回:
            float: 剩余秒数。
        """
        last_trigger = self._legacy_cache.get(key, 0.0)
        return max(0.0, (last_trigger + self.default_cd) - time.time())


class CountLimiter(BaseLimiter):
    """
    次数限制器 (每日重置)。
    """

    def __init__(self, max_count: int):
        """
        初始化次数限制器。

        参数:
            max_count: 每日最大允许次数。
        """
        self.max_count = max_count
        self._legacy_cache: dict[Any, dict] = {}

    def check(
        self,
        state: dict[str, Any] | Any | None,
        increase: int = 1,
        **kwargs: Any,
    ) -> Any:
        """
        检查是否超过次数限制。

        支持新版状态字典检查和旧版基于key的检查。

        参数:
            state: 状态字典 {'count': int, 'date': str} 或旧版 key。
            increase: 本次调用增加的次数。
            kwargs: 额外参数。

        返回:
            Any: LimitResult (新版) 或 bool (旧版)。
        """
        if not isinstance(state, dict) and state is not None:
            key = state
            return self._legacy_check(key, increase)

        if not isinstance(state, dict):
            state = {}
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        tomorrow = now + timedelta(days=1)
        end_of_day = datetime(
            year=tomorrow.year, month=tomorrow.month, day=tomorrow.day
        )
        expire_at = end_of_day.timestamp()

        record_date = state.get("date", "")
        current_count = state.get("count", 0)

        if record_date != today_str:
            current_count = 0

        if current_count + increase > self.max_count:
            return LimitResult(
                passed=False,
                new_state={"count": current_count, "date": today_str},
                retry_after=max(0.0, expire_at - now.timestamp()),
                expire_at=expire_at,
            )

        new_state = {"count": current_count + increase, "date": today_str}
        return LimitResult(
            passed=True,
            new_state=new_state,
            retry_after=0.0,
            expire_at=expire_at,
        )

    def _get_legacy_data(self, key: Any) -> dict:
        """[内部] 获取旧版兼容数据"""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        data = self._legacy_cache.get(key, {"count": 0, "date": today_str})
        if data["date"] != today_str:
            data = {"count": 0, "date": today_str}
            self._legacy_cache[key] = data
        return data

    def _legacy_check(self, key: Any, increase: int = 1) -> bool:
        """[内部] 旧版兼容检查逻辑"""
        data = self._get_legacy_data(key)
        return data["count"] + increase <= self.max_count

    def get_num(self, key: Any) -> int:
        """
        [兼容旧版] 获取当前已使用次数。

        参数:
            key: 标识键。

        返回:
            int: 已使用次数。
        """
        data = self._get_legacy_data(key)
        return data["count"]

    def increase(self, key: Any, num: int = 1):
        """
        [兼容旧版] 增加使用次数。

        参数:
            key: 标识键。
            num: 增加的数量。
        """
        data = self._get_legacy_data(key)
        data["count"] += num
        self._legacy_cache[key] = data


class UserBlockLimiter(BaseLimiter):
    """
    用户阻塞限制器 (检测用户是否正在调用命令)。
    采用阻塞锁 + 超时机制。
    """

    def __init__(self, default_timeout: int = 30):
        """
        初始化用户阻塞限制器。

        参数:
            default_timeout: 默认阻塞超时时间 (秒)。
        """
        self.default_timeout = default_timeout
        self._legacy_cache: dict[Any, float] = {}

    def check(self, state: dict[str, Any] | Any | None, **kwargs: Any) -> Any:
        """
        检查用户是否被阻塞。

        支持新版状态字典检查和旧版基于key的检查。

        参数:
            state: 状态字典 {'blocked_until': float} 或旧版 key。
            kwargs: 额外参数。

        返回:
            Any: LimitResult (新版) 或 bool (旧版)。
        """
        if not isinstance(state, dict) and state is not None:
            key = state
            return self._legacy_check(key)

        if not isinstance(state, dict):
            state = {}
        now = time.time()
        blocked_until = state.get("blocked_until", 0.0)

        if now < blocked_until:
            return LimitResult(
                passed=False,
                new_state=state,
                retry_after=blocked_until - now,
                expire_at=blocked_until,
            )

        return LimitResult(passed=True, new_state={}, retry_after=0.0, expire_at=0.0)

    def set_block(self, duration: int | None = None) -> LimitResult:
        """
        设置用户阻塞状态。

        参数:
            duration: 阻塞时长 (秒)，默认为初始化时的 default_timeout。

        返回:
            LimitResult: 包含新状态的结果对象。
        """
        now = time.time()
        timeout = duration if duration is not None else self.default_timeout
        expire_at = now + timeout
        return LimitResult(
            passed=True,
            new_state={"blocked_until": expire_at},
            retry_after=0.0,
            expire_at=expire_at,
        )

    def set_true(self, key: Any):
        """
        [兼容旧版] 设置阻塞。
        """
        self._legacy_cache[key] = time.time() + self.default_timeout

    def set_false(self, key: Any):
        """
        [兼容旧版] 解除阻塞。
        """
        if key in self._legacy_cache:
            del self._legacy_cache[key]

    def _legacy_check(self, key: Any) -> bool:
        if key not in self._legacy_cache:
            return True
        if time.time() > self._legacy_cache[key]:
            del self._legacy_cache[key]
            return True
        return False


class RateLimiter(BaseLimiter):
    """
    速率限制器 (基于滑动窗口)。
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        初始化速率限制器。

        参数:
            max_calls: 时间窗口内允许的最大调用次数。
            time_window: 时间窗口大小 (秒)。
        """
        self.max_calls = max_calls
        self.time_window = time_window

    def check(self, state: dict[str, Any] | None, **kwargs: Any) -> LimitResult:
        """
        检查是否超过速率限制。

        参数:
            state: 状态字典 {'history': [timestamp1, timestamp2, ...]}。
            kwargs: 额外参数。

        返回:
            LimitResult: 检查结果。
        """
        if not isinstance(state, dict):
            state = {}
        now = time.time()
        history = state.get("history", [])
        valid_since = now - self.time_window
        new_history = [ts for ts in history if ts > valid_since]

        if len(new_history) >= self.max_calls:
            earliest_ts = new_history[0] if new_history else now
            retry_after = (earliest_ts + self.time_window) - now
            expire_at = now + self.time_window
            return LimitResult(
                passed=False,
                new_state={"history": new_history},
                retry_after=max(0.0, retry_after),
                expire_at=expire_at,
            )

        new_history.append(now)
        expire_at = now + self.time_window
        return LimitResult(
            passed=True,
            new_state={"history": new_history},
            retry_after=0.0,
            expire_at=expire_at,
        )


class ConcurrencyLimiter:
    """
    并发限制器 (基于 asyncio.Semaphore)。
    """

    def __init__(self, max_concurrent: int):
        """
        初始化并发限制器。

        参数:
            max_concurrent: 最大并发数。
        """
        self._semaphores: dict[Any, asyncio.Semaphore] = {}
        self.max_concurrent = max_concurrent
        self._active_tasks: dict[Any, int] = defaultdict(int)

    def _get_semaphore(self, key: Any) -> asyncio.Semaphore:
        """获取或创建信号量"""
        if key not in self._semaphores:
            self._semaphores[key] = asyncio.Semaphore(self.max_concurrent)
        return self._semaphores[key]

    async def acquire(self, key: Any):
        """
        获取一个信号量。

        如果达到并发上限，则会阻塞等待。

        参数:
            key: 标识键。
        """
        semaphore = self._get_semaphore(key)
        await semaphore.acquire()
        self._active_tasks[key] += 1

    def release(self, key: Any):
        """
        释放一个信号量。

        参数:
            key: 标识键。
        """
        if key in self._semaphores:
            if self._active_tasks[key] > 0:
                self._semaphores[key].release()
                self._active_tasks[key] -= 1
            else:
                import logging

                logging.warning(f"尝试释放键 '{key}' 的信号量时，计数已经为零。")
