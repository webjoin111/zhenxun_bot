"""
LLM 核心基础设施模块

包含执行 LLM 请求所需的底层组件，如 HTTP 客户端、API Key 存储和智能重试逻辑。
"""

import asyncio
from enum import Enum
import json
import os
import time
from typing import Any

import aiofiles
import httpx
import nonebot
from pydantic import BaseModel, Field

from zhenxun.configs.config import BotConfig
from zhenxun.configs.path_config import DATA_PATH
from zhenxun.services.ai.config import ProviderConfig
from zhenxun.services.ai.core.exceptions import LLMErrorCode, LLMException
from zhenxun.services.log import logger
from zhenxun.utils.user_agent import get_user_agent

driver = nonebot.get_driver()


class HttpClientConfig(BaseModel):
    """HTTP客户端配置"""

    timeout: int = 180
    max_connections: int = 100
    max_keepalive_connections: int = 20


class LLMHttpClient:
    """[内部 API] LLM 服务专用异步 HTTP 客户端封装。"""

    def __init__(self, config: HttpClientConfig | None = None):
        self.config = config or HttpClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._active_requests = 0
        self._lock = asyncio.Lock()

    async def _ensure_client_initialized(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._lock:
                if self._client is None or self._client.is_closed:
                    logger.debug(
                        f"LLMHttpClient: 正在初始化新的 httpx.AsyncClient "
                        f"配置: {self.config}"
                    )
                    headers = get_user_agent()
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                    )
                    timeout = httpx.Timeout(self.config.timeout)

                    client_kwargs = {}
                    if BotConfig.system_proxy:
                        try:
                            version_parts = httpx.__version__.split(".")
                            major = int(
                                "".join(c for c in version_parts[0] if c.isdigit())
                            )
                            minor = (
                                int("".join(c for c in version_parts[1] if c.isdigit()))
                                if len(version_parts) > 1
                                else 0
                            )
                            if (major, minor) >= (0, 28):
                                client_kwargs["proxy"] = BotConfig.system_proxy
                            else:
                                client_kwargs["proxies"] = BotConfig.system_proxy
                        except (ValueError, IndexError):
                            client_kwargs["proxies"] = BotConfig.system_proxy
                            logger.warning(
                                f"无法解析 httpx 版本 '{httpx.__version__}'，"
                                "LLM模块将默认使用旧版 'proxies' 参数语法。"
                            )

                    self._client = httpx.AsyncClient(
                        headers=headers,
                        limits=limits,
                        timeout=timeout,
                        follow_redirects=True,
                        **client_kwargs,
                    )
        if self._client is None:
            raise LLMException(
                "HTTP 客户端初始化失败。", LLMErrorCode.CONFIGURATION_ERROR
            )
        return self._client

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        client = await self._ensure_client_initialized()
        async with self._lock:
            self._active_requests += 1
        try:
            return await client.request(method, url, **kwargs)
        finally:
            async with self._lock:
                self._active_requests -= 1

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def close(self):
        async with self._lock:
            if self._client and not self._client.is_closed:
                logger.debug(
                    f"LLMHttpClient: 正在关闭，配置: {self.config}. "
                    f"活跃请求数: {self._active_requests}"
                )
                if self._active_requests > 0:
                    logger.warning(
                        f"LLMHttpClient: 关闭时仍有 {self._active_requests} "
                        f"个请求处于活跃状态。"
                    )
                await self._client.aclose()
            self._client = None
        logger.debug(f"配置为 {self.config} 的 LLMHttpClient 已完全关闭。")

    @property
    def is_closed(self) -> bool:
        return self._client is None or self._client.is_closed


class LLMHttpClientManager:
    """[内部 API] 负责管理与复用 LLMHttpClient 连接池。"""

    def __init__(self):
        self._clients: dict[tuple[int], LLMHttpClient] = {}
        self._lock = asyncio.Lock()

    def _get_client_key(self, provider_config: ProviderConfig) -> tuple[int]:
        return (provider_config.timeout,)

    async def get_client(self, provider_config: ProviderConfig) -> LLMHttpClient:
        key = self._get_client_key(provider_config)
        async with self._lock:
            client = self._clients.get(key)
            if client and not client.is_closed:
                logger.debug(
                    f"LLMHttpClientManager: 复用现有的 LLMHttpClient 密钥: {key}"
                )
                return client

            if client and client.is_closed:
                logger.debug(
                    f"LLMHttpClientManager: 发现密钥 {key} 对应的客户端已关闭。"
                    f"正在创建新的客户端。"
                )

            logger.debug(f"LLMHttpClientManager: 为密钥 {key} 创建新的 LLMHttpClient")
            http_client_config = HttpClientConfig(timeout=provider_config.timeout)
            new_client = LLMHttpClient(config=http_client_config)
            self._clients[key] = new_client
            return new_client

    async def shutdown(self):
        async with self._lock:
            logger.info(
                f"LLMHttpClientManager: 正在关闭。关闭 {len(self._clients)} 个客户端。"
            )
            close_tasks = [
                client.close()
                for client in self._clients.values()
                if client and not client.is_closed
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            self._clients.clear()
        logger.info("LLMHttpClientManager: 关闭完成。")


http_client_manager = LLMHttpClientManager()


async def create_llm_http_client(
    timeout: int = 180,
) -> LLMHttpClient:
    """创建LLM HTTP客户端"""
    config = HttpClientConfig(timeout=timeout)
    return LLMHttpClient(config)


class RouteHealthState(str, Enum):
    """路由端点健康状态枚举 (L7)"""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class KeyHealthStatus(BaseModel):
    """单个 API Key 的详细状态信息 (L4)"""

    status: str = "HEALTHY"
    successes: int = 0
    failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None


class RouteHealthStatus(BaseModel):
    """单个模型路由端点的遥测状态信息 (L7)"""

    state: RouteHealthState = RouteHealthState.CLOSED
    cooldown_until: float = 0.0
    success_rate: float = 100.0
    successes: int = 0
    failures: int = 0
    latency_ema: float = 0.0
    last_error: str | None = None


class ProviderHealthStatus(BaseModel):
    """单个提供商的健康状态 (L4)"""

    api_keys: dict[str, KeyHealthStatus] = Field(default_factory=dict)


class GlobalHealthState(BaseModel):
    """全局 AI 遥测与健康状态基座"""

    providers: dict[str, ProviderHealthStatus] = Field(default_factory=dict)
    routes: dict[str, RouteHealthStatus] = Field(default_factory=dict)


class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
        key_rotation: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.key_rotation = key_rotation


def _should_retry_llm_error(
    error: LLMException, attempt: int, max_retries: int
) -> bool:
    """判断LLM错误是否应该重试"""
    non_retryable_errors = {
        LLMErrorCode.MODEL_NOT_FOUND,
        LLMErrorCode.CONTEXT_LENGTH_EXCEEDED,
        LLMErrorCode.USER_LOCATION_NOT_SUPPORTED,
        LLMErrorCode.INVALID_PARAMETER,
        LLMErrorCode.CONFIGURATION_ERROR,
        LLMErrorCode.API_KEY_INVALID,
    }

    if error.code in non_retryable_errors:
        return False

    retryable_errors = {
        LLMErrorCode.API_REQUEST_FAILED,
        LLMErrorCode.API_TIMEOUT,
        LLMErrorCode.API_RATE_LIMITED,
        LLMErrorCode.API_RESPONSE_INVALID,
        LLMErrorCode.RESPONSE_PARSE_ERROR,
        LLMErrorCode.GENERATION_FAILED,
        LLMErrorCode.CONTENT_FILTERED,
        LLMErrorCode.API_QUOTA_EXCEEDED,
    }

    if error.code in retryable_errors:
        if error.code == LLMErrorCode.API_QUOTA_EXCEEDED:
            return attempt < min(2, max_retries)
        return True

    return False


class HealthManager:
    """全局 AI 健康与遥测管理器 (支持 L4 & L7 熔断)"""

    def __init__(self):
        self.state = GlobalHealthState()
        self._provider_key_index: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._file_path = DATA_PATH / "ai" / "api_key.json"

    async def initialize(self):
        """从文件异步加载遥测状态"""
        async with self._lock:
            if not self._file_path.exists():
                logger.info("未找到遥测状态文件，将使用内存状态启动。")
                return

            try:
                logger.info(f"正在从 {self._file_path} 加载密钥状态...")
                async with aiofiles.open(self._file_path, encoding="utf-8") as f:
                    content = await f.read()
                    if not content:
                        return
                    from zhenxun.utils.pydantic_compat import parse_as

                    self.state = parse_as(GlobalHealthState, json.loads(content))
                total_keys = sum(
                    len(provider.api_keys) for provider in self.state.providers.values()
                )
                logger.info(f"成功加载 {total_keys} 个密钥的状态。")

            except json.JSONDecodeError:
                logger.error(f"遥测状态文件 {self._file_path} 格式错误，无法解析。")
            except Exception as e:
                logger.error(f"加载遥测状态文件时发生错误: {e}", e=e)

    async def _save_to_file_internal(self):
        """
        [内部方法] 将当前密钥状态安全地写入JSON文件。
        假定调用方已持有锁。
        """
        from zhenxun.utils.pydantic_compat import model_dump

        data_to_save = model_dump(self.state)

        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._file_path.with_suffix(".json.tmp")

            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data_to_save, ensure_ascii=False, indent=2))

            if self._file_path.exists():
                self._file_path.unlink()
            os.rename(temp_path, self._file_path)
        except Exception as e:
            logger.error(f"保存密钥状态到文件失败: {e}", e=e)

    async def shutdown(self):
        """在应用关闭时安全地保存状态"""
        async with self._lock:
            await self._save_to_file_internal()
        logger.info("HealthManager 已在关闭前保存遥测状态。")

    async def get_next_available_key(
        self,
        provider_name: str,
        api_keys: list[str],
        exclude_keys: set[str] | None = None,
        strict_mode: bool = False,
    ) -> str | None:
        """
        获取下一个可用的API密钥（轮询策略）
        """
        if not api_keys:
            return None

        exclude_keys = exclude_keys or set()

        async with self._lock:
            provider_state = self.state.providers.setdefault(
                provider_name, ProviderHealthStatus()
            )
            for key in api_keys:
                if key not in provider_state.api_keys:
                    provider_state.api_keys[key] = KeyHealthStatus()

            now = time.time()
            available_keys = [
                key
                for key in api_keys
                if key not in exclude_keys
                and provider_state.api_keys[key].cooldown_until <= now
            ]

            if not available_keys:
                if strict_mode:
                    return None
                return api_keys[0]

            current_index = self._provider_key_index.get(provider_name, 0)
            selected_key = available_keys[current_index % len(available_keys)]
            self._provider_key_index[provider_name] = current_index + 1

            stats = provider_state.api_keys[selected_key]
            total_usage = stats.successes + stats.failures
            logger.debug(
                f"轮询选择API密钥: {self._get_key_id(selected_key)} "
                f"(使用次数: {total_usage})"
            )
            return selected_key

    def is_route_healthy(self, route_name: str) -> bool:
        """
        [L7 熔断查询] 检查指定模型路由当前是否健康。
        负责触发 OPEN -> HALF_OPEN 的状态流转。
        """
        stats = self.state.routes.get(route_name)
        if not stats:
            return True

        if stats.state == RouteHealthState.CLOSED:
            return True

        if stats.state == RouteHealthState.OPEN:
            if time.time() >= stats.cooldown_until:
                stats.state = RouteHealthState.HALF_OPEN
                logger.info(
                    f"🔄 [L7 Router] 节点 '{route_name}' "
                    "冷却期结束，进入 HALF_OPEN 半开试探状态。"
                )
                return True
            return False

        if stats.state == RouteHealthState.HALF_OPEN:
            return False

        return True

    async def record_route_success(self, route_name: str, latency: float):
        """记录路由成功，恢复健康状态并更新 EMA 延迟"""
        async with self._lock:
            stats = self.state.routes.setdefault(route_name, RouteHealthStatus())
            stats.successes += 1
            total = stats.successes + stats.failures
            stats.success_rate = (stats.successes / total) * 100

            if stats.latency_ema == 0.0:
                stats.latency_ema = latency
            else:
                stats.latency_ema = 0.2 * latency + 0.8 * stats.latency_ema

            if stats.state != RouteHealthState.CLOSED:
                logger.info(
                    f"✅ [L7 Router] 节点 '{route_name}' "
                    "试探成功！已完全恢复健康状态 (CLOSED)。"
                )
                stats.state = RouteHealthState.CLOSED
                stats.cooldown_until = 0.0

            stats.last_error = None
            await self._save_to_file_internal()

    async def record_route_failure(self, route_name: str, error_msg: str):
        """记录服务端致命错误，熔断该模型节点"""
        now = time.time()
        cooldown_duration = 180

        async with self._lock:
            stats = self.state.routes.setdefault(route_name, RouteHealthStatus())
            stats.failures += 1
            total = stats.successes + stats.failures
            stats.success_rate = (stats.successes / total) * 100

            stats.state = RouteHealthState.OPEN
            stats.cooldown_until = now + cooldown_duration
            stats.last_error = error_msg[:256]
            await self._save_to_file_internal()

        logger.warning(
            f"🚨 [L7 Router] 节点 '{route_name}' 发生服务端故障，"
            f"已触发熔断 (OPEN)，冷却 {cooldown_duration} 秒。错误: {error_msg}"
        )

    def get_best_fallback_route(self, route_names: list[str]) -> str:
        """
        [全死保底机制] 当组内所有节点全部宕机时，挑出一个最有可能恢复的节点强制探活。
        策略：优先选择剩余冷却时间最短的节点。
        """

        def get_cooldown(name: str) -> float:
            stats = self.state.routes.get(name)
            return stats.cooldown_until if stats else 0.0

        return sorted(route_names, key=get_cooldown)[0]

    async def record_key_success(self, provider_name: str, api_key: str):
        """记录 L4 Key 级成功使用，解除冷却"""
        async with self._lock:
            provider_state = self.state.providers.setdefault(
                provider_name, ProviderHealthStatus()
            )
            stats = provider_state.api_keys.setdefault(api_key, KeyHealthStatus())
            stats.cooldown_until = 0.0
            stats.successes += 1
            stats.status = "HEALTHY"
            stats.last_error = None
            await self._save_to_file_internal()

    async def record_key_failure(
        self,
        provider_name: str,
        api_key: str,
        status_code: int | None,
        error_message: str,
    ):
        """
        记录 L4 Key 级失败。仅对强相关的限流、鉴权错误进行冷却。
        """
        key_id = self._get_key_id(api_key)
        now = time.time()
        cooldown_duration = 0

        location_not_supported = error_message and (
            "USER_LOCATION_NOT_SUPPORTED" in error_message
            or "User location is not supported" in error_message
        )
        if location_not_supported:
            logger.warning(
                f"API Key {key_id} 请求失败，原因是地区不支持 (Gemini)。"
                " 这通常是代理节点问题，Key 本身可能是正常的。跳过冷却。"
            )
            async with self._lock:
                provider_state = self.state.providers.setdefault(
                    provider_name, ProviderHealthStatus()
                )
                stats = provider_state.api_keys.setdefault(api_key, KeyHealthStatus())
                stats.failures += 1
                stats.last_error = error_message[:256]
                await self._save_to_file_internal()
            return

        if error_message and (
            "API_QUOTA_EXCEEDED" in error_message
            or "insufficient_quota" in error_message.lower()
        ):
            cooldown_duration = 3600
            logger.warning(f"API Key {key_id} 额度耗尽，冷却 1 小时。")

        is_key_invalid = status_code == 401 or (
            status_code == 400
            and error_message
            and (
                "API_KEY_INVALID" in error_message
                or "API key not valid" in error_message
            )
        )

        if is_key_invalid:
            cooldown_duration = 31536000
            log_level = "error"
            log_message = f"API密钥认证/权限/路径错误，将永久禁用: {key_id}"
        elif status_code == 403:
            cooldown_duration = 3600
            log_level = "warning"
            log_message = f"API密钥权限不足或地区不支持(403)，冷却1小时: {key_id}"
        elif status_code == 429:
            cooldown_duration = 60
            log_level = "warning"
            log_message = f"API密钥被限流，冷却60秒: {key_id}"
        else:
            log_level = "debug"
            log_message = (
                f"API请求发生服务异常，已交由L7路由接管熔断，不冷却密钥本身: {key_id}"
            )

        async with self._lock:
            provider_state = self.state.providers.setdefault(
                provider_name, ProviderHealthStatus()
            )
            stats = provider_state.api_keys.setdefault(api_key, KeyHealthStatus())
            if cooldown_duration > 0:
                stats.cooldown_until = now + cooldown_duration
                stats.status = (
                    "COOLDOWN" if cooldown_duration < 31536000 else "DISABLED"
                )
            stats.failures += 1
            stats.last_error = error_message[:256]
            await self._save_to_file_internal()

        getattr(logger, log_level)(log_message)

    async def reset_key_status(self, provider_name: str, api_key: str):
        """重置密钥状态，并持久化"""
        async with self._lock:
            provider_state = self.state.providers.setdefault(
                provider_name, ProviderHealthStatus()
            )
            stats = provider_state.api_keys.setdefault(api_key, KeyHealthStatus())
            stats.cooldown_until = 0.0
            stats.last_error = None
            stats.status = "HEALTHY"
            await self._save_to_file_internal()
        logger.info(f"重置API密钥状态: {self._get_key_id(api_key)}")

    def _get_key_id(self, api_key: str) -> str:
        """获取API密钥的标识符（用于日志）"""
        if len(api_key) <= 8:
            return api_key
        return f"{api_key[:4]}...{api_key[-4:]}"


health_manager = HealthManager()


@driver.on_shutdown
async def _shutdown_health_manager():
    await health_manager.shutdown()
