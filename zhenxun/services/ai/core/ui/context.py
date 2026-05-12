import time

from .manager import UIStreamerRegistry, active_streamers


class UIStreamerContext:
    """动态 UI 状态流收集上下文管理器"""

    def __init__(self, session_id: str, streamer_type: str = "markdown"):
        self.session_id = session_id
        self.streamer_type = streamer_type
        self.start_time = 0.0
        self.streamer_instance = None

    async def __aenter__(self):
        self.start_time = time.monotonic()
        streamer_cls = UIStreamerRegistry.get(self.streamer_type)
        self.streamer_instance = streamer_cls(self.session_id)
        active_streamers[self.session_id] = self.streamer_instance
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        active_streamers.pop(self.session_id, None)

    def render(self) -> str:
        if not self.streamer_instance:
            return ""
        duration = time.monotonic() - self.start_time
        return self.streamer_instance.render(duration)
