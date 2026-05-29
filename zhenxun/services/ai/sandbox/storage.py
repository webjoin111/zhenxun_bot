import hashlib
from pathlib import PurePath, PurePosixPath
from typing import Final
from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field

_INSTALL_MARKER: Final[str] = "INSTALL_VFS_HELPER_V1"

_RESOLVE_WORKSPACE_PATH_SCRIPT: Final[str] = """
#!/bin/sh
set -eu

root="$1"
candidate="$2"
for_write="$3"
max_symlink_depth=64

case "$for_write" in
    0|1) ;;
    *)
        printf 'for_write must be 0 or 1\\n' >&2
        exit 64
        ;;
esac

resolve_path() {
    path="$1"
    depth="${2:-0}"
    seen="${3:-}"
    if [ "$path" = "/" ]; then
        printf '/\\n'
        return 0
    fi

    if [ "$depth" -ge "$max_symlink_depth" ]; then
        printf 'symlink resolution depth exceeded: %s\\n' "$path" >&2
        exit 112
    fi

    if [ -d "$path" ]; then
        (
            cd "$path"
            pwd -P
        )
        return 0
    fi

    parent=${path%/*}
    base=${path##*/}
    if [ -z "$parent" ] || [ "$parent" = "$path" ]; then
        parent="/"
    fi

    resolved_parent=$(resolve_path "$parent" "$depth" "$seen")
    candidate_path="$resolved_parent/$base"
    
    if [ -L "$candidate_path" ]; then
        case ":$seen:" in
            *":$candidate_path:"*)
                printf 'symlink resolution depth exceeded: %s\\n' "$candidate_path" >&2
                exit 112
                ;;
        esac
        target=$(readlink "$candidate_path")
        next_depth=$((depth + 1))
        next_seen="${seen}:$candidate_path"
        case "$target" in
            /*) resolve_path "$target" "$next_depth" "$next_seen" ;;
            *) resolve_path "$resolved_parent/$target" "$next_depth" "$next_seen" ;;
        esac
        return 0
    fi

    printf '%s\\n' "$candidate_path"
}

resolved_candidate=$(resolve_path "$candidate" 0)
resolved_root=$(resolve_path "$root" 0)

case "$resolved_candidate" in
    "$resolved_root"|"$resolved_root"/*)
        printf '%s\\n' "$resolved_candidate"
        exit 0
        ;;
esac

printf 'workspace escape: %s\\n' "$resolved_candidate" >&2
exit 111
""".strip()


class VfsHelperScript:
    """VFS 运行时探针"""

    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        self.install_path = (
            PurePosixPath("/tmp/zhenxun-sandbox/bin") / f"{name}-{digest}"
        )

    def install_command(self) -> list[str]:
        tmp_template = f"{self.install_path}.tmp.$$"
        heredoc = (
            f"ZHENXUN_VFS_HELPER_{self.install_path.name.upper().replace('-', '_')}"
        )
        script = f"""
# {_INSTALL_MARKER}
set -eu
dest="$1"
tmp="{tmp_template}"
mkdir -p -- "$(dirname -- "$dest")"
cat > "$tmp" <<'{heredoc}'
{self.content}
{heredoc}
chmod 0555 "$tmp"
rm -f -- "$dest"
mv -f -- "$tmp" "$dest"
"""
        return ["sh", "-c", script.strip(), "sh", str(self.install_path)]


RESOLVE_PATH_HELPER = VfsHelperScript(
    "resolve-workspace-path", _RESOLVE_WORKSPACE_PATH_SCRIPT
)


def coerce_posix_path(path: str | PurePath) -> PurePosixPath:
    if isinstance(path, PurePath):
        path = path.as_posix()
    else:
        path = path.replace("\\", "/")
    return PurePosixPath(path)


class BaseMount(BaseModel, ABC):
    """虚拟挂载抽象基类"""

    target_path: str = Field(description="沙箱内部的挂载目标路径")
    read_only: bool = Field(default=True)

    @abstractmethod
    async def mount(self, driver: Any) -> bool:
        """执行具体的挂载逻辑 (由 Driver 回调)"""
        pass


class RcloneMount(BaseMount):
    """基于 Rclone 的云原生对象存储挂载器"""

    type: Literal["rclone"] = "rclone"
    remote_type: str = Field(description="rclone 后端类型，如 s3, webdav, oss")
    config: dict[str, str] = Field(description="rclone 配置键值对")
    remote_path: str = Field(default="", description="远端路径，如 bucket_name/path")

    async def mount(self, driver: Any) -> bool:
        from zhenxun.services.log import logger

        check = await driver.execute_raw_command("command -v rclone")
        if check.exit_code != 0:
            logger.info(f"[{driver.session_id}] 正在热装配 rclone...")
            await driver.execute_raw_command(
                "apt-get update -qq && apt-get install -y -qq rclone fuse3"
            )

        config_lines = ["[zx_remote]"]
        config_lines.append(f"type = {self.remote_type}")
        for k, v in self.config.items():
            config_lines.append(f"{k} = {v}")
        config_content = "\n".join(config_lines)

        await driver.write_raw_file("/workspace/.rclone.conf", config_content)
        await driver.execute_raw_command(f"mkdir -p {self.target_path}")

        cmd = (
            f"rclone mount zx_remote:{self.remote_path} {self.target_path} "
            f"--config /workspace/.rclone.conf --daemon "
        )
        if self.read_only:
            cmd += "--read-only "

        res = await driver.execute_raw_command(cmd)
        if res.exit_code != 0:
            logger.error(f"Rclone 挂载失败: {res.stderr}")
            return False

        logger.info(f"[{driver.session_id}] 成功挂载 Rclone 存储至 {self.target_path}")
        return True


class LocalDirMount(BaseMount):
    """基于本地目录映射的挂载器"""
    type: Literal["local_dir"] = "local_dir"
    source_path: str = Field(description="宿主机源目录路径")

    async def mount(self, driver: Any) -> bool:
        return True
