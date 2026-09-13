"""递归记录源码摘要及运行时间，供攻击和问答实验共同使用。"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from .paths import PROJECT_ROOT
from .io import sha256_file, stable_key

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def implementation_snapshot() -> dict[str, Any]:
    """记录实际执行代码内容，供无独立版本库的运行之间核对。"""
    project_root = PROJECT_ROOT
    candidates = [project_root / "pyproject.toml", project_root / "tools" / "run_cli.py"]
    candidates.extend(sorted((project_root / "src" / "frequency_rag").rglob("*.py")))
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in candidates
        if path.is_file()
    ]
    return {
        "created_at": utc_timestamp(),
        "project_root": str(project_root),
        "files": files,
        "content_identity": stable_key(
            f"{record['path']}:{record['sha256']}" for record in files
        ),
    }
