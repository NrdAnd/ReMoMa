"""Atomic, content-identified artifacts for long-running experiments."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


@contextmanager
def exclusive_lock(path: str | Path):
    """Fail explicitly on concurrent writers; never guess whether a lock is stale."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another writer or an interrupted process owns {path}. "
            "Check its recorded PID/host before removing a stale lock."
        ) from exc
    try:
        import socket
        with handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)
