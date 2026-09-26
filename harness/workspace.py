from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harness.config import Config
from harness.contracts import HarnessError

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".harness-runs",
    "build",
    "dist",
    ".ssh",
}
EXCLUDED_FILES = {".DS_Store", ".npmrc", ".pypirc", ".netrc"}


def excluded(name: str) -> bool:
    return (
        name in EXCLUDED_DIRS
        or name in EXCLUDED_FILES
        or name == ".env"
        or name.startswith(".env.")
        or name.endswith((".pem", ".key", ".pyc", ".pyo", ".egg-info"))
    )


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _walk_error(error: OSError) -> None:
    raise HarnessError(
        "Cannot enumerate the entire repository; check directory permissions"
    ) from error


def atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".harness-write-", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())


@dataclass(frozen=True)
class FileState:
    content: bytes
    mode: int


class Workspace:
    def __init__(self, source: Path, run_dir: Path, config: Config):
        self.source = source.resolve()
        self.root = run_dir / "workspace"
        self.config = config
        self.omitted: list[str] = []
        if not self.source.is_dir():
            raise HarnessError("Repository must be an existing directory")
        if run_dir.resolve().is_relative_to(self.source):
            raise HarnessError("Output directory must be outside the source repository")
        self.root.mkdir()
        self.initial = self._scan(self.source, record_omitted=True)
        for relative, item in self.initial.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.content)
            os.chmod(target, item.mode)
        if not self.initial:
            raise HarnessError("Repository contains no eligible regular files")

    def _scan(self, root: Path, record_omitted: bool = False) -> dict[str, FileState]:
        result: dict[str, FileState] = {}
        total = 0
        for parent, directories, files in os.walk(root, followlinks=False, onerror=_walk_error):
            directories.sort()
            files.sort()
            kept = []
            for name in directories:
                relative = (Path(parent) / name).relative_to(root).as_posix()
                if excluded(name):
                    if record_omitted:
                        self.omitted.append(relative + "/")
                elif (Path(parent) / name).is_symlink():
                    raise HarnessError(f"Symlink directories are not supported: {relative}")
                else:
                    kept.append(name)
            directories[:] = kept
            for name in files:
                path = Path(parent) / name
                relative = path.relative_to(root).as_posix()
                if excluded(name):
                    if record_omitted:
                        self.omitted.append(relative)
                    continue
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise HarnessError(f"Only regular files are supported: {relative}")
                if info.st_size > self.config.limits.max_file_bytes:
                    raise HarnessError(f"File size limit exceeded: {relative}")
                content = path.read_bytes()
                if len(content) > self.config.limits.max_file_bytes:
                    raise HarnessError(
                        f"File grew beyond the size limit during reading: {relative}"
                    )
                total += len(content)
                if total > self.config.limits.max_repo_bytes:
                    raise HarnessError("Repository byte limit exceeded")
                if len(result) >= self.config.limits.max_files:
                    raise HarnessError("Repository file count limit exceeded")
                result[relative] = FileState(content, stat.S_IMODE(info.st_mode) & 0o777)
        return result

    def scan(self) -> dict[str, FileState]:
        return self._scan(self.root)

    def path(self, value: str) -> Path:
        if not value or "\x00" in value or "\\" in value:
            raise HarnessError("Use a nonempty repository-relative POSIX path")
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(part in (".", "..") for part in value.split("/")):
            raise HarnessError("Absolute paths and dot traversal are forbidden")
        if relative.as_posix() != value:
            raise HarnessError(
                "Use a canonical repository-relative path without repeated separators"
            )
        if any(excluded(part) for part in relative.parts):
            raise HarnessError("Access to excluded paths is forbidden")
        path = self.root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise HarnessError("Symlink access is forbidden")
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise HarnessError("Path leaves the workspace")
        return path

    def is_protected(self, value: str) -> bool:
        return any(
            fnmatch.fnmatchcase(value.casefold(), pattern.casefold())
            for pattern in self.config.protected
        )

    def assert_integrity(self) -> None:
        current = self.scan()
        for name in self.initial.keys() | current.keys():
            if self.is_protected(name) and current.get(name) != self.initial.get(name):
                raise HarnessError(f"Protected verification file changed: {name}")

    def fingerprint(self) -> str:
        state = [
            (name, digest(item.content), item.mode) for name, item in sorted(self.scan().items())
        ]
        return digest(json.dumps(state, separators=(",", ":")).encode())

    def changes(self) -> list[dict[str, Any]]:
        current = self.scan()
        changes = []
        for name in sorted(self.initial.keys() | current.keys()):
            before, after = self.initial.get(name), current.get(name)
            if before == after:
                continue
            changes.append(
                {
                    "path": name,
                    "kind": "added"
                    if before is None
                    else "deleted"
                    if after is None
                    else "modified",
                    "before_sha256": digest(before.content) if before else None,
                    "after_sha256": digest(after.content) if after else None,
                    "before_mode": before.mode if before else None,
                    "after_mode": after.mode if after else None,
                }
            )
        return changes

    def export(self, run_dir: Path) -> list[dict[str, Any]]:
        changes = self.changes()
        current = self.scan()
        patch: list[str] = []
        export_root = run_dir / "changed-files"
        for change in changes:
            name = change["path"]
            before, after = self.initial.get(name), current.get(name)
            if after:
                target = export_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(target, after.content, after.mode)
            try:
                left = before.content.decode("utf-8").splitlines(keepends=True) if before else []
                right = after.content.decode("utf-8").splitlines(keepends=True) if after else []
                if (before and b"\x00" in before.content) or (after and b"\x00" in after.content):
                    raise UnicodeError("binary file")
                difference = difflib.unified_diff(
                    left,
                    right,
                    fromfile="a/" + name if before else "/dev/null",
                    tofile="b/" + name if after else "/dev/null",
                )
                for line in difference:
                    patch.append(
                        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                    )
            except UnicodeError:
                patch.append(f"Binary change: {name} (see manifest and changed-files)\n")
        atomic_write(run_dir / "changes.patch", "".join(patch).encode())
        write_json(run_dir / "changes.json", changes)
        return changes

    def read_text(self, value: str) -> tuple[Path, str, str]:
        path = self.path(value)
        if not path.is_file():
            raise HarnessError(f"Not a regular file: {value}")
        if path.stat().st_size > self.config.limits.max_file_bytes:
            raise HarnessError("File size limit exceeded")
        content = path.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeError as exc:
            raise HarnessError("Only UTF-8 text can be read or edited") from exc
        if "\x00" in text:
            raise HarnessError("Binary files cannot be read or edited")
        return path, text, digest(content)

    def edit(self, value: str, expected: str, old: str, new: str) -> dict[str, str]:
        path = self.path(value)
        if self.is_protected(value):
            raise HarnessError("Editing protected verification files is forbidden")
        if expected == "NEW":
            if path.exists() or old:
                raise HarnessError("Creating a file requires a missing path and empty old_text")
            content = new.encode("utf-8")
            mode = 0o644
        else:
            path, text, actual = self.read_text(value)
            if expected != actual:
                raise HarnessError("Stale file hash: read the current file before editing")
            if not old or text.count(old) != 1:
                raise HarnessError("old_text must be nonempty and match exactly once")
            content = text.replace(old, new, 1).encode("utf-8")
            mode = stat.S_IMODE(path.stat().st_mode)
        if b"\x00" in content:
            raise HarnessError("NUL bytes are not allowed in text edits")
        if len(content) > self.config.limits.max_file_bytes:
            raise HarnessError("Edited file would exceed the file size limit")
        current = self.scan()
        previous_size = len(current[value].content) if value in current else 0
        if sum(len(item.content) for item in current.values()) - previous_size + len(content) > (
            self.config.limits.max_repo_bytes
        ):
            raise HarnessError("Edit would exceed the repository byte limit")
        if value not in current and len(current) >= self.config.limits.max_files:
            raise HarnessError("Edit would exceed the repository file count limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, content, mode)
        return {"path": value, "sha256": digest(content)}

    def delete(self, value: str, expected: str) -> dict[str, str]:
        if self.is_protected(value):
            raise HarnessError("Deleting protected verification files is forbidden")
        path, _, actual = self.read_text(value)
        if actual != expected:
            raise HarnessError("Stale file hash: read the current file before deleting")
        path.unlink()
        return {"deleted": value}
