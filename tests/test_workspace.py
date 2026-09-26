from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from harness.config import Config, Limits
from harness.contracts import HarnessError
from harness.workspace import Workspace, digest


def workspace(
    tmp_path: Path, content: bytes = b"one\ntwo\n", config: Config | None = None
) -> Workspace:
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    (source / "code.py").write_bytes(content)
    return Workspace(source, run, config or Config())


def test_edit_is_atomic_and_preserves_original_and_mode(tmp_path: Path) -> None:
    work = workspace(tmp_path, b"one\r\ntwo\r\n")
    path = work.root / "code.py"
    path.chmod(0o755)
    original = path.read_bytes()
    work.edit("code.py", digest(original), "two", "three")
    assert path.read_bytes() == b"one\r\nthree\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o755
    assert (work.source / "code.py").read_bytes() == original
    assert not list(work.root.glob(".harness-write-*"))


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/etc/passwd",
        "a/../../outside",
        "./code.py",
        ".env",
        "a/.env.local",
        ".git/config",
        "a\\..\\outside",
        "",
        "a\x00b",
    ],
)
def test_disallowed_paths(tmp_path: Path, path: str) -> None:
    work = workspace(tmp_path)
    with pytest.raises(HarnessError):
        work.path(path)


def test_symlinks_are_rejected_in_source_and_tools(tmp_path: Path) -> None:
    work = workspace(tmp_path)
    (work.root / "link").symlink_to(work.source / "code.py")
    with pytest.raises(HarnessError, match="Symlink"):
        work.path("link")
    with pytest.raises(HarnessError, match="regular"):
        work.scan()
    run = tmp_path / "another-run"
    run.mkdir()
    (work.source / "link").symlink_to(work.source / "code.py")
    with pytest.raises(HarnessError):
        Workspace(work.source, run, Config())


def test_secret_names_are_not_copied(tmp_path: Path) -> None:
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    for name in ("code.py", ".env", ".env.local", "private.pem", ".npmrc"):
        (source / name).write_text("data")
    work = Workspace(source, run, Config())
    assert list(work.initial) == ["code.py"]
    assert len(work.omitted) == 4


def test_stale_and_ambiguous_edits_do_not_write(tmp_path: Path) -> None:
    work = workspace(tmp_path, b"same same\n")
    original = work.fingerprint()
    with pytest.raises(HarnessError, match="Stale"):
        work.edit("code.py", "invalid", "same", "different")
    with pytest.raises(HarnessError, match="exactly once"):
        work.edit("code.py", digest(b"same same\n"), "same", "different")
    assert work.fingerprint() == original


def test_create_delete_and_export_include_exact_bytes(tmp_path: Path) -> None:
    work = workspace(tmp_path, b"one")
    work.edit("nested/space name.py", "NEW", "", "new")
    work.delete("code.py", digest(b"one"))
    changes = work.export(tmp_path / "run")
    assert [(item["path"], item["kind"]) for item in changes] == [
        ("code.py", "deleted"),
        ("nested/space name.py", "added"),
    ]
    assert (tmp_path / "run/changed-files/nested/space name.py").read_bytes() == b"new"
    assert "\\ No newline at end of file" in (tmp_path / "run/changes.patch").read_text()
    assert json.loads((tmp_path / "run/changes.json").read_text()) == changes


def test_existing_creation_and_protected_files_rejected(tmp_path: Path) -> None:
    work = workspace(tmp_path)
    with pytest.raises(HarnessError, match="missing"):
        work.edit("code.py", "NEW", "", "replacement")
    with pytest.raises(HarnessError, match="protected"):
        work.edit("tests/test_new.py", "NEW", "", "pass")
    with pytest.raises(HarnessError, match="protected"):
        work.delete("tests/test_new.py", "irrelevant")


def test_protected_file_mutation_by_command_detected(tmp_path: Path) -> None:
    work = workspace(tmp_path)
    (work.root / "tests").mkdir()
    (work.root / "tests/test_injected.py").write_text("pass")
    with pytest.raises(HarnessError, match="Protected"):
        work.assert_integrity()


@pytest.mark.parametrize("content", [b"\xff\xfe", b"abc\x00def"])
def test_binary_read_rejected(tmp_path: Path, content: bytes) -> None:
    work = workspace(tmp_path, content)
    with pytest.raises(HarnessError):
        work.read_text("code.py")


def test_file_and_repository_limits_apply_before_edit(tmp_path: Path) -> None:
    work = workspace(tmp_path, b"abc", Config(limits=Limits(max_file_bytes=4, max_repo_bytes=4)))
    with pytest.raises(HarnessError, match="file size"):
        work.edit("code.py", digest(b"abc"), "abc", "abcde")
    with pytest.raises(HarnessError, match="repository byte"):
        work.edit("new.py", "NEW", "", "ab")
    assert not (work.root / "new.py").exists()


def test_file_count_limit_applies_to_new_files(tmp_path: Path) -> None:
    work = workspace(tmp_path, config=Config(limits=Limits(max_files=1)))
    with pytest.raises(HarnessError, match="file count"):
        work.edit("new.py", "NEW", "", "abc")


def test_snapshot_rejects_oversized_file(tmp_path: Path) -> None:
    with pytest.raises(HarnessError, match="File size"):
        workspace(tmp_path, b"too long", Config(limits=Limits(max_file_bytes=2)))


def test_fifo_rejected_without_blocking(tmp_path: Path) -> None:
    work = workspace(tmp_path)
    os.mkfifo(work.root / "pipe")
    with pytest.raises(HarnessError, match="regular"):
        work.scan()


def test_failed_atomic_replace_preserves_existing_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = workspace(tmp_path)
    original = (work.root / "code.py").read_bytes()

    def fail_replace(source: object, target: object) -> None:
        raise OSError("simulated failure before replace")

    monkeypatch.setattr("harness.workspace.os.replace", fail_replace)
    with pytest.raises(OSError):
        work.edit("code.py", digest(original), "two", "three")
    assert (work.root / "code.py").read_bytes() == original
    assert not list(work.root.glob(".harness-write-*"))


def test_case_alias_of_protected_file_cannot_be_edited(tmp_path: Path) -> None:
    work = workspace(tmp_path)
    with pytest.raises(HarnessError, match="protected"):
        work.edit("TESTS/TEST_NEW.PY", "NEW", "", "pass")


@pytest.mark.parametrize("name", ["a//b.py", "code.py/"])
def test_noncanonical_paths_rejected(tmp_path: Path, name: str) -> None:
    work = workspace(tmp_path)
    with pytest.raises(HarnessError, match="canonical"):
        work.path(name)
