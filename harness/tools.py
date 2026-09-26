from __future__ import annotations

import fnmatch
from typing import Any, Callable

from harness.contracts import Action, HarnessError
from harness.workspace import Workspace


def _string(description: str, maximum: int = 8192) -> dict[str, Any]:
    return {"type": "string", "description": description, "maxLength": maximum}


def _integer(description: str, minimum: int, maximum: int) -> dict[str, Any]:
    return {
        "type": "integer",
        "description": description,
        "minimum": minimum,
        "maximum": maximum,
    }


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool(
        "list_files",
        "List eligible files, at most 100 per page.",
        {
            "pattern": _string("Repository-relative glob, e.g. * or *.py", 256),
            "offset": _integer("Zero-based pagination offset", 0, 100_000),
        },
    ),
    _tool(
        "read_file",
        "Read UTF-8 text with 1-based line numbers and a hash for editing.",
        {
            "path": _string("Repository-relative file path", 512),
            "start_line": _integer("First line, inclusive", 1, 10_000_000),
            "end_line": _integer("Last line, inclusive; no more than 200 lines", 1, 10_000_000),
        },
    ),
    _tool(
        "search",
        "Search literal, case-sensitive text; returns at most 50 matching lines.",
        {
            "query": _string("Nonempty literal search text", 512),
            "pattern": _string("Repository-relative glob, e.g. *.py", 256),
            "offset": _integer("Zero-based match pagination offset", 0, 100_000),
        },
    ),
    _tool(
        "read_chunk",
        "Read a character slice of a UTF-8 file or known check log, including very long lines.",
        {
            "source": {"type": "string", "enum": ["file", "check_log"], "maxLength": 9},
            "path": _string("Repository-relative file path or exact recorded check log path", 512),
            "offset": _integer("Zero-based character offset", 0, 100_000_000),
            "length": _integer("Maximum characters to return", 1, 12_000),
        },
    ),
    _tool(
        "edit_file",
        "Replace one exact occurrence in a previously read file, or create a file.",
        {
            "path": _string("Repository-relative file path", 512),
            "expected_sha256": _string("Hash from read_file, or NEW for a missing file", 64),
            "old_text": _string("Exact unique text to replace; empty only for a new file", 16_000),
            "new_text": _string("Replacement content", 16_000),
        },
    ),
    _tool(
        "delete_file",
        "Delete a previously read, non-protected file.",
        {
            "path": _string("Repository-relative file path", 512),
            "expected_sha256": _string("Hash from read_file", 64),
        },
    ),
    _tool(
        "record_plan",
        "Persist the current concise plan, findings, and unresolved issues.",
        {
            "plan": _string("Plan and important facts to retain across context trimming", 4000),
        },
    ),
    _tool(
        "run_check",
        "Execute one check configured by the user. Cannot choose arbitrary commands.",
        {
            "name": _string("Configured check name", 40),
        },
    ),
    _tool(
        "read_check_log",
        "Read retained output from a recorded check, up to 200 lines.",
        {
            "log": _string("Exact log path returned by run_check", 512),
            "start_line": _integer("First line, inclusive", 1, 10_000_000),
            "end_line": _integer("Last line, inclusive", 1, 10_000_000),
        },
    ),
    _tool(
        "finish",
        "Request completion. The harness independently verifies before deciding status.",
        {
            "summary": _string(
                "Summary of changes, reasoning, and any remaining limitations", 4000
            ),
        },
    ),
]


def validate_action(action: Action) -> None:
    specification = next((tool for tool in TOOLS if tool["name"] == action.name), None)
    if specification is None:
        raise HarnessError(f"Unknown tool: {action.name}")
    properties = specification["parameters"]["properties"]
    if not isinstance(action.arguments, dict) or set(action.arguments) != set(properties):
        raise HarnessError(f"Invalid arguments for {action.name}; all schema fields are required")
    for name, schema in properties.items():
        value = action.arguments[name]
        if schema["type"] == "string":
            if not isinstance(value, str) or len(value) > schema["maxLength"] or "\x00" in value:
                raise HarnessError(f"Invalid string argument: {name}")
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise HarnessError(f"Invalid Unicode string argument: {name}") from exc
            if "enum" in schema and value not in schema["enum"]:
                raise HarnessError(f"Invalid choice for argument: {name}")
        elif type(value) is not int or not schema["minimum"] <= value <= schema["maximum"]:
            raise HarnessError(f"Invalid integer argument: {name}")


class Tools:
    def __init__(
        self,
        workspace: Workspace,
        check: Callable[[str], dict[str, Any]],
        record_plan: Callable[[str], None],
        read_log: Callable[[str, int, int, bool], dict[str, Any]],
    ):
        self.workspace, self.check, self.record_plan = workspace, check, record_plan
        self.read_log = read_log

    def execute(self, action: Action) -> dict[str, Any]:
        validate_action(action)
        args = action.arguments
        if action.name == "list_files":
            names = [
                name
                for name in sorted(self.workspace.scan())
                if fnmatch.fnmatchcase(name, args["pattern"])
            ]
            end = args["offset"] + 100
            return {
                "files": names[args["offset"] : end],
                "total": len(names),
                "next_offset": end if len(names) > end else None,
            }
        if action.name == "read_file":
            start, end = args["start_line"], args["end_line"]
            if end < start or end - start >= 200:
                raise HarnessError("Read an inclusive range of at most 200 lines")
            _, text, sha = self.workspace.read_text(args["path"])
            lines = text.splitlines(keepends=True)
            if start > max(1, len(lines)):
                raise HarnessError("start_line is beyond the end of the file")
            content = "".join(lines[start - 1 : end])
            truncated = len(content) > 12_000
            return {
                "path": args["path"],
                "sha256": sha,
                "total_lines": len(lines),
                "start_line": start,
                "end_line": min(end, len(lines)),
                "content": content[:12_000],
                "truncated": truncated,
            }
        if action.name == "read_chunk":
            if args["source"] == "check_log":
                return self.read_log(args["path"], args["offset"], args["length"], True)
            _, text, sha = self.workspace.read_text(args["path"])
            if args["offset"] > len(text):
                raise HarnessError("Character offset is beyond the end of the file")
            end = args["offset"] + args["length"]
            return {
                "path": args["path"],
                "sha256": sha,
                "content": text[args["offset"] : end],
                "total_characters": len(text),
                "next_offset": end if end < len(text) else None,
            }
        if action.name == "search":
            if not args["query"]:
                raise HarnessError("Search query cannot be empty")
            matches = []
            total = 0
            end = args["offset"] + 50
            skipped_binary = 0
            for name, item in sorted(self.workspace.scan().items()):
                if not fnmatch.fnmatchcase(name, args["pattern"]):
                    continue
                try:
                    text = item.content.decode("utf-8")
                    if "\x00" in text:
                        raise UnicodeError("binary")
                except UnicodeError:
                    skipped_binary += 1
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if args["query"] in line:
                        if args["offset"] <= total < end:
                            matches.append(
                                {
                                    "path": name,
                                    "line": number,
                                    "text": line[:300],
                                    "truncated": len(line) > 300,
                                }
                            )
                        total += 1
            return {
                "matches": matches,
                "total": total,
                "next_offset": end if total > end else None,
                "skipped_binary_files": skipped_binary,
            }
        if action.name == "edit_file":
            return self.workspace.edit(
                args["path"],
                args["expected_sha256"],
                args["old_text"],
                args["new_text"],
            )
        if action.name == "delete_file":
            return self.workspace.delete(args["path"], args["expected_sha256"])
        if action.name == "record_plan":
            if not args["plan"].strip():
                raise HarnessError("Plan must not be empty")
            self.record_plan(args["plan"])
            return {"recorded": True}
        if action.name == "run_check":
            return self.check(args["name"])
        if action.name == "read_check_log":
            return self.read_log(args["log"], args["start_line"], args["end_line"], False)
        raise HarnessError("finish is handled only by the run controller")
