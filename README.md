# AI Coding Harness

A first runnable milestone for the coding-harness brief: a Python CLI that
inspects a repository, plans changes, edits a copy, runs configured checks,
recovers from tool failures, and exports verification evidence.

**This is a development harness, not yet a competition-ready autonomous system.**
The offline demo is scripted. It proves the execution pipeline, not a model's
ability to solve unfamiliar tasks. Real model requests require your explicit
permission and your own locally configured credentials.

The organizers have announced **DeepSeek and Qwen** for final evaluation and
**`AI_API_KEY`** for runtime credentials. Both model families have a configurable
Chat Completions adapter. Exact evaluation model IDs, hosting endpoints, and
supported model options still need confirmation; a model family is not an API
specification.

## Requirements and installation

- Python 3.9 or newer. Python 3.11+ is recommended for new installations.
- macOS or Linux for command execution.
- A tool-capable model for live runs from a DeepSeek/Qwen host
  supporting non-streaming OpenAI-compatible Chat Completions. Model ID and
  provider are selected independently; there is no automatic model fallback.
- No Docker, Node.js, database, or API key is needed for the offline demo.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The resolved development environment is recorded in
[requirements-dev.lock](requirements-dev.lock). To reproduce those versions,
install with `python -m pip install -c requirements-dev.lock -e '.[dev]'`.
The lock was validated on macOS with Python 3.9; validate other platforms rather
than assuming identical native wheels. The application wheel is built with
`python -m pip wheel --no-deps --wheel-dir dist .`.

## Run the offline demonstration

From the project directory, with the virtual environment activated:

```sh
ai-harness demo --allow-host-execution --output .harness-runs
```

This command creates a temporary non-sensitive repository with a closed-interval
boundary bug. It runs seven baseline tests, follows six scripted agent actions,
and independently reruns the checks before reporting `verified`.

Expected evidence:

- Baseline: seven tests execute, three fail.
- Change: make both interval endpoints inclusive.
- Agent-requested verification: seven tests pass.
- Final controller verification: seven tests pass again.
- Original input files remain unchanged.
- Output is labelled `scripted-demo-not-an-llm`, with zero model token usage.

The copied workspace and artifacts persist under the printed run directory.
The temporary input repository is removed when the demo finishes.

## Runtime credential: AI_API_KEY

**Every live provider reads only `AI_API_KEY`.** The SDK receives this value
explicitly, so it cannot fall back to `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
`QWEN_API_KEY`, or `DASHSCOPE_API_KEY`. The variable is read when creating each
live client, not saved in source code or configuration. Missing, blank, or
whitespace-containing credentials fail before any model request.

Do not paste your key into chat, source files, configuration JSON, or a shell
command that will be saved in history. Do not create a credential `.env` file;
this application does not load one. Enter your development key privately in
your terminal (Bash or Zsh):

```sh
printf 'API key for the selected provider (hidden input): '
read -r -s AI_API_KEY
printf '\n'
export AI_API_KEY
```

For evaluation, the judges export `AI_API_KEY` themselves. **No key change in
the code, Makefile, documentation, or any committed file is needed.** Child
processes inherit the exported variable normally; the Makefile intentionally
does not interpolate or echo its value. Repository checks run with a separate
allowlisted environment that excludes it.

The value must belong to the selected provider and endpoint: a DeepSeek key
does not authenticate against a Qwen hosting service. This cannot be inferred from the key.
When switching providers, replace the variable's value privately and select
the corresponding endpoint/model. Run `unset AI_API_KEY` when finished locally.

## First live test with DeepSeek

With the virtual environment activated and a DeepSeek-issued key in
`AI_API_KEY`, create a fresh non-sensitive example repository:

```sh
EXAMPLE_DIR="$(mktemp -d)"
ai-harness init-demo --destination "$EXAMPLE_DIR/repo"

ai-harness run \
  --provider deepseek \
  --base-url https://api.deepseek.com/v1 \
  --model deepseek-chat \
  --repo "$EXAMPLE_DIR/repo" \
  --task "Fix contains so both closed-interval endpoints are included; preserve invalid-bound checks." \
  --config examples/checks.json \
  --allow-model-upload \
  --allow-host-execution \
  --output .harness-runs
```

`deepseek-chat` is an explicit development model selection, not an assertion
about the judges' model. Confirm that this ID is available at your endpoint,
or replace it with the documented tool-capable model ID. These are paid API
requests. The harness cannot validate account access without a live request.
Start with this example, a provider-side spending limit, and the run budgets.
The example input directory can be removed after reviewing the saved results.

For an existing **non-sensitive, trusted** repository, change `--repo`, supply
the task through `--task` or `--task-file`, and configure that repository's real
checks. The artifact output directory must be outside the input repository.
`init-demo` refuses to overwrite an existing destination.

## Provider selection and evaluation

| `--provider` | Protocol | Endpoint configuration | Credential |
| --- | --- | --- | --- |
| `deepseek` | Chat Completions | Explicit `--base-url` required | `AI_API_KEY` |
| `qwen` | Chat Completions | Explicit `--base-url` required | `AI_API_KEY` |

`--provider` and `--base-url` are mandatory for every live run. The API key alone
does not select a provider, model, or endpoint.

The `openai` Python package is the API client library for these compatible
endpoints; it does not select OpenAI's service or require an OpenAI-issued key.
There is no separate OpenAI model adapter or live OpenAI provider option.

For Qwen, use `--provider qwen --model YOUR_QWEN_MODEL_ID --base-url YOUR_HTTPS_API_BASE`.
For example, Alibaba Cloud's Singapore OpenAI-compatible base URL is
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. This is a development
example, **not an assumed evaluation host or region**. Other hosts may require
different model identifiers, tool templates, or reasoning settings.

For both Chat adapters:

- Supply a trusted HTTPS **base** URL, not a full `/chat/completions` URL.
  Credentials in URLs, query strings, fragments, and cleartext HTTP are rejected.
  CLI-created clients do not follow redirects or forward ambient OpenAI
  organization/project headers to another provider.
- Internal tool definitions are translated to Chat Completions
  function envelopes. No server-side `strict`, `store`, or parallel-call option
  is assumed. Arguments are still validated by the harness before execution.
- `tool_choice="auto"` avoids assuming forced-tool support. A plain-text answer,
  invalid function JSON, truncated response, or multiple calls executes nothing.
  Up to two corrective responses are allowed, with actual reported usage counted;
  three unusable responses stop the run. Refusals are not treated as requests to
  bypass the provider's restrictions.
- Extra reasoning fields are not executable instructions and are not stored or
  replayed. Requests are stateless; this is not native multi-turn reasoning
  history preservation.
- `--qwen-thinking provider-default` sends no thinking override. Explicit
  `enabled` or `disabled` sends Qwen's `enable_thinking` field; use it only when
  supported by the host and permitted by evaluation rules. Some Qwen models
  require streaming in thinking mode; that mode is not supported here yet.
  The Qwen flag cannot be used with DeepSeek.
- `--token-limit-parameter max_tokens` is the Chat default.
  On Qwen, this can limit only the answer, excluding reasoning. Where the host
  supports it, select `max_completion_tokens` to bound both. The adapter never
  silently changes the field after a failed request. Limits and semantics must
  match the final model's documentation.

Run DeepSeek and Qwen evaluations separately with the same task/check
configuration and budgets. Results record the provider, protocol, base URL,
selected model, and model controls, but never the credential. API compatibility
tests use mocked HTTP, not real model benchmarks or a claim of competition readiness.

References: [DeepSeek function calling](https://api-docs.deepseek.com/guides/function_calling),
[Qwen function calling](https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling),
and [Qwen Chat Completions parameters](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions).

## Makefile entrypoints

```sh
make setup
make test
make lint
make demo
```

`make test` is offline and needs no key. `make demo` opts into running the bundled,
trusted fixture on this host; it is still scripted, not a live model.

The live entrypoint accepts the same CLI arguments, with credentials inherited
only from the exported environment:

```sh
make run ARGS='--provider deepseek --base-url https://api.deepseek.com/v1 --model deepseek-chat --repo /path/to/trusted/repo --task "Fix the described bug" --config examples/checks.json --allow-model-upload --allow-host-execution'
```

Replace the non-secret paths, task, and model as appropriate. Do not pass a key
in `ARGS` or as a Make command-line assignment. The targets are convenience
entrypoints; the organizers' exact task-input and submission protocol is still
needed before claiming compatibility with their runner.

## Execution and data boundaries

**Host execution is not a sandbox.** A Python test, build script, or dependency
can execute arbitrary code, access the host, or make network requests. Only run
trusted repositories in this mode. A copied directory is not OS-level isolation.
Do not use this version for adversarial or untrusted evaluation repositories;
container or equivalent isolation is a prerequisite for that use.

Two independent opt-ins are required:

1. `--allow-model-upload` permits transmitting the task, filenames, selected
   file excerpts, and selected check output to the selected provider. Use only code you are
   authorized to share, and never confidential repositories.
2. `--allow-host-execution` permits configured checks to run on the host.
   Without it, configurations containing checks are rejected before model calls.

Additional controls:

- Model tools cannot choose arbitrary shell commands. `run_check` selects a
  named command from the trusted configuration; arguments are passed without a
  shell. Dependencies are not installed automatically.
- Check processes receive a small environment allowlist, a private `HOME` and
  temporary directory, and no inherited API key. This reduces accidental leaks;
  it does not prevent hostile code from accessing the host.
- File tools reject traversal, symlinks, special files, stale edit hashes,
  ambiguous replacements, and oversized files.
- Common test paths are protected against creation, modification, and deletion.
  Protection is case-insensitive to work consistently on macOS filesystems.
  Add repository-specific evaluator/configuration paths to `protected`.
- Repository scans omit version-control metadata, common dependency/cache
  directories, `.env` files, and some common credential filenames.
  The complete omission list is persisted. **This is not a secret scanner.**
- Git ignore rules are not interpreted. Eligible untracked files are copied,
  including local changes. Review the source before authorizing model upload.
- A check that changes eligible repository files invalidates its evidence.

Artifacts contain code and command output. The run directory is private to the
current user by default; do not publish it without reviewing its contents.

## Configuration

See [examples/checks.json](examples/checks.json). All configured checks are
required. Unknown fields, invalid limits, and duplicate check names are errors.

```json
{
  "checks": [
    {
      "name": "tests",
      "argv": ["{python}", "-m", "unittest", "discover", "-s", "tests", "-v"],
      "kind": "unittest"
    }
  ],
  "protected": ["pyproject.toml", "setup.cfg"],
  "limits": {
    "max_steps": 24,
    "max_total_tokens": 50000,
    "max_output_tokens": 4096,
    "max_seconds": 180,
    "command_timeout": 30
  }
}
```

`{python}` expands only when it is a complete argument, to the harness's Python
interpreter. Supply an absolute executable path for a different environment.
Inherited environment variables other than `PATH`, `LANG`, `LC_ALL`, and `TZ`
are not forwarded to checks.

Supported check kinds:

| Kind | Success criterion |
| --- | --- |
| `unittest` | Exit zero and a standard summary showing at least one non-skipped test |
| `pytest` | Exit zero and a standard summary showing at least one passed test |
| `command` | Exit zero; use `success_pattern` to require particular output |

An optional `success_pattern` regular expression must also match. Test counts
are parsed from standard runner output, not an independent anti-tampering
oracle. Custom reporters and third-party test frameworks require a suitable
adapter or an explicit `command` check; do not pretend a generic command ran tests.

Additional default limits are 128 KiB per command log, 24,000 context characters,
1 MiB per file, 16 MiB of eligible repository content, and 2,000 files. File
reads, search pages, and log reads are bounded. `read_chunk` handles long lines.

Token accounting includes provider-reported usage. Before each request, a
conservative byte-based input estimate reserves room for output; failed requests
without usage consume an estimated reservation and are labelled accordingly.
This is not an exact dollar cap or a tokenizer-independent billing guarantee.
All attempts count towards the step limit, including at most two retries of
transient provider failures. SDK retries are disabled.
`max_output_tokens` is the configured per-call output ceiling. For Chat models
it is mapped to the selected token-limit parameter described above. Whether
reasoning is included depends on that parameter and the host. Increase the
limit within your total budget when a model cannot finish a tool call;
incomplete responses are rejected, not executed.

Command deadlines terminate the owned process group. The overall deadline is
checked around actions and reserves time before model calls for verification.
SDK network timeouts are per I/O operation, not a hard whole-process watchdog;
slow filesystems or unusual network behavior can exceed the nominal wall budget.

## Architecture

```text
CLI and trusted configuration
              |
              v
preflight -> baseline -> inspect / plan / edit / check -> final verification
                            ^             |
                            +-- feedback -+
              |
              v
copied workspace + exported changes + persistent evidence

Provider boundary: Model.choose_action(...) -> ModelReply(Action, usage)
Provider adapter: DeepSeek/Qwen Chat Completions
Offline demo: scripted adapter
```

- [engine.py](harness/engine.py): controller, context, budgets, recovery, completion.
- [chat_model.py](harness/chat_model.py): DeepSeek/Qwen-compatible protocol and model controls.
- [model_common.py](harness/model_common.py): runtime credential and shared response validation.
- [tools.py](harness/tools.py): tool schemas and local argument validation.
- [workspace.py](harness/workspace.py): snapshots, edits, hashes, exports.
- [runner.py](harness/runner.py): named checks, process cleanup, evidence.
- [contracts.py](harness/contracts.py): provider-independent interface.

Each model request is stateless. The task, plan, current changes and check
evidence are pinned; recent observations fit into the remaining context.
Omitted or truncated observations are marked. The model can reread files and
logs rather than relying on lossy summaries. If pinned state cannot fit, the
run stops explicitly instead of dropping requirements.

## Results and artifacts

| Status | Exit code | Meaning |
| --- | --- | --- |
| `verified` | 0 | Every configured check passed on the final copied workspace |
| `failed` | 1 | An unrecoverable provider, tool, integrity, or I/O failure |
| CLI/configuration error | 2 | Invalid invocation, missing key, or missing permission |
| `unverified` | 3 | Agent finished, but no verification checks were configured |
| `budget_exhausted` | 4 | Step, token, time, context, or repetition limit reached |
| `cancelled` | 130 | Run interrupted by the user |

`verified` means **the configured checks passed**, not that all possible
requirements or hidden tests are satisfied. A model's completion claim never
substitutes for check execution. Pre-existing failures are recorded in the
baseline, not silently ignored. A later edit requires fresh final verification.

Each run contains:

- `request.json`: task, provider identity, and non-secret run configuration.
- `source-manifest.json`: eligible starting files, hashes, permissions, exclusions.
- `workspace/`: final working copy; the original input is not edited by file tools.
- `events.jsonl`: action and tool-result history.
- `checkpoint.json`: latest durable controller state.
- `logs/`: bounded command output.
- `changes.json`: added/modified/deleted paths, hashes and permissions.
- `changed-files/`: exact final bytes of added/modified files.
- `changes.patch`: human-readable unified text diff.
- `result.json`: final outcome, checks, logs, usage, and artifact locations.

Review changes before applying them to your repository. The manifest and exact
file copies are authoritative: the text diff is not a complete representation
of binary or permission-only changes, and unusual filenames may need special
handling. Automatic application to the original repository is intentionally
not implemented.

Checkpoints are diagnostic in this milestone, **not automatic crash-resume**.
After a crash, inspect the checkpoint and copied files; do not blindly replay
state-changing actions. Rerun from a known starting state.

## Development and validation

```sh
python -m pytest
python -m ruff check harness tests
python -m ruff format --check harness tests
python -m mypy harness
```

Tests use scripted models and mocked HTTP transports, never paid API calls.
They cover the full pipeline, fresh-copy replay, independent boundary checks,
provider failures, malformed arguments, file protections, zero-test detection,
stale verification, output/time limits, process cleanup, and budget exhaustion.

## Remaining work before competition use

1. Validate the live loop with the development DeepSeek key configured privately.
2. Add real execution isolation before running untrusted repositories.
3. Match the organizers' model/API, runner format, limits, and evaluator contract.
4. Add representative multi-file tasks and measure correctness versus cost.
5. Add dependency/environment provisioning, new regression-test support without
   weakening protected checks, safe crash recovery, and further context strategies
   only as justified by those evaluations.

Replacing your development credential with the judges' credential requires only
runtime `AI_API_KEY` injection. Different endpoints, model IDs, or provider
protocols still require the correct configuration and end-to-end validation.
No competition-model performance or first-run guarantee is claimed.
